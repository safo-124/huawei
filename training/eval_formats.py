"""Format suite: the standard evaluation (and augmentation source) of every model.

The hidden data may present the signals in a different composition than the
kit (for example several sessions of one person in a single input, or short
excerpts). Every model is therefore scored on four input formats:

  full    each recording as it is in the kit
  joined  all sessions of a person joined into one continuous input (random order)
  crop    each recording cut into consecutive pieces of random length (300 to 900 s)
  jcrop   the joined input cut into such pieces (pieces can straddle two sessions)

robust score = mean of the four macro-F1 values; the minimum is reported too.

Protocol: GroupKFold(5) by user_id (seed 0), 3 model seeds. Kalman noise is
tuned inside each training fold. One decoder (per-class offsets) is tuned per
outer fold on the other folds' out-of-fold predictions pooled over the four
formats, because at inference the format is unknown.

Inputs are assembled from the cached per-second aggregates (sums and counts),
so a format costs no extra pass over the raw rows. Pieces start on whole
seconds of the source recording; the only difference to cutting raw rows is
the handful of sample-to-sample differences across a cut.

Training views (augmentation, --mix): originals, joined (all pairs and the
triple of a training person, random order), crop and jcrop, never mixing
people. --dropout blanks one signal in a share of the training views (whole
view or a 1 to 5 min stretch).

Usage:
    python training/eval_formats.py --name F --groups raw kalman norm resp ctr
    python training/eval_formats.py --name F_aug --mix orig:0.4,joined:0.3,crop:0.15,jcrop:0.15
Results: artifacts/robust/<name>.json
"""

import argparse
import itertools
import json
import time
import zlib

import numpy as np

from common import M, ROOT, user_folds
import train as T
import tune_decoder as TD

OUT = ROOT / "artifacts" / "robust"
OUT.mkdir(parents=True, exist_ok=True)
FORMATS = ("full", "joined", "crop", "jcrop")
SIGNALS = ("EDA", "HR", "IBI", "Temperature", "BVP")
BEAT_KEYS = ("beat_t", "beat_d", "beat_miss")
SEEDS = (0, 1, 2)
TOTAL_ROWS = 90_000
F_GROUPS = ["raw", "kalman", "norm", "resp", "ctr"]
BLANK = {
    "HR": (("hr_c", "hr_s", "hr_q"), ()),
    "EDA": (("eda_c", "eda_s", "eda_q", "eda_rise"), ()),
    "Temperature": (("tmp_c", "tmp_s"), ()),
    "BVP": (("bvp_c", "bvp_s", "bvp_q", "bvp_d", "bvp_dc"), ("bvp_rng",)),
    "IBI": (("ibi_c", "ibi_s", "ibi_q", "ibi_miss", "ibi_gap", "ibi_dd", "ibi_ddc"), ()),
}


def _rng(*key):
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


# --------------------------------------------------------------------------
# Views: a view is a tuple of segments (recording index, first second, end second)
# --------------------------------------------------------------------------
def partition(n, rng, lo=300, hi=900):
    """Consecutive pieces of random length in [lo, hi] covering 0..n."""
    cuts, a = [], 0
    while n - a > hi:
        a += int(rng.integers(lo, hi + 1))
        cuts.append(a)
    if cuts and n - cuts[-1] < lo // 2:  # a very short tail joins the previous piece
        cuts.pop()
    edges = [0] + cuts + [n]
    return list(zip(edges[:-1], edges[1:]))


def cut_view(view, a, e):
    """Seconds [a, e) of a view, as a view."""
    out, off = [], 0
    for i, s0, s1 in view:
        ln = s1 - s0
        lo, hi = max(a, off), min(e, off + ln)
        if hi > lo:
            out.append((i, s0 + lo - off, s0 + hi - off))
        off += ln
    return tuple(out)


def view_len(view):
    return sum(e - a for _, a, e in view)


class Suite:
    def __init__(self, recs):
        self.recs = recs
        self.n = [r["base"][1] for r in recs]
        self.by_user = {}
        for i, r in enumerate(recs):
            self.by_user.setdefault(r["uid"], []).append(i)
        self.cache = {}

    def whole(self, i):
        return ((i, 0, self.n[i]),)

    def joined(self, idx, rng):
        return tuple(self.whole(i)[0] for i in rng.permutation(idx))

    def eval_views(self, fmt, users):
        """Views of a format that cover every recording of `users` exactly once."""
        out = []
        for u in users:
            idx = self.by_user[u]
            rng = _rng("eval", fmt, u)
            if fmt == "full":
                out += [self.whole(i) for i in idx]
            elif fmt == "joined":
                out.append(self.joined(idx, rng))
            elif fmt == "crop":
                for i in idx:
                    out += [((i, a, e),) for a, e in partition(self.n[i], rng)]
            else:
                j = self.joined(idx, rng)
                out += [cut_view(j, a, e) for a, e in partition(view_len(j), rng)]
        return out

    def train_views(self, kind, users):
        out = []
        for u in users:
            idx = self.by_user[u]
            rng = _rng("train", kind, u)
            if kind == "orig":
                out += [self.whole(i) for i in idx]
            elif kind == "joined":
                for k in range(2, len(idx) + 1):
                    for comb in itertools.combinations(idx, k):
                        out.append(self.joined(list(comb), rng))
            elif kind == "crop":
                for i in idx:
                    out += [((i, a, e),) for a, e in partition(self.n[i], rng)]
            elif kind == "jcrop" and len(idx) > 1:
                j = self.joined(idx, rng)
                out += [cut_view(j, a, e) for a, e in partition(view_len(j), rng)]
        return out

    # ---------------------------------------------------------------- bases
    def base(self, view, blank=None):
        parts = []
        for i, a, e in view:
            b = self.recs[i]["base"][2]
            n = self.n[i]
            p = {k: v[a:e] for k, v in b.items() if k not in BEAT_KEYS and k != "eda4"}
            p["eda4"] = b["eda4"][4 * a:4 * e]
            m = (b["beat_t"] >= a) & (b["beat_t"] < e)
            p["beat_t"], p["beat_d"], p["beat_miss"] = b["beat_t"][m] - a, b["beat_d"][m], b["beat_miss"][m]
            parts.append((p, e - a))
        if len(parts) == 1:
            out, n = dict(parts[0][0]), parts[0][1]
        else:
            out, off = {}, 0
            for k in parts[0][0]:
                if k == "beat_t":
                    continue
                out[k] = np.concatenate([p[k] for p, _ in parts])
            bt = []
            for p, ln in parts:
                bt.append(p["beat_t"] + off)
                off += ln
            out["beat_t"] = np.concatenate(bt)
            n = off
        if blank is not None:
            sig, lo, hi = blank
            out = {k: v.copy() for k, v in out.items()}
            zero, nan = BLANK[sig]
            for k in zero:
                out[k][lo:hi] = 0.0
            for k in nan:
                out[k][lo:hi] = np.nan
            if sig == "EDA":
                out["eda4"][4 * lo:4 * hi] = np.nan
            if sig == "IBI":
                m = ~((out["beat_t"] >= lo) & (out["beat_t"] < hi))
                out["beat_t"], out["beat_d"], out["beat_miss"] = out["beat_t"][m], out["beat_d"][m], out["beat_miss"][m]
        return n, out

    def labels(self, view):
        return np.concatenate([self.recs[i]["y_sec"][a:e] for i, a, e in view])

    def features(self, view, cfg, cfg_key, blank=None):
        key = (view, blank, cfg_key)
        if key not in self.cache:
            n, b = self.base(view, blank)
            _, names, X = M.compute_features(None, cfg, (None, n, b))
            self.cache[key] = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
            self.names = names
        return self.cache[key]


# --------------------------------------------------------------------------
# Training set and prediction
# --------------------------------------------------------------------------
def parse_mix(s):
    mix = {k: float(v) for k, v in (kv.split(":") for kv in s.split(","))}
    tot = sum(mix.values())
    return {k: v / tot for k, v in mix.items() if v > 0}


def blank_for(view, dropout, n):
    """Deterministic channel dropout decision for a training view."""
    if dropout <= 0:
        return None
    rng = _rng("blank", view)
    if rng.random() >= dropout:
        return None
    sig = SIGNALS[int(rng.integers(len(SIGNALS)))]
    if rng.random() < 0.5 or n <= 120:
        return (sig, 0, n)
    ln = int(rng.integers(60, 301))
    lo = int(rng.integers(0, max(n - ln, 1)))
    return (sig, lo, min(n, lo + ln))


def training_set(suite, users, cfg, cfg_key, mix, dropout):
    Xs, ys = [], []
    for kind, share in mix.items():
        views = suite.train_views(kind, users)
        tot = sum(view_len(v) for v in views)
        stride = max(1, int(round(tot / (share * TOTAL_ROWS))))
        for v in views:
            n = view_len(v)
            X = suite.features(v, cfg, cfg_key, blank_for(v, dropout, n))
            off = int(_rng("off", v).integers(stride))
            Xs.append(X[off::stride])
            ys.append(suite.labels(v)[off::stride])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    m = y >= 0
    X, y = X[m], y[m]
    cnt = np.bincount(y, minlength=3).astype(float)
    w = (len(y) / (3 * np.maximum(cnt, 1)))[y]  # balanced classes over the whole training set
    return X, y, w


def predict_views(models, suite, views, cfg, cfg_key, step, blank_sig=None):
    """Per-second probabilities of each view (mean over models is NOT taken:
    returns one list per model). One batched call per model."""
    pts, backs, Xs = [], [], []
    for v in views:
        n = view_len(v)
        X = suite.features(v, cfg, cfg_key, (blank_sig, 0, n) if blank_sig else None)
        p, b = M.step_points(n, step)
        Xs.append(X[p])
        backs.append(b)
    Xall = np.concatenate(Xs)
    cuts = np.cumsum([len(x) for x in Xs])[:-1]
    return [[pp[b] for pp, b in zip(np.split(m.predict_proba(Xall), cuts), backs)] for m in models]


def scatter(suite, views, probas, store):
    for v, p in zip(views, probas):
        off = 0
        for i, a, e in v:
            if store[i] is None:
                store[i] = np.zeros((suite.n[i], 3), dtype=np.float32)
            store[i][a:e] = p[off:off + e - a]
            off += e - a


# --------------------------------------------------------------------------
# Scoring: one decoder for all formats, nested
# --------------------------------------------------------------------------
def pooled_offsets(data, oofs, idx, fmts=FORMATS):
    N = len(data)
    data4 = data * len(fmts)
    oof4 = [o for f in fmts for o in oofs[f]]
    idx4 = np.concatenate([np.asarray(idx) + k * N for k in range(len(fmts))])
    cfg, p, _ = TD.tune(data4, oof4, idx4, 5, use_hmm=False)
    return cfg, p


def score(data, oofs, fold_of, extra=()):
    """Returns per-format (mean over folds, folds, per-class F1) with pooled nested offsets."""
    K = fold_of.max() + 1
    res = {f: dict(folds=[], cm=np.zeros((3, 3), dtype=np.int64)) for f in FORMATS + tuple(extra)}
    offs = []
    for k in range(K):
        tr, te = np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)
        cfg, p = pooled_offsets(data, oofs, tr)
        offs.append([p["oA"], p["oS"]])
        for f in res:
            f1, _, cm = TD.evaluate(data, oofs[f], te, cfg)
            res[f]["folds"].append(f1)
            res[f]["cm"] += cm
    out = {}
    for f, r in res.items():
        c = r["cm"]
        tp = np.diag(c).astype(float)
        pc = np.where(tp > 0, 2 * tp / np.maximum(2 * tp + c.sum(0) - tp + c.sum(1) - tp, 1), 0.0)
        out[f] = dict(mean=float(np.mean(r["folds"])), folds=r["folds"], per_class=pc.tolist())
    return out, offs


def fit_nonact(recs, idx):
    """First-pass Activity detector on absolute raw features (logistic
    regression, fitted on training recordings); used by norm_mode 'nonact'."""
    from sklearn.linear_model import LogisticRegression
    Xs, ys = [], []
    for i in idx:
        _, n, b = recs[i]["base"]
        f = M.features_raw(b, n)
        Xs.append(np.stack([f[k] for k in M.NONACT_KEYS], 1)[::5])
        ys.append(recs[i]["y_sec"][::5])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    m = y >= 0
    X, y = X[m], (y[m] == 1).astype(int)
    med = np.nanmedian(X, 0)
    X = np.where(np.isfinite(X), X, med)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced").fit((X - mu) / sd, y)
    return dict(med=med.tolist(), mu=mu.tolist(), sd=sd.tolist(), coef=clf.coef_[0].tolist(),
                b=float(clf.intercept_[0]))


def build_cfg(a, kal, extra_params=None):
    params = {"ctr_past_only": True}
    if a.calm:
        params["norm_calm"] = True
    if a.norm_mode != "global":
        params["norm_mode"] = a.norm_mode
    if a.norm_win:
        params["norm_win"] = a.norm_win
    params.update(extra_params or {})
    return T.make_cfg("robust", kal, a.pred_step, a.groups, params)


def make_model(a, seed):
    import lightgbm as lgb
    p = dict(T.LGB_PARAMS)
    if a.trees:
        p["n_estimators"] = a.trees
    if a.leaves:
        p["num_leaves"] = a.leaves
    return lgb.LGBMClassifier(random_state=seed, n_jobs=4, **p)


def add_args(ap):
    ap.add_argument("--name", required=True)
    ap.add_argument("--groups", nargs="*", default=F_GROUPS)
    ap.add_argument("--calm", action="store_true")
    ap.add_argument("--norm-mode", default="global", choices=["global", "roll", "nonact", "seg"])
    ap.add_argument("--norm-win", type=int, default=0)
    ap.add_argument("--mix", default="orig:1")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--pred-step", type=int, default=5)
    ap.add_argument("--trees", type=int, default=0)
    ap.add_argument("--leaves", type=int, default=0)
    ap.add_argument("--missing", action="store_true", help="also score the full format with each signal removed")


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    a = ap.parse_args()
    t0 = time.time()
    recs = T.load_all()
    data = TD.prepare()
    suite = Suite(recs)
    uids = np.array([r["uid"] for r in recs])
    mix = parse_mix(a.mix)
    folds = user_folds(uids, 5, seed=0)
    fold_of = np.zeros(len(recs), dtype=int)
    extra = tuple(f"no_{s}" for s in SIGNALS) if a.missing else ()
    oofs = {s: {f: [None] * len(recs) for f in FORMATS + extra} for s in SEEDS}
    for k, (tr, te) in enumerate(folds):
        fold_of[te] = k
        kal = T.tune_kalman([recs[i] for i in tr])
        ep = {"nonact": fit_nonact(recs, tr)} if a.norm_mode == "nonact" else None
        cfg = build_cfg(a, kal, ep)
        ck = json.dumps(cfg, sort_keys=True)
        tr_users, te_users = sorted(set(uids[tr])), sorted(set(uids[te]))
        X, y, w = training_set(suite, tr_users, cfg, ck, mix, a.dropout)
        models = [make_model(a, s).fit(X, y, sample_weight=w) for s in SEEDS]
        for f in FORMATS:
            views = suite.eval_views(f, te_users)
            for s, pr in zip(SEEDS, predict_views(models, suite, views, cfg, ck, a.pred_step)):
                scatter(suite, views, pr, oofs[s][f])
        for sig in (SIGNALS if a.missing else ()):
            views = suite.eval_views("full", te_users)
            for s, pr in zip(SEEDS, predict_views(models, suite, views, cfg, ck, a.pred_step, sig)):
                scatter(suite, views, pr, oofs[s][f"no_{sig}"])
        print(f"[{a.name}] fold {k}: {len(y)} rows, {X.shape[1]} features ({time.time()-t0:.0f}s)", flush=True)
    runs = []
    for s in SEEDS:
        sc, offs = score(data, oofs[s], fold_of, extra)
        vals = [sc[f]["mean"] for f in FORMATS]
        runs.append(dict(seed=s, formats=sc, robust=float(np.mean(vals)), minimum=float(np.min(vals)), offsets=offs))
        np.save(OUT / f"{a.name}_s{s}.npy", np.array([oofs[s][f] for f in FORMATS], dtype=object), allow_pickle=True)
        print(f"[{a.name}] seed {s}: " + " ".join(f"{f} {sc[f]['mean']:.4f}" for f in sc)
              + f" | robust {runs[-1]['robust']:.4f} min {runs[-1]['minimum']:.4f}", flush=True)
    np.save(OUT / "folds.npy", fold_of)
    summ = {f: dict(mean=float(np.mean([r["formats"][f]["mean"] for r in runs])),
                    per_class=dict(zip(M.CLASSES, np.mean([r["formats"][f]["per_class"] for r in runs], 0).round(4).tolist())))
            for f in FORMATS + extra}
    res = dict(name=a.name, flags=vars(a), formats=summ, robust=float(np.mean([r["robust"] for r in runs])),
               minimum=float(np.min([summ[f]["mean"] for f in FORMATS])),
               robust_seeds=[r["robust"] for r in runs], runs=runs, minutes=(time.time() - t0) / 60)
    (OUT / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    print(f"[{a.name}] DONE " + " ".join(f"{f} {summ[f]['mean']:.4f}" for f in summ)
          + f" | robust {res['robust']:.4f} min {res['minimum']:.4f} | Stress F1 "
          + " ".join(f"{summ[f]['per_class']['Stress']:.3f}" for f in FORMATS) + f" ({res['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
