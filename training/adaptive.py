"""Step 2: per-recording adaptive Stress threshold (2a) and ranking score (2b).

Works on saved out-of-fold 3-class probabilities (artifacts/stat/<base>_r*_s*.npy).
For each run (3 repeats x 3 seeds) and outer fold k:
  - global decoder: per-class offsets tuned on the other folds (as stat_ext)
  - per recording, on its non-Activity seconds (global decision != Activity),
    the Stress score s (2a: log p(Stress) - log p(Rest); 2b: ranking model score)
  - rules (parameters tuned on the other folds' recordings, applied to fold k):
      gmm   : two-component Gaussian mixture of s within the recording (numpy EM).
              If separation (mu1 - mu0) / sqrt((v0 + v1) / 2) >= D, upper weight
              >= Wmin and upper mean >= Mmin, non-Activity seconds become Stress
              where the upper component's posterior > 0.5, else Rest. Otherwise
              the global decision is kept.
      z     : robust z of s within the recording (median, IQR over non-Activity
              seconds); non-Activity seconds become Stress where z > tau.
      zfloor: as z, but Stress also needs s > smin.
2b trains XGBoost rank:pairwise inside each training fold (groups = recording,
target Stress vs Rest on non-Activity seconds of STRESS recordings) and caches
its out-of-fold scores (artifacts/stat/rank_r*_s*.npy).

Usage:
    python training/adaptive.py --score logodds
    python training/adaptive.py --score rank
"""

import argparse
import json
import time

import numpy as np

from common import M, ROOT, per_class_f1, user_folds
import train as T
import tune_decoder as TD

OUT = ROOT / "artifacts" / "stat"
LOW = ("S15", "S17", "f11", "f17", "f18", "S01", "f13")
STEP = 5
GMM_GRID = [(d, w, m) for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0) for w in (0.05, 0.1, 0.2, 0.3)
            for m in (-np.inf, -4.0, -2.0, 0.0)]
Z_GRID = [(t, -np.inf) for t in np.arange(0.0, 4.01, 0.25)]
ZF_GRID = [(t, f) for t in np.arange(0.0, 4.01, 0.25) for f in (-6.0, -4.0, -3.0, -2.0, -1.0, 0.0)]


def gmm2(x, iters=30):
    """Two-component 1D Gaussian mixture by EM; returns mu, var, w (sorted by mu)."""
    q = np.percentile(x, (25, 75))
    mu = q.astype(float) + np.array([-1e-3, 1e-3])
    vx = float(np.var(x)) + 1e-6
    var = np.full(2, vx / 2)
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = -0.5 * ((x[:, None] - mu) ** 2 / var + np.log(2 * np.pi * var)) + np.log(w)
        ll -= ll.max(1, keepdims=True)
        r = np.exp(ll)
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        w = nk / len(x)
        mu = (r * x[:, None]).sum(0) / nk
        var = (r * (x[:, None] - mu) ** 2).sum(0) / nk + 1e-3 * vx + 1e-6
    o = np.argsort(mu)
    return mu[o], var[o], w[o]


def gmm_post_upper(x, mu, var, w):
    ll = -0.5 * ((x[:, None] - mu) ** 2 / var + np.log(2 * np.pi * var)) + np.log(w)
    ll -= ll.max(1, keepdims=True)
    r = np.exp(ll)
    return r[:, 1] / r.sum(1)


class Rec:
    """Per recording, per outer-fold decoder: base codes and score statistics."""

    def __init__(self, d, codes, s):
        self.d = d
        self.codes = codes
        self.s = s
        self.na = codes != 1
        t = d["wtrue"]
        self.base_cm = cm(t, TD.window_preds(d, codes))
        x = s[self.na]
        self.ok = len(x) >= 60
        if self.ok:
            self.g = gmm2(x)
            mu, var, w = self.g
            self.sep = (mu[1] - mu[0]) / np.sqrt(0.5 * (var[0] + var[1]))
            self.up = gmm_post_upper(s, *self.g) > 0.5
            lo, med, hi = np.percentile(x, (25, 50, 75))
            self.z = (s - med) / max(hi - lo, 1e-6)
        self.cache = {}

    def confusion(self, method, p):
        key = (method, p)
        if key in self.cache:
            return self.cache[key]
        out = self.base_cm
        if self.ok:
            new = None
            if method == "gmm":
                D, W, Mm = p
                mu, _, w = self.g
                if self.sep >= D and w[1] >= W and mu[1] >= Mm:
                    new = np.where(self.na, np.where(self.up, 2, 0), self.codes)
            else:
                tau, floor = p
                new = np.where(self.na, np.where((self.z > tau) & (self.s > floor), 2, 0), self.codes)
            if new is not None:
                out = cm(self.d["wtrue"], TD.window_preds(self.d, new))
        self.cache[key] = out
        return out


def cm(t, p):
    return np.bincount(t * 3 + p, minlength=9).reshape(3, 3)


def f1_from_cm(c):
    tp = np.diag(c).astype(float)
    fp = c.sum(0) - tp
    fn = c.sum(1) - tp
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def score_set(recs, method, p):
    c = sum(r.confusion(method, p) for r in recs)
    return float(f1_from_cm(c).mean())


GRIDS = {"gmm": GMM_GRID, "z": Z_GRID, "zfloor": ZF_GRID}


# --------------------------------------------------------------------------
# 2b: ranking model
# --------------------------------------------------------------------------
def rank_oof(recs_all, uids, rep, seeds, kal_cache, fc, chosen):
    import xgboost as xgb

    folds = user_folds(uids, 5, seed=rep)
    out = {s: [None] * len(recs_all) for s in seeds}
    for k, (tr, te) in enumerate(folds):
        key = tuple(sorted(tr))
        if key not in kal_cache:
            kal_cache[key] = T.tune_kalman([recs_all[i] for i in tr])
        cfg = T.make_cfg("stat", kal_cache[key], STEP, ["raw", "kalman", "norm", "resp", "ctr"],
                         {"ctr_past_only": True})
        Xs, ys, qs = [], [], []
        for i in tr:
            r = recs_all[i]
            if r["prot"] != "STRESS":
                continue
            X = fc.get(i, cfg)[1][::4]
            y = r["y_sec"][::4]
            m = (y == 0) | (y == 2)
            Xs.append(X[m])
            ys.append((y[m] == 2).astype(int))
            qs.append(np.full(m.sum(), i))
        X, y, q = np.concatenate(Xs), np.concatenate(ys), np.concatenate(qs)
        p = chosen[(rep, k)]
        for s in seeds:
            mdl = xgb.XGBRanker(objective="rank:pairwise", tree_method="hist", n_estimators=200,
                                learning_rate=0.1, max_depth=int(p.get("max_depth", 6)),
                                min_child_weight=p.get("min_child_weight", 5),
                                colsample_bytree=p.get("colsample_bytree", 0.7), subsample=0.7,
                                random_state=s, n_jobs=4)
            mdl.fit(X, y, qid=q)
            for i in te:
                Xi = fc.get(i, cfg)[1]
                pts, back = M.step_points(len(Xi), STEP)
                out[s][i] = mdl.predict(np.ascontiguousarray(Xi[pts]))[back]
        print(f"  rank rep {rep} fold {k} ({len(y)} rows)", flush=True)
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="logodds", choices=["logodds", "rank"])
    ap.add_argument("--base", default="base")
    a = ap.parse_args()
    t0 = time.time()
    data = TD.prepare()
    lst = __import__("common").list_recordings()
    prot = [p for p, _, _ in lst]
    uid = [d["uid"] for d in data]
    recs_all = fc = None
    if a.score == "rank":
        recs_all = T.load_all()
        fc = T.FeatureCache(recs_all)
        zoo = json.loads((ROOT / "artifacts" / "zoo" / "xgb.json").read_text())
        chosen = {(c["rep"], c["fold"]): c["params"] for c in zoo["chosen"]}
        kal_cache = {}
    runs = []
    for rep in (0, 1, 2):
        fold_of = np.load(OUT / f"folds_r{rep}.npy")
        if a.score == "rank":
            rk = {}
            for s in (0, 1, 2):
                f = OUT / f"rank_r{rep}_s{s}.npy"
                if f.exists():
                    rk[s] = np.load(f, allow_pickle=True)
            if len(rk) < 3:
                new = rank_oof(recs_all, np.array(uid), rep, (0, 1, 2), kal_cache, fc, chosen)
                for s in (0, 1, 2):
                    rk[s] = np.array(new[s], dtype=object)
                    np.save(OUT / f"rank_r{rep}_s{s}.npy", rk[s], allow_pickle=True)
        for s in (0, 1, 2):
            oof = np.load(OUT / f"{a.base}_r{rep}_s{s}.npy", allow_pickle=True)
            res = {m: dict(folds=[], cm=np.zeros((3, 3), int), params=[]) for m in GRIDS}
            res["offsets"] = dict(folds=[], cm=np.zeros((3, 3), int), params=[])
            user = {m: {} for m in res}
            for k in range(fold_of.max() + 1):
                tr, te = np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)
                cfg, _, _ = TD.tune(data, oof, tr, STEP, use_hmm=False)
                R = {}
                for i in range(len(data)):
                    p = np.clip(oof[i], 1e-6, 1)
                    sc = np.log(p[:, 2]) - np.log(p[:, 0]) if a.score == "logodds" else rk[s][i]
                    R[i] = Rec(data[i], M.decode(oof[i], cfg), sc)
                Rtr, Rte = [R[i] for i in tr], [R[i] for i in te]
                c = sum(r.base_cm for r in Rte)
                res["offsets"]["folds"].append(float(f1_from_cm(c).mean()))
                res["offsets"]["cm"] += c
                for i in te:
                    if prot[i] == "STRESS":
                        user["offsets"][uid[i]] = float(f1_from_cm(R[i].base_cm)[2])
                for m, grid in GRIDS.items():
                    best = max(grid, key=lambda q: score_set(Rtr, m, q))
                    c = sum(r.confusion(m, best) for r in Rte)
                    res[m]["folds"].append(float(f1_from_cm(c).mean()))
                    res[m]["cm"] += c
                    res[m]["params"].append([float(v) for v in best])
                    for i in te:
                        if prot[i] == "STRESS":
                            user[m][uid[i]] = float(f1_from_cm(R[i].confusion(m, best))[2])
            for m, rr in res.items():
                pc = f1_from_cm(rr["cm"])
                runs.append(dict(method=m, rep=rep, seed=s, mean=float(np.mean(rr["folds"])), folds=rr["folds"],
                                 per_class=pc.tolist(), params=rr["params"], user_stress=user[m]))
            print(f"rep {rep} seed {s}: " + "  ".join(
                f"{m} {np.mean(res[m]['folds']):.4f}" for m in res) + f" ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for m in ["offsets"] + list(GRIDS):
        rr = [r for r in runs if r["method"] == m]
        reps = [float(np.mean([r["mean"] for r in rr if r["rep"] == rep])) for rep in (0, 1, 2)]
        users = {u: float(np.mean([r["user_stress"][u] for r in rr if u in r["user_stress"]])) for u in LOW}
        summary[m] = dict(mean=float(np.mean([r["mean"] for r in rr])),
                          std=float(np.std([f for r in rr for f in r["folds"]])), reps=reps,
                          per_class=dict(zip(M.CLASSES, np.mean([r["per_class"] for r in rr], 0).round(4).tolist())),
                          low=users)
        print(f"[{a.score} {m}] mean {summary[m]['mean']:.4f} +- {summary[m]['std']:.4f} reps "
              f"{[round(v, 4) for v in reps]} per-class {summary[m]['per_class']} low {{"
              + ", ".join(f"{u}: {v:.2f}" for u, v in users.items()) + "}", flush=True)
    (OUT / f"adaptive_{a.score}.json").write_text(json.dumps(dict(summary=summary, runs=runs), indent=1))


if __name__ == "__main__":
    main()
