"""Statistical extensions on top of XGBoost (variant F features).

Protocol (same for every item, paired against the XGBoost baseline):
  - 3 repeats of GroupKFold(5) by user_id (shuffle seed = repeat) x 3 seeds.
  - XGBoost hyperparameters: the ones chosen by the inner search of the model
    zoo run for the same (repeat, fold) (artifacts/zoo/xgb.json). They were
    chosen inside the training folds, so reusing them leaks nothing and makes
    every comparison paired. With --search the inner search is run again
    (extended space with reg_alpha, item 5).
  - Kalman noise (and the vector Kalman noise, item 2) by maximum likelihood,
    shrinkage population statistics and kappa (item 1), and the QDA (item 3)
    are all fitted inside each training fold.
  - Decoder: per-class offsets (or the cost matrix, item 4) tuned per outer
    fold on the other folds' out-of-fold predictions, competition window metric.

Usage:
    python training/stat_ext.py --name base
    python training/stat_ext.py --name shrink --shrink
    python training/stat_ext.py --name calm --calm
    python training/stat_ext.py --name vkf --vkf
    python training/stat_ext.py --name qda --qda
    python training/stat_ext.py --name base --cost-only        (item 4, reuses saved OOF)
    python training/stat_ext.py --name reg --search            (item 5)
    python training/stat_ext.py --name combo --vkf --qda --export
Results: artifacts/stat/<name>.json
"""

import argparse
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from common import M, ROOT, user_folds, per_class_f1  # noqa: E402
import train as T  # noqa: E402
import tune_decoder as TD  # noqa: E402
import model_zoo as Z  # noqa: E402

OUT = ROOT / "artifacts" / "stat"
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = ["raw", "kalman", "norm", "resp", "ctr"]
PARAMS = {"ctr_past_only": True}
REPEATS = (0, 1, 2)
SEEDS = (0, 1, 2)
STEP = 5
LOW = ("S15", "S17", "f11", "f17", "f18")
KAPPA_GRID = (60, 300, 900, 3600)
LAMBDA_GRID = (0.0, 0.1, 0.3, 0.6, 0.9)
COST_GRID = np.round(2.0 ** np.arange(-3, 3.01, 0.5), 4)
VKF_GRID = [dict(vkf_q=q, vkf_rh=rh, vkf_ri=ri, vkf_infl=fl)
            for q in (0.001, 0.003, 0.01, 0.03) for rh in (0.01, 0.03, 0.1, 0.3)
            for ri in (4.0, 16.0, 64.0, 256.0) for fl in (1.0, 10.0, 100.0)]
XGB_SEARCH = dict(n_estimators=[200, 300, 500], learning_rate=[0.03, 0.05, 0.1], max_depth=[4, 6, 8],
                  min_child_weight=[1, 5, 20], colsample_bytree=[0.5, 0.7, 0.9],
                  reg_lambda=[0.1, 1.0, 5.0, 20.0], reg_alpha=[0.0, 0.1, 1.0, 5.0])
JOBS = 4


# --------------------------------------------------------------------------
def make(model, p, seed):
    if model == "lgbF":  # the submission F model: fixed LightGBM parameters
        import lightgbm as lgb
        return lgb.LGBMClassifier(random_state=seed, n_jobs=JOBS, **T.LGB_PARAMS)
    return Z.make_model("xgb", p, seed, JOBS)


def tune_vkf(train_recs):
    """Maximum likelihood (grid) for the two-measurement HR filter."""
    bs = [r["base"] for r in train_recs]
    ll = [sum(M.vkf_loglik(b, n, g) for _, n, b in bs) for g in VKF_GRID]
    return dict(VKF_GRID[int(np.argmax(ll))])


def calm_from_cols(X, names):
    prev = {k: X[:, names.index(k)].astype(np.float64) for k in ("hr_mean_60", "kf_eda_phasic_std_60")}
    return M.calm_mask(prev)


class Shrinker:
    """Recomputes the z_ columns of a feature matrix with shrunk statistics."""

    def __init__(self, names, calm):
        self.names, self.calm = names, calm
        self.keys = [k for k in M.NORM_KEYS if f"z_{k}" in names]
        self.jx = [names.index(k) for k in self.keys]
        self.jz = [names.index(f"z_{k}") for k in self.keys]
        self.stats = {}

    def stat(self, i, X):
        if i not in self.stats:
            mask = calm_from_cols(X, self.names) if self.calm else None
            self.stats[i] = [M.norm_stat(X[:, j].astype(np.float64), mask) for j in self.jx]
        return self.stats[i]

    def pop(self, idx, get):
        out = {}
        for a, k in enumerate(self.keys):
            v = np.array([self.stat(i, get(i))[a][1:] for i in idx], dtype=np.float64)
            v = v[np.isfinite(v).all(1)]
            out[k] = [float(np.median(v[:, 0])), float(np.median(v[:, 1]))]
        return out

    def apply(self, i, X, kappa, pop):
        X = X.copy()
        for a, k in enumerate(self.keys):
            ne, med, sc = self.stat(i, X)[a]
            if ne == 0:
                continue
            med, sc = M.shrink_stat(ne, med, sc, kappa.get(M._signal_group(k), 0), pop.get(k))
            X[:, self.jz[a]] = (X[:, self.jx[a]] - med) / sc
        return X


def qda_cols(names):
    return [names.index(k) for k in M.QDA_KEYS]


def qda_features(recs, Xget, tr, te, names, uids):
    """Cross-fitted LLR columns for training recordings (QDA fitted on other
    training people) and full-fold LLR columns for the test recordings."""
    jq = qda_cols(names)

    def zy(idx):
        Zs = [Xget(i)[::2, jq].astype(np.float64) for i in idx]
        ys = [recs[i]["y_sec"][::2] for i in idx]
        Zc, yc = np.concatenate(Zs), np.concatenate(ys)
        m = yc >= 0
        return Zc[m], yc[m]

    inner = user_folds(uids[tr], 5, seed=0)
    best_lam, best_ll, cross = None, -np.inf, None
    for lam in LAMBDA_GRID:
        tot, out = 0.0, {}
        for itr, ite in inner:
            q = M.qda_fit(*zy(tr[itr]), lam)
            Zv, yv = zy(tr[ite])
            ll = M.qda_loglik(Zv, q)
            mx = ll.max(1, keepdims=True)
            tot += float(np.sum(ll[np.arange(len(yv)), yv] - (mx[:, 0] + np.log(np.exp(ll - mx).sum(1)))))
            for i in tr[ite]:
                out[i] = M.qda_llr(Xget(i)[:, jq].astype(np.float64), q)
        if tot > best_ll:
            best_lam, best_ll, cross = lam, tot, out
    qfull = M.qda_fit(*zy(tr), best_lam)
    cols = dict(cross)
    for i in te:
        cols[i] = M.qda_llr(Xget(i)[:, jq].astype(np.float64), qfull)
    return {i: np.stack(v, 1).astype(np.float32) for i, v in cols.items()}, best_lam, qfull


# --------------------------------------------------------------------------
# Decoders
# --------------------------------------------------------------------------
def cost_cfg(c):
    C = np.zeros((3, 3))
    for (i, j), v in c.items():
        C[i, j] = v
    return {"cost": C.tolist()}


def tune_cost(data, oof, idx, rounds=3):
    c = {(i, j): 1.0 for i in range(3) for j in range(3) if i != j}
    best = TD.evaluate(data, oof, idx, cost_cfg(c))[0]
    for _ in range(rounds):
        changed = False
        for k in list(c):
            for v in COST_GRID:
                if v == c[k]:
                    continue
                q = dict(c)
                q[k] = float(v)
                s = TD.evaluate(data, oof, idx, cost_cfg(q))[0]
                if s > best + 1e-6:
                    best, c, changed = s, q, True
        if not changed:
            break
    return cost_cfg(c), c


def nested(data, oof, fold_of, recs, decoder="offsets"):
    scores, Tw, Pw, chosen, user = [], [], [], [], {}
    for k in range(fold_of.max() + 1):
        tr, te = np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)
        if decoder == "cost":
            cfg, c = tune_cost(data, oof, tr)
            chosen.append({f"{a}{b}": v for (a, b), v in c.items()})
        else:
            cfg, p, _ = TD.tune(data, oof, tr, STEP, use_hmm=False)
            chosen.append({"oA": p["oA"], "oS": p["oS"]})
        f1, _, _ = TD.evaluate(data, oof, te, cfg)
        scores.append(f1)
        for i in te:
            t, pr = data[i]["wtrue"], TD.window_preds(data[i], M.decode(oof[i], cfg))
            Tw.append(t)
            Pw.append(pr)
            if recs[i]["prot"] == "STRESS":
                user[recs[i]["uid"]] = float(per_class_f1(t, pr)[2])
    pc = per_class_f1(np.concatenate(Tw), np.concatenate(Pw))
    return float(np.mean(scores)), scores, pc, chosen, user


def summarize(name, runs, extra):
    res = {"name": name, "runs": runs, **extra}
    res["mean"] = float(np.mean([r["mean"] for r in runs]))
    res["std_folds"] = float(np.std([f for r in runs for f in r["folds"]]))
    res["rep_means"] = {rep: float(np.mean([r["mean"] for r in runs if r["rep"] == rep])) for rep in REPEATS}
    res["per_class"] = dict(zip(M.CLASSES, np.mean([r["per_class"] for r in runs], 0).round(4).tolist()))
    res["auc"] = float(np.mean([r["auc"] for r in runs]))
    users = sorted({u for r in runs for u in r["user_stress"]})
    res["user_stress"] = {u: float(np.mean([r["user_stress"][u] for r in runs if u in r["user_stress"]]))
                          for u in users}
    res["low_responders"] = float(np.mean([res["user_stress"][u] for u in LOW if u in res["user_stress"]]))
    return res


def report(res):
    print(f"[{res['name']}] mean {res['mean']:.4f} +- {res['std_folds']:.4f} reps "
          f"{[round(v, 4) for v in res['rep_means'].values()]} per-class {res['per_class']} "
          f"AUC {res['auc']:.4f} low-resp Stress F1 {res['low_responders']:.3f}", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="xgb", choices=["xgb", "lgbF"])
    ap.add_argument("--shrink", action="store_true")
    ap.add_argument("--calm", action="store_true")
    ap.add_argument("--vkf", action="store_true")
    ap.add_argument("--qda", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--cost", action="store_true", help="also score the cost-matrix decoder")
    ap.add_argument("--cost-only", action="store_true", help="score the cost decoder on saved OOF")
    ap.add_argument("--repeats", type=int, nargs="*", default=list(REPEATS))
    ap.add_argument("--export-only", action="store_true",
                    help="fit on all data and export ONNX + reference (needs the CV json of --name)")
    a = ap.parse_args()
    t0 = time.time()

    recs = T.load_all()
    data = TD.prepare()
    uids = np.array([r["uid"] for r in recs])
    zoo = json.loads((ROOT / "artifacts" / "zoo" / "xgb.json").read_text())
    chosen_xgb = {(c["rep"], c["fold"]): c["params"] for c in zoo["chosen"]}

    if a.export_only:
        export_final(a, recs, data, uids, zoo)
        return

    if a.cost_only:
        runs = {"offsets": [], "cost": []}
        for rep in a.repeats:
            fold_of = np.load(OUT / f"folds_r{rep}.npy")
            for s in SEEDS:
                oof = np.load(OUT / f"{a.name}_r{rep}_s{s}.npy", allow_pickle=True)
                auc = Z.window_auc(data, oof, range(len(recs)))
                for dec in runs:
                    mean, fs, pc, ch, us = nested(data, oof, fold_of, recs, dec)
                    runs[dec].append(dict(rep=rep, seed=s, mean=mean, folds=fs, per_class=pc.tolist(),
                                          auc=auc, user_stress=us, decoder=ch))
                    print(f"  {dec} rep {rep} seed {s}: {mean:.4f} {np.round(pc, 3)}", flush=True)
        for dec, rr in runs.items():
            res = summarize(f"{a.name}_{dec}", rr, {})
            if dec == "cost":
                allc = [c for r in rr for c in r["decoder"]]
                res["cost_mean"] = {k: float(np.mean([c[k] for c in allc])) for k in allc[0]}
                res["cost_geomean"] = {k: float(np.exp(np.mean(np.log([c[k] for c in allc])))) for k in allc[0]}
            (OUT / f"{a.name}_{dec}.json").write_text(json.dumps(res, indent=1))
            report(res)
        return

    groups = list(GROUPS)
    if a.vkf:
        groups.insert(groups.index("kalman") + 1, "vkf")
    fc = T.FeatureCache(recs)
    kal_cache, vkf_cache = {}, {}
    runs, chosen = [], []
    for rep in a.repeats:
        folds = user_folds(uids, 5, seed=rep)
        fold_of = np.zeros(len(recs), dtype=int)
        oofs = {s: [None] * len(recs) for s in SEEDS}
        for k, (tr, te) in enumerate(folds):
            fold_of[te] = k
            key = tuple(sorted(tr))
            if key not in kal_cache:
                kal_cache[key] = T.tune_kalman([recs[i] for i in tr])
            params = dict(PARAMS)
            info = {"rep": rep, "fold": k}
            if a.vkf:
                if key not in vkf_cache:
                    vkf_cache[key] = tune_vkf([recs[i] for i in tr])
                params.update(vkf_cache[key])
                info["vkf"] = vkf_cache[key]
            if a.calm:
                params["norm_calm"] = True
            cfg = T.make_cfg("stat", kal_cache[key], STEP, groups, params)
            names = fc.get(0, cfg)[0]

            def raw(i, cfg=cfg):
                X = fc.get(i, cfg)[1]
                return np.where(np.isfinite(X), X, np.nan).astype(np.float32)

            get = raw
            if a.shrink:
                sh = Shrinker(names, a.calm)
                pop = sh.pop(tr, raw)
                kappa = tune_kappa(recs, raw, sh, pop, tr, uids, rep, k, a.model, chosen_xgb[(rep, k)])
                info["kappa"] = kappa

                def get(i, sh=sh, kappa=kappa, pop=pop):
                    return sh.apply(i, raw(i), kappa, pop)
            if a.qda:
                cols, lam, _ = qda_features(recs, get, tr, te, names, uids)
                info["qda_lambda"] = lam
                prev_get = get

                def get(i, prev_get=prev_get, cols=cols):
                    return np.concatenate([prev_get(i), cols[i]], axis=1)

            p = chosen_xgb[(rep, k)]
            if a.search:
                p = search_params(recs, get, tr, uids, rep, k)
                info["params"] = p

            def stack(idx, stride):
                X = np.concatenate([get(i)[::stride] for i in idx])
                y = np.concatenate([recs[i]["y_sec"][::stride] for i in idx])
                w = np.concatenate([T.sample_weights(recs[i]["y_sec"], 1.0, 0)[::stride] for i in idx])
                m = y >= 0
                return X[m], y[m], w[m]

            Xtr, ytr, wtr = stack(tr, 2)
            for s in SEEDS:
                m = make(a.model, p, s).fit(Xtr, ytr, sample_weight=wtr)
                for i in te:
                    oofs[s][i] = T.predict_proba(m, get(i), STEP)
            chosen.append(info)
            print(f"[{a.name}] rep {rep} fold {k} {json.dumps({x: y for x, y in info.items() if x not in ('rep', 'fold')})} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        np.save(OUT / f"folds_r{rep}.npy", fold_of)
        for s in SEEDS:
            np.save(OUT / f"{a.name}_r{rep}_s{s}.npy", np.array(oofs[s], dtype=object), allow_pickle=True)
            auc = Z.window_auc(data, oofs[s], range(len(recs)))
            decs = ["offsets"] + (["cost"] if a.cost else [])
            for dec in decs:
                mean, fs, pc, ch, us = nested(data, oofs[s], fold_of, recs, dec)
                runs.append(dict(rep=rep, seed=s, mean=mean, folds=fs, per_class=pc.tolist(), auc=auc,
                                 user_stress=us, decoder=ch, dec=dec))
                print(f"[{a.name}] rep {rep} seed {s} {dec}: macroF1 {mean:.4f} per-class {np.round(pc, 3)} "
                      f"AUC {auc:.4f}", flush=True)
    res = summarize(a.name, [r for r in runs if r["dec"] == "offsets"],
                    {"chosen": chosen, "groups": groups, "flags": vars(a), "minutes": (time.time() - t0) / 60})
    (OUT / f"{a.name}.json").write_text(json.dumps(res, indent=1, default=str))
    report(res)


def inner_split(tr, uids, rep, k):
    rng = np.random.default_rng(1000 * rep + k)
    tr_users = np.unique(uids[tr])
    val = set(rng.choice(tr_users, size=max(1, len(tr_users) // 4), replace=False))
    return [i for i in tr if uids[i] not in val], [i for i in tr if uids[i] in val], rng


def inner_stack(recs, get, idx, stride):
    X = np.concatenate([get(i)[::stride] for i in idx])
    y = np.concatenate([recs[i]["y_sec"][::stride] for i in idx])
    w = np.concatenate([T.sample_weights(recs[i]["y_sec"], 1.0, 0)[::stride] for i in idx])
    m = y >= 0
    return X[m], y[m], w[m]


def tune_kappa(recs, raw, sh, pop, tr, uids, rep, k, model, p):
    """One pass of coordinate search over kappa per signal group, scored on
    one grouped 75/25 user split of the training fold (as the zoo search)."""
    itr, iva, _ = inner_split(tr, uids, rep, k)

    def score(kappa):
        def g(i):
            return sh.apply(i, raw(i), kappa, pop)
        Xi, yi, wi = inner_stack(recs, g, itr, 6)
        Xv, yv, _ = inner_stack(recs, g, iva, 3)
        m = make(model, p, 0).fit(Xi, yi, sample_weight=wi)
        return float(per_class_f1(yv, m.predict_proba(Xv).argmax(1)).mean())

    kappa = {g: 0 for g in M.NORM_GROUPS}
    best = score(kappa)
    for g in M.NORM_GROUPS:
        for v in KAPPA_GRID:
            q = dict(kappa, **{g: v})
            s = score(q)
            if s > best + 1e-6:
                best, kappa = s, q
    return kappa


def search_params(recs, get, tr, uids, rep, k, trials=12):
    """Inner random search (zoo protocol) over the extended XGBoost space."""
    itr, iva, rng = inner_split(tr, uids, rep, k)
    Xi, yi, wi = inner_stack(recs, get, itr, 6)
    Xv, yv, _ = inner_stack(recs, get, iva, 3)
    keys = list(XGB_SEARCH)
    seen, best, best_p = set(), -1.0, None
    while len(seen) < trials:
        p = {kk: XGB_SEARCH[kk][rng.integers(len(XGB_SEARCH[kk]))] for kk in keys}
        key = json.dumps(p, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        m = make("xgb", p, 0).fit(Xi, yi, sample_weight=wi)
        s = float(per_class_f1(yv, m.predict_proba(Xv).argmax(1)).mean())
        if s > best:
            best, best_p = s, p
    return best_p


def _mode(values):
    vals = [json.dumps(v, sort_keys=True) for v in values]
    return json.loads(max(set(vals), key=vals.count))


def export_final(a, recs, data, uids, zoo):
    """Fit on all recordings with the settings of the CV run and export.

    XGBoost parameters: the most frequently chosen set of the zoo search (as
    the zoo export). Kalman / vector Kalman noise: maximum likelihood on all
    data. kappa: most frequent per group across the CV folds. QDA: LLR
    training columns cross-fitted by user, final QDA on all data (metadata).
    Offsets: tuned on the out-of-fold probabilities averaged over the 9 runs.
    """
    import onnx

    res = json.loads((OUT / f"{a.name}.json").read_text())
    groups = res["groups"] + (["qda"] if a.qda else [])
    params = dict(PARAMS)
    if a.vkf:
        params.update(tune_vkf(recs))
    if a.calm:
        params["norm_calm"] = True
    kal = T.tune_kalman(recs)
    cfg = T.make_cfg("stat", kal, STEP, [g for g in groups if g != "qda"], params)
    fc = T.FeatureCache(recs)
    names = fc.get(0, cfg)[0]

    def get(i):
        X = fc.get(i, cfg)[1]
        return np.where(np.isfinite(X), X, np.nan).astype(np.float32)

    allidx = np.arange(len(recs))
    if a.shrink:
        sh = Shrinker(names, a.calm)
        pop = sh.pop(allidx, get)
        kappa = {g: _mode([c["kappa"][g] for c in res["chosen"]]) for g in M.NORM_GROUPS}
        params.update(shr_kappa=kappa, shr_pop=pop)
        base_get = get

        def get(i, base_get=base_get):
            return sh.apply(i, base_get(i), kappa, pop)
    if a.qda:
        cols, lam, qfull = qda_features(recs, get, allidx, np.array([], dtype=int), names, uids)
        params["qda"] = qfull
        base_get2 = get

        def get(i, base_get2=base_get2):
            return np.concatenate([base_get2(i), cols[i]], axis=1)
        names = names + ["qda_llr_stress", "qda_llr_act"]
    cfg = T.make_cfg("stat", kal, STEP, groups, params)

    p = zoo["export"]["params"] if a.model == "xgb" else None
    X = np.concatenate([get(i)[::2] for i in allidx])
    y = np.concatenate([r["y_sec"][::2] for r in recs])
    w = np.concatenate([T.sample_weights(r["y_sec"], 1.0, 0)[::2] for r in recs])
    mk = y >= 0
    m = make(a.model, p, 0).fit(X[mk], y[mk], sample_weight=w[mk])

    # decoder: offsets on the averaged out-of-fold probabilities
    oofs = [np.load(OUT / f"{a.name}_r{r}_s{s}.npy", allow_pickle=True) for r in REPEATS for s in SEEDS]
    avg = [np.mean([o[i] for o in oofs], axis=0) for i in allidx]
    dec, dp, _ = TD.tune(data, avg, allidx, STEP, use_hmm=False)
    cfg.update(dec)
    cfg["features"] = names
    cfg["variant_label"] = a.name
    cfg["model"] = a.model

    if a.model == "xgb":
        b = Z.export("xgb", m, len(names))
        onx = onnx.load_from_string(b)
    else:
        path = ROOT / "artifacts" / f"model_{a.name}.onnx"
        T.export_onnx(m, len(names), cfg, path)
        onx = onnx.load(str(path))
    for prop in list(onx.metadata_props):
        if prop.key == "adm_config":
            onx.metadata_props.remove(prop)
    meta = onx.metadata_props.add()
    meta.key = "adm_config"
    meta.value = json.dumps(cfg)
    path = ROOT / "artifacts" / f"model_{a.name}.onnx"
    onnx.save_model(onx, str(path))

    # reference: inference-path features (as run() computes them) through the fitted model
    j = next(i for i, r in enumerate(recs) if r["prot"] == "STRESS" and r["name"] == "f01")
    _, nref, Xf = M.compute_features(recs[j]["df"], cfg)
    assert list(nref) == names, "feature names differ between training and inference path"
    np.savez(ROOT / "artifacts" / f"ref_{a.name}.npz", proba=T.predict_proba(m, Xf, STEP),
             recording=str(ROOT / "data" / "STRESS" / "f01.parquet"))
    print(f"exported {path.name}: {len(names)} features, offsets {dp}, params {p}")


if __name__ == "__main__":
    main()
