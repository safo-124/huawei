"""Block A: tabular model comparison on the variant F feature set.

Protocol per model:
  - 3 repeats of GroupKFold(5) by user_id, each with a different grouping
    (shuffle seed = repeat).
  - Inside every outer training fold: Kalman noise tuned (maximum likelihood),
    then a small random hyperparameter search scored on one grouped 75/25
    user split of the training part (subsampled seconds, second-level
    macro-F1). The chosen parameters are refit on the whole training part.
  - Random models are refit with 3 seeds; deterministic models once.
  - Per-class offsets tuned per outer fold on the other folds' out-of-fold
    predictions (nested), then scored with the competition window metric.
  - Stress-vs-Rest AUC on 10 s windows (score = mean P(Stress)).
  - Final export: fit on all data with the most frequently chosen
    parameters, ONNX parity, size and single-thread inference time.

Usage:
    python training/model_zoo.py lgbm [--jobs 6]
Results: artifacts/zoo/<model>.json, OOF arrays artifacts/zoo/<model>_r<rep>_s<seed>.npy
"""

import argparse
import collections
import json
import os
import tempfile
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from common import M, ROOT, user_folds, per_class_f1  # noqa: E402
import train as T  # noqa: E402
import tune_decoder as TD  # noqa: E402

ZOO = ROOT / "artifacts" / "zoo"
ZOO.mkdir(parents=True, exist_ok=True)
GROUPS = ["raw", "kalman", "norm", "resp", "ctr"]
PARAMS = {"ctr_past_only": True}
REPEATS = (0, 1, 2)
SEEDS = (0, 1, 2)
STEP = 5


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
def _pipe(est, scale):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("clf", est))
    return Pipeline(steps)


def make_model(name, p, seed, jobs):
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="multiclass", subsample=0.7, subsample_freq=1, verbose=-1,
                                  random_state=seed, n_jobs=jobs, **p)
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(tree_method="hist", subsample=0.7, random_state=seed, n_jobs=jobs, **p)
    if name == "catboost":
        import catboost as cb
        return cb.CatBoostClassifier(loss_function="MultiClass", verbose=0, random_seed=seed,
                                     thread_count=jobs, **p)
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **p)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return _pipe(RandomForestClassifier(random_state=seed, n_jobs=jobs, **p), False)
    if name == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return _pipe(ExtraTreesClassifier(random_state=seed, n_jobs=jobs, **p), False)
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return _pipe(LogisticRegression(max_iter=500, **p), True)
    if name == "enet":
        from sklearn.linear_model import LogisticRegression
        return _pipe(LogisticRegression(solver="saga", l1_ratio=p["l1_ratio"], C=p["C"], max_iter=300,
                                        random_state=seed), True)
    if name == "qda":
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        return _pipe(QuadraticDiscriminantAnalysis(priors=[1 / 3] * 3, **p), True)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return _pipe(MLPClassifier(early_stopping=True, max_iter=200, batch_size=512, n_iter_no_change=8,
                                   random_state=seed, **p), True)
    raise KeyError(name)


SPACES = {
    "lgbm": dict(n_estimators=[200, 300, 500], learning_rate=[0.03, 0.05, 0.1], num_leaves=[15, 31, 63],
                 min_child_samples=[50, 200, 500], colsample_bytree=[0.5, 0.7, 0.9], reg_lambda=[0.0, 1.0, 5.0]),
    "xgb": dict(n_estimators=[200, 300, 500], learning_rate=[0.03, 0.05, 0.1], max_depth=[4, 6, 8],
                min_child_weight=[1, 5, 20], colsample_bytree=[0.5, 0.7, 0.9], reg_lambda=[1.0, 5.0]),
    "catboost": dict(iterations=[300, 500], learning_rate=[0.05, 0.1], depth=[4, 6, 8], l2_leaf_reg=[1, 3, 10]),
    "hgb": dict(max_iter=[200, 400], learning_rate=[0.05, 0.1], max_leaf_nodes=[15, 31, 63],
                min_samples_leaf=[20, 100, 300], l2_regularization=[0.0, 1.0]),
    "rf": dict(n_estimators=[100, 150], max_depth=[8, 10, 12], max_features=["sqrt", 0.2, 0.4],
               min_samples_leaf=[5, 20, 50]),
    "et": dict(n_estimators=[100, 150], max_depth=[8, 10, 12], max_features=["sqrt", 0.2, 0.4],
               min_samples_leaf=[5, 20, 50]),
    "logreg": dict(C=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]),
    "enet": dict(C=[0.01, 0.03, 0.1, 0.3, 1.0], l1_ratio=[0.2, 0.5, 0.8]),
    "qda": dict(reg_param=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]),
    "mlp": dict(hidden_layer_sizes=[(64,), (128,), (64, 32), (128, 64)], alpha=[1e-4, 1e-3, 1e-2],
                learning_rate_init=[1e-3, 3e-3]),
}
TRIALS = {"lgbm": 12, "xgb": 12, "catboost": 8, "hgb": 12, "rf": 8, "et": 8,
          "logreg": 8, "enet": 8, "qda": 8, "mlp": 8}
RANDOM = {"lgbm", "xgb", "catboost", "hgb", "rf", "et", "mlp", "enet"}


def fit_model(name, p, seed, jobs, X, y, w):
    m = make_model(name, p, seed, jobs)
    if name == "qda":
        m.fit(X, y)  # balanced through equal priors
    elif name in ("rf", "et", "logreg", "enet", "mlp"):
        m.fit(X, y, clf__sample_weight=w)
    else:
        m.fit(X, y, sample_weight=w)
    return m


def sample_params(name, rng, n):
    space = SPACES[name]
    keys = list(space)
    grid_size = int(np.prod([len(space[k]) for k in keys]))
    seen, out = set(), []
    while len(out) < min(n, grid_size):
        p = {k: space[k][rng.integers(len(space[k]))] for k in keys}
        key = json.dumps(p, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export(name, m, n_features):
    """Return ONNX bytes whose output 'probabilities' is a float tensor."""
    import onnx

    if name in ("lgbm", "xgb"):
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        conv = onnxmltools.convert_lightgbm if name == "lgbm" else onnxmltools.convert_xgboost
        kw = {"zipmap": False} if name == "lgbm" else {}
        onx = conv(m, initial_types=[("input", FloatTensorType([None, n_features]))], target_opset=15, **kw)
        return onx.SerializeToString()
    if name == "catboost":
        d = tempfile.mkdtemp()
        path = os.path.join(d, "c.onnx")
        m.save_model(path, format="onnx")
        onx = onnx.load(path)
        g = onx.graph
        zm = [nd for nd in g.node if nd.op_type == "ZipMap"]
        if zm:
            # drop the ZipMap (sequence of maps) and expose its input tensor as 'probabilities'
            src, dst = zm[0].input[0], zm[0].output[0]
            g.node.remove(zm[0])
            for o in list(g.output):
                if o.name == dst:
                    g.output.remove(o)
            g.node.append(onnx.helper.make_node("Identity", [src], ["probabilities"]))
            g.output.append(onnx.helper.make_tensor_value_info("probabilities", onnx.TensorProto.FLOAT, [None, 3]))
        return onx.SerializeToString()
    from skl2onnx import to_onnx
    X0 = np.zeros((1, n_features), dtype=np.float32)
    onx = to_onnx(m, X0, options={"zipmap": False}, target_opset={"": 15, "ai.onnx.ml": 3})
    return onx.SerializeToString()


def onnx_session(b):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(b, so, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def window_auc(data, oof, idx):
    from sklearn.metrics import roc_auc_score
    s, y = [], []
    for i in idx:
        d = data[i]
        ps = oof[i][:, 2]
        num = np.bincount(d["pw"], weights=d["pc"] * ps[d["ps"]], minlength=d["nw"])
        den = np.bincount(d["pw"], weights=d["pc"], minlength=d["nw"])
        sc = (num / np.maximum(den, 1))[d["keep"]]
        t = d["wtrue"]
        m = t != 1
        s.append(sc[m])
        y.append(t[m] == 2)
    return float(roc_auc_score(np.concatenate(y), np.concatenate(s)))


def nested_score(data, oof, fold_of):
    scores, Tw, Pw = [], [], []
    for k in range(fold_of.max() + 1):
        tr, te = np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)
        cfg, _, _ = TD.tune(data, oof, tr, STEP, use_hmm=False)
        f1, _, _ = TD.evaluate(data, oof, te, cfg)
        scores.append(f1)
        for i in te:
            Tw.append(data[i]["wtrue"])
            Pw.append(TD.window_preds(data[i], M.decode(oof[i], cfg)))
    pc = per_class_f1(np.concatenate(Tw), np.concatenate(Pw))
    return float(np.mean(scores)), scores, pc


def second_macro_f1(m, X, y):
    p = m.predict_proba(X).argmax(1)
    return float(per_class_f1(y, p).mean())


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--repeats", type=int, nargs="*", default=list(REPEATS))
    a = ap.parse_args()
    name = a.model
    seeds = SEEDS if name in RANDOM else (0,)
    t_start = time.time()

    recs = T.load_all()
    fc = T.FeatureCache(recs)
    data = TD.prepare()
    uids = np.array([r["uid"] for r in recs])
    kal_cache = {}

    def feats(i, cfg):
        X = fc.get(i, cfg)[1]
        return np.where(np.isfinite(X), X, np.nan).astype(np.float32)

    results = {"model": name, "repeats": {}, "chosen": []}
    for rep in a.repeats:
        folds = user_folds(uids, 5, seed=rep)
        fold_of = np.zeros(len(recs), dtype=int)
        oofs = {s: [None] * len(recs) for s in seeds}
        for k, (tr, te) in enumerate(folds):
            fold_of[te] = k
            key = tuple(sorted(tr))
            if key not in kal_cache:
                kal_cache[key] = T.tune_kalman([recs[i] for i in tr])
            cfg = T.make_cfg("zoo", kal_cache[key], STEP, GROUPS, PARAMS)

            def stack(idx, stride):
                X = np.concatenate([feats(i, cfg)[::stride] for i in idx])
                y = np.concatenate([recs[i]["y_sec"][::stride] for i in idx])
                w = np.concatenate([T.sample_weights(recs[i]["y_sec"], 1.0, 0)[::stride] for i in idx])
                m = y >= 0
                return X[m], y[m], w[m]

            # inner search: one grouped 75/25 user split of the training part
            rng = np.random.default_rng(1000 * rep + k)
            tr_users = np.unique(uids[tr])
            val_users = set(rng.choice(tr_users, size=max(1, len(tr_users) // 4), replace=False))
            itr = [i for i in tr if uids[i] not in val_users]
            iva = [i for i in tr if uids[i] in val_users]
            Xi, yi, wi = stack(itr, 6)
            Xv, yv, _ = stack(iva, 3)
            best, best_p = -1.0, None
            for p in sample_params(name, rng, TRIALS[name]):
                try:
                    m = fit_model(name, p, 0, a.jobs, Xi, yi, wi)
                    s = second_macro_f1(m, Xv, yv)
                except Exception as e:  # a bad trial must not stop the run
                    print("  trial failed", p, type(e).__name__, str(e)[:100])
                    s = -1.0
                if s > best:
                    best, best_p = s, p
            results["chosen"].append({"rep": rep, "fold": k, "params": best_p, "inner": best})
            Xtr, ytr, wtr = stack(tr, 2)
            for s in seeds:
                m = fit_model(name, best_p, s, a.jobs, Xtr, ytr, wtr)
                for i in te:
                    oofs[s][i] = T.predict_proba(m, feats(i, cfg), STEP)
            print(f"[{name}] rep {rep} fold {k} params {best_p} inner {best:.4f} ({time.time()-t_start:.0f}s)", flush=True)
        rep_res = []
        for s in seeds:
            np.save(ZOO / f"{name}_r{rep}_s{s}.npy", np.array(oofs[s], dtype=object), allow_pickle=True)
            np.save(ZOO / f"folds_r{rep}.npy", fold_of)
            mean, fs, pc = nested_score(data, oofs[s], fold_of)
            auc = window_auc(data, oofs[s], range(len(recs)))
            rep_res.append(dict(seed=s, mean=mean, folds=fs, per_class=pc.tolist(), auc=auc))
            print(f"[{name}] rep {rep} seed {s}: macroF1 {mean:.4f} per-class {np.round(pc, 3)} AUC {auc:.4f}", flush=True)
        results["repeats"][rep] = rep_res

    allr = [r for rr in results["repeats"].values() for r in rr]
    results["mean"] = float(np.mean([r["mean"] for r in allr]))
    results["std_folds"] = float(np.std([f for r in allr for f in r["folds"]]))
    results["rep_means"] = {rep: float(np.mean([r["mean"] for r in rr])) for rep, rr in results["repeats"].items()}
    results["per_class"] = dict(zip(M.CLASSES, np.mean([r["per_class"] for r in allr], axis=0).round(4).tolist()))
    results["auc"] = float(np.mean([r["auc"] for r in allr]))

    # final export with the most frequently chosen parameters
    try:
        cnt = collections.Counter(json.dumps(c["params"], sort_keys=True, default=str) for c in results["chosen"])
        p = json.loads(cnt.most_common(1)[0][0])
        if "hidden_layer_sizes" in p:
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        kal = T.tune_kalman(recs)
        cfg = T.make_cfg("zoo", kal, STEP, GROUPS, PARAMS)
        X = np.concatenate([feats(i, cfg)[::2] for i in range(len(recs))])
        y = np.concatenate([r["y_sec"][::2] for r in recs])
        w = np.concatenate([T.sample_weights(r["y_sec"], 1.0, 0)[::2] for r in recs])
        mk = y >= 0
        m = fit_model(name, p, 0, a.jobs, X[mk], y[mk], w[mk])
        b = export(name, m, X.shape[1])
        (ZOO / f"{name}.onnx").write_bytes(b)
        sess = onnx_session(b)
        j = next(i for i, r in enumerate(recs) if r["prot"] == "STRESS" and r["name"] == "f01")
        Xf = feats(j, cfg)
        pts, _ = M.step_points(len(Xf), STEP)
        Xp = np.ascontiguousarray(Xf[pts])
        inp = sess.get_inputs()[0].name
        po = sess.run(["probabilities"], {inp: Xp})[0]
        dp = float(np.abs(po - m.predict_proba(Xp)).max())
        t = time.perf_counter()
        for _ in range(20):
            sess.run(["probabilities"], {inp: Xp})
        ms = (time.perf_counter() - t) / 20 * 1000
        results["export"] = dict(params=p, size_mb=len(b) / 1e6, onnx_ms=ms, parity=dp, ok=True)
    except Exception as e:
        results["export"] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    results["minutes"] = (time.time() - t_start) / 60
    (ZOO / f"{name}.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"[{name}] DONE mean {results['mean']:.4f} rep means {results['rep_means']} per-class {results['per_class']} "
          f"AUC {results['auc']:.4f} export {results['export']}", flush=True)


if __name__ == "__main__":
    main()
