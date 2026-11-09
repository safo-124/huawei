"""Cross-validate a variant with GroupKFold by user_id, then fit and export ONNX.

Usage:
    python training/train.py --variant C --pred-step 5 [--chunks 60 300] [--export]
    python training/train.py --variant D --name D_sw2 --stress-weight 2 --trans-drop 30

All tuning (Kalman noise parameters) happens inside the training folds.
Out-of-fold per-second probabilities are saved for decoder tuning
(tune_decoder.py), together with the fold assignment.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import (
    M, ROOT, list_recordings, load_recording, second_labels, window_majority,
    macro_f1, per_class_f1, confusion, user_folds,
)

ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

LGB_PARAMS = dict(
    objective="multiclass",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=200,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    verbose=-1,
)

HR_GRID = [(q, r) for q in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1) for r in (0.03, 0.1, 0.25, 0.5, 1.0, 2.0)]
TMP_GRID = [(q, r) for q in (1e-5, 1e-4, 1e-3, 1e-2) for r in (1e-3, 1e-2, 3e-2, 0.1, 0.3)]


# --------------------------------------------------------------------------
def load_all():
    recs = []
    t0 = time.time()
    for prot, name, path in list_recordings():
        X_in, y_row, uid = load_recording(path)
        base = M.per_second(X_in)
        sec, n, b = base
        recs.append(dict(prot=prot, name=name, uid=uid, df=X_in, y_row=y_row, base=base,
                         sec=sec, y_sec=second_labels(sec, y_row, n),
                         hr1=M._ratio(b["hr_s"], b["hr_c"]), tmp1=M._ratio(b["tmp_s"], b["tmp_c"])))
    print(f"loaded {len(recs)} recordings in {time.time()-t0:.1f}s")
    return recs


def tune_kalman(train_recs):
    """Maximum likelihood choice of HR and temperature noise on training recordings."""
    def best(key, grid):
        ll = [sum(M.cv_loglik(r[key], q, rr) for r in train_recs if np.isfinite(r[key]).sum() > 30)
              for q, rr in grid]
        return grid[int(np.argmax(ll))]

    hq, hr = best("hr1", HR_GRID)
    tq, tr = best("tmp1", TMP_GRID)
    return {"hr_q": hq, "hr_r": hr, "tmp_q": tq, "tmp_r": tr}


class FeatureCache:
    def __init__(self, recs):
        self.recs, self.cache = recs, {}

    def get(self, i, cfg):
        key = (i, json.dumps(cfg, sort_keys=True))
        if key not in self.cache:
            r = self.recs[i]
            _, names, X = M.compute_features(r["df"], cfg, r["base"])
            self.cache[key] = (names, X)
        return self.cache[key]


def sample_weights(y, stress_weight, trans_drop, trans_w=0.0):
    """Balanced class weights, Stress multiplied by stress_weight, and
    seconds within trans_drop s after a label change weighted by trans_w."""
    m = y >= 0
    cnt = np.bincount(y[m], minlength=3).astype(float)
    cw = m.sum() / (3 * np.maximum(cnt, 1))
    cw[2] *= stress_weight
    w = np.where(m, cw[np.clip(y, 0, 2)], 0.0)
    if trans_drop > 0:
        ch = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] >= 0)) + 1
        near = np.zeros(len(y), dtype=bool)
        for c in ch:
            near[c:c + trans_drop] = True
        w[near] *= trans_w
    return w


def fit(Xs, ys, opts, stride=2, seed=0):
    import lightgbm as lgb

    X = np.concatenate([x[::stride] for x in Xs])
    ws = [sample_weights(y, opts.get("stress_weight", 1.0), opts.get("trans_drop", 0))[::stride] for y in ys]
    y = np.concatenate([y[::stride] for y in ys])
    w = np.concatenate(ws)
    m = (y >= 0) & (w > 0)
    params = dict(LGB_PARAMS)
    params.update(opts.get("lgb", {}))
    clf = lgb.LGBMClassifier(random_state=seed, **params)
    clf.fit(X[m], y[m], sample_weight=w[m])
    return clf


def predict_proba(clf, X, step=1):
    """Per-second probabilities exactly as run() computes them (with pred_step)."""
    pts, back = M.step_points(X.shape[0], step)
    return clf.predict_proba(X[pts])[back]


def score_rows(parts):
    """parts: list of (index, y_row, pred_row). Pools all windows."""
    T, P = [], []
    for idx, yt, yp in parts:
        t, p = window_majority(idx, yt, yp)
        T.append(t)
        P.append(p)
    T, P = np.concatenate(T), np.concatenate(P)
    return macro_f1(T, P), per_class_f1(T, P), confusion(T, P)


def predict_chunked(clf, r, cfg, chunk_s, step=1):
    """Run feature extraction independently on chunks of chunk_s seconds."""
    df = r["df"]
    t = df.index.values.astype(np.int64)
    cid = (t - t[0]) // (chunk_s * M.NS)
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    ends = np.r_[starts[1:], len(df)]
    out = np.empty(len(df), dtype=np.int64)
    for a, b in zip(starts, ends):
        sec, _, X = M.compute_features(df.iloc[a:b], cfg)
        out[a:b] = predict_proba(clf, X, step).argmax(1)[sec]
    return out


def make_cfg(variant, kalman, step, groups=None, params=None):
    cfg = {"variant": variant, "pred_step": step, "kalman": kalman}
    if groups:
        cfg["groups"] = list(groups)
    if params:
        cfg["params"] = dict(params)
    return cfg


def cross_validate(recs, fc, variant, opts, chunks=(), n_splits=5, step=1, groups=None, params=None):
    uids = [r["uid"] for r in recs]
    folds = user_folds(uids, n_splits)
    uses_kalman = "kalman" in M.resolve({"variant": variant, "groups": groups})[0]
    res = {"full": []}
    for c in chunks:
        res[f"chunk{c}"] = []
    oof = [None] * len(recs)
    fold_of = np.zeros(len(recs), dtype=int)
    all_parts = []
    for k, (tr, te) in enumerate(folds):
        kal = tune_kalman([recs[i] for i in tr]) if uses_kalman else {}
        cfg = make_cfg(variant, kal, step, groups, params)
        clf = fit([fc.get(i, cfg)[1] for i in tr], [recs[i]["y_sec"] for i in tr], opts)
        parts = []
        for i in te:
            r = recs[i]
            proba = predict_proba(clf, fc.get(i, cfg)[1], step)
            oof[i] = proba
            fold_of[i] = k
            parts.append((r["df"].index, r["y_row"], proba.argmax(1)[r["sec"]]))
        all_parts += parts
        f1, pc, _ = score_rows(parts)
        res["full"].append(f1)
        msg = f"fold {k}: macroF1 {f1:.4f} [Rest {pc[0]:.3f} Activity {pc[1]:.3f} Stress {pc[2]:.3f}]"
        if uses_kalman:
            msg += f" kalman {kal}"
        for c in chunks:
            cp = [(recs[i]["df"].index, recs[i]["y_row"], predict_chunked(clf, recs[i], cfg, c, step)) for i in te]
            res[f"chunk{c}"].append(score_rows(cp)[0])
            msg += f" | chunk{c} {res[f'chunk{c}'][-1]:.4f}"
        print(msg, flush=True)
    f1, pc, cm = score_rows(all_parts)
    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}
    summary["folds"] = res["full"]
    summary["pooled"] = f1
    summary["per_class"] = dict(zip(M.CLASSES, pc.tolist()))
    summary["confusion"] = cm.tolist()
    return summary, oof, fold_of


def export_onnx(clf, n_features, cfg, path):
    import onnx
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    onx = onnxmltools.convert_lightgbm(
        clf, initial_types=[("input", FloatTensorType([None, n_features]))],
        zipmap=False, target_opset=15,
    )
    meta = onx.metadata_props.add()
    meta.key = "adm_config"
    meta.value = json.dumps(cfg)
    onnx.save_model(onx, str(path))
    return path


def check_parity(clf, path, Xs):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = [o.name for o in sess.get_outputs()]
    worst, flips, tot = 0.0, 0, 0
    for X in Xs[::10]:
        p1 = clf.predict_proba(X)
        p2 = sess.run([out[-1]], {"input": X})[0]
        worst = max(worst, float(np.abs(p1 - p2).max()))
        flips += int((p1.argmax(1) != p2.argmax(1)).sum())
        tot += len(X)
    print(f"ONNX vs LightGBM max |dp| = {worst:.2e}, argmax flips {flips}/{tot}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B")
    ap.add_argument("--name", default=None, help="artifact name (defaults to variant)")
    ap.add_argument("--groups", nargs="*", default=None, help="override feature groups")
    ap.add_argument("--chunks", type=int, nargs="*", default=[])
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--pred-step", type=int, default=5)
    ap.add_argument("--stress-weight", type=float, default=1.0)
    ap.add_argument("--trans-drop", type=int, default=0)
    ap.add_argument("--params", default=None, help='feature params as JSON, e.g. {"ctr_past_only": true}')
    a = ap.parse_args()
    name = a.name or a.variant
    opts = {"stress_weight": a.stress_weight, "trans_drop": a.trans_drop}

    recs = load_all()
    fc = FeatureCache(recs)
    t0 = time.time()
    summary, oof, fold_of = cross_validate(recs, fc, a.variant, opts, a.chunks, step=a.pred_step, groups=a.groups,
                                           params=json.loads(a.params) if a.params else None)
    summary.update(pred_step=a.pred_step, opts=opts, groups=a.groups, variant=a.variant)
    print(f"CV {name}: macroF1 {summary['full'][0]:.4f} +- {summary['full'][1]:.4f} "
          f"pooled {summary['pooled']:.4f} per-class {summary['per_class']} ({time.time()-t0:.0f}s)")
    for c in a.chunks:
        print(f"  chunk{c}: {summary[f'chunk{c}'][0]:.4f} +- {summary[f'chunk{c}'][1]:.4f}")
    (ART / f"cv_{name}.json").write_text(json.dumps(summary, indent=1))
    np.save(ART / f"oof_{name}.npy", np.array(oof, dtype=object), allow_pickle=True)
    np.save(ART / f"folds_{name}.npy", fold_of)

    if a.export:
        uses_kalman = "kalman" in M.resolve({"variant": a.variant, "groups": a.groups})[0]
        kal = tune_kalman(recs) if uses_kalman else {}
        cfg = make_cfg(a.variant, kal, a.pred_step, a.groups, json.loads(a.params) if a.params else None)
        Xs = [fc.get(i, cfg)[1] for i in range(len(recs))]
        names = fc.get(0, cfg)[0]
        clf = fit(Xs, [r["y_sec"] for r in recs], opts)
        cfg["features"] = names
        path = ART / f"model_{name}.onnx"
        export_onnx(clf, len(names), cfg, path)
        check_parity(clf, path, Xs)
        # reference probabilities through the training path, for the submission parity check
        j = next(i for i, r in enumerate(recs) if r["prot"] == "STRESS" and r["name"] == "f01")
        np.savez(ART / f"ref_{name}.npz", proba=predict_proba(clf, Xs[j], a.pred_step),
                 recording=str(ROOT / "data" / "STRESS" / "f01.parquet"))
        print("final kalman params:", kal)
        imp = pd.Series(clf.booster_.feature_importance("gain"), index=names).sort_values(ascending=False)
        print("top features:\n", imp.head(15).round(0).to_string())


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Per-fold tuning of GP hyperparameters and SCR detector (items 6 and 8b)
# --------------------------------------------------------------------------
GP_GRIDS = {
    "hr": [dict(hr_ell=l, hr_sig=s, hr_r=r) for l in (30, 60, 120, 300, 600)
           for s in (10, 20, 40) for r in (0.003, 0.01, 0.03, 0.1)],
    "tmp": [dict(tmp_ell=l, tmp_sig=s, tmp_r=r) for l in (10, 20, 60, 180)
            for s in (1.0, 3.0, 10.0) for r in (0.01, 0.03, 0.1, 0.3)],
    "eda": [dict(edat_ell=lt, edat_sig=st, edap_ell=lp, edap_sig=sp, eda_r=r)
            for lt in (15, 30, 60, 120) for st in (0.3, 1.0, 3.0) for lp in (2, 5, 15)
            for sp in (0.05, 0.15, 0.5) for r in (1e-5, 1e-4)],
    "qp": [dict(qp_period=pp, qp_ell=l, qp_sig=s, qp_r=r) for pp in (3, 4, 5, 6, 8)
           for l in (1, 2, 5, 15) for s in (0.03, 0.1, 0.3) for r in (1e-5, 1e-4)],
}


def tune_gp(train_recs):
    """Marginal likelihood (Kalman prediction error) maximization per signal."""
    inputs = [M.gp_inputs(r["base"][2], r["base"][1]) for r in train_recs]
    best = dict(M.DEFAULT_GP)
    for sig, grid in GP_GRIDS.items():
        scores = []
        for g in grid:
            p = dict(best, **g)
            sy = M.gp_systems(p)[sig]
            scores.append(sum(M.gp_loglik(x[sig], sy) for x in inputs))
        best.update(grid[int(np.argmax(scores))])
    return {f"gp_{k}": v for k, v in best.items()}


SCR_GRID = [dict(scr_rise=r, scr_decay=d, scr_thr=t) for r in (0.5, 1.0, 2.0)
            for d in (3.0, 6.0, 10.0) for t in (1.5, 2.0, 3.0, 4.0) if r < d]


def _auc(pos, neg):
    from scipy.stats import rankdata

    x = np.r_[pos, neg]
    rk = rankdata(x)
    return (rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def tune_scr(train_recs):
    """Choose template constants and threshold that best separate Stress from
    Rest seconds (AUC of the 30 s detection count) in the training STRESS sessions."""
    rs = [r for r in train_recs if r["prot"] == "STRESS"]
    best, best_auc = SCR_GRID[0], -1
    for g in SCR_GRID:
        pos, neg = [], []
        for r in rs:
            sec, n, b = r["base"]
            f = M.features_scr(b, n, g)
            x = f["scr_count_30"] + 0.01 * f["scr_sum_30"]  # sum breaks ties between counts
            y = r["y_sec"]
            pos.append(x[y == 2][::5])
            neg.append(x[y == 0][::5])
        pos, neg = np.concatenate(pos), np.concatenate(neg)
        ok_p, ok_n = np.isfinite(pos), np.isfinite(neg)
        a = _auc(pos[ok_p], neg[ok_n])
        if a > best_auc:
            best, best_auc = g, a
    return dict(best), float(best_auc)
