"""Step 3: session-type classifier and session specialists with soft gating.

Per repeat (3) and outer fold (5), everything fitted on the training recordings:
  - session classifier: logistic regression on whole-recording physiology
    summaries (HR range relative to its own low quantile, HR spread, beat
    dropout rate, pulse motion level and burstiness). It does not use the
    3-class model, so its training inputs never depend on test people.
  - Stress specialist: XGBoost Stress vs Rest on STRESS training recordings.
  - Exercise specialist: XGBoost Activity vs Rest on AEROBIC / ANAEROBIC ones.
  - soft gating: p = p(stress) * [pR_s, 0, pS_s] + p(exercise) * [pR_e, pA_e, 0]
  - decoders scored as in stat_ext (nested per-class offsets), plus the z
    threshold of Step 2 applied only to recordings the classifier calls stress
    sessions (on the single 3-class model and on the gated specialists).
XGBoost parameters: the zoo's in-fold choices for the same (repeat, fold).

Usage: python training/specialists.py
"""

import json
import time

import numpy as np

from common import M, ROOT, per_class_f1
import train as T
import tune_decoder as TD
import stat_ext as SE
import adaptive as AD

OUT = ROOT / "artifacts" / "stat"
SEEDS = (0, 1, 2)
STEP = 5
LOW = ("S15", "S17", "f11", "f17", "f18", "S01", "f13")
SESSION_KEYS = ("hr_rise", "hr_spread", "miss_rate", "motion", "motion_burst")


def session_features(b):
    """Whole-recording summaries (no time or length information)."""
    hr = M._ratio(b["hr_s"], b["hr_c"])
    hr = hr[np.isfinite(hr)]
    p05, p50, p95 = np.percentile(hr, (5, 50, 95)) if len(hr) > 30 else (np.nan,) * 3
    miss = b["ibi_miss"].sum() / max(b["ibi_c"].sum(), 1)
    mot = M._ratio(b["bvp_d"], b["bvp_dc"])
    mot = mot[np.isfinite(mot)]
    m50, m90 = np.percentile(mot, (50, 90)) if len(mot) > 30 else (np.nan, np.nan)
    return np.array([np.log(p95 / p05), (p95 - p50) / max(p50 - p05, 1.0), miss,
                     np.log(m50 + 1e-6), np.log((m90 + 1e-6) / (m50 + 1e-6))])


def fit_session(F, y):
    from sklearn.linear_model import LogisticRegression
    med = np.nanmedian(F, 0)
    Fi = np.where(np.isfinite(F), F, med)
    mu, sd = Fi.mean(0), Fi.std(0) + 1e-9
    clf = LogisticRegression(C=1.0, max_iter=1000).fit((Fi - mu) / sd, y)
    return dict(med=med.tolist(), mu=mu.tolist(), sd=sd.tolist(), coef=clf.coef_[0].tolist(),
                intercept=float(clf.intercept_[0]))


def session_prob(F, s):
    Fi = np.where(np.isfinite(F), F, s["med"])
    z = ((Fi - s["mu"]) / s["sd"]) @ np.asarray(s["coef"]) + s["intercept"]
    return 1.0 / (1.0 + np.exp(-z))  # probability of a STRESS session


def gated_z_nested(data, oof, fold_of, gate, prot, uid):
    """Nested: offsets, then the zfloor rule on gated recordings only."""
    grid = AD.ZF_GRID
    folds, C, user = [], np.zeros((3, 3), int), {}
    for k in range(fold_of.max() + 1):
        tr, te = np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)
        cfg, _, _ = TD.tune(data, oof, tr, STEP, use_hmm=False)
        R = {}
        for i in range(len(data)):
            p = np.clip(oof[i], 1e-6, 1)
            R[i] = AD.Rec(data[i], M.decode(oof[i], cfg), np.log(p[:, 2]) - np.log(p[:, 0]))

        def conf(i, q):
            return R[i].confusion("zfloor", q) if gate[i] else R[i].base_cm

        best = max(grid, key=lambda q: float(AD.f1_from_cm(sum(conf(i, q) for i in tr)).mean()))
        c = sum(conf(i, best) for i in te)
        folds.append(float(AD.f1_from_cm(c).mean()))
        C += c
        for i in te:
            if prot[i] == "STRESS":
                user[uid[i]] = float(AD.f1_from_cm(conf(i, best))[2])
    return float(np.mean(folds)), folds, AD.f1_from_cm(C), user


def session_rest(data, oof, fold_of, prot):
    """Rest F1 inside stress sessions and inside exercise sessions (nested offsets)."""
    C = {"STRESS": np.zeros((3, 3), int), "EXERCISE": np.zeros((3, 3), int)}
    for k in range(fold_of.max() + 1):
        tr, te = np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)
        cfg, _, _ = TD.tune(data, oof, tr, STEP, use_hmm=False)
        for i in te:
            key = "STRESS" if prot[i] == "STRESS" else "EXERCISE"
            C[key] += AD.cm(data[i]["wtrue"], TD.window_preds(data[i], M.decode(oof[i], cfg)))
    return {k: float(AD.f1_from_cm(v)[0]) for k, v in C.items()}


def main():
    t0 = time.time()
    recs = T.load_all()
    data = TD.prepare()
    prot = [r["prot"] for r in recs]
    uid = [r["uid"] for r in recs]
    y_sess = np.array([p == "STRESS" for p in prot], dtype=int)
    SF = np.array([session_features(r["base"][2]) for r in recs])
    zoo = json.loads((ROOT / "artifacts" / "zoo" / "xgb.json").read_text())
    chosen = {(c["rep"], c["fold"]): c["params"] for c in zoo["chosen"]}
    fc = T.FeatureCache(recs)
    kal = {}
    runs, sess_acc, sess_models = [], [], []
    for rep in (0, 1, 2):
        fold_of = np.load(OUT / f"folds_r{rep}.npy")
        spec = {s: [None] * len(recs) for s in SEEDS}
        gate_p = np.zeros(len(recs))
        for k in range(fold_of.max() + 1):
            tr, te = np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)
            sm = fit_session(SF[tr], y_sess[tr])
            sess_models.append(sm)
            gate_p[te] = session_prob(SF[te], sm)
            key = tuple(sorted(tr))
            if key not in kal:
                kal[key] = T.tune_kalman([recs[i] for i in tr])
            cfg = T.make_cfg("stat", kal[key], STEP, SE.GROUPS, SE.PARAMS)

            def get(i):
                X = fc.get(i, cfg)[1]
                return np.where(np.isfinite(X), X, np.nan).astype(np.float32)

            def stack(idx, keep):
                X = np.concatenate([get(i)[::2] for i in idx])
                y = np.concatenate([recs[i]["y_sec"][::2] for i in idx])
                w = np.concatenate([T.sample_weights(recs[i]["y_sec"], 1.0, 0)[::2] for i in idx])
                m = np.isin(y, keep)
                return X[m], y[m], w[m]

            str_idx = [i for i in tr if prot[i] == "STRESS"]
            ex_idx = [i for i in tr if prot[i] != "STRESS"]
            Xs, ys, ws = stack(str_idx, (0, 2))
            Xe, ye, we = stack(ex_idx, (0, 1))
            p = chosen[(rep, k)]
            for s in SEEDS:
                ms = SE.make("xgb", p, s).fit(Xs, (ys == 2).astype(int), sample_weight=ws)
                me = SE.make("xgb", p, s).fit(Xe, (ye == 1).astype(int), sample_weight=we)
                for i in te:
                    X = get(i)
                    ps = T.predict_proba(ms, X, STEP)[:, 1]
                    pe = T.predict_proba(me, X, STEP)[:, 1]
                    g = gate_p[i]
                    spec[s][i] = np.stack([g * (1 - ps) + (1 - g) * (1 - pe), (1 - g) * pe, g * ps], 1)
            print(f"rep {rep} fold {k} ({time.time()-t0:.0f}s)", flush=True)
        acc = float(np.mean((gate_p > 0.5) == y_sess))
        sess_acc.append(acc)
        gate = gate_p > 0.5
        for s in SEEDS:
            np.save(OUT / f"spec_r{rep}_s{s}.npy", np.array(spec[s], dtype=object), allow_pickle=True)
            base = np.load(OUT / f"base_r{rep}_s{s}.npy", allow_pickle=True)
            for name, oof in (("single", base), ("specialists", spec[s])):
                mean, fs, pc, _, us = SE.nested(data, oof, fold_of, recs)
                rest = session_rest(data, oof, fold_of, prot)
                runs.append(dict(method=name, rep=rep, seed=s, mean=mean, folds=fs, per_class=pc.tolist(),
                                 user_stress=us, rest_in=rest))
                zm, zf, zpc, zus = gated_z_nested(data, oof, fold_of, gate, prot, uid)
                runs.append(dict(method=name + "+gated_z", rep=rep, seed=s, mean=zm, folds=zf,
                                 per_class=zpc.tolist(), user_stress=zus, rest_in=None))
            print(f"rep {rep} seed {s}: " + "  ".join(f"{r['method']} {r['mean']:.4f}" for r in runs[-4:])
                  + f" | session acc {acc:.3f}", flush=True)
    summary = {}
    for m in ("single", "specialists", "single+gated_z", "specialists+gated_z"):
        rr = [r for r in runs if r["method"] == m]
        summary[m] = dict(
            mean=float(np.mean([r["mean"] for r in rr])),
            std=float(np.std([f for r in rr for f in r["folds"]])),
            reps=[float(np.mean([r["mean"] for r in rr if r["rep"] == rep])) for rep in (0, 1, 2)],
            per_class=dict(zip(M.CLASSES, np.mean([r["per_class"] for r in rr], 0).round(4).tolist())),
            low={u: float(np.mean([r["user_stress"][u] for r in rr if u in r["user_stress"]])) for u in LOW},
            rest_in=({k: float(np.mean([r["rest_in"][k] for r in rr])) for k in ("STRESS", "EXERCISE")}
                     if rr[0]["rest_in"] else None))
        s = summary[m]
        print(f"[{m}] mean {s['mean']:.4f} +- {s['std']:.4f} reps {[round(v, 4) for v in s['reps']]} "
              f"per-class {s['per_class']} rest-in {s['rest_in']} low "
              + ", ".join(f"{u}: {v:.2f}" for u, v in s["low"].items()), flush=True)
    print(f"session classifier accuracy per repeat: {sess_acc}")
    coefs = np.mean([m["coef"] for m in sess_models], 0)
    (OUT / "specialists.json").write_text(json.dumps(dict(
        summary=summary, session_acc=sess_acc, session_keys=SESSION_KEYS, session_coef_mean=coefs.tolist(),
        runs=runs), indent=1))


if __name__ == "__main__":
    main()
