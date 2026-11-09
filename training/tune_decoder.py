"""Variant E: tune per-class log offsets and HMM stickiness on out-of-fold
probabilities, maximizing the competition metric.

Nested evaluation: for each outer fold k, the decoder (transition matrix,
priors, offsets, stickiness) is estimated only from recordings of the other
folds, then applied to fold k. The final decoder is tuned on all OOF
predictions and written into the ONNX metadata.

Usage:
    python training/tune_decoder.py --name D [--write-model artifacts/model_D.onnx --out artifacts/model_E.onnx]
"""

import argparse
import json
import time

import numpy as np

from common import M, ROOT, list_recordings, load_recording, second_labels, per_class_f1, confusion

ART = ROOT / "artifacts"
OFFSET_GRID = np.round(np.arange(-2.0, 2.01, 0.25), 2)
ALPHA_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
STRESS_ALPHA_GRID = [1.0, 3.0, 10.0, 30.0]
EMIS_GRID = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
SHIFT_GRID = list(range(-60, 61, 5))


# --------------------------------------------------------------------------
def prepare():
    """Per recording: second labels and the compressed (window, second, count) table."""
    data = []
    for prot, name, path in list_recordings():
        df, y, uid = load_recording(path)
        t = df.index.values.astype(np.int64)
        sec = (t - t[0]) // M.NS
        n = int(sec[-1]) + 1
        m = y >= 0
        w = t[m] // (10 * M.NS)
        w = w - w.min()
        nw = int(w.max()) + 1
        ytrue = np.bincount(w * 3 + y[m], minlength=nw * 3).reshape(nw, 3)
        keep = ytrue.sum(1) > 0
        # unique (window, second) pairs with row counts, scored rows only
        key = w * (n + 1) + sec[m]
        u, cnt = np.unique(key, return_counts=True)
        data.append(dict(prot=prot, name=name, uid=uid, y_sec=second_labels(sec, y, n),
                         pw=u // (n + 1), ps=u % (n + 1), pc=cnt, nw=nw, keep=keep,
                         wtrue=ytrue[keep].argmax(1)))
    return data


def window_preds(d, codes):
    cp = np.bincount(d["pw"] * 3 + codes[d["ps"]], weights=d["pc"], minlength=d["nw"] * 3).reshape(d["nw"], 3)
    return cp[d["keep"]].argmax(1)


def evaluate(data, oof, idx, cfg):
    T, P = [], []
    for i in idx:
        T.append(data[i]["wtrue"])
        P.append(window_preds(data[i], M.decode(oof[i], cfg)))
    T, P = np.concatenate(T), np.concatenate(P)
    pc = per_class_f1(T, P)
    return float(pc.mean()), pc, confusion(T, P)


# --------------------------------------------------------------------------
def estimate_hmm(data, idx):
    """Priors and 1 s transition counts from training label sequences."""
    C = np.ones((3, 3))  # add-one smoothing
    pri = np.zeros(3)
    for i in idx:
        y = data[i]["y_sec"]
        y = y[y >= 0]  # unscored seconds removed, sequence compressed
        pri += np.bincount(y, minlength=3)
        np.add.at(C, (y[:-1], y[1:]), 1)
    T = C / C.sum(1, keepdims=True)
    return T, pri / pri.sum()


def build_cfg(T1, priors, offsets, alpha, alpha_s, step, use_hmm=True, emis_w=1.0):
    cfg = {"priors": priors.tolist(), "offsets": [0.0, float(offsets[0]), float(offsets[1])]}
    if use_hmm:
        cfg["emis_w"] = float(emis_w)
    if use_hmm:
        T = T1.copy()
        off = ~np.eye(3, dtype=bool)
        T[off] *= alpha
        T[2, :2] *= alpha_s  # leaving Stress
        T[:2, 2] *= alpha_s  # entering Stress
        np.fill_diagonal(T, 0.0)
        T = np.minimum(T, 0.5)
        np.fill_diagonal(T, 1.0 - T.sum(1))
        Ts = np.linalg.matrix_power(T, step)
        cfg["log_trans"] = np.log(np.maximum(Ts, 1e-12)).tolist()
        cfg["hmm_step"] = step
    return cfg


def tune(data, oof, idx, step, use_hmm=True, rounds=3, verbose=False, shift_mode=None):
    T1, priors = estimate_hmm(data, idx)
    p = {"oA": 0.0, "oS": 0.0, "alpha": 1.0, "alpha_s": 1.0, "emis_w": 1.0}
    grids = {"oA": OFFSET_GRID, "oS": OFFSET_GRID}
    if use_hmm:
        grids.update(emis_w=EMIS_GRID, alpha=ALPHA_GRID, alpha_s=STRESS_ALPHA_GRID)
    if shift_mode == "global":
        p["shift"] = 0
        grids["shift"] = SHIFT_GRID
    elif shift_mode == "per":
        for k in ("enS", "lvS", "enA", "lvA"):
            p[k] = 0
            grids[k] = SHIFT_GRID

    def score(q):
        return evaluate(data, oof, idx, full_cfg(q))[0]

    def full_cfg(q):
        cfg = build_cfg(T1, priors, (q["oA"], q["oS"]), q["alpha"], q["alpha_s"], step, use_hmm, q["emis_w"])
        if "shift" in q:
            cfg["shift"] = int(q["shift"])
        if "enS" in q:
            cfg["enter"] = [0, int(q["enA"]), int(q["enS"])]
            cfg["leave"] = [0, int(q["lvA"]), int(q["lvS"])]
        return cfg

    best = score(p)
    for r in range(rounds):
        changed = False
        for k, g in grids.items():
            for v in g:
                if v == p[k]:
                    continue
                q = dict(p, **{k: v})
                s = score(q)
                if s > best + 1e-6:
                    best, p, changed = s, q, True
        if verbose:
            print(f"  round {r}: {best:.4f} {p}", flush=True)
        if not changed:
            break
    return full_cfg(p), p, best


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="D")
    ap.add_argument("--write-model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--final-hmm", action="store_true", help="ship the HMM decoder (default: offsets only)")
    a = ap.parse_args()

    summary = json.loads((ART / f"cv_{a.name}.json").read_text())
    step = int(summary.get("pred_step", 5))
    oof = np.load(ART / f"oof_{a.name}.npy", allow_pickle=True)
    fold_of = np.load(ART / f"folds_{a.name}.npy")
    data = prepare()
    K = fold_of.max() + 1
    t0 = time.time()

    modes = {
        "argmax": None,
        "offsets": dict(use_hmm=False),
        "offsets+HMM": dict(use_hmm=True),
    }
    report = {}
    for mode, kw in modes.items():
        fold_scores, T, P = [], [], []
        for k in range(K):
            tr = np.flatnonzero(fold_of != k)
            te = np.flatnonzero(fold_of == k)
            if kw is None:
                cfg = {}
            else:
                cfg, p, _ = tune(data, oof, tr, step, **kw)
            f1, pc, _ = evaluate(data, oof, te, cfg)
            fold_scores.append(f1)
            for i in te:
                T.append(data[i]["wtrue"])
                P.append(window_preds(data[i], M.decode(oof[i], cfg)))
        T, P = np.concatenate(T), np.concatenate(P)
        pc = per_class_f1(T, P)
        cm = confusion(T, P)
        report[mode] = dict(mean=float(np.mean(fold_scores)), std=float(np.std(fold_scores)),
                            folds=fold_scores, per_class=dict(zip(M.CLASSES, pc.tolist())),
                            confusion=cm.tolist())
        print(f"{mode:12s} macroF1 {np.mean(fold_scores):.4f} +- {np.std(fold_scores):.4f} "
              f"[Rest {pc[0]:.3f} Activity {pc[1]:.3f} Stress {pc[2]:.3f}] ({time.time()-t0:.0f}s)", flush=True)
        print("   confusion (rows true Rest/Activity/Stress):", cm.tolist())

    print("tuning final decoder on all OOF predictions")
    cfg, p, best = tune(data, oof, np.arange(len(oof)), step, use_hmm=a.final_hmm, verbose=True)
    print("final decoder params:", p, f"(in-sample {best:.4f})")
    report["final_params"] = p
    (ART / f"decoder_{a.name}.json").write_text(json.dumps(dict(report=report, cfg=cfg), indent=1))

    if a.write_model:
        import onnx

        model = onnx.load(a.write_model)
        for prop in model.metadata_props:
            if prop.key == "adm_config":
                base = json.loads(prop.value)
                base.update(cfg)
                base["variant_label"] = "E"
                prop.value = json.dumps(base)
        onnx.save_model(model, a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
