"""Item 8c: decision-stage diagnostics for Stress vs not-Stress.

Uses the out-of-fold probabilities of the shipped variant (E = D + offsets,
seed 0). Window score = row-weighted mean of P(Stress) over the scored rows of
each 10 s window. Reports the ROC, AUC, the Stress detection rate at fixed
false-alarm rates (Neyman-Pearson operating points), and the operating point
actually produced by the nested-tuned offsets.

Usage:
    python training/roc_stress.py --name D
"""

import argparse
import json

import numpy as np

from common import M, ROOT
import tune_decoder as TD

ART = ROOT / "artifacts"


def window_scores(d, proba):
    ps = proba[:, 2]
    num = np.bincount(d["pw"], weights=d["pc"] * ps[d["ps"]], minlength=d["nw"])
    den = np.bincount(d["pw"], weights=d["pc"], minlength=d["nw"])
    return (num / np.maximum(den, 1))[d["keep"]]


def main():
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="D")
    a = ap.parse_args()
    oof = np.load(ART / f"oof_{a.name}.npy", allow_pickle=True)
    fold_of = np.load(ART / f"folds_{a.name}.npy")
    data = TD.prepare()

    y, sc, dec = [], [], []
    for k in range(fold_of.max() + 1):
        tr, te = np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)
        cfg, _, _ = TD.tune(data, oof, tr, 5, use_hmm=False)
        for i in te:
            d = data[i]
            y.append(d["wtrue"] == 2)
            sc.append(window_scores(d, oof[i]))
            dec.append(TD.window_preds(d, M.decode(oof[i], cfg)) == 2)
    y, sc, dec = np.concatenate(y), np.concatenate(sc), np.concatenate(dec)
    fpr, tpr, thr = roc_curve(y, sc)
    auc = roc_auc_score(y, sc)
    np_points = {f"{int(f*100)}%": float(np.interp(f, fpr, tpr)) for f in (0.05, 0.10, 0.20)}
    op = {"fpr": float(dec[~y].mean()), "tpr": float(dec[y].mean())}
    res = {"auc": float(auc), "detection_at_false_alarm": np_points, "offsets_operating_point": op,
           "n_stress_windows": int(y.sum()), "n_other_windows": int((~y).sum())}
    print(json.dumps(res, indent=1))
    (ART / "roc_stress.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#2a6fb0", lw=2, label=f"OOF Stress vs not-Stress (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
    for f, t in zip((0.05, 0.10, 0.20), np_points.values()):
        ax.plot([f], [t], "o", color="#2a6fb0", ms=5)
        ax.annotate(f"{t:.2f} at {int(f*100)}%", (f, t), textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax.plot([op["fpr"]], [op["tpr"]], "s", color="#d1495b", ms=8, label="tuned offsets (nested)")
    ax.set_xlabel("False alarm rate (not-Stress windows called Stress)")
    ax.set_ylabel("Stress detection rate")
    ax.set_title("Stress detector, out-of-fold 10 s windows")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "roc_stress.png", dpi=150)
    print("saved figures/roc_stress.png")


if __name__ == "__main__":
    main()
