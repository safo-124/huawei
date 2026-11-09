"""Out-of-fold diagnostics for a variant: per-class F1, normalized confusion,
per-protocol and per-phase confusions on the 10 s window metric.

Usage:
    python training/diagnostics.py --variant B
"""

import argparse

import numpy as np
import pandas as pd

from common import M, ROOT, CLASSES_ORDER, list_recordings, window_majority, per_class_f1, confusion

ART = ROOT / "artifacts"


def window_table(variant, oof=None, decoder=None):
    """One row per scored 10 s window: protocol, file, phase, true, pred."""
    if oof is None:
        oof = np.load(ART / f"oof_{variant}.npy", allow_pickle=True)
    rows = []
    for (prot, name, path), proba in zip(list_recordings(), oof):
        df = pd.read_parquet(path, columns=["Phase"])
        from common import label_codes
        y = label_codes(df["Phase"])
        t = df.index.values.astype(np.int64)
        sec = (t - t[0]) // M.NS
        codes = decoder(proba) if decoder else proba.argmax(1)
        yp = codes[sec]
        m = y >= 0
        w = t[m] // (10 * M.NS)
        d = pd.DataFrame({"w": w, "y": y[m], "p": yp[m], "phase": df["Phase"].to_numpy()[m]})
        g = d.groupby("w")
        out = pd.DataFrame({
            "y": g["y"].agg(lambda s: np.bincount(s, minlength=3).argmax()),
            "p": g["p"].agg(lambda s: np.bincount(s, minlength=3).argmax()),
            "phase": g["phase"].agg(lambda s: s.value_counts().index[0]),
        })
        out["prot"], out["file"] = prot, name
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def norm_cm(t, p):
    cm = confusion(t, p).astype(float)
    return pd.DataFrame(cm / cm.sum(1, keepdims=True), index=[f"true {c}" for c in CLASSES_ORDER],
                        columns=[f"pred {c}" for c in CLASSES_ORDER]).round(3)


def report(W):
    t, p = W.y.to_numpy(), W.p.to_numpy()
    f = per_class_f1(t, p)
    print("pooled macro-F1 %.4f" % f.mean())
    print("per-class F1:", {c: round(float(x), 3) for c, x in zip(CLASSES_ORDER, f)})
    print("\nnormalized confusion (rows true, all protocols):\n", norm_cm(t, p).to_string())
    for prot in ("STRESS", "AEROBIC", "ANAEROBIC"):
        x = W[W.prot == prot]
        cm = pd.crosstab(x.y.map(dict(enumerate(CLASSES_ORDER))), x.p.map(dict(enumerate(CLASSES_ORDER))),
                         normalize="index").round(3)
        print(f"\n[{prot}] windows {len(x)}, rows true, normalized:\n{cm.to_string()}")
    print("\nper-phase prediction share (rows: majority phase of window):")
    ph = W.copy()
    ph["phase"] = np.where(ph.phase.str.contains("rpm"), "rpm stages", ph.phase)
    ph["prot_phase"] = ph.prot + " / " + ph.phase
    tab = pd.crosstab(ph.prot_phase, ph.p.map(dict(enumerate(CLASSES_ORDER))), normalize="index").round(3)
    tab["windows"] = ph.prot_phase.value_counts()
    print(tab.to_string())
    print("\nworst files by Stress recall (STRESS protocol):")
    s = W[(W.prot == "STRESS") & (W.y == 2)]
    print(s.groupby("file").p.apply(lambda v: round(float((v == 2).mean()), 2)).sort_values().head(10).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B")
    a = ap.parse_args()
    report(window_table(a.variant))


if __name__ == "__main__":
    main()
