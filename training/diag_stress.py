"""Step 1 diagnostic: is Stress missed because of the threshold or because the
signal is invisible? Uses saved out-of-fold probabilities only (no training).

For each of the 9 runs (3 repeats x 3 seeds) of a stat_ext experiment:
  - nested offsets decoder per outer fold (as in stat_ext), window predictions
  - per STRESS recording: within-recording Stress-vs-Rest AUC on 10 s windows
    (score = mean Stress log-odds against Rest over the window), Stress F1,
    median score in Stress and in Rest windows
  - origin of Stress false alarms by protocol
Values are averaged over the 9 runs.

Usage: python training/diag_stress.py [--name base]
"""

import argparse
import json

import numpy as np

from common import M, ROOT, per_class_f1
import train as T
import tune_decoder as TD

OUT = ROOT / "artifacts" / "stat"
LOW = ("S15", "S17", "f11", "f17", "f18")


def window_score(d, s):
    num = np.bincount(d["pw"], weights=d["pc"] * s[d["ps"]], minlength=d["nw"])
    den = np.bincount(d["pw"], weights=d["pc"], minlength=d["nw"])
    return (num / np.maximum(den, 1))[d["keep"]]


def auc(pos, neg):
    from scipy.stats import rankdata
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.r_[pos, neg])
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    a = ap.parse_args()
    recs = [(p, n) for p, n, _ in __import__("common").list_recordings()]
    uids = [T.load_recording(path)[2] for _, _, path in __import__("common").list_recordings()]
    data = TD.prepare()
    per = {}  # recording index -> lists of metrics
    fa = {"STRESS": [], "AEROBIC": [], "ANAEROBIC": []}
    for rep in (0, 1, 2):
        fold_of = np.load(OUT / f"folds_r{rep}.npy")
        for s in (0, 1, 2):
            oof = np.load(OUT / f"{a.name}_r{rep}_s{s}.npy", allow_pickle=True)
            counts = {k: 0 for k in fa}
            for k in range(fold_of.max() + 1):
                tr, te = np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)
                cfg, _, _ = TD.tune(data, oof, tr, 5, use_hmm=False)
                for i in te:
                    d = data[i]
                    pred = TD.window_preds(d, M.decode(oof[i], cfg))
                    t = d["wtrue"]
                    counts[recs[i][0]] += int(((pred == 2) & (t != 2)).sum())
                    if recs[i][0] != "STRESS":
                        continue
                    p = np.clip(oof[i], 1e-6, 1)
                    lo = np.log(p[:, 2]) - np.log(p[:, 0])
                    w = window_score(d, lo)
                    m = per.setdefault(i, {"auc": [], "f1": [], "ms": [], "mr": []})
                    m["auc"].append(auc(w[t == 2], w[t == 0]))
                    m["f1"].append(float(per_class_f1(t, pred)[2]))
                    m["ms"].append(float(np.median(w[t == 2])) if (t == 2).any() else np.nan)
                    m["mr"].append(float(np.median(w[t == 0])) if (t == 0).any() else np.nan)
            tot = max(sum(counts.values()), 1)
            for kk in fa:
                fa[kk].append(counts[kk] / tot)
    rows = []
    for i, m in per.items():
        rows.append(dict(rec=f"{recs[i][0]}/{recs[i][1]}", user=uids[i], auc=float(np.nanmean(m["auc"])),
                         f1=float(np.mean(m["f1"])), med_stress=float(np.nanmean(m["ms"])),
                         med_rest=float(np.nanmean(m["mr"]))))
    rows.sort(key=lambda r: r["f1"])
    print(f"{'person':8s} {'AUC':>6s} {'StressF1':>9s} {'med S':>7s} {'med R':>7s}")
    for r in rows:
        flag = " *" if r["user"] in LOW else ""
        print(f"{r['user']:8s} {r['auc']:6.3f} {r['f1']:9.3f} {r['med_stress']:7.2f} {r['med_rest']:7.2f}{flag}")
    share = {k: float(np.mean(v)) for k, v in fa.items()}
    print("Stress false alarms by protocol:", {k: round(v, 3) for k, v in share.items()})
    low = [r for r in rows if r["user"] in LOW]
    print("low responders with AUC >= 0.75:", sum(r["auc"] >= 0.75 for r in low), "of", len(low))
    (OUT / f"diag_{a.name}.json").write_text(json.dumps(dict(rows=rows, false_alarms=share), indent=1))


if __name__ == "__main__":
    main()
