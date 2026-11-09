"""Stress experiments on top of variant E with nested grouped CV.

Each experiment produces out-of-fold per-second probabilities (models trained
with GroupKFold by user_id, Kalman noise tuned inside each training fold).
The decoder (per-class offsets, and optionally the choice among several
training settings such as the Stress weight) is selected for each outer fold
using only the other folds' out-of-fold predictions, then applied to that fold.

Usage:
    python training/experiments.py E ctr two_stage ...   (names from EXPERIMENTS)
"""

import json
import sys
import time

import numpy as np

from common import M, ROOT, per_class_f1, confusion, user_folds
import train as T
import tune_decoder as TD

ART = ROOT / "artifacts"
BASE_GROUPS = ["raw", "kalman", "norm"]

EXPERIMENTS = {
    "E": dict(groups=BASE_GROUPS),
    "ctr": dict(groups=BASE_GROUPS + ["ctr"]),
    "ctr_past": dict(groups=BASE_GROUPS + ["ctr"], params={"ctr_past_only": True}),
    "two_stage": dict(groups=BASE_GROUPS + ["ctr"], two_stage=True),
    "two_stage_past": dict(groups=BASE_GROUPS + ["ctr"], params={"ctr_past_only": True}, two_stage=True),
    "resp": dict(groups=BASE_GROUPS + ["resp"]),
    "resp_ctr": dict(groups=BASE_GROUPS + ["resp", "ctr"]),
    "vaso": dict(groups=BASE_GROUPS + ["vaso"]),
    "vaso_ctr": dict(groups=BASE_GROUPS + ["vaso", "ctr"]),
    "sw": dict(groups=BASE_GROUPS, stress_weights=[1, 2, 3, 4, 6]),
    "E_seed1": dict(groups=BASE_GROUPS, seed=1),
    "E_seed2": dict(groups=BASE_GROUPS, seed=2),
    "resp_ctr_past": dict(groups=BASE_GROUPS + ["resp", "ctr"], params={"ctr_past_only": True}),
    "gp": dict(groups=BASE_GROUPS + ["gp"]),
    "glrt": dict(groups=BASE_GROUPS + ["glrt"]),
    "scr": dict(groups=BASE_GROUPS + ["scr"]),
    "shift_global": dict(groups=BASE_GROUPS, shift="global"),
    "shift_per": dict(groups=BASE_GROUPS, shift="per"),
    "combo_explore": dict(groups=BASE_GROUPS + ["resp", "glrt", "scr", "ctr"], params={"ctr_past_only": True}),
}


# --------------------------------------------------------------------------
class Runner:
    def __init__(self):
        self.recs = T.load_all()
        self.fc = T.FeatureCache(self.recs)
        self.folds = user_folds([r["uid"] for r in self.recs], 5)
        self.fold_of = np.zeros(len(self.recs), dtype=int)
        for k, (_, te) in enumerate(self.folds):
            self.fold_of[te] = k
        self.kal = {}
        self.data = TD.prepare()

    def kalman(self, k):
        if k not in self.kal:
            tr = self.folds[k][0]
            self.kal[k] = T.tune_kalman([self.recs[i] for i in tr])
        return self.kal[k]

    def tuned(self, k, groups):
        """Feature hyperparameters tuned on the training part of fold k."""
        out = {}
        tr = [self.recs[i] for i in self.folds[k][0]]
        if "gp" in groups:
            if ("gp", k) not in self.kal:
                self.kal[("gp", k)] = T.tune_gp(tr)
            out.update(self.kal[("gp", k)])
        if "scr" in groups:
            if ("scr", k) not in self.kal:
                self.kal[("scr", k)] = T.tune_scr(tr)[0]
            out.update(self.kal[("scr", k)])
        return out

    def cfg(self, k, groups, params):
        c = T.make_cfg("custom", self.kalman(k), 5, groups)
        pp = self.tuned(k, groups)
        pp.update(params or {})
        if pp:
            c["params"] = pp
        return c

    def oof(self, groups, params=None, stress_weight=1.0, two_stage=False, stage1=None, seed=0):
        """Out-of-fold per-second probabilities for every recording."""
        out = [None] * len(self.recs)
        for k, (tr, te) in enumerate(self.folds):
            cfg = self.cfg(k, groups, params)
            Xs = [self.fc.get(i, cfg)[1] for i in tr]
            ys = [self.recs[i]["y_sec"] for i in tr]
            if not two_stage:
                clf = T.fit(Xs, ys, {"stress_weight": stress_weight}, seed=seed)
                for i in te:
                    out[i] = T.predict_proba(clf, self.fc.get(i, cfg)[1], 5)
                continue
            # stage 2: Stress vs Rest on non-Activity seconds
            ys2 = [np.where(y == 1, -1, y) for y in ys]
            clf2 = T.fit(Xs, ys2, {"stress_weight": stress_weight, "lgb": {"objective": "binary"}})
            for i in te:
                p1 = stage1[i]
                p2 = T.predict_proba(clf2, self.fc.get(i, cfg)[1], 5)
                ps = p2[:, -1] / p2.sum(1)  # classes seen: Rest, Stress
                na = 1.0 - p1[:, 1]
                out[i] = np.stack([na * (1 - ps), p1[:, 1], na * ps], axis=1)
        return out

    # ------------------------------------------------------------------
    def nested(self, oofs, shift_mode=None):
        """oofs: dict setting -> OOF list. Select setting and offsets per outer
        fold on the other folds, evaluate on the fold."""
        scores, Tw, Pw, rec_ids, chosen = [], [], [], [], []
        for k in range(5):
            tr = np.flatnonzero(self.fold_of != k)
            te = np.flatnonzero(self.fold_of == k)
            best = (-1, None, None)
            for s, oof in oofs.items():
                cfg, p, sc = TD.tune(self.data, oof, tr, 5, use_hmm=False, shift_mode=shift_mode)
                if sc > best[0]:
                    best = (sc, s, cfg)
            _, s, cfg = best
            if shift_mode:
                chosen.append({"setting": s, "shift": cfg.get("shift"), "enter": cfg.get("enter"),
                               "leave": cfg.get("leave"), "offsets": cfg["offsets"]})
            else:
                chosen.append(s)
            f1, _, _ = TD.evaluate(self.data, oofs[s], te, cfg)
            scores.append(f1)
            for i in te:
                Tw.append(self.data[i]["wtrue"])
                Pw.append(TD.window_preds(self.data[i], M.decode(oofs[s][i], cfg)))
                rec_ids.append(np.full(len(Tw[-1]), i))
        t, p, rid = np.concatenate(Tw), np.concatenate(Pw), np.concatenate(rec_ids)
        pc = per_class_f1(t, p)
        cm = confusion(t, p)
        srow = cm[2] / cm[2].sum()
        low = []
        for i, d in enumerate(self.data):
            if d["prot"] != "STRESS":
                continue
            m = rid == i
            f = per_class_f1(t[m], p[m])[2]
            if f < 0.3:
                low.append((d["uid"], round(float(f), 2)))
        return dict(mean=float(np.mean(scores)), std=float(np.std(scores)), folds=scores,
                    per_class=dict(zip(M.CLASSES, pc.round(4).tolist())), confusion=cm.tolist(),
                    stress_row={"Rest": round(float(srow[0]), 3), "Activity": round(float(srow[1]), 3),
                                "Stress": round(float(srow[2]), 3)},
                    low_stress_users=sorted(low), chosen=[str(c) for c in chosen])


def main():
    args = sys.argv[1:]
    seeds = [0]
    if "--seeds" in args:
        j = args.index("--seeds")
        seeds = [int(x) for x in args[j + 1:]]
        args = args[:j]
    names = args or ["E"]
    R = Runner()
    stage1 = None
    for name in names:
        e = EXPERIMENTS[name]
        t0 = time.time()
        runs = []
        for seed in seeds if len(seeds) > 1 else [e.get("seed", 0)]:
            if e.get("two_stage") and stage1 is None:
                stage1 = R.oof(BASE_GROUPS)
            sws = e.get("stress_weights", [1.0])
            oofs = {sw: R.oof(e["groups"], e.get("params"), sw, e.get("two_stage", False), stage1, seed) for sw in sws}
            runs.append(R.nested(oofs, e.get("shift")))
        res = runs[0] if len(runs) == 1 else dict(
            mean=float(np.mean([r["mean"] for r in runs])),
            std=float(np.mean([r["std"] for r in runs])),
            seed_means=[r["mean"] for r in runs],
            per_class={c: round(float(np.mean([r["per_class"][c] for r in runs])), 4) for c in M.CLASSES},
            stress_row={c: round(float(np.mean([r["stress_row"][c] for r in runs])), 3) for c in M.CLASSES},
            low_stress_users=sorted({u for r in runs for u, _ in r["low_stress_users"]}),
            low_by_seed=[r["low_stress_users"] for r in runs],
            chosen=[r["chosen"] for r in runs], runs=runs)
        res["experiment"] = name
        res["fold_params"] = {str(k): v for k, v in R.kal.items()}
        res["spec"] = {k: v for k, v in e.items()}
        tag = name if len(seeds) == 1 else f"{name}_s{len(seeds)}"
        (ART / f"exp_{tag}.json").write_text(json.dumps(res, indent=1))
        pc = res["per_class"]
        extra = f" seeds {np.round(res['seed_means'], 4).tolist()}" if "seed_means" in res else ""
        print(f"{tag:18s} {res['mean']:.4f} +- {res['std']:.4f}  Activity {pc['Activity']:.3f} "
              f"Rest {pc['Rest']:.3f} Stress {pc['Stress']:.3f}  stress_row {res['stress_row']}{extra}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        print(f"{'':18s} low Stress F1 users: {res['low_stress_users']}", flush=True)


if __name__ == "__main__":
    main()
