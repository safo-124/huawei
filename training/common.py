"""Shared training utilities: loading, labels, competition metric, folds."""

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))

import adm_model as M  # noqa: E402

DATA = ROOT / "data"
PROTOCOLS = ("STRESS", "AEROBIC", "ANAEROBIC")
INPUT_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
CODE = {c: i for i, c in enumerate(M.CLASSES)}  # Rest 0, Activity 1, Stress 2
UNSCORED = -1


def label_codes(phase: pd.Series) -> np.ndarray:
    """Phase column to class codes; unscored phases get -1."""
    cats = pd.Series(phase.unique())
    lut = {p: CODE.get(M.phase_to_label(p), UNSCORED) for p in cats}
    return phase.map(lut).to_numpy(dtype=np.int64)


def list_recordings():
    out = []
    for p in PROTOCOLS:
        for f in sorted((DATA / p).glob("*.parquet")):
            out.append((p, f.stem, f))
    return out


def load_recording(path):
    """Returns (inputs only DataFrame, row label codes, user_id)."""
    df = pd.read_parquet(path)
    y = label_codes(df["Phase"])
    uid = str(df["user_id"].iloc[0])
    return df[INPUT_COLS], y, uid


def second_labels(sec, y, n):
    """Majority scored label per second (-1 when the second has no scored rows)."""
    m = y >= 0
    cnt = np.bincount(sec[m] * 3 + y[m], minlength=n * 3).reshape(n, 3)
    lab = cnt.argmax(1)
    lab[cnt.sum(1) == 0] = UNSCORED
    return lab


# --------------------------------------------------------------------------
# Competition metric
# --------------------------------------------------------------------------
WIN_NS = 10 * 1_000_000_000


def window_majority(index, y_true, y_pred, rec_id=0):
    """Per 10 s window (aligned to epoch): majority true and predicted label.

    Rows with unscored truth are ignored for both majorities. Windows with no
    scored rows are dropped.
    """
    t = index.values.astype("datetime64[ns]").astype(np.int64)
    m = y_true >= 0
    w = t[m] // WIN_NS
    w = w - w.min() if len(w) else w
    nw = int(w.max()) + 1 if len(w) else 0
    ct = np.bincount(w * 3 + y_true[m], minlength=nw * 3).reshape(nw, 3)
    cp = np.bincount(w * 3 + y_pred[m], minlength=nw * 3).reshape(nw, 3)
    keep = ct.sum(1) > 0
    return ct[keep].argmax(1), cp[keep].argmax(1)


def macro_f1(t, p, k=3):
    f = per_class_f1(t, p, k)
    return float(np.mean(f))


def per_class_f1(t, p, k=3):
    out = []
    for c in range(k):
        tp = np.sum((t == c) & (p == c))
        fp = np.sum((t != c) & (p == c))
        fn = np.sum((t == c) & (p != c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.array(out)


def confusion(t, p, k=3):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    return cm


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------
def user_folds(uids, n_splits=5, seed=0):
    """GroupKFold by user_id with a shuffled but deterministic group order."""
    from sklearn.model_selection import GroupKFold

    uids = np.asarray(uids)
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in gkf.split(np.zeros(len(uids)), groups=uids)]
CLASSES_ORDER = M.CLASSES
