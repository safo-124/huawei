import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from common import M, window_majority, macro_f1, per_class_f1, label_codes, second_labels  # noqa: E402


# ---------------------------------------------------------------- labels
@pytest.mark.parametrize("phase,label", [
    ("Stroop", "Stress"), ("TMCT", "Stress"), ("Real opinion", "Stress"),
    ("Opposite opinion", "Stress"), ("Subtraction", "Stress"),
    ("Warm-up", "Activity"), ("85 rpm", "Activity"), ("90-95 rpm", "Activity"),
    ("Sprint", "Activity"), ("Cooldown", "Activity"),
    ("Baseline", "Rest"), ("Rest", "Rest"),
    ("Undefined", None), ("Something else", None), (None, None),
])
def test_phase_mapping(phase, label):
    assert M.phase_to_label(phase) == label


def test_label_codes():
    codes = label_codes(pd.Series(["Rest", "Sprint", "TMCT", "Undefined"]))
    assert codes.tolist() == [0, 1, 2, -1]


# ---------------------------------------------------------------- metric
def _idx(n, hz=1):
    return pd.date_range("2020-01-01", periods=n, freq=pd.Timedelta(seconds=1 / hz))


def test_window_majority_and_unscored():
    idx = _idx(30)  # three 10 s windows aligned to epoch
    yt = np.array([0] * 10 + [1] * 6 + [-1] * 4 + [-1] * 10)
    # window 2 scored rows: 4 Stress, 2 Activity; its 4 unscored rows say Activity
    yp = np.array([0] * 10 + [2] * 4 + [1] * 2 + [1] * 4 + [0] * 10)
    t, p = window_majority(idx, yt, yp)
    # third window fully unscored is dropped; unscored rows in window 2 ignored
    assert t.tolist() == [0, 1]
    assert p.tolist() == [0, 2]


def test_window_majority_is_majority():
    idx = _idx(10)
    yt = np.zeros(10, dtype=int)
    yp = np.array([1] * 6 + [0] * 4)
    t, p = window_majority(idx, yt, yp)
    assert p.tolist() == [1]


def test_macro_f1():
    t = np.array([0, 0, 1, 1, 2, 2])
    assert macro_f1(t, t) == 1.0
    p = np.array([0, 0, 1, 1, 1, 1])
    f = per_class_f1(t, p)
    assert f[2] == 0.0 and f[0] == 1.0
    assert abs(f[1] - 2 * 2 / (2 * 2 + 2)) < 1e-12


def test_second_labels_majority():
    sec = np.array([0, 0, 0, 1, 1, 2])
    y = np.array([0, 2, 2, -1, -1, 1])
    assert second_labels(sec, y, 3).tolist() == [2, -1, 1]


# ---------------------------------------------------------------- helpers / HMM
def test_step_points_cover_all():
    for n in (1, 2, 4, 5, 6, 11, 100):
        for s in (1, 3, 5):
            pts, back = M.step_points(n, s)
            assert len(back) == n
            assert pts.max() < n and back.max() < len(pts)


def test_viterbi_smooths_isolated_flip():
    p = np.full((50, 3), 0.1)
    p[:, 0] = 0.8
    p[25] = [0.3, 0.6, 0.1]  # one noisy second
    stay = np.log(np.full((3, 3), 0.001 / 2))
    np.fill_diagonal(stay, np.log(0.999))
    path = M.viterbi(np.log(p), stay, np.full(3, -np.log(3)))
    assert (path == 0).all()


def test_viterbi_follows_real_change():
    p = np.vstack([np.tile([0.9, 0.05, 0.05], (30, 1)), np.tile([0.05, 0.9, 0.05], (30, 1))])
    lt = np.log(np.full((3, 3), 0.01 / 2))
    np.fill_diagonal(lt, np.log(0.99))
    path = M.viterbi(np.log(p), lt, np.full(3, -np.log(3)))
    assert path[:30].tolist() == [0] * 30 and path[30:].tolist() == [1] * 30


# ---------------------------------------------------------------- features
def _synthetic(n_sec=300, gap_at=None):
    t = pd.date_range("2020-01-01", periods=n_sec * 64, freq=pd.Timedelta(seconds=1 / 64))
    if gap_at:
        t = t.where(t < t[0] + pd.Timedelta(seconds=gap_at), t + pd.Timedelta(seconds=33))
    df = pd.DataFrame(np.nan, index=pd.DatetimeIndex(t, name="t"),
                      columns=["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"])
    rng = np.random.default_rng(0)
    df["BVP"] = rng.normal(0, 50, len(df))
    df.iloc[::16, df.columns.get_loc("EDA")] = 2.0
    df.iloc[::16, df.columns.get_loc("Temperature")] = 212.0  # corrupted, must be ignored
    df.iloc[::64, df.columns.get_loc("HR")] = 75.0
    df.iloc[::50, df.columns.get_loc("IBI_duration")] = 0.8
    df.iloc[::50, df.columns.get_loc("IBI_dt")] = 0.8
    return df


def test_features_shape_and_ranges():
    df = _synthetic()
    sec, names, X = M.compute_features(df)
    assert len(sec) == len(df) and X.shape == (300, len(names))
    j = names.index("hr_mean_60")
    assert np.allclose(X[:, j], 75.0)
    assert np.isnan(X[:, names.index("tmp_mean_60")]).all()  # out of range temperature dropped


def test_features_respect_time_gaps():
    df = _synthetic(gap_at=100)
    sec, names, X = M.compute_features(df)
    assert X.shape[0] == 300 + 33  # grid follows time, not row count


# ---------------------------------------------------------------- submission format
def _built():
    d = ROOT / "build"
    cands = sorted(d.glob("submission_*/adm_model.onnx")) if d.exists() else []
    return cands[0].parent if cands else None


@pytest.mark.skipif(_built() is None, reason="no built submission")
def test_submission_format():
    import importlib.util

    sys.path.insert(0, str(ROOT))
    sub = _built()
    spec = importlib.util.spec_from_file_location("adm_model_built", sub / "adm_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import adm_base_class

    model = mod.get_model()
    assert isinstance(model, adm_base_class.ADM_Base)
    for df in (_synthetic(), _synthetic(gap_at=50), _synthetic(n_sec=3), _synthetic().iloc[:1]):
        out = model.run(df)
        assert list(out.columns) == ["Activity"]
        assert out.index.equals(df.index)
        assert set(out["Activity"]) <= {"Stress", "Activity", "Rest"}


# ---------------------------------------------------------------- Kalman
def test_kalman_tracks_ramp_and_slope():
    rng = np.random.default_rng(1)
    truth = np.r_[np.full(200, 70.0), np.linspace(70, 140, 100), np.full(200, 140.0)]
    y = truth + rng.normal(0, 2, truth.size)
    y[100:130] = np.nan  # missing samples are bridged
    lev, slo, inn, S = M.kalman_cv(y, 0.01, 4.0)
    assert np.abs(lev[350:450] - 140).mean() < 1.0
    assert abs(slo[250] - 70 / 99) < 0.1
    assert np.isnan(inn[100:130]).all()  # no innovation where nothing was measured
    assert 0.8 < np.nanstd(inn) < 1.2  # normalized innovations have unit variance


def test_kalman_ml_prefers_true_noise():
    rng = np.random.default_rng(2)
    y = 70 + np.cumsum(rng.normal(0, 0.05, 2000)) + rng.normal(0, 1.0, 2000)
    assert M.cv_loglik(y, 0.01, 1.0) > M.cv_loglik(y, 0.01, 0.01)


def test_slow_level_is_slow():
    y = np.r_[np.full(1000, 60.0), np.full(60, 120.0), np.full(1000, 60.0)]
    base = M.slow_level(y, 600.0)
    assert base.max() < 70.0


def test_decode_offsets_shift_decision():
    p = np.tile([0.5, 0.3, 0.2], (5, 1))
    assert (M.decode(p, {}) == 0).all()
    assert (M.decode(p, {"offsets": [0.0, 0.0, 1.0]}) == 2).all()
