"""run() must answer every input with one valid label per row, in the row
order it was given, whatever is wrong with the input.

Uses the built submission (build/submission_R, else the newest build) with
the repository's submission/adm_model.py.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
VALID = {"Stress", "Activity", "Rest"}


def _onnx():
    d = ROOT / "build"
    pref = d / "submission_R" / "adm_model.onnx"
    if pref.exists():
        return pref
    c = sorted(d.glob("submission_*/adm_model.onnx"), key=lambda p: p.stat().st_mtime) if d.exists() else []
    return c[-1] if c else None


@pytest.fixture(scope="module")
def model():
    onx = _onnx()
    if onx is None:
        pytest.skip("no built submission")
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("adm_model_abn", ROOT / "submission" / "adm_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.ADMModel()
    m.load_model(onx)
    return m


def synth(n_sec=400, seed=0):
    """A plausible recording: 64 Hz BVP, 4 Hz EDA and temperature, 1 Hz HR, beats."""
    n = n_sec * 64
    t = pd.date_range("2024-03-01 10:00:00", periods=n, freq=pd.Timedelta(seconds=1 / 64))
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(np.nan, index=pd.DatetimeIndex(t, name="t"), columns=COLS)
    df["BVP"] = rng.normal(0, 50, n)
    df.iloc[::16, 1] = 2.0 + rng.normal(0, 0.05, n // 16)
    df.iloc[::16, 5] = 33.0 + rng.normal(0, 0.05, n // 16)
    df.iloc[::64, 2] = 70.0 + rng.normal(0, 2, n // 64)
    k = len(df.iloc[::51])
    df.iloc[::51, 3] = 0.8
    df.iloc[::51, 4] = 0.8 + rng.normal(0, 0.03, k)
    return df


def check(model, df):
    out = model.run(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["Activity"]
    assert len(out) == len(df)
    assert out.index.equals(df.index)
    assert out["Activity"].notna().all()
    assert set(out["Activity"].unique()) <= VALID
    return out


def _gap(df, at_s, gap):
    t = df.index
    return df.set_axis(t.where(t < t[0] + pd.Timedelta(seconds=at_s), t + gap))


CASES = {
    "empty": lambda d: d.iloc[:0],
    "one_row": lambda d: d.iloc[:1],
    "200_rows": lambda d: d.iloc[:200],
    "ten_seconds": lambda d: d.iloc[:640],
    "duplicate_timestamps": lambda d: pd.concat([d.iloc[:3000], d.iloc[:3000]]).sort_index(),
    "all_signals_nan": lambda d: d * np.nan,
    "constant_signals": lambda d: d.where(d.isna(), 1.0),
    "zeros": lambda d: d.where(d.isna(), 0.0),
    "infinite_values": lambda d: d.mask(d.notna() & (np.arange(len(d))[:, None] % 97 == 0), np.inf),
    "out_of_range_values": lambda d: d.assign(HR=d["HR"] * 10, Temperature=d["Temperature"] + 180, EDA=-d["EDA"]),
    "gap_33_seconds": lambda d: _gap(d, 100, pd.Timedelta(seconds=33)),
    "gap_3_days": lambda d: _gap(d, 200, pd.Timedelta(days=3)),
    "gap_1_year": lambda d: _gap(d, 200, pd.Timedelta(days=365)),
    "extra_columns": lambda d: d.assign(Phase="Rest", user_id="x", other=1.0),
    "float32": lambda d: d.astype(np.float32),
    "columns_reordered": lambda d: d[COLS[::-1]],
    "timezone_aware_index": lambda d: d.tz_localize("Europe/Helsinki"),
    "microsecond_index": lambda d: d.set_axis(d.index.as_unit("us")),
    "range_index": lambda d: d.reset_index(drop=True),
    "missing_timestamps": lambda d: d.set_axis(d.index.where(np.arange(len(d)) % 500 != 7, pd.NaT)),
    "only_bvp_rows": lambda d: d[d["EDA"].isna() & d["HR"].isna()],
}
for _c in COLS:
    CASES[f"no_column_{_c}"] = lambda d, c=_c: d.drop(columns=[c])
    CASES[f"all_nan_{_c}"] = lambda d, c=_c: d.assign(**{c: np.nan})


@pytest.mark.parametrize("name", sorted(CASES))
def test_abnormal_input(model, name):
    check(model, CASES[name](synth()))


@pytest.mark.parametrize("how", ["shuffled", "reversed", "two_halves_swapped"])
def test_row_order_does_not_matter(model, how):
    df = synth()
    ref = check(model, df)["Activity"]
    n = len(df)
    order = {"shuffled": np.random.default_rng(1).permutation(n), "reversed": np.arange(n)[::-1],
             "two_halves_swapped": np.r_[np.arange(n // 2, n), np.arange(n // 2)]}[how]
    out = check(model, df.iloc[order])["Activity"]
    assert (out.to_numpy() == ref.to_numpy()[order]).all()  # same answer for the same row


def test_long_gap_costs_nothing(model):
    """A break of a year must not allocate a year of seconds."""
    import time
    df = _gap(synth(), 200, pd.Timedelta(days=365))
    t = time.perf_counter()
    check(model, df)
    assert time.perf_counter() - t < 2.0
