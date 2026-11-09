"""ADM submission model: Rest / Activity / Stress from wrist signals.

This module is the single source of truth for preprocessing and features.
Training code imports it, so training and inference cannot diverge.

Allowed imports in the evaluation environment: numpy, pandas, pyarrow, scipy,
scikit-learn, onnxruntime (plus the standard library).

Pipeline inside run():
  1. Extract each signal at its native rate, apply range checks.
  2. Aggregate every signal onto a 1 Hz grid anchored at the first input
     timestamp (by time, never by row count), keeping per-second counts so
     gaps stay visible.
  3. Window features (10 s, 60 s, 180 s, centred) computed with cumulative sums.
  4. ONNX classifier gives per-second class probabilities.
  5. Optional per-class log offsets and HMM smoothing (configured through the
     ONNX metadata, so the .py never changes between variants).
  6. Per-second decisions are mapped back onto every input row.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import onnxruntime as ort

try:
    from adm_base_class import ADM_Base
except ImportError:  # local fallback with the identical interface
    import abc

    class ADM_Base(abc.ABC):
        def load_model(self, onnx_file: pathlib.Path) -> None:
            self.session = ort.InferenceSession(
                onnx_file, providers=["CPUExecutionProvider"]
            )

        @abc.abstractmethod
        def run(self, input: pd.DataFrame) -> pd.DataFrame:
            pass


CLASSES = ("Rest", "Activity", "Stress")
WINDOWS = (10, 60, 180)
NS = 1_000_000_000

# Physiological range checks; values outside are treated as missing.
HR_RANGE = (30.0, 220.0)
IBI_RANGE = (0.25, 2.0)
TEMP_RANGE = (20.0, 42.0)
MISS_TOL = 0.05  # IBI_dt - IBI_duration above this means at least one missed beat


# --------------------------------------------------------------------------
# Labels (official mapping from phases_to_activity.txt)
# --------------------------------------------------------------------------
STRESS_PHASES = {"Stroop", "TMCT", "Real opinion", "Opposite opinion", "Subtraction"}
ACTIVITY_PHASES = {"Warm-up", "Sprint", "Cooldown"}
REST_PHASES = {"Baseline", "Rest"}


def phase_to_label(phase):
    """Map a protocol phase to Rest/Activity/Stress, or None when not scored."""
    if not isinstance(phase, str):
        return None
    p = phase.strip()
    if p in STRESS_PHASES:
        return "Stress"
    if p in ACTIVITY_PHASES or "rpm" in p:
        return "Activity"
    if p in REST_PHASES:
        return "Rest"
    return None


# --------------------------------------------------------------------------
# Per-second aggregation
# --------------------------------------------------------------------------
def _col(df, name):
    if name in df.columns:
        return df[name].to_numpy(dtype=np.float64, na_value=np.nan)
    return np.full(len(df), np.nan)


def seconds_index(index):
    """Integer second of each row relative to the first timestamp."""
    t = index.values.astype("datetime64[ns]").astype(np.int64)
    return (t - t[0]) // NS, t


def per_second(df):
    """Aggregate raw signals to 1 Hz sums and counts. Returns (sec, n, base dict)."""
    sec, t_ns = seconds_index(df.index)
    n = int(sec[-1]) + 1
    b = {}

    def bc(s, w=None):
        return np.bincount(s, weights=w, minlength=n).astype(np.float64)

    # HR (1 Hz)
    hr = _col(df, "HR")
    m = (hr >= HR_RANGE[0]) & (hr <= HR_RANGE[1])
    s = sec[m]
    b["hr_c"] = bc(s)
    b["hr_s"] = bc(s, hr[m])
    b["hr_q"] = bc(s, hr[m] ** 2)

    # EDA (4 Hz)
    eda = _col(df, "EDA")
    m = eda >= 0
    s, v = sec[m], eda[m]
    b["eda_c"] = bc(s)
    b["eda_s"] = bc(s, v)
    b["eda_q"] = bc(s, v * v)
    if len(v) > 1:
        dv = np.diff(v)
        dt = np.diff(t_ns[m]) / NS
        ok = dt < 0.5  # only consecutive 4 Hz samples
        rise = np.where(ok & (dv > 0), dv, 0.0)
        b["eda_rise"] = bc(s[1:], rise)
    else:
        b["eda_rise"] = np.zeros(n)
    # 4 Hz EDA grid (mean per quarter second, NaN when empty) for SCR detection
    q4 = (t_ns[m] - t_ns[0]) * 4 // NS
    c4 = np.bincount(q4, minlength=4 * n)[: 4 * n].astype(np.float64)
    s4 = np.bincount(q4, weights=v, minlength=4 * n)[: 4 * n]
    b["eda4"] = _ratio(s4, c4)

    # Temperature (4 Hz)
    tmp = _col(df, "Temperature")
    m = (tmp >= TEMP_RANGE[0]) & (tmp <= TEMP_RANGE[1])
    s = sec[m]
    b["tmp_c"] = bc(s)
    b["tmp_s"] = bc(s, tmp[m])

    # BVP (64 Hz)
    bvp = _col(df, "BVP")
    m = np.isfinite(bvp)
    s, v = sec[m], bvp[m]
    b["bvp_c"] = bc(s)
    b["bvp_s"] = bc(s, v)
    b["bvp_q"] = bc(s, v * v)
    rng = np.full(n, np.nan)
    if len(v) > 1:
        dv = np.abs(np.diff(v))
        ok = (np.diff(t_ns[m]) <= NS // 32)
        b["bvp_d"] = bc(s[1:], np.where(ok, dv, 0.0))
        b["bvp_dc"] = bc(s[1:], ok.astype(np.float64))
        starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        rng[s[starts]] = np.maximum.reduceat(v, starts) - np.minimum.reduceat(v, starts)
    else:
        b["bvp_d"] = np.zeros(n)
        b["bvp_dc"] = np.zeros(n)
    b["bvp_rng"] = rng

    # IBI beats
    dur = _col(df, "IBI_duration")
    idt = _col(df, "IBI_dt")
    m = (dur >= IBI_RANGE[0]) & (dur <= IBI_RANGE[1])
    s, d = sec[m], dur[m]
    gap = np.clip(np.nan_to_num(idt[m] - d, nan=0.0), 0.0, 10.0)
    missed = gap > MISS_TOL
    b["ibi_c"] = bc(s)
    b["ibi_s"] = bc(s, d)
    b["ibi_q"] = bc(s, d * d)
    b["ibi_miss"] = bc(s, missed.astype(np.float64))
    b["ibi_gap"] = bc(s, gap)
    b["beat_t"] = (t_ns[m] - t_ns[0]) / NS
    b["beat_d"] = d
    b["beat_miss"] = missed
    if len(d) > 1:
        dd = np.diff(d)
        ok = ~missed[1:]  # successive difference only between adjacent beats
        b["ibi_dd"] = bc(s[1:], np.where(ok, dd * dd, 0.0))
        b["ibi_ddc"] = bc(s[1:], ok.astype(np.float64))
    else:
        b["ibi_dd"] = np.zeros(n)
        b["ibi_ddc"] = np.zeros(n)
    return sec, n, b


# --------------------------------------------------------------------------
# Window helpers (centred windows via cumulative sums)
# --------------------------------------------------------------------------
def _wsum(x, w):
    """Centred moving sum of x (NaN treated as 0) over w seconds."""
    n = len(x)
    c = np.concatenate(([0.0], np.cumsum(np.nan_to_num(x))))
    h = w // 2
    i = np.arange(n)
    lo = np.clip(i - h, 0, n)
    hi = np.clip(i - h + w, 0, n)
    return c[hi] - c[lo]


def _ratio(a, b, min_b=1e-9):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > min_b, a / np.where(b > min_b, b, 1.0), np.nan)


def _wmean(s, c, w):
    return _ratio(_wsum(s, w), _wsum(c, w))


def _wstd(s, q, c, w):
    cs = _wsum(c, w)
    mu = _ratio(_wsum(s, w), cs)
    var = _ratio(_wsum(q, w), cs) - mu * mu
    return np.sqrt(np.maximum(var, 0.0))


def _wslope(y, w):
    """Least squares slope (units per second) of a 1 Hz series with NaNs."""
    n = len(y)
    m = np.isfinite(y).astype(np.float64)
    t = np.arange(n, dtype=np.float64) - n / 2.0
    yy = np.where(m > 0, y, 0.0)
    sn = _wsum(m, w)
    st = _wsum(t * m, w)
    sy = _wsum(yy, w)
    stt = _wsum(t * t * m, w)
    sty = _wsum(t * yy, w)
    den = sn * stt - st * st
    return _ratio(sn * sty - st * sy, den * (sn >= max(3, w // 4)), 1e-9)


def _lag(x, k):
    out = np.full_like(x, np.nan)
    if k < len(x):
        out[k:] = x[k:] - x[:-k]
    return out


# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------
def features_raw(b, n, params=None, prev=None):
    """Variant B: raw window features. Returns an ordered dict name -> 1 Hz array."""
    f = {}
    hr1 = _ratio(b["hr_s"], b["hr_c"])
    rng = b["bvp_rng"]
    rng_c = np.isfinite(rng).astype(np.float64)
    rng0 = np.nan_to_num(rng)
    for w in WINDOWS:
        f[f"hr_mean_{w}"] = _wmean(b["hr_s"], b["hr_c"], w)
        f[f"hr_std_{w}"] = _wstd(b["hr_s"], b["hr_q"], b["hr_c"], w)
        f[f"ibi_mean_{w}"] = _wmean(b["ibi_s"], b["ibi_c"], w)
        f[f"ibi_sdnn_{w}"] = _wstd(b["ibi_s"], b["ibi_q"], b["ibi_c"], w)
        f[f"ibi_rmssd_{w}"] = np.sqrt(_wmean(b["ibi_dd"], b["ibi_ddc"], w))
        cnt = _wsum(b["ibi_c"], w)
        f[f"beat_rate_{w}"] = cnt / w
        f[f"miss_rate_{w}"] = _ratio(_wsum(b["ibi_miss"], w), cnt)
        f[f"gap_mean_{w}"] = _ratio(_wsum(b["ibi_gap"], w), cnt)
        f[f"eda_mean_{w}"] = _wmean(b["eda_s"], b["eda_c"], w)
        f[f"eda_std_{w}"] = _wstd(b["eda_s"], b["eda_q"], b["eda_c"], w)
        f[f"eda_rise_{w}"] = _wmean(b["eda_rise"], (b["eda_c"] > 0).astype(float), w)
        f[f"tmp_mean_{w}"] = _wmean(b["tmp_s"], b["tmp_c"], w)
        f[f"bvp_std_{w}"] = _wstd(b["bvp_s"], b["bvp_q"], b["bvp_c"], w)
        f[f"bvp_hf_{w}"] = _wmean(b["bvp_d"], b["bvp_dc"], w)
        f[f"bvp_amp_{w}"] = _wmean(rng0, rng_c, w)
        f[f"bvp_amp_std_{w}"] = _wstd(rng0, rng0 * rng0, rng_c, w)
    eda1 = _ratio(b["eda_s"], b["eda_c"])
    tmp1 = _ratio(b["tmp_s"], b["tmp_c"])
    for w in (60, 180):
        f[f"hr_slope_{w}"] = _wslope(hr1, w)
        f[f"eda_slope_{w}"] = _wslope(eda1, w)
        f[f"tmp_slope_{w}"] = _wslope(tmp1, w)
    for key in ("hr_mean_60", "eda_mean_60", "tmp_mean_60", "bvp_std_60"):
        for k in (60, 180):
            f[f"{key}_d{k}"] = _lag(f[key], k)
    return f


def step_points(n, step):
    """Seconds at which to evaluate the classifier and the map back to all seconds.

    The classifier runs every `step` seconds (at the centre of each block) and
    each second takes the prediction of its block. Scoring uses 10 s windows,
    so a step of a few seconds costs little accuracy and saves inference time.
    """
    if step <= 1:
        i = np.arange(n)
        return i, i
    pts = np.minimum(np.arange(step // 2, n + step - 1, step), n - 1)
    pts = pts[: (n + step - 1) // step]
    back = np.minimum(np.arange(n) // step, len(pts) - 1)
    return pts, back


# --------------------------------------------------------------------------
# Kalman filtering (steady state, numpy + scipy.signal only)
#
# Every filter here has constant noise parameters, so its gain converges to
# the steady-state value from the discrete Riccati equation within a few
# samples. The steady-state filter and RTS smoother are linear time-invariant
# systems, so they run as IIR filters (scipy.signal.lfilter, compiled code)
# instead of a Python loop. Missing seconds are bridged by linear
# interpolation; how much was missing is exposed as a feature, and the
# innovation statistics only use seconds that were really measured.
# --------------------------------------------------------------------------
DEFAULT_KALMAN = {
    "hr_q": 0.03, "hr_r": 2.0,          # HR constant velocity model
    "tmp_q": 1e-6, "tmp_r": 1e-3,       # temperature local linear trend
    "eda_q": 1e-6, "eda_r": 0.05,       # tonic EDA local linear trend (slow)
    "base_tau": 600.0,                  # personal baseline time constant (s)
}


def _fill(y):
    """Linear interpolation over NaNs; returns (filled, measured mask)."""
    m = np.isfinite(y)
    if m.all():
        return y, m
    if not m.any():
        return np.zeros_like(y), m
    i = np.arange(len(y))
    return np.interp(i, i[m], y[m]), m


_GAIN_CACHE = {}


def _cv_system(q, r):
    """Steady-state constant velocity Kalman filter and RTS smoother.

    State [level, slope], x[t] = A x[t-1] + w, y[t] = level + v,
    Q = q * [[1/3, 1/2], [1/2, 1]] (white noise acceleration), R = r.
    Returns transfer functions for the filter, the smoother and S.
    """
    key = (q, r)
    if key in _GAIN_CACHE:
        return _GAIN_CACHE[key]
    from scipy import linalg, signal

    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    P = linalg.solve_discrete_are(A.T, H.T, Q, np.array([[r]]))  # prior covariance
    S = float((H @ P @ H.T)[0, 0]) + r
    K = (P @ H.T) / S
    F = (np.eye(2) - K @ H) @ A
    Pf = (np.eye(2) - K @ H) @ P
    J = Pf @ A.T @ np.linalg.inv(P)
    B = np.eye(2) - J @ A
    fnum, fden = signal.ss2tf(F, K, F, K)
    snum, sden = [], None
    for i in range(2):
        nu, sden = signal.ss2tf(J, B, J, B, input=i)
        snum.append(nu)
    sys = dict(fnum=fnum, fden=fden, snum=snum, sden=sden, S=S,
               fzi=[signal.lfilter_zi(fnum[o], fden) for o in range(2)],
               szi=[[signal.lfilter_zi(snum[i][o], sden) for o in range(2)] for i in range(2)])
    _GAIN_CACHE[key] = sys
    return sys


def kalman_cv(y, q, r, smooth=True):
    """Constant velocity Kalman on a 1 Hz series with NaNs.

    Returns level, slope (per second), normalized innovation (NaN where not
    measured) and the innovation variance S.
    """
    from scipy.signal import lfilter

    yf, m = _fill(y)
    sys = _cv_system(q, r)
    y0 = yf[0]
    lev = lfilter(sys["fnum"][0], sys["fden"], yf, zi=sys["fzi"][0] * y0)[0]
    slo = lfilter(sys["fnum"][1], sys["fden"], yf, zi=sys["fzi"][1] * y0)[0]
    pred = np.r_[y0, lev[:-1] + slo[:-1]]
    innov = np.where(m, (yf - pred) / np.sqrt(sys["S"]), np.nan)
    if smooth:
        u = (lev[::-1], slo[::-1])
        out = []
        for o in range(2):
            acc = 0.0
            for i in range(2):
                acc = acc + lfilter(sys["snum"][i][o], sys["sden"], u[i],
                                    zi=sys["szi"][i][o] * u[i][0])[0]
            out.append(acc[::-1])
        lev, slo = out
    return lev, slo, innov, sys["S"]


def cv_loglik(y, q, r):
    """Gaussian innovation log-likelihood of the steady-state filter (measured seconds)."""
    _, _, e, S = kalman_cv(y, q, r, smooth=False)
    e = e[np.isfinite(e)][10:]  # skip the start-up transient
    return float(-0.5 * np.sum(np.log(2 * np.pi * S) + e * e))


def slow_level(y, tau):
    """Very slow local level filter, forward plus backward (RTS) pass.

    For a local level model the steady-state filter is exponential smoothing
    with gain K and the RTS smoother is the same recursion run backwards.
    """
    from scipy.signal import lfilter

    yf, _ = _fill(y)
    k = 1.0 / tau
    a = [1.0, -(1.0 - k)]
    f = lfilter([k], a, yf, zi=[(1 - k) * yf[0]])[0]
    s = lfilter([k], a, f[::-1], zi=[(1 - k) * f[-1]])[0][::-1]
    return s


def _wmax(x, w):
    """Centred moving max (NaN aware) via sliding windows."""
    n = len(x)
    h = w // 2
    xp = np.pad(np.where(np.isfinite(x), x, -np.inf), (h, w - h - 1), constant_values=-np.inf)
    v = np.lib.stride_tricks.sliding_window_view(xp, w).max(axis=1)[:n]
    return np.where(np.isfinite(v), v, np.nan)


def features_kalman(b, n, params, prev=None):
    """Variant C: Kalman HR tracker, temperature and tonic EDA trend models,
    phasic EDA residual and a slow personal baseline."""
    p = params
    f = {}
    hr1 = _ratio(b["hr_s"], b["hr_c"])
    eda1 = _ratio(b["eda_s"], b["eda_c"])
    tmp1 = _ratio(b["tmp_s"], b["tmp_c"])

    # Heart rate tracker
    lev, slo, inn, _ = kalman_cv(hr1, p["hr_q"], p["hr_r"])
    base = slow_level(hr1, p["base_tau"])
    ok = np.isfinite(inn).astype(np.float64)
    i0 = np.nan_to_num(inn)
    f["kf_hr"] = lev
    f["kf_hr_slope"] = slo
    f["kf_hr_base"] = base
    f["kf_hr_minus_base"] = lev - base
    f["kf_hr_recovery"] = -slo / np.maximum(lev - base, 2.0)
    for w in (10, 60):
        f[f"kf_hr_innov2_{w}"] = _wmean(i0 * i0, ok, w)
        f[f"kf_hr_innov_{w}"] = _wmean(i0, ok, w)
    f["kf_hr_slope_absmean_60"] = _wmean(np.abs(slo), np.ones(n), 60)
    f["hr_cover_60"] = _wsum(ok, 60) / 60.0

    # Temperature trend
    if np.isfinite(tmp1).any():
        tl, ts, _, _ = kalman_cv(tmp1, p["tmp_q"], p["tmp_r"])
        tb = slow_level(tmp1, p["base_tau"])
        f["kf_tmp_slope"] = ts * 60.0
        f["kf_tmp_minus_base"] = tl - tb
    else:
        f["kf_tmp_slope"] = np.full(n, np.nan)
        f["kf_tmp_minus_base"] = np.full(n, np.nan)

    # EDA: tonic trend, phasic residual, baseline
    if np.isfinite(eda1).any():
        el, es, _, _ = kalman_cv(eda1, p["eda_q"], p["eda_r"])
        eb = slow_level(eda1, p["base_tau"])
        ef, em = _fill(eda1)
        ph = np.where(em, ef - el, np.nan)
        phc = em.astype(np.float64)
        ph0 = np.nan_to_num(ph)
        f["kf_eda_tonic_slope"] = es * 60.0
        f["kf_eda_tonic_minus_base"] = el - eb
        f["kf_eda_tonic_rel"] = _ratio(el - eb, np.abs(eb) + 0.05, 0.0)
        # phasic peaks: local maxima of the residual above a small threshold
        pk = np.zeros(n)
        if n > 2:
            mid = (ph0[1:-1] > ph0[:-2]) & (ph0[1:-1] >= ph0[2:]) & (ph0[1:-1] > 0.01)
            pk[1:-1] = mid
        pamp = np.where(pk > 0, ph0, 0.0)
        for w in (10, 60):
            f[f"kf_eda_phasic_std_{w}"] = _wstd(ph0, ph0 * ph0, phc, w)
            f[f"kf_eda_peaks_{w}"] = _wsum(pk, w) * (60.0 / w)
            f[f"kf_eda_peak_amp_{w}"] = _ratio(_wsum(pamp, w), _wsum(pk, w))
        f["kf_eda_phasic_max_10"] = _wmax(ph, 10)
        # rise time proxy: share of seconds with a rising phasic component
        rise = np.r_[0.0, (np.diff(ph0) > 0).astype(np.float64)]
        f["kf_eda_rise_frac_30"] = _wmean(rise, phc, 30)
    else:
        for k in ("kf_eda_tonic_slope", "kf_eda_tonic_minus_base", "kf_eda_tonic_rel",
                  "kf_eda_phasic_std_10", "kf_eda_peaks_10", "kf_eda_peak_amp_10",
                  "kf_eda_phasic_std_60", "kf_eda_peaks_60", "kf_eda_peak_amp_60",
                  "kf_eda_phasic_max_10", "kf_eda_rise_frac_30"):
            f[k] = np.full(n, np.nan)
    return f


# --------------------------------------------------------------------------
# Per-recording normalization and recording context
# --------------------------------------------------------------------------
NORM_KEYS = (
    "hr_mean_10", "hr_mean_60", "kf_hr", "kf_hr_minus_base", "ibi_mean_60", "ibi_rmssd_60",
    "ibi_sdnn_60", "beat_rate_60", "miss_rate_60", "eda_mean_60", "eda_std_60",
    "kf_eda_phasic_std_60", "tmp_mean_60", "bvp_std_10", "bvp_std_60", "bvp_hf_10",
    "bvp_hf_60", "bvp_amp_60", "bvp_amp_std_60",
)
CTX_KEYS = ("kf_hr", "bvp_hf_60", "bvp_std_60", "beat_rate_60", "miss_rate_60", "eda_mean_60")


def features_norm(b, n, params, prev):
    """Variant D: features expressed relative to the input recording.

    Robust z-score (median and IQR over the recording) plus the position of
    each second within the recording's own distribution (rank in 0..1).
    Raw features stay in the model, so between-state level differences are
    not lost.
    """
    f = {}
    for k in NORM_KEYS:
        x = prev[k]
        ok = np.isfinite(x)
        if ok.sum() < 30:
            f[f"z_{k}"] = np.full(n, np.nan)
            f[f"q_{k}"] = np.full(n, np.nan)
            continue
        v = x[ok]
        lo, med, hi = np.percentile(v, (25, 50, 75))
        f[f"z_{k}"] = (x - med) / max(hi - lo, 1e-6)
        r = np.full(n, np.nan)
        r[ok] = np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
        f[f"q_{k}"] = r
    return f


def features_ctx(b, n, params, prev):
    """Recording context: quantiles of a few signals over the whole input."""
    f = {}
    for k in CTX_KEYS:
        x = prev[k]
        ok = np.isfinite(x)
        qs = np.percentile(x[ok], (10, 50, 90)) if ok.sum() >= 30 else (np.nan,) * 3
        for qn, qv in zip((10, 50, 90), qs):
            f[f"ctx_{k}_p{qn}"] = np.full(n, qv)
    return f


# --------------------------------------------------------------------------
# Stress-oriented groups
# --------------------------------------------------------------------------
def _band(x, lo, hi, fs=1.0):
    """Zero-phase Butterworth band-pass of a 1 Hz series (NaNs bridged first)."""
    from scipy.signal import butter, sosfiltfilt

    xf, m = _fill(x)
    if len(xf) < 40 or not m.any():
        return np.full(len(x), np.nan), m
    sos = _SOS_CACHE.get((lo, hi, fs))
    if sos is None:
        sos = butter(2, [lo, hi], btype="band", fs=fs, output="sos")
        _SOS_CACHE[(lo, hi, fs)] = sos
    return sosfiltfilt(sos, xf - xf.mean()), m


_SOS_CACHE = {}


def features_resp(b, n, params, prev):
    """Speech and breathing: HRV bands from the IBI series and BVP-derived
    respiration (pulse amplitude modulation and baseline wander, 0.1 to 0.45 Hz)."""
    f = {}
    ibi1 = _ratio(b["ibi_s"], b["ibi_c"])
    cov = np.isfinite(ibi1).astype(np.float64)
    hf, _ = _band(ibi1, 0.15, 0.4)
    lf, _ = _band(ibi1, 0.04, 0.15)
    hf2, lf2 = hf * hf, lf * lf
    for w in (30, 60):
        hp = _wmean(hf2, np.ones(n), w)
        lp = _wmean(lf2, np.ones(n), w)
        f[f"ibi_hf_{w}"] = np.log(hp + 1e-8)
        f[f"ibi_lfhf_{w}"] = np.log((lp + 1e-8) / (hp + 1e-8))
        f[f"ibi_cov_{w}"] = _wsum(cov, w) / w
    med = np.nanmedian(f["ibi_hf_60"])
    f["ibi_hf_60_rel"] = f["ibi_hf_60"] - med
    # BVP-derived respiration from two 1 Hz carriers
    amp = b["bvp_rng"]
    wander = _ratio(b["bvp_s"], b["bvp_c"])
    for name, x in (("amp", amp), ("wan", wander)):
        r, _ = _band(x, 0.1, 0.45)
        sd = np.sqrt(_wmean(r * r, np.ones(n), 60)) + 1e-9
        zc = np.r_[0.0, (np.sign(r[1:]) != np.sign(r[:-1])).astype(np.float64)]
        for w in (30, 60):
            # breaths per minute from zero crossings of the band-passed carrier
            f[f"resp_{name}_rate_{w}"] = _wsum(zc, w) / w * 30.0
            pw = _wmean(r * r, np.ones(n), w)
            tot = _wstd(np.nan_to_num(x), np.nan_to_num(x) ** 2, np.isfinite(x).astype(float), w) ** 2
            f[f"resp_{name}_frac_{w}"] = _ratio(pw, tot + 1e-9)
        # regularity: variability of the respiratory envelope over 60 s
        env = np.abs(r) / sd
        f[f"resp_{name}_irreg_60"] = _wstd(env, env * env, np.ones(n), 60)
    return f


def features_vaso(b, n, params, prev):
    """Vasoconstriction and temperature: pulse amplitude relative to the
    recording median and short-scale temperature slopes."""
    f = {}
    rng = b["bvp_rng"]
    c = np.isfinite(rng).astype(np.float64)
    r0 = np.nan_to_num(rng)
    med = np.nanmedian(rng) if c.any() else np.nan
    for w in (10, 30, 60):
        a = _wmean(r0, c, w)
        f[f"pamp_rel_{w}"] = np.log(_ratio(a, med) + 1e-6)
    tmp1 = _ratio(b["tmp_s"], b["tmp_c"])
    f["tmp_slope_30"] = _wslope(tmp1, 30) * 60.0
    f["tmp_slope_60s"] = _wslope(tmp1, 60) * 60.0
    return f


CONTRAST_KEYS = (
    "hr_mean_10", "kf_hr", "ibi_mean_10", "ibi_rmssd_60", "beat_rate_10", "miss_rate_60",
    "eda_mean_10", "kf_eda_phasic_std_10", "kf_eda_peaks_60", "kf_eda_tonic_slope",
    "tmp_mean_10", "bvp_amp_10", "bvp_hf_10", "bvp_std_10",
    "ibi_hf_30", "ibi_lfhf_60", "resp_amp_rate_30", "resp_wan_rate_30", "resp_amp_irreg_60",
    "pamp_rel_30", "tmp_slope_30",
)


def _ring_mean(x, inner, outer, past_only):
    """Mean of x over [t-outer, t-inner] and, unless past_only, [t+inner, t+outer]."""
    n = len(x)
    m = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(m, x, 0.0))))
    k = np.concatenate(([0.0], np.cumsum(m)))
    i = np.arange(n)

    def seg(a, bb):
        lo = np.clip(i + a, 0, n)
        hi = np.clip(i + bb, 0, n)
        return c[hi] - c[lo], k[hi] - k[lo]

    s, cnt = seg(-outer, -inner + 1)
    if not past_only:
        s2, c2 = seg(inner, outer + 1)
        s, cnt = s + s2, cnt + c2
    return _ratio(s, cnt, 0.5)


def features_contrast(b, n, params, prev):
    """Within-recording contrast: feature at t minus its mean over a ring of
    context around t (central part excluded), two-sided or past-only."""
    inner = int(params.get("ctr_inner", 30))
    outer = int(params.get("ctr_outer", 180))
    past = bool(params.get("ctr_past_only", False))
    f = {}
    for k in CONTRAST_KEYS:
        if k in prev:
            f[f"ctr_{k}"] = prev[k] - _ring_mean(prev[k], inner, outer, past)
    return f


# --------------------------------------------------------------------------
# State-space Gaussian processes (item 6)
#
# A Matern-nu GP with nu = p + 1/2 is exactly a linear SDE with p + 1 states,
# so GP regression is a Kalman filter. Each kernel is discretized at the
# sampling step (A = expm(F dt), Q = Pinf - A Pinf A^T) and run as a
# steady-state filter through scipy.signal.lfilter, as for the variant C
# filters. Hyperparameters (lengthscale, variance, noise) are learned by
# maximizing the Kalman prediction error likelihood on training folds.
# --------------------------------------------------------------------------
DEFAULT_GP = {
    "hr_ell": 60.0, "hr_sig": 10.0, "hr_r": 0.5,                 # Matern-3/2
    "tmp_ell": 180.0, "tmp_sig": 1.0, "tmp_r": 1e-2,             # Matern-5/2
    "edat_ell": 300.0, "edat_sig": 0.3,                          # tonic, Matern-5/2
    "edap_ell": 5.0, "edap_sig": 0.05, "eda_r": 1e-3,            # phasic, Matern-1/2
    "qp_period": 4.0, "qp_ell": 15.0, "qp_sig": 0.02, "qp_r": 1e-3,  # IBI quasi-periodic
}


def _matern(nu, ell, sig2, dt):
    """Discrete state-space form (A, Q) of a Matern kernel with smoothness nu."""
    from scipy import linalg

    if nu == 0.5:
        lam = 1.0 / ell
        F = np.array([[-lam]])
        q = 2 * lam * sig2
    elif nu == 1.5:
        lam = np.sqrt(3.0) / ell
        F = np.array([[0.0, 1.0], [-lam ** 2, -2 * lam]])
        q = 4 * lam ** 3 * sig2
    else:  # 2.5
        lam = np.sqrt(5.0) / ell
        F = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-lam ** 3, -3 * lam ** 2, -3 * lam]])
        q = 16.0 / 3.0 * sig2 * lam ** 5
    d = F.shape[0]
    Lq = np.zeros((d, d))
    Lq[-1, -1] = q
    Pinf = linalg.solve_continuous_lyapunov(F, -Lq)
    A = linalg.expm(F * dt)
    Q = Pinf - A @ Pinf @ A.T
    return A, (Q + Q.T) / 2


def _resonator(period, ell, sig2, dt):
    """Quasi-periodic (damped oscillator) state-space component."""
    a = np.exp(-dt / ell)
    w = 2 * np.pi * dt / period
    A = a * np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
    Q = sig2 * (1 - a * a) * np.eye(2)
    return A, Q


def _blockdiag(*mats):
    d = sum(m.shape[0] for m in mats)
    out = np.zeros((d, d))
    i = 0
    for m in mats:
        k = m.shape[0]
        out[i:i + k, i:i + k] = m
        i += k
    return out


_SS_CACHE = {}


def ss_system(key, A, Q, H, r):
    """Steady-state Kalman filter of (A, Q, H, r) as transfer functions per state."""
    if key in _SS_CACHE:
        return _SS_CACHE[key]
    from scipy import linalg, signal

    H = np.atleast_2d(H)
    P = linalg.solve_discrete_are(A.T, H.T, Q, np.array([[r]]))
    S = float((H @ P @ H.T)[0, 0]) + r
    K = (P @ H.T) / S
    F = (np.eye(A.shape[0]) - K @ H) @ A
    num, den = signal.ss2tf(F, K, F, K)
    zi = [signal.lfilter_zi(num[o], den) for o in range(A.shape[0])]
    sysd = dict(num=num, den=den, zi=zi, S=S, HA=(H @ A).ravel())
    _SS_CACHE[key] = sysd
    return sysd


def ss_run(y, sysd):
    """Filter a series with NaNs. Returns states (T x d), normalized innovation
    and per-sample log likelihood (NaN where not measured)."""
    from scipy.signal import lfilter

    z, m = _fill(y)
    z0 = z[0]
    X = np.stack([lfilter(sysd["num"][o], sysd["den"], z, zi=sysd["zi"][o] * z0)[0]
                  for o in range(len(sysd["zi"]))], axis=1)
    pred = np.r_[z0, X[:-1] @ sysd["HA"]]
    e = z - pred
    S = sysd["S"]
    en = np.where(m, e / np.sqrt(S), np.nan)
    ll = np.where(m, -0.5 * (np.log(2 * np.pi * S) + e * e / S), np.nan)
    return X, en, ll


def gp_systems(p):
    """Build the four GP systems from hyperparameters p (cached by value)."""
    out = {}
    A, Q = _matern(1.5, p["hr_ell"], p["hr_sig"] ** 2, 1.0)
    out["hr"] = ss_system(("hr", p["hr_ell"], p["hr_sig"], p["hr_r"]), A, Q, [1, 0], p["hr_r"])
    A, Q = _matern(2.5, p["tmp_ell"], p["tmp_sig"] ** 2, 1.0)
    out["tmp"] = ss_system(("tmp", p["tmp_ell"], p["tmp_sig"], p["tmp_r"]), A, Q, [1, 0, 0], p["tmp_r"])
    At, Qt = _matern(2.5, p["edat_ell"], p["edat_sig"] ** 2, 1.0)
    Ap, Qp = _matern(0.5, p["edap_ell"], p["edap_sig"] ** 2, 1.0)
    key = ("eda", p["edat_ell"], p["edat_sig"], p["edap_ell"], p["edap_sig"], p["eda_r"])
    out["eda"] = ss_system(key, _blockdiag(At, Ap), _blockdiag(Qt, Qp), [1, 0, 0, 1], p["eda_r"])
    A, Q = _resonator(p["qp_period"], p["qp_ell"], p["qp_sig"] ** 2, 0.25)
    key = ("qp", p["qp_period"], p["qp_ell"], p["qp_sig"], p["qp_r"])
    out["qp"] = ss_system(key, A, Q, [1, 0], p["qp_r"])
    return out


def gp_inputs(b, n):
    """The four model inputs: HR, temperature, relative EDA (1 Hz, demeaned by
    the recording median) and the detrended 4 Hz IBI series."""
    hr1 = _ratio(b["hr_s"], b["hr_c"])
    tmp1 = _ratio(b["tmp_s"], b["tmp_c"])
    eda1 = _ratio(b["eda_s"], b["eda_c"])
    med = lambda x: np.nanmedian(x) if np.isfinite(x).any() else 0.0  # noqa: E731
    emed = med(eda1)
    eda_rel = eda1 / emed - 1.0 if emed > 0.01 else eda1 * np.nan
    ibi4 = np.full(4 * n, np.nan)
    bt, bd = b["beat_t"], b["beat_d"]
    if len(bt) > 4:
        t4 = np.arange(4 * n) / 4.0
        j = np.searchsorted(bt, t4)
        inside = (j > 0) & (j < len(bt))
        jj = np.clip(j, 1, len(bt) - 1)
        span = bt[jj] - bt[jj - 1]
        ok = inside & (span < 2.5)
        v = np.interp(t4, bt, bd)
        ibi4 = np.where(ok, v, np.nan)
        cnt = np.isfinite(ibi4).astype(np.float64)
        trend = _wmean(np.nan_to_num(ibi4), cnt, 40)  # 10 s moving mean
        ibi4 = ibi4 - trend
    return {"hr": hr1 - med(hr1), "tmp": tmp1 - med(tmp1), "eda": eda_rel, "qp": ibi4}


def gp_loglik(x, sysd):
    """Total prediction error log likelihood over measured samples."""
    if not np.isfinite(x).any():
        return 0.0
    _, _, ll = ss_run(x, sysd)
    ll = ll[np.isfinite(ll)][10:]
    return float(ll.sum())


def features_gp(b, n, params, prev=None):
    p = dict(DEFAULT_GP)
    p.update({k[3:]: v for k, v in params.items() if k.startswith("gp_")})
    sy = gp_systems(p)
    x = gp_inputs(b, n)
    f = {}

    def llw(ll, w, name):
        ok = np.isfinite(ll).astype(np.float64)
        f[name] = _wmean(np.nan_to_num(ll), ok, w)

    X, en, ll = ss_run(x["hr"], sy["hr"])
    f["gp_hr_level"] = X[:, 0]
    f["gp_hr_slope"] = X[:, 1]
    e0, ok = np.nan_to_num(en), np.isfinite(en).astype(np.float64)
    f["gp_hr_innov2_10"] = _wmean(e0 * e0, ok, 10)
    llw(ll, 10, "gp_hr_ll_10")
    llw(ll, 60, "gp_hr_ll_60")

    if np.isfinite(x["tmp"]).any():
        X, en, ll = ss_run(x["tmp"], sy["tmp"])
        f["gp_tmp_level"] = X[:, 0]
        f["gp_tmp_slope"] = X[:, 1] * 60.0
        llw(ll, 60, "gp_tmp_ll_60")
    else:
        for k in ("gp_tmp_level", "gp_tmp_slope", "gp_tmp_ll_60"):
            f[k] = np.full(n, np.nan)

    if np.isfinite(x["eda"]).any():
        X, en, ll = ss_run(x["eda"], sy["eda"])
        ph = X[:, 3]
        f["gp_eda_tonic"] = X[:, 0]
        f["gp_eda_tonic_slope"] = X[:, 1] * 60.0
        f["gp_eda_phasic"] = ph
        for w in (10, 60):
            f[f"gp_eda_phasic_std_{w}"] = _wstd(ph, ph * ph, np.ones(n), w)
            llw(ll, w, f"gp_eda_ll_{w}")
    else:
        for k in ("gp_eda_tonic", "gp_eda_tonic_slope", "gp_eda_phasic", "gp_eda_phasic_std_10",
                  "gp_eda_ll_10", "gp_eda_phasic_std_60", "gp_eda_ll_60"):
            f[k] = np.full(n, np.nan)

    # quasi-periodic respiration component on the 4 Hz IBI series
    if np.isfinite(x["qp"]).sum() > 40:
        X, en, ll = ss_run(x["qp"], sy["qp"])
        amp2 = (X ** 2).sum(1).reshape(n, 4).mean(1)
        llok = np.isfinite(ll).reshape(n, 4).sum(1).astype(np.float64)
        ll1 = _ratio(np.nan_to_num(ll).reshape(n, 4).sum(1), llok)
        e2 = np.nan_to_num(en) ** 2
        ok4 = np.isfinite(en).reshape(n, 4).sum(1).astype(np.float64)
        e2s = e2.reshape(n, 4).sum(1)
        for w in (30, 60):
            f[f"gp_qp_amp_{w}"] = np.sqrt(_wmean(amp2, np.ones(n), w))
            f[f"gp_qp_innov2_{w}"] = _ratio(_wsum(e2s, w), _wsum(ok4, w))
            llw(ll1, w, f"gp_qp_ll_{w}")
    else:
        for w in (30, 60):
            for k in (f"gp_qp_amp_{w}", f"gp_qp_innov2_{w}", f"gp_qp_ll_{w}"):
                f[k] = np.full(n, np.nan)
    return f


# --------------------------------------------------------------------------
# Detection theory (items 8a and 8b)
# --------------------------------------------------------------------------
def _seg_stats(x, a, bb):
    """Sum, sum of squares and count of x over [t+a, t+bb) for every t."""
    n = len(x)
    m = np.isfinite(x)
    x0 = np.where(m, x, 0.0)
    c1 = np.concatenate(([0.0], np.cumsum(x0)))
    c2 = np.concatenate(([0.0], np.cumsum(x0 * x0)))
    cn = np.concatenate(([0.0], np.cumsum(m)))
    i = np.arange(n)
    lo = np.clip(i + a, 0, n)
    hi = np.clip(i + bb, 0, n)
    return c1[hi] - c1[lo], c2[hi] - c2[lo], cn[hi] - cn[lo]


def _glrt(s1, q1, n1, s2, q2, n2):
    """Mean shift statistic between two windows, scaled by the pooled std."""
    with np.errstate(invalid="ignore", divide="ignore"):
        m1, m2 = s1 / n1, s2 / n2
        v1 = q1 / n1 - m1 * m1
        v2 = q2 / n2 - m2 * m2
        vp = (n1 * v1 + n2 * v2) / (n1 + n2)
        t = (m1 - m2) / np.sqrt(np.maximum(vp, 1e-12))
    ok = (n1 >= 3) & (n2 >= 10)
    return np.where(ok, np.clip(t, -20, 20), np.nan)


def features_glrt(b, n, params, prev=None):
    """GLRT for a mean shift: recent window vs the preceding 120 s, and a
    centred window vs its surrounding ring (two-sided)."""
    p = dict(DEFAULT_KALMAN)
    p.update(params)
    sig = {"hr": _ratio(b["hr_s"], b["hr_c"]), "tmp": _ratio(b["tmp_s"], b["tmp_c"])}
    eda1 = _ratio(b["eda_s"], b["eda_c"])
    if np.isfinite(eda1).any():
        sig["eda"] = kalman_cv(eda1, p["eda_q"], p["eda_r"])[0]
        sig["eda"][~np.isfinite(eda1)] = np.nan
    else:
        sig["eda"] = eda1
    f = {}
    for k, x in sig.items():
        for w in (10, 30, 60):
            r = _seg_stats(x, -w + 1, 1)
            ref = _seg_stats(x, -w - 120 + 1, -w + 1)
            f[f"glrt_{k}_past_{w}"] = _glrt(*r, *ref)
            h = w // 2
            c = _seg_stats(x, -h, w - h)
            l1 = _seg_stats(x, -h - 150, -h)
            l2 = _seg_stats(x, w - h, w - h + 150)
            f[f"glrt_{k}_ctr_{w}"] = _glrt(*c, l1[0] + l2[0], l1[1] + l2[1], l1[2] + l2[2])
    return f


DEFAULT_SCR = {"scr_rise": 1.5, "scr_decay": 6.0, "scr_thr": 3.0}


def scr_template(rise, decay, fs=4.0):
    """Bateman-shaped SCR: fast rise, exponential decay, unit energy."""
    rise = min(rise, 0.8 * decay)  # keep the template non-degenerate
    t = np.arange(0, 5 * decay, 1.0 / fs)
    h = np.exp(-t / decay) - np.exp(-t / rise)
    return h / np.sqrt(np.sum(h * h))


def scr_detector(b, n, params):
    """Matched filter output (noise normalized) on the 4 Hz phasic EDA."""
    p = dict(DEFAULT_KALMAN)
    p.update(DEFAULT_SCR)
    p.update(params)
    eda1 = _ratio(b["eda_s"], b["eda_c"])
    e4 = b["eda4"]
    if not np.isfinite(e4).any() or not np.isfinite(eda1).any():
        return None
    tonic = kalman_cv(eda1, p["eda_q"], p["eda_r"])[0]
    e4f, m4 = _fill(e4)
    ph = e4f - np.repeat(tonic, 4)
    h = scr_template(p["scr_rise"], p["scr_decay"])
    y = np.convolve(ph, h[::-1], mode="full")[len(h) - 1:]  # correlation, aligned at SCR onset
    sd = _wstd(y, y * y, m4.astype(np.float64), 1200)  # local noise over 300 s
    z = np.where(m4, _ratio(y - _wmean(y, m4.astype(np.float64), 1200), sd), np.nan)
    return z, p["scr_thr"]


def features_scr(b, n, params, prev=None):
    out = scr_detector(b, n, params)
    f = {}
    if out is None:
        for w in (10, 30, 60):
            for k in ("count", "sum", "max"):
                f[f"scr_{k}_{w}"] = np.full(n, np.nan)
        return f
    z, thr = out
    z0 = np.nan_to_num(z)
    pk = np.zeros_like(z0)
    pk[1:-1] = (z0[1:-1] > z0[:-2]) & (z0[1:-1] >= z0[2:]) & (z0[1:-1] > thr)
    cnt = pk.reshape(n, 4).sum(1)
    pos = np.maximum(z0, 0).reshape(n, 4).mean(1)
    mx = np.where(np.isfinite(z), z, -np.inf).reshape(n, 4).max(1)
    mx = np.where(np.isfinite(mx), mx, np.nan)
    for w in (10, 30, 60):
        f[f"scr_count_{w}"] = _wsum(cnt, w) * (60.0 / w)
        f[f"scr_sum_{w}"] = _wmean(pos, np.ones(n), w)
        f[f"scr_max_{w}"] = _wmax(mx, w)
    return f


GROUP_FUNCS = {
    "raw": features_raw,
    "kalman": features_kalman,
    "norm": features_norm,
    "ctx": features_ctx,
    "resp": features_resp,
    "vaso": features_vaso,
    "ctr": features_contrast,
    "gp": features_gp,
    "glrt": features_glrt,
    "scr": features_scr,
}


VARIANT_GROUPS = {
    "B": ("raw",),
    "C": ("raw", "kalman"),
    "D": ("raw", "kalman", "norm"),
    "Dctx": ("raw", "kalman", "norm", "ctx"),
}


def resolve(cfg):
    """Accept a variant name or a config dict; return (groups, params)."""
    if isinstance(cfg, str):
        cfg = {"variant": cfg}
    groups = cfg.get("groups") or VARIANT_GROUPS[cfg.get("variant", "B")]
    params = dict(DEFAULT_KALMAN)
    params.update(cfg.get("kalman", {}))
    params.update(cfg.get("params", {}))
    return tuple(groups), params


def compute_features(df, cfg="B", base=None):
    """Returns (sec, names, X): row-to-second map and per-second feature matrix.

    `base` lets training reuse the per-second aggregation across configs.
    """
    groups, params = resolve(cfg)
    sec, n, b = base if base is not None else per_second(df)
    f = {}
    for g in groups:
        f.update(GROUP_FUNCS[g](b, n, params, f))
    names = list(f)
    X = np.empty((n, len(names)), dtype=np.float32)
    for j, k in enumerate(names):
        X[:, j] = f[k]
    return sec, names, X


# --------------------------------------------------------------------------
# Decoding: offsets and HMM
# --------------------------------------------------------------------------
def viterbi(logp, log_trans, log_start):
    """Viterbi path for per-step log emissions (T x K)."""
    T, K = logp.shape
    back = np.empty((T, K), dtype=np.int8)
    d = log_start + logp[0]
    for t in range(1, T):
        sc = d[:, None] + log_trans
        back[t] = np.argmax(sc, axis=0)
        d = sc[back[t], np.arange(K)] + logp[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmax(d))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def shift_boundaries(codes, shift=0, enter=(0, 0, 0), leave=(0, 0, 0)):
    """Move every decision boundary by shift + enter[new class] + leave[old class]
    seconds (positive = later). Boundaries keep their order; the recording
    start and end do not move."""
    n = len(codes)
    ch = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    if len(ch) == 0:
        return codes
    a, bb = codes[ch - 1], codes[ch]
    nb = ch + int(shift) + np.asarray(enter, dtype=np.int64)[bb] + np.asarray(leave, dtype=np.int64)[a]
    nb = np.maximum.accumulate(np.clip(nb, 0, n))
    starts = np.r_[0, nb]
    ends = np.r_[nb, n]
    vals = np.r_[codes[0], bb]
    return np.repeat(vals, np.maximum(ends - starts, 0))


def decode(proba, cfg):
    """Turn per-second probabilities into class codes using cfg."""
    codes = _decode_core(proba, cfg)
    if "shift" in cfg or "enter" in cfg or "leave" in cfg:
        codes = shift_boundaries(codes, cfg.get("shift", 0), cfg.get("enter", (0, 0, 0)),
                                 cfg.get("leave", (0, 0, 0)))
    return codes


def _decode_core(proba, cfg):
    logp = np.log(np.clip(proba, 1e-6, 1.0))
    if "priors" in cfg:
        logp = logp - np.log(np.asarray(cfg["priors"]))
    if "emis_w" in cfg:
        # consecutive seconds share most of their 60 to 180 s feature windows,
        # so their evidence is not independent; temper it before Viterbi
        logp = logp * float(cfg["emis_w"])
    if "offsets" in cfg:
        logp = logp + np.asarray(cfg["offsets"])
    if "log_trans" in cfg:
        lt = np.asarray(cfg["log_trans"])
        step = int(cfg.get("hmm_step", 1))
        if step > 1:
            path = viterbi(logp[::step], lt, np.full(3, -np.log(3)))
            return np.repeat(path, step)[: len(logp)]
        return viterbi(logp, lt, np.full(3, -np.log(3)))
    return np.argmax(logp, axis=1)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class ADMModel(ADM_Base):
    def load_model(self, onnx_file: pathlib.Path) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_file), sess_options=so, providers=["CPUExecutionProvider"]
        )
        meta = self.session.get_modelmeta().custom_metadata_map
        self.cfg = json.loads(meta.get("adm_config", "{}"))
        self.variant = self.cfg.get("variant", "B")
        self.input_name = self.session.get_inputs()[0].name
        outs = [o.name for o in self.session.get_outputs()]
        self.prob_name = "probabilities" if "probabilities" in outs else outs[-1]
        self.labels = np.array(CLASSES, dtype=object)
        self._warmup()

    def _warmup(self):
        """Run once on a synthetic recording so that one-time costs (scipy
        imports, Riccati solutions, filter designs) happen at load time and
        not inside the timed run() calls."""
        n = 400 * 64
        t = pd.date_range("2000-01-01", periods=n, freq=pd.Timedelta(seconds=1 / 64))
        rng = np.random.default_rng(0)
        df = pd.DataFrame(np.nan, index=t, columns=["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"])
        df["BVP"] = rng.normal(0, 50, n)
        df.iloc[::16, 1] = 2.0 + rng.normal(0, 0.05, n // 16)
        df.iloc[::16, 5] = 33.0 + rng.normal(0, 0.05, n // 16)
        df.iloc[::64, 2] = 70.0 + rng.normal(0, 2, n // 64)
        df.iloc[::51, 3] = 0.8
        df.iloc[::51, 4] = 0.8 + rng.normal(0, 0.03, len(df.iloc[::51]))
        try:
            self.run(df)
        except Exception:
            pass

    def predict_seconds(self, input: pd.DataFrame):
        """Returns (sec, per-second class probabilities)."""
        # the full config carries the feature groups and the tuned filter parameters
        sec, _, X = compute_features(input, self.cfg if self.cfg else self.variant)
        step = int(self.cfg.get("pred_step", 1))
        pts, back = step_points(X.shape[0], step)
        Xs = np.ascontiguousarray(X[pts])
        proba = self.session.run([self.prob_name], {self.input_name: Xs})[0][back]
        return sec, proba

    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        if len(input) == 0:
            return pd.DataFrame({"Activity": pd.Series([], dtype=object)}, index=input.index)
        sec, proba = self.predict_seconds(input)
        codes = decode(proba, self.cfg)
        out = pd.Series(self.labels[codes[sec]], index=input.index, dtype=object)
        return pd.DataFrame({"Activity": out})


def get_model():
    model = ADMModel()
    model.load_model(pathlib.Path(__file__).with_suffix(".onnx"))
    return model
