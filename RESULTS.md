# Results

## Protocol

- Data: official kit only (85 recordings, 36 users).
- CV: `GroupKFold(n_splits=5, shuffle=True, random_state=0)` grouped by `user_id`. A person never appears in both the train and test parts of a fold.
- Metric (mirrors the competition): rows with unscored truth are ignored. For each 10 s window aligned to the epoch, take the majority true label and the majority predicted label over the scored rows. Macro-F1 is computed over Rest, Activity and Stress; each fold pools all its test windows. Tables show mean and std over folds.
- Per-class F1 is computed on all out-of-fold windows pooled.
- All tuning happens inside training folds: Kalman noise uses maximum likelihood on the training recordings of each fold. Decoder parameters (offsets, HMM) are tuned with nested evaluation: for each outer fold, only on the other folds' out-of-fold predictions.
- MET: mean of 50 `run()` calls on STRESS/f01 (201,305 rows, the longest file), one thread. Official speed score = (1 - MET / 0.3 s) x 20.
- Leaderboard: submission_B scored 83 / 100, which confirms that `run()` receives whole recordings.

## Ablation

| Variant | Macro-F1 (mean ± std) | Activity F1 | Rest F1 | Stress F1 | MET (ms) | Notes |
|---|---|---:|---:|---:|---:|---|
| A: Kalman-only IMM | pending | | | | | |
| B: raw window features + LightGBM (submitted, 83 on LB) | 0.695 ± 0.031 | 0.867 | 0.783 | 0.449 | 27 | 300 s chunks 0.611, 60 s chunks 0.505 |
| B (rerun, explicit balanced sample weights) | 0.697 ± 0.026 | 0.867 | 0.785 | 0.448 | 27 | reference for C to E |
| C: B + Kalman features | 0.735 ± 0.025 | 0.889 | 0.809 | 0.518 | about 28 | +0.039 over B |
| D: C + per-recording normalization | 0.767 ± 0.022 | 0.901 | 0.824 | 0.578 | 30.5 | +0.032 over C |
| D + recording context quantiles | 0.763 ± 0.018 | 0.901 | 0.819 | 0.574 | 31 | no gain, dropped |
| **E: D + per-class offsets (shipped as submission_E)** | **0.773 ± 0.024** | 0.909 | 0.820 | 0.587 | **30.5** | nested decoder tuning |
| D + offsets + HMM | 0.766 ± 0.026 | 0.904 | 0.813 | 0.585 | about 33 | HMM tested, rejected |
| E + autocorrelation | pending | | | | | |
| E + cross-correlation | pending | | | | | |

E improves on B by +0.078 CV macro-F1 (+6.2 quality points at equal generalization). Its speed score is 17.97 / 20 against 18.2 for B.

## Section 1 diagnostics (variant B, out-of-fold, 10 s windows)

Per-class F1: Rest 0.783, Activity 0.867, Stress 0.449. Normalized confusion (rows true):

| | pred Rest | pred Activity | pred Stress |
|---|---:|---:|---:|
| true Rest | 0.789 | 0.144 | 0.067 |
| true Activity | 0.109 | 0.868 | 0.023 |
| true Stress | 0.446 | 0.124 | 0.429 |

- Stress (only in STRESS sessions) is missed mostly as Rest (45%), and 12% is called Activity, which is impossible within that session. The four files S12, f11, f17 and f18 have 0% Stress recall.
- Inside AEROBIC and ANAEROBIC, true Rest is called Activity 42% of the time. Per phase: AEROBIC Baseline 45%, AEROBIC Rest 39%, ANAEROBIC Rest 55%. Warm-up is called Rest 46% (AEROBIC) and 32% (ANAEROBIC). Cooldown is fine (82% and 94% correct).
- MET 26 ms on one thread.

## Variant C: Kalman features

- Steady-state Kalman filters. With constant noise the gain converges to the discrete Riccati solution within a few samples, so the filter and the RTS smoother are linear time-invariant systems. They run as IIR filters (`scipy.signal.lfilter`), not a Python loop, which keeps `run()` fast.
- HR tracker: constant-velocity model `[HR, dHR/dt]`, RTS smoothed. Outputs: level, slope, normalized innovation (only on measured seconds; mean and power over 10 s and 60 s windows serve as the adaptive-noise signal), mean absolute slope, a recovery-decay rate `-slope / (HR - baseline)`, and HR coverage.
- Temperature and tonic EDA: local linear trend models. Phasic EDA is the residual of EDA minus the tonic level. Phasic std, peak count, peak amplitude, 10 s max and rise fraction.
- Personal baseline: a very slow local level filter (time constant 600 s), forward plus backward, on HR, EDA and temperature. Features relative to it.
- Noise tuned by maximum likelihood inside each training fold. The final values (all data) are HR q = 0.003, r = 0.03 and temperature q = 1e-4, r = 0.03, and every fold chose nearly the same values. HR prefers the smallest r, consistent with the wristband HR already being a smoothed average.
- Deviations from the original plan: (1) the gain is steady-state, and innovation size enters as a feature rather than adapting the gain online; (2) missing seconds are bridged by linear interpolation instead of skipping the update (coverage is a feature). Both keep the filters fully vectorized.

## Variant D: per-recording normalization

- 19 key features (HR, IBI, HRV, EDA, temperature, BVP) are added as a robust z-score (median and IQR over the input recording) and as a within-recording rank in 0 to 1. Raw features are kept.
- Rest and Activity are not hurt: Rest F1 goes 0.809 to 0.824 and Activity 0.889 to 0.901. Stress gains most (0.518 to 0.578).
- Recording-level context quantiles (p10, p50, p90 of six signals) did not help (0.763) and were dropped. With only 85 recordings, constant-per-recording features are easy to memorize.

## Variant E: decoder

Nested evaluation on D's out-of-fold probabilities:

| Decoder | Macro-F1 | Rest | Activity | Stress |
|---|---|---:|---:|---:|
| argmax | 0.767 ± 0.022 | 0.824 | 0.901 | 0.578 |
| offsets | **0.773 ± 0.024** | 0.820 | 0.909 | 0.587 |
| offsets + HMM (stickiness, Stress-specific stickiness, emission weight) | 0.766 ± 0.026 | 0.813 | 0.904 | 0.585 |

- HMM: 3 states, 1 s transitions estimated from training label sequences (unscored seconds removed), raised to the 5 s prediction step. Emissions are probabilities divided by class priors. Tuned jointly: a stickiness factor on all off-diagonal transitions, a separate factor on transitions into and out of Stress, and an emission weight to compensate for the strong correlation between consecutive steps (they share most of their 60 to 180 s windows).
- Stress F1 before the HMM is 0.587 and after it 0.585, so the HMM does not erase short Stress blocks. It trades fewer missed Stress windows (408 to 419 called Rest is about flat, recall up slightly) for more Rest to Stress false alarms (427 to 499).
- Tuned in-sample, the HMM reaches 0.787, but nested only 0.766, so its gain is decoder overfitting. The remaining errors are whole phases misclassified, not short flickers, and a smoother cannot repair those. The shipped E therefore uses offsets only.
- Final offsets (tuned on all OOF, added after dividing by priors): Rest 0, Activity +1.25, Stress -1.0.

Confusion matrices, out-of-fold windows (rows true Rest, Activity, Stress):

| | D argmax | E offsets |
|---|---|---|
| Rest | 5122 / 633 / 292 | 4815 / 805 / 427 |
| Activity | 760 / 6849 / 37 | 479 / 7115 / 52 |
| Stress | 501 / 75 / 621 | 408 / 92 / 697 |

Top features of D by gain: `miss_rate_180`, `z_ibi_mean_60`, `eda_mean_60_d180`, `kf_hr_base`, `hr_mean_60_d180`, `q_ibi_mean_60`, `ibi_rmssd_180`, `ibi_mean_60`, `kf_eda_tonic_rel`, `eda_slope_180`.

## Timing (submission_E)

| Recording | Rows | MET | Speed score |
|---|---:|---:|---:|
| STRESS/f01 | 201,305 | 30.5 ms | 17.97 |
| AEROBIC/f01 | 158,556 | 25.3 ms | 18.31 |
| ANAEROBIC/f13 | 142,797 | 24.2 ms | 18.39 |

## Variant B details (first submission)

- 62 features from a 1 Hz grid, using centred windows of 10, 60 and 180 s:
  - HR mean and std.
  - IBI mean, SDNN, RMSSD, beat rate, missed-beat rate, mean `IBI_dt - IBI_duration`.
  - EDA mean, std and rise; temperature mean.
  - BVP std, mean absolute derivative, per-second amplitude mean and spread.
  - Slopes over 60 and 180 s for HR, EDA and temperature.
  - Lagged changes (t vs t-60 s and t-180 s) for the 60 s means of HR, EDA, temperature and BVP std.
- LightGBM with 300 trees and 31 leaves, `min_child_samples=200`, balanced class weights, trained on every second second. Exported with onnxmltools (`zipmap=False`).
- The classifier runs every 5 s (`pred_step=5`) and each second inherits its block's prediction. CV macro-F1 is 0.6951 at step 1, 0.6947 at step 3 and 0.6954 at step 5, while ONNX time drops from 35 ms to 8 ms.

## Stress fixes on top of E (nested grouped CV)

Method: every idea is trained with the same GroupKFold by `user_id` (5 folds), Kalman noise tuned inside each training fold, and per-class offsets (plus the Stress weight, where it applies) chosen per outer fold on the other folds' out-of-fold predictions only. Keep rule: nested macro-F1 at least +0.005 over E and Stress F1 improving.

### Noise floor

E re-run with three LightGBM seeds (same folds, same features): 0.7725, 0.7716, 0.7644. The seed-to-seed spread (0.008) is larger than the +0.005 keep threshold, so single-seed rows below are indicative only. Decisions use the 3-seed means.

### Single seed (seed 0)

| Idea | Macro-F1 | Activity | Rest | Stress | Stress row (Rest / Activity / Stress) | MET est. (ms) |
|---|---|---:|---:|---:|---|---:|
| E (reference) | 0.7725 ± 0.024 | 0.909 | 0.820 | 0.587 | 0.341 / 0.077 / 0.582 | 30.5 |
| 1. contrast two-sided (single model) | 0.7664 ± 0.018 | 0.904 | 0.812 | 0.586 | 0.348 / 0.094 / 0.558 | 32 |
| 1. contrast past-only (single model) | 0.7740 ± 0.022 | 0.904 | 0.811 | 0.605 | 0.295 / 0.080 / 0.625 | 32 |
| 1. two-stage, contrast two-sided | 0.7721 ± 0.019 | 0.909 | 0.821 | 0.587 | 0.357 / 0.084 / 0.559 | 40 |
| 1. two-stage, contrast past-only | 0.7739 ± 0.021 | 0.909 | 0.820 | 0.590 | 0.338 / 0.079 / 0.583 | 40 |
| 2. speech and breathing (resp) | 0.7734 ± 0.021 | 0.901 | 0.812 | 0.606 | 0.305 / 0.087 / 0.608 | 35 |
| 2. resp + contrast two-sided | 0.7726 ± 0.019 | 0.903 | 0.811 | 0.604 | 0.293 / 0.094 / 0.613 | 37 |
| 3. vasoconstriction and temperature (vaso) | 0.7639 ± 0.023 | 0.905 | 0.817 | 0.571 | 0.367 / 0.089 / 0.544 | 31 |
| 3. vaso + contrast two-sided | 0.7679 ± 0.024 | 0.904 | 0.820 | 0.582 | 0.385 / 0.075 / 0.540 | 33 |
| 4. Stress weight {1,2,3,4,6} with offsets (chosen 4,4,4,2,2) | 0.7707 ± 0.021 | 0.906 | 0.819 | 0.589 | 0.353 / 0.079 / 0.568 | 30.5 |
| exploratory: resp + contrast past-only | 0.7769 ± 0.021 | 0.907 | 0.812 | 0.611 | 0.267 / 0.088 / 0.645 | 37 |

MET est.: measured MET of E plus the measured extra feature time on STRESS/f01 (resp 4.6 ms, contrast 1.9 ms, vaso 0.8 ms; the two-stage model adds a second ONNX pass, about 8 ms). None of these was exported, so these are estimates.

### Three seeds (decision basis)

| Idea | Macro-F1 (mean of seeds) | Seed scores | Activity | Rest | Stress | Stress row (Rest / Activity / Stress) | Decision |
|---|---|---|---:|---:|---:|---|---|
| E (reference) | 0.7695 | 0.7725, 0.7716, 0.7644 | 0.906 | 0.817 | 0.585 | 0.340 / 0.078 / 0.582 | |
| 1. contrast past-only | 0.7706 (+0.0011) | 0.7740, 0.7694, 0.7683 | 0.905 | 0.812 | 0.595 | 0.306 / 0.079 / 0.615 | drop |
| 2. resp | 0.7741 (+0.0046) | 0.7734, 0.7761, 0.7728 | 0.904 | 0.814 | 0.603 | 0.305 / 0.084 / 0.611 | drop (just under +0.005) |
| 4. Stress weight | 0.7701 (+0.0006) | 0.7707, 0.7682, 0.7715 | 0.906 | 0.817 | 0.589 | 0.338 / 0.085 / 0.577 | drop |
| exploratory: resp + contrast past-only | 0.7746 (+0.0051) | 0.7769, 0.7760, 0.7710 | 0.904 | 0.814 | 0.606 | 0.283 / 0.085 / 0.632 | below the +0.01 bar for submission_F |

Dropped after one seed (clearly negative or flat): two-sided contrast, both two-stage variants, vaso, vaso + contrast.

Observations:
- The section 1 two-stage design gives nothing over the single 3-class model: Stage 2 sees the same features, and the offsets already set the Stress vs Rest trade-off.
- Two-sided contrast hurts. The ring of context around a 30 s opinion task often contains the neighbouring Stress task or its recovery, so the contrast is diluted. Past-only context is cleaner.
- Respiration features are the only group that helps on every seed (paired gains over E +0.0009, +0.0045, +0.0084), and they move Stress windows from Rest to Stress (Stress recall 0.582 to 0.611). Combined with past-only contrast, Stress recall reaches 0.632.
- Activity and Rest F1 stay within 0.005 of E in every row, so none of these ideas damages the other classes.
- The Stress weight is redundant with the Stress offset: both shift the same decision boundary.

People whose Stress F1 stays below 0.3 (any seed, E and the best variants): S15, S17, f11, f17, f18 in nearly every run; S01 and f13 in some runs. These are low responders whose whole Stress tasks look like the surrounding Rest.

Final combination: no idea passed the keep rule on its own, so the combination is E itself and no submission_F was built. The best combination tried (resp + past-only contrast) gains +0.005, half the +0.01 required for a new submission.

## State-space GP, timing shift and detection theory (items 6 to 8)

Same protocol as above: GroupKFold by `user_id`, every hyperparameter tuned inside the training folds, offsets (and shifts) tuned per outer fold on the other folds' out-of-fold predictions, three LightGBM seeds per row. Reference: E at 0.7695 (3-seed mean). MET est. = measured MET of E (30.5 ms) plus the measured extra feature time on STRESS/f01.

| Item | Macro-F1 (3 seeds) | Seed scores | Activity | Rest | Stress | Stress row (Rest / Activity / Stress) | MET est. (ms) | Decision |
|---|---|---|---:|---:|---:|---|---:|---|
| E (reference) | 0.7695 ± 0.022 | 0.7725, 0.7716, 0.7644 | 0.906 | 0.817 | 0.585 | 0.340 / 0.078 / 0.582 | 30.5 | |
| 6. State-space GP layer | 0.7674 ± 0.022 | 0.7688, 0.7686, 0.7649 | 0.907 | 0.820 | 0.578 | 0.367 / 0.069 / 0.564 | 35 | drop (-0.002) |
| 7. Global time shift | 0.7679 ± 0.022 | 0.7711, 0.7693, 0.7634 | 0.906 | 0.816 | 0.581 | 0.344 / 0.075 / 0.581 | 30.5 | drop (-0.002) |
| 7. Per-transition shifts | 0.7685 ± 0.021 | 0.7691, 0.7726, 0.7638 | 0.904 | 0.818 | 0.583 | 0.359 / 0.066 / 0.575 | 30.5 | drop (-0.001, unstable) |
| 8a. GLRT change statistics | 0.7709 ± 0.021 | 0.7708, 0.7685, 0.7734 | 0.908 | 0.815 | 0.590 | 0.319 / 0.088 / 0.594 | 35 | drop (+0.0014) |
| 8b. SCR matched filter | 0.7703 ± 0.021 | 0.7732, 0.7702, 0.7675 | 0.905 | 0.817 | 0.588 | 0.340 / 0.077 / 0.583 | 34.5 | drop (+0.0008) |

### Item 6: kernels and learned hyperparameters

Each signal is modelled as a GP in state-space form, discretized at its sampling step (A = expm(F dt), Q = Pinf - A Pinf A^T with Pinf from the continuous Lyapunov equation) and run as a steady-state Kalman filter through `scipy.signal.lfilter`. Hyperparameters maximize the Kalman prediction error likelihood (the GP marginal likelihood) on the training recordings of each fold, by grid search per signal.

| Signal | Kernel (state) | Input | Learned (all 5 folds unless a range is given) |
|---|---|---|---|
| HR | Matern-3/2 (level, slope) | 1 Hz HR minus recording median | lengthscale 120 s, sigma 20 bpm, noise r 0.003 |
| Temperature | Matern-5/2 (3 states) | 1 Hz minus median | lengthscale 10 to 20 s, sigma 1.0 C, r 0.01 to 0.03 |
| EDA tonic | Matern-5/2 | EDA / median - 1 (relative) | lengthscale 120 s, sigma 3.0 |
| EDA phasic | Matern-1/2 (Ornstein-Uhlenbeck) | summed with the tonic GP, shared r | lengthscale 5 s, sigma 0.15, r 1e-5 to 1e-4 |
| Respiration | quasi-periodic damped resonator (2 states) | 4 Hz IBI series, 10 s trend removed | period 6 s, damping lengthscale 2 s, sigma 0.03 s, r 1e-5 |

Features: filtered states (HR level and slope, temperature level and slope, tonic level and slope, phasic level), normalized innovation power, and the per-window mean log likelihood under each kernel (10 s and 60 s; 30 s and 60 s for respiration), plus respiratory amplitude.

Interpretation: hyperparameters are very stable across folds, but the GP layer adds no information beyond the variant C Kalman features (same models up to stationarity). The respiration resonator learns a damping lengthscale of only 2 s, so the IBI series carries little coherent respiratory oscillation at the wristband's beat detection quality. The HR and EDA measurement noise go to the lowest grid values because the device HR is already an averaged series and EDA sensor noise is tiny compared with physiological variation.

### Item 7: timing shift

- Global shift, chosen per fold in 5 s steps over -60 to +60 s: seed 0 chose 0, 5, 0, 0, 0; seed 1 chose 10, 0, 0, 0, 10; seed 2 chose 0, 0, 0, 0, 10. The best shift is essentially 0. The centred 60 to 180 s feature windows already absorb the physiological lag.
- Per-transition shifts (enter and leave Stress, enter and leave Activity) are unstable: leaving Activity varies from -50 s to +5 s across folds and seeds, entering Activity from -10 to +25 s, Stress shifts from -10 to +10 s. Dropped.

### Item 8a: GLRT

Mean-shift statistic (m_recent - m_ref) / pooled std, with recent windows of 10, 30 and 60 s against the preceding 120 s (causal), and a two-sided version (centred window against a 150 s ring on both sides), for HR, tonic EDA and temperature. All computed with cumulative sums. It moves some Stress windows from Rest to Stress (Stress row Rest share 0.340 to 0.319) but adds Rest to Stress false alarms; net +0.0014.

### Item 8b: SCR matched filter

Bateman template exp(-t/decay) - exp(-t/rise) at 4 Hz with unit energy, correlated with the 4 Hz phasic EDA (EDA minus the Kalman tonic level), normalized by the local mean and std of the filter output over 300 s. Detections are local maxima above a threshold. Rise, decay and threshold are tuned inside each training fold by the AUC of the 30 s detection count for Stress vs Rest seconds in training STRESS sessions. All folds chose rise 0.5 s, decay 3 s, threshold 1.5. The best AUC is only 0.58: phasic responses occur in Rest as often as in Stress at the wrist.

### Item 8c: decision stage as a detector

The decision stage is a Bayesian three-hypothesis detector. The LightGBM model is trained with balanced class weights, so its output approximates p(x | c) up to a common factor. The decoder chooses the class c maximizing log p(c | x) - log pi_c + o_c, where pi_c are the class priors from training and o_c are the tuned offsets. This is a MAP rule with effective priors and costs proportional to exp(o_c). Final offsets: Rest 0, Activity +1.25, Stress -1.0.

Out-of-fold Stress vs not-Stress (10 s windows, score = mean P(Stress) over the window), shipped variant E, seed 0:

| Quantity | Value |
|---|---|
| ROC AUC | 0.944 |
| Stress detection at 5% false alarm | 0.66 |
| Stress detection at 10% false alarm | 0.80 |
| Stress detection at 20% false alarm | 0.93 |
| Operating point of the tuned offsets (nested) | false alarm 0.035, detection 0.58 |

![Stress ROC](figures/roc_stress.png)

The tuned offsets sit at a low false-alarm rate on purpose. Stress is only 8% of windows, so with about 13,700 not-Stress windows a 10% false-alarm rate would produce about 1,370 false alarms against about 950 detections (precision 0.41, F1 about 0.55), worse than the current 0.585. Gains must come from better ranking (higher AUC), not from moving the threshold.

People whose Stress F1 stays below 0.3 across items 6 to 8: S15, S17, f11, f17, f18 in nearly every run; S01 and f13 in some.

## Final combination (item 9)

No item from sections 1 to 8 passed the keep rule (+0.005 nested macro-F1 on the 3-seed mean, plus a Stress F1 gain for Stress items), so the final combination is E itself and no new submission was built. For reference, an exploratory stack of every slightly positive group:

| Exploratory stack | Macro-F1 (3 seeds) | Seed scores | Activity | Rest | Stress | Stress row (Rest / Activity / Stress) | MET est. (ms) |
|---|---|---|---:|---:|---:|---|---:|
| E + resp + past-only contrast + GLRT + SCR | 0.7745 ± 0.020 | 0.7755, 0.7773, 0.7706 | 0.904 | 0.812 | 0.607 | 0.281 / 0.085 / 0.635 | 43 |
| E + resp + past-only contrast | 0.7746 ± 0.021 | 0.7769, 0.7760, 0.7710 | 0.904 | 0.814 | 0.606 | 0.283 / 0.085 / 0.632 | 37 |

GLRT and SCR add nothing on top of breathing plus past-only contrast. The +0.005 gain is half of the +0.01 required for a new submission.

## Full ablation (for the report and pitch)

Nested grouped 5-fold CV by `user_id`, competition window metric. Rows marked (3 seeds) are means over LightGBM seeds 0, 1, 2; the others are seed 0. LB = public leaderboard.

| Variant | Macro-F1 | Activity | Rest | Stress | MET (ms) | Status |
|---|---|---:|---:|---:|---:|---|
| B: raw window features + LightGBM | 0.695 ± 0.031 | 0.867 | 0.783 | 0.449 | 27 | submitted, LB 83 |
| C: B + Kalman features | 0.735 ± 0.025 | 0.889 | 0.809 | 0.518 | about 28 | step |
| D: C + per-recording normalization | 0.767 ± 0.022 | 0.901 | 0.824 | 0.578 | 30.5 | step |
| D + recording context quantiles | 0.763 ± 0.018 | 0.901 | 0.819 | 0.574 | 31 | dropped |
| D + offsets + HMM | 0.766 ± 0.026 | 0.904 | 0.813 | 0.585 | about 33 | dropped |
| **E: D + per-class offsets** | **0.773 ± 0.024 (3 seeds: 0.7695)** | 0.909 | 0.820 | 0.587 | **30.5** | **submitted, LB 85** |
| 1. contrast two-sided | 0.766 ± 0.018 | 0.904 | 0.812 | 0.586 | 32 | dropped |
| 1. contrast past-only (3 seeds) | 0.7706 | 0.905 | 0.812 | 0.595 | 32 | dropped |
| 1. two-stage (two-sided / past-only) | 0.772 / 0.774 | 0.909 | 0.821 / 0.820 | 0.587 / 0.590 | 40 | dropped |
| 2. speech and breathing (3 seeds) | 0.7741 | 0.904 | 0.814 | 0.603 | 35 | dropped (+0.0046) |
| 3. vasoconstriction and temperature | 0.764 ± 0.023 | 0.905 | 0.817 | 0.571 | 31 | dropped |
| 4. Stress weight, nested (3 seeds) | 0.7701 | 0.906 | 0.817 | 0.589 | 30.5 | dropped |
| 6. state-space GP layer (3 seeds) | 0.7674 | 0.907 | 0.820 | 0.578 | 35 | dropped |
| 7. global time shift (3 seeds) | 0.7679 | 0.906 | 0.816 | 0.581 | 30.5 | dropped |
| 7. per-transition shifts (3 seeds) | 0.7685 | 0.904 | 0.818 | 0.583 | 30.5 | dropped |
| 8a. GLRT (3 seeds) | 0.7709 | 0.908 | 0.815 | 0.590 | 35 | dropped |
| 8b. SCR matched filter (3 seeds) | 0.7703 | 0.905 | 0.817 | 0.588 | 34.5 | dropped |
| best stack: E + breathing + past-only contrast (3 seeds) | 0.7746 | 0.904 | 0.814 | 0.606 | 37 | below +0.01 |
| A: Kalman-only IMM | pending | | | | | |

Seed-to-seed spread of a single configuration is about 0.008, so differences under about 0.005 between rows are not meaningful.

## submission_F (built on request)

Variant F = E + speech and breathing features + past-only contrast, per-class offsets (final: Rest 0, Activity +1.5, Stress -0.25). Built at the user's request although its CV gain (+0.005 on the 3-seed mean, +0.0044 to +0.0066 paired per seed) is below the +0.01 rule.

| | Macro-F1 (nested, seed 0) | 3-seed mean | Activity | Rest | Stress | MET (STRESS/f01) |
|---|---|---|---:|---:|---:|---:|
| E | 0.7725 ± 0.024 | 0.7695 | 0.909 | 0.820 | 0.587 | 30.5 ms |
| F | 0.7769 ± 0.021 | 0.7746 | 0.907 | 0.812 | 0.611 | 38 to 40 ms |

Two inference fixes went into F (they did not exist in submission_B or submission_E):

1. Configuration bug: `run()` passed only the variant name to the feature code, so the Kalman noise parameters tuned in training (HR q 0.003, r 0.03; temperature q 1e-4, r 0.03) were replaced by the defaults (HR q 0.03, r 2.0; temperature q 1e-6, r 1e-3) at inference. submission_E was therefore evaluated with Kalman features that differ from those it was trained on. `run()` now passes the full configuration from the ONNX metadata, and `make_submission.py` checks parity: per-second probabilities from `run()` must match the training path on STRESS/f01 (max |dp| 1.5e-7 for F) and the feature names must match.
2. Cold start: the first `run()` call paid about 700 ms of one-time cost (scipy imports, Riccati solutions, filter design). If the evaluator does not warm up, that adds about 14 ms to a 50-call mean. `load_model()` now runs a synthetic 400 s warm-up (about 0.8 s at load time), and the first timed call costs the same as the others (40 ms).

## XGBoost + statistical extensions (items 1 to 6)

### Protocol

- 3 repeats of GroupKFold(5) by `user_id` (shuffle seed = repeat) x 3 model seeds = 9 runs per row. Every row uses the same folds.
- Features: the variant F set (raw, Kalman, per-recording normalization, breathing, past-only contrast).
- XGBoost hyperparameters: the ones the model zoo search chose inside each (repeat, fold), reused for every item so that comparisons are paired. Item 5 reruns the search.
- Fitted inside each training fold only: Kalman noise (maximum likelihood), vector Kalman noise (maximum likelihood), shrinkage population statistics and kappa, QDA (cross-fitted on training people), decoder offsets or costs (nested, on the other folds' out-of-fold predictions).
- Keep rule: mean macro-F1 at least +0.005 over the reference, a win on at least 2 of 3 repeats, and Stress F1 not lower.
- Macro-F1 is the mean over the 9 runs, +/- the std over all 45 fold scores. AUC is Stress vs Rest on 10 s windows. Low responders = mean Stress F1 of S15, S17, f11, f17, f18.
- MET was measured on this container, which is about 1.45x slower than the machine used for the earlier MET figures (submission F: 56.5 ms here, 38 to 40 ms there). Extra feature time is measured on STRESS/f01. Code: `training/stat_ext.py`, results: `artifacts/stat/*.json`.

### Results on top of XGBoost

| Item | Variant | Macro-F1 | Repeat 0 / 1 / 2 | Activity | Rest | Stress | AUC | Low resp. | Extra time | Decision |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| ref | XGBoost, variant F features | 0.7652 +/- 0.040 | 0.7690 / 0.7415 / 0.7852 | 0.905 | 0.807 | 0.582 | 0.892 | 0.238 | | reference |
| 1 | MAP / LMMSE shrinkage of the normalization statistics | 0.7641 +/- 0.035 | 0.7667 / 0.7468 / 0.7789 | 0.904 | 0.808 | 0.583 | 0.892 | 0.216 | < 0.1 ms | drop (-0.0011, 1/3) |
| 1 | Calm baseline (calmest 25% of the recording) | 0.7688 +/- 0.036 | 0.7751 / 0.7487 / 0.7828 | 0.907 | 0.810 | 0.595 | 0.894 | 0.192 | < 0.1 ms | drop by the rule (+0.0036, 2/3) |
| 1 | Calm baseline + shrinkage | 0.7646 +/- 0.040 | 0.7735 / 0.7375 / 0.7829 | 0.904 | 0.805 | 0.587 | 0.892 | 0.180 | < 0.1 ms | drop (-0.0006, 1/3) |
| 4 | Cost matrix decoder instead of offsets | 0.7596 +/- 0.043 | 0.7653 / 0.7308 / 0.7827 | 0.906 | 0.805 | 0.567 | 0.892 | 0.241 | 0 | drop (-0.0056, 0/3) |
| 2 | Two-measurement HR Kalman filter | 0.7651 +/- 0.037 | 0.7704 / 0.7389 / 0.7859 | 0.904 | 0.809 | 0.584 | 0.894 | 0.235 | 1.2 ms | drop (-0.0001) |
| 3 | Shrinkage QDA log-likelihood ratios | 0.7657 +/- 0.040 | 0.7684 / 0.7414 / 0.7872 | 0.905 | 0.808 | 0.584 | 0.892 | 0.222 | about 0.3 ms | drop (+0.0005, 1/3) |
| 5 | XGBoost search with reg_lambda and reg_alpha | 0.7640 +/- 0.031 | 0.7667 / 0.7535 / 0.7717 | 0.905 | 0.807 | 0.586 | 0.896 | 0.247 | 0 | report only |

#### Item 1: Bayesian shrinkage of personal baselines

- baseline = (n_eff * recording statistic + kappa * population statistic) / (n_eff + kappa), applied to the median and the IQR of the 19 normalized keys. n_eff = valid seconds of that key. The population statistic is the median over the training fold's recordings. kappa is tuned per signal group (heart, EDA, temperature, BVP) over {0, 60, 300, 900, 3600} by coordinate search on a grouped 75/25 inner split of the training fold (inner model: XGBoost with 150 trees on every 8th second).
- Chosen kappa over 15 folds: heart 0 x6, 60 x4, 300 x2, 900 x2, 3600 x1; EDA 0 x6, 60 x2, 300 x3, 900 x2, 3600 x2; temperature 0 x9, 300 x2, 900 x3, 3600 x1; BVP 0 x14, 60 x1. No shrinkage is the most common choice everywhere and the rest is unstable. Each recording has 900 to 3000 valid seconds, so its own statistic is already precise and shrinking it mostly removes real between-person information.
- The calm baseline ranks every second by HR level and phasic EDA activity and takes the median and IQR over the calmest 25%. It wins 2 of 3 repeats and raises Stress F1 by 0.013, but +0.0036 is under the bar. Combined with shrinkage the gain disappears.
- Low responders do not improve with any variant (0.238 to 0.192 for calm; S15 0.23 to 0.02, f11 0.51 to 0.64, f17 0.05 to 0.00). Their Stress tasks look like their Rest in every signal, so no baseline estimate creates contrast that is not there.

#### Item 4: Bayesian cost-matrix decision

- Decision = argmin over j of sum over i of p(i | x) C[i][j] on the balanced XGBoost probabilities, 6 free off-diagonal costs on a log2 grid from 1/8 to 8, coordinate search, nested per outer fold.
- Chosen costs (geometric mean over folds, truth -> decision): Rest -> Activity 0.45, Rest -> Stress 0.57, Activity -> Rest 0.99, Activity -> Stress 0.84, Stress -> Rest 0.95, Stress -> Activity 1.54. They encode the same preferences as the offsets (borderline Rest goes to Activity; Stress is never called Activity) but with 6 parameters instead of 2 the nested tuning overfits and loses on all 9 runs.

#### Item 2: two-measurement HR Kalman filter

- State [HR, dHR/dt]. Measurements per second: the HR column (noise r_h) and 60 / IBI_duration of each beat in that second (noise r_i per beat, times an inflation factor for beats after a missed beat), fused by precision weighting. Steady-state gain for the nominal fused noise (keeps it an IIR filter, 1.2 ms per recording); the time-varying fused variance enters the innovation variance. Features: fused HR, slope, HR minus slow baseline, innovation power (10, 60 s), measurement agreement (squared normalized difference over 10 and 60 s, signed mean over 60 s) and the share of seconds with both measurements.
- Maximum likelihood (grid, inside each fold): q 0.003, r_h 0.01, r_i 64 in all 15 folds; inflation 10 (10 folds) or 100 (5 folds). One beat is about 6400 times less reliable than the device HR, which is already a smoothed average, so the fused HR equals the existing Kalman HR and the agreement features repeat what the missed-beat rate already says.

#### Item 3: QDA log-likelihood ratios

- Inputs: z_hr_mean_60, z_ibi_rmssd_60, z_eda_mean_60, resp_wan_rate_60, kf_tmp_slope (missing filled with training medians, clipped to the training 0.5 to 99.5% range). Class covariances shrunk towards the pooled covariance with lambda chosen by cross-fitted held-out likelihood; all 15 folds chose 0.9, so the QDA is close to an LDA. Training rows get LLR values from an inner 5-fold cross-fit by user; test rows from the QDA fitted on the whole training fold.
- log p(x | Stress) - log p(x | Rest) and log p(x | Activity) - log p(x | Rest) add nothing: the trees already split on the same five inputs.

#### Item 5: regularization as priors

- An L2 penalty on leaf weights (reg_lambda) is a zero-mean Gaussian prior, so the fit is a MAP estimate; an L1 penalty (reg_alpha) is a Laplace prior that sets weak leaves to zero. Searched inside each fold with the zoo protocol (12 trials, grouped 75/25 inner split): reg_lambda in {0.1, 1, 5, 20}, reg_alpha in {0, 0.1, 1, 5}.
- Chosen over 15 folds: reg_lambda 0.1 x5, 1 x4, 5 x3, 20 x3; reg_alpha 0 x4, 0.1 x4, 1 x4, 5 x3. The choices are spread over the whole grid: with about 70,000 training seconds per fold the likelihood dominates any reasonable prior on leaf weights, and the effective regularization comes from min_child_weight, depth and subsampling. Score within noise (-0.0012).

### Item 6: combination and submission

No item passed the keep rule, so the combination of kept items is plain XGBoost. The reference for "current best submission" is submission F scored with the same protocol:

| Model | Macro-F1 | Repeat 0 / 1 / 2 | Activity | Rest | Stress | AUC | Low resp. | MET STRESS/f01 (this container) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Submission F (LightGBM, fixed parameters) | 0.7706 +/- 0.031 | 0.7719 / 0.7599 / 0.7800 | 0.907 | 0.815 | 0.596 | 0.906 | 0.229 | 56.5 ms |
| XGBoost (plain) | 0.7652 +/- 0.040 | 0.7690 / 0.7415 / 0.7852 | 0.905 | 0.807 | 0.582 | 0.892 | 0.238 | about 59 ms |
| submission_G: XGBoost + calm baseline | 0.7688 +/- 0.036 | 0.7751 / 0.7487 / 0.7828 | 0.907 | 0.810 | 0.595 | 0.894 | 0.192 | 59.4 ms |
| **submission_H: LightGBM F + calm baseline** | **0.7754 +/- 0.033** | 0.7812 / 0.7651 / 0.7798 | 0.910 | 0.814 | 0.610 | 0.906 | 0.189 | 52.9 to 60.8 ms |

- XGBoost is not better than submission F. The zoo ranking (XGBoost 0.763 vs LightGBM 0.759) compared XGBoost with a LightGBM whose parameters came from the zoo's own inner search; the fixed parameters of submission F beat them. Measured with the same folds, F leads plain XGBoost by 0.005.
- submission_G (built on request, XGBoost + calm baseline): -0.0018 vs F; wins repeats 0 and 2 by about 0.003, loses the hardest repeat by 0.011. Not an expected gain.
- submission_H (LightGBM F + calm baseline, offsets Activity +1.5, Stress -0.75): +0.0048 vs F, wins 2 of 3 repeats (+0.0094, +0.0052, -0.0002), Stress F1 +0.014. This misses the +0.005 rule by 0.0002, so it is a borderline candidate, built because the calm baseline was asked for and this is its stronger form. Expected gain about +0.4 quality points (0.0048 x 80) at equal speed, which is within leaderboard noise.
- Speed: G and H take about the same time as F (all within timer noise of each other on this container). On the machine where F measured 38 to 40 ms they should be around 40 ms.
- Both zips pass every `make_submission.py` check: allowed imports, size (357 KB and 379 KB), fresh-process load, output format, and parity of `run()` against the training path (max |dp| 4.8e-7 for G, 1.7e-7 for H).

## Reframing experiments

Reference for the keep rule: XGBoost on the variant F features (0.7652 under the repeated protocol, `artifacts/stat/base.json`). Note that submission F's LightGBM scores 0.7706 under the same protocol, and submission H 0.7754.

### Step 1: diagnostic of missed Stress (no training)

Out-of-fold XGBoost probabilities of the 9 runs (3 repeats x 3 seeds), nested offsets decoder, 10 s windows. Score = mean Stress log-odds against Rest, log p(Stress) - log p(Rest), over the window. AUC = within-recording Stress vs Rest. Values are means over the 9 runs. Low responders marked *.

| Person | Within-recording AUC | Stress F1 | Median score, Stress windows | Median score, Rest windows |
|---|---:|---:|---:|---:|
| f17 * | 0.721 | 0.046 | -2.08 | -3.38 |
| f18 * | 0.811 | 0.201 | -1.89 | -4.21 |
| S17 * | 0.510 | 0.208 | -1.97 | -1.91 |
| S15 * | 0.814 | 0.227 | -1.38 | -4.96 |
| f13 | 0.715 | 0.249 | -3.22 | -4.75 |
| S01 | 0.714 | 0.399 | -1.15 | -3.16 |
| f12 | 0.805 | 0.412 | -0.64 | -5.99 |
| f03 | 0.877 | 0.472 | 0.59 | -3.13 |
| f11 * | 0.962 | 0.505 | 0.41 | -8.53 |
| f16 | 0.785 | 0.528 | 1.20 | -4.45 |
| f06 | 0.952 | 0.553 | 2.63 | -2.69 |
| f01 | 0.871 | 0.556 | -0.08 | -5.65 |
| S13 | 0.922 | 0.563 | -0.21 | -5.90 |
| f09 | 0.930 | 0.605 | -0.67 | -7.22 |
| f04 | 0.980 | 0.620 | 1.96 | -3.05 |
| S05 | 0.912 | 0.637 | -0.09 | -6.28 |
| f15 | 0.958 | 0.663 | 2.42 | -3.51 |
| S09 | 0.964 | 0.678 | 0.08 | -6.86 |
| f05 | 0.969 | 0.684 | 5.78 | -7.65 |
| f10 | 0.925 | 0.688 | 1.76 | -5.86 |
| S18 | 0.938 | 0.695 | -0.19 | -5.71 |
| S11 | 0.920 | 0.703 | -0.16 | -5.71 |
| S16 | 0.970 | 0.714 | 0.45 | -7.33 |
| S03 | 0.972 | 0.720 | 1.15 | -6.61 |
| S12 | 0.933 | 0.727 | 0.25 | -4.48 |
| f02 | 0.939 | 0.758 | 1.62 | -6.40 |
| S06 | 0.933 | 0.761 | 2.15 | -4.85 |
| S10 | 0.973 | 0.808 | 2.70 | -6.01 |
| S08 | 0.971 | 0.829 | 0.65 | -5.74 |
| S07 | 0.976 | 0.839 | 3.45 | -4.57 |
| S14 | 0.983 | 0.886 | 4.43 | -7.49 |

Stress false alarms by protocol: STRESS 92.6%, AEROBIC 5.3%, ANAEROBIC 2.1%.

- Mostly a threshold problem: 3 of the 5 low responders (S15 0.81, f18 0.81, f11 0.96) rank their own Stress windows well above their Rest windows, but the whole recording sits below the global threshold (median Stress-window score -1.4 to -1.9 for S15 and f18). S17 (0.51) is invisible and f17 (0.72) is weak. The rule for Step 2 (at least half with AUC >= 0.75) is met.
- Almost all Stress false alarms (93%) are inside STRESS sessions, so session gating (Step 3) can remove at most about 7% of them; its value would have to come from better Rest/Activity and Stress/Rest specialists.

### Step 2: per-recording adaptive threshold (2a) and ranking score (2b)

Out-of-fold XGBoost probabilities of the 9 runs (2a needs no training). Per outer fold: global offsets tuned on the other folds, then the rule parameters tuned on the other folds' recordings and applied to the fold. Rules act on the non-Activity seconds of every recording (the global decision for Activity is kept):

- gmm: two-component Gaussian mixture (numpy EM) of the Stress score within the recording. If separation (mu1 - mu0) / sqrt((v0 + v1) / 2) >= D, upper weight >= Wmin and upper mean >= Mmin, the upper component becomes Stress and the rest Rest; otherwise the global decision stays. Grid D 0.5 to 4, Wmin 0.05 to 0.3, Mmin none, -4, -2, 0.
- z: robust z-score of the Stress score within the recording (median, IQR); Stress where z > tau (tau 0 to 4).
- zfloor: as z, but the raw score must also exceed a floor (-6 to 0).

2a uses the Stress log-odds against Rest. 2b replaces it by an XGBoost `rank:pairwise` model trained inside each fold (groups = recording, Stress vs Rest seconds of STRESS training recordings, every 4th second, 200 trees, zoo depth / child weight / colsample).

| Step | Rule | Macro-F1 | Repeat 0 / 1 / 2 | Activity | Rest | Stress | S15 | S17 | f11 | f17 | f18 | S01 | f13 | Decision |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ref | global offsets | 0.7652 +/- 0.040 | 0.7690 / 0.7415 / 0.7852 | 0.905 | 0.807 | 0.582 | 0.23 | 0.21 | 0.50 | 0.05 | 0.20 | 0.40 | 0.25 | |
| 2a | gmm | 0.7632 +/- 0.039 | 0.7677 / 0.7390 / 0.7830 | 0.905 | 0.803 | 0.580 | 0.23 | 0.28 | 0.49 | 0.05 | 0.20 | 0.41 | 0.25 | drop |
| 2a | z | 0.7473 +/- 0.034 | 0.7454 / 0.7360 / 0.7604 | 0.905 | 0.803 | 0.536 | 0.33 | 0.15 | 0.49 | 0.09 | 0.44 | 0.38 | 0.27 | drop |
| 2a | zfloor | 0.7649 +/- 0.037 | 0.7684 / 0.7401 / 0.7862 | 0.905 | 0.809 | 0.582 | 0.23 | 0.21 | 0.52 | 0.05 | 0.23 | 0.41 | 0.26 | drop (-0.0003) |
| 2b | rank + gmm | 0.7612 +/- 0.040 | 0.7650 / 0.7382 / 0.7803 | 0.905 | 0.805 | 0.574 | 0.23 | 0.21 | 0.50 | 0.05 | 0.20 | 0.40 | 0.25 | drop |
| 2b | rank + z | 0.7351 +/- 0.032 | 0.7338 / 0.7279 / 0.7437 | 0.905 | 0.807 | 0.497 | 0.05 | 0.07 | 0.54 | 0.04 | 0.37 | 0.26 | 0.21 | drop |
| 2b | rank + zfloor | 0.7389 +/- 0.031 | 0.7391 / 0.7316 / 0.7460 | 0.905 | 0.810 | 0.504 | 0.06 | 0.06 | 0.50 | 0.05 | 0.35 | 0.24 | 0.14 | drop |

- The mixture rule tunes itself off: the most frequent choice is D = 4 (never reached), so it collapses to the global offsets. Within a recording the Stress score is not bimodal enough to split reliably; Stress tasks are short (30 to 200 s) against long Rest, and the score drifts.
- The z rule lifts S15 (0.23 to 0.33) and f18 (0.20 to 0.44), confirming the threshold diagnosis for them, but it imposes a Stress share on every recording, including exercise sessions and strong responders, and costs 0.018 overall. The floor removes the damage and the gain together.
- The ranking model is worse than the 3-class log-odds in every rule: trained only on the 24 to 25 STRESS training recordings per fold, it sees less data, and its scores are not comparable across recordings, so floors and global fallbacks do not transfer.
- Step 2 kept nothing. A per-recording threshold only helps if it is applied to the right recordings, which links to the session classifier of Step 3.

### Step 3: session classifier and session specialists

Per repeat and outer fold, fitted on the training recordings only (`training/specialists.py`):
- Session classifier: logistic regression on five whole-recording summaries (log ratio of HR p95 to p5, HR upper vs lower spread, missed-beat rate, median pulse motion, pulse motion burstiness). It does not use the 3-class model. Accuracy (stress vs exercise session, out of fold): 0.882, 0.906, 0.882 on the three repeats.
- Specialists (XGBoost, zoo in-fold parameters, 3 seeds): Stress vs Rest on STRESS training recordings, Activity vs Rest on AEROBIC and ANAEROBIC ones.
- Soft gating: p = p(stress session) x [p_Rest, 0, p_Stress] from the Stress specialist + p(exercise session) x [p_Rest, p_Activity, 0] from the exercise specialist, then the nested per-class offsets.
- Gated z: the Step 2 zfloor rule applied only to recordings the classifier calls stress sessions. (The gate of a training recording comes from its own out-of-fold classifier; the classifier is near-deterministic, so the effect on tuning is negligible.)

| Variant | Macro-F1 | Repeat 0 / 1 / 2 | Activity | Rest | Stress | Rest F1 in stress / exercise sessions | S15 | S17 | f11 | f17 | f18 | S01 | f13 | Decision |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Single 3-class model (reference) | 0.7652 +/- 0.040 | 0.7690 / 0.7415 / 0.7852 | 0.905 | 0.807 | 0.582 | 0.880 / 0.551 | 0.23 | 0.21 | 0.50 | 0.05 | 0.20 | 0.40 | 0.25 | |
| Gated specialists | 0.7045 +/- 0.049 | 0.6954 / 0.7175 / 0.7006 | 0.846 | 0.770 | 0.511 | 0.855 / 0.508 | 0.32 | 0.18 | 0.81 | 0.29 | 0.19 | 0.10 | 0.01 | drop (-0.061) |
| Single + gated z | 0.7656 +/- 0.037 | 0.7680 / 0.7414 / 0.7872 | 0.905 | 0.809 | 0.584 | | 0.25 | 0.21 | 0.54 | 0.05 | 0.23 | 0.40 | 0.27 | drop (+0.0004, 1/3) |
| Gated specialists + gated z | 0.7041 +/- 0.048 | 0.6987 / 0.7075 / 0.7062 | 0.846 | 0.765 | 0.517 | | 0.29 | 0.14 | 0.82 | 0.28 | 0.27 | 0.10 | 0.04 | drop |

- The session classifier misroutes about 1 recording in 9. A misrouted stress session can no longer produce Stress, and a misrouted exercise session can no longer produce Activity, so whole recordings are lost. This outweighs any specialist gain.
- The Stress specialist does help some low responders (f11 0.50 to 0.81, f17 0.05 to 0.29, S15 0.23 to 0.32), but loses S01 and f13 almost completely. Rest F1 is lower in both session types.
- Rest inside exercise sessions (F1 0.55) is the weakest cell of the single model; the exercise specialist does not fix it (0.51).
- The single 3-class model already infers the session from its 180 s context features (93% of Stress false alarms are inside STRESS sessions, Step 1), so hard session structure adds little and costs a lot when the session guess is wrong.

### Step 4: combination

Nothing from Steps 2 and 3 passed the keep rule. The remaining near-miss not yet measured, averaging the class probabilities of the 3 seeds, was scored on the cached out-of-fold predictions (nested offsets, same folds):

| Model | Single seed (mean of 9 runs) | 3-seed average | Repeat 0 / 1 / 2 (3-seed average) | Activity | Rest | Stress |
|---|---|---|---|---:|---:|---:|
| XGBoost | 0.7652 | 0.7676 | 0.7666 / 0.7472 / 0.7890 | 0.906 | 0.811 | 0.587 |
| Submission F (LightGBM) | 0.7706 | 0.7722 | 0.7732 / 0.7616 / 0.7818 | 0.908 | 0.814 | 0.598 |
| Submission H (F + calm baseline) | 0.7754 | 0.7781 | 0.7787 / 0.7695 / 0.7861 | 0.911 | 0.818 | 0.613 |

- 3-seed averaging adds +0.0016 to +0.0027 on every model. On top of submission H it gives +0.0027 (repeats -0.0025, +0.0044, +0.0063), under the +0.005 rule, and it would triple the ONNX inference time (about +16 ms per call on this container, which would break the 50 ms budget there).
- No combination beats the current best submission (H) by the required margin, so no new submission was built. Submission H remains the recommended model.
