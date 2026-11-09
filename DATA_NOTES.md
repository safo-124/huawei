# Data notes (official starting kit)

All numbers below were measured on the official kit only. Durations are
estimated as rows / 64 (rows sit on the 64 Hz BVP clock).

## Layout

| Folder | Files | Participants |
|---|---|---|
| `data/STRESS` | 31 | S01, S03, S05 to S18, f01 to f06, f09 to f13, f15 to f18 |
| `data/AEROBIC` | 27 | S01 to S10, S14 to S18, f01 to f05, f07 to f13 |
| `data/ANAEROBIC` | 27 | S02 to S14, S17, S18, f01 to f09, f11 to f13 |

- 85 Parquet files, 36 distinct `user_id`s. Most people appear in two or three protocols, so every split must be grouped by `user_id`.
- Index `t`: `datetime64[ns]`, monotonic, no duplicates, every timestamp on the 1/64 s grid.
- Columns: `BVP, EDA, HR, IBI_dt, IBI_duration, Phase, Temperature, user_id`.
  - BVP on almost every row (64 Hz), EDA and Temperature on about 6.25% (4 Hz), HR on about 1.56% (1 Hz), IBI on 0 to 2% (one row per detected beat).
  - `Phase` has no NaNs. `user_id` is constant per file.
- Model inputs are only `BVP, EDA, HR, IBI_dt, IBI_duration, Temperature`. `Phase` gives labels and `user_id` gives CV groups. `subject-info.csv` is never used as input.
- Recording lengths: STRESS 22 to 52 min, AEROBIC 24 to 41 min, ANAEROBIC 14 to 37 min. The longest file (STRESS/f01) has 201,305 rows.

## Labels (official mapping)

| Protocol | Stress | Activity | Rest | Not scored |
|---|---:|---:|---:|---:|
| STRESS | 191.0 | 0 | 773.6 | 87.9 |
| AEROBIC | 0 | 795.1 | 99.7 | 27.1 |
| ANAEROBIC | 0 | 476.8 | 117.4 | 67.7 |
| **Total (min)** | **191.0 (8%)** | **1271.9 (52%)** | **990.7 (40%)** | 182.7 |

Your measured facts are confirmed: Activity about 1273 min, Rest about 990, Stress about 191; 774 Rest minutes come from STRESS sessions (Rest 677.4 plus Baseline 96.2); ANAEROBIC Cooldown is 344.0 min.

Stress phase lengths (seconds per segment, mean and range):

| Phase | Segments | Mean | Min to max |
|---|---:|---:|---:|
| TMCT | 31 | 208 | 125 to 452 |
| Stroop | 16 | 111 | 92 to 152 |
| Real opinion | 31 | 35 | 28 to 45 |
| Opposite opinion | 31 | 34 | 27 to 48 |
| Subtraction | 31 | 36 | 26 to 65 |

Typical sequences:
- STRESS (V2): Baseline, Undefined, TMCT, Rest, Real opinion, Undefined, Opposite opinion, Rest, Subtraction, Rest. V1 sessions also include Stroop.
- AEROBIC: Baseline, Warm-up, rpm stages (60 to 110 rpm), Cooldown, Rest.
- ANAEROBIC: Baseline, Warm-up, 4 times (Sprint about 30 s, Cooldown about 200 s), Rest, Undefined.

Structural point: a STRESS session never contains Activity and an exercise session never contains Stress. Long-context features (180 s) carry much of this information, which is why scores drop on short chunks (see RESULTS.md).

## Issues found and how the code handles them

| Issue | Where | Handling |
|---|---|---|
| Corrupted temperature | AEROBIC/S07 (mean 212 C), AEROBIC/S09 (mean 48, max 235), ANAEROBIC/S02 (mean 201), ANAEROBIC/S05 (mean 143) | Values outside 20 to 42 C treated as missing |
| HR out of range | none observed in the kit (49 to 197 bpm) | 30 to 220 bpm check kept for unseen data |
| IBI_duration | 0.31 to 1.64 s observed | 0.25 to 2.0 s check |
| EDA | minimum 0.0 (sensor floor) | negative values dropped |
| Timestamp gaps | gaps over 1 s in 20 files, largest 270 s (AEROBIC/f02), 269 s (ANAEROBIC/f04), 191 s (AEROBIC/f11), 33 s (AEROBIC/f12, which also has 25 sub-second gaps) | 1 Hz grid built from timestamps, never row counts; seconds without samples get count 0 and features become NaN |
| No IBI at all | AEROBIC/f04 | IBI features NaN, handled by the trees |
| Missed beats | `IBI_dt - IBI_duration` up to 1392 s | Beat counted as "missed" when the difference exceeds 0.05 s; used for dropout rate and gap mean; RMSSD only uses adjacent beats |
| Incomplete protocols | AEROBIC S03, S07; ANAEROBIC S06 | Kept; still labelled |

## Probable hidden test recordings

Comparing the kit with `subject-info.csv`, these protocol and participant pairs exist in the study but are missing from the kit, so the hidden set may well include them: STRESS S02, S04, f07, f08, f14; AEROBIC S11, S13, f06; ANAEROBIC S01, S15, S16, f10. This is a guess, used only to motivate grouped CV. No external copy of the data is used.
