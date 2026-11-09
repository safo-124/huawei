HUAWEI FINLAND TECH ARENA 2026: STARTING KIT

Classify a person's state as Rest, Stress or Activity from wrist-worn BVP, IBI, EDA and skin-temperature signals. The task, the model interface, an example submission and the scoring are described on the topic description page:
https://huawei.agorize.com/en/challenges/2026-finland-tech-arena/pages/topic-description?lang=en

FILES
- data/ contains one folder per protocol: STRESS, AEROBIC and ANAEROBIC. Each includes one Parquet file per participant, named S01.parquet, S02.parquet, f01.parquet, etc.
- phases_to_activity.txt maps the protocol phases in the Phase column to the three target states (Stress, Activity, Rest). Phases not listed, such as Undefined, are not scored.
- adm_base_class.py defines the ADM_Base interface that submitted models must implement.
- Data_Dictionary.csv describes the sensor signals, their sampling frequencies and units.
- subject-info.csv contains demographic data such as age, weight and height for all participants.
- data_constraints.txt describes issues such as incorrect wristband placements, incomplete protocols and connection problems.
- requirements.txt lists the package versions of the evaluation environment (Python 3.13).
- submission_guide.pdf explains how to submit on Agorize and read your results.
- LICENSE.txt is the license of the dataset.

Participants from the first stage of data collection are labeled Sxx, and those from the second stage fxx.
