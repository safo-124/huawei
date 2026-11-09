"""Time 50 run() calls on a full recording with a single thread.

Usage:
    python benchmark.py [submission_dir] [recording.parquet]

Environment variables limiting threads are set before numpy / onnxruntime load.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
INPUT_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


def load_submission(sub_dir):
    sub_dir = pathlib.Path(sub_dir)
    py = [p for p in sub_dir.glob("*.py")]
    assert len(py) == 1, f"expected one .py in {sub_dir}, found {py}"
    sys.path.insert(0, str(ROOT))  # official adm_base_class.py
    spec = importlib.util.spec_from_file_location(py[0].stem, py[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_model()


def bench(model, df, n=50):
    model.run(df)  # warm-up
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        model.run(df)
        ts.append(time.perf_counter() - t)
    return sum(ts) / n, max(ts)


def main():
    sub_dir = sys.argv[1] if len(sys.argv) > 1 else ROOT / "build" / "submission_B"
    rec = sys.argv[2] if len(sys.argv) > 2 else ROOT / "data" / "STRESS" / "f01.parquet"
    df = pd.read_parquet(rec)[INPUT_COLS]
    model = load_submission(sub_dir)
    met, worst = bench(model, df)
    print(f"{pathlib.Path(rec).parent.name}/{pathlib.Path(rec).name}: {len(df)} rows, "
          f"MET {met*1000:.1f} ms (max {worst*1000:.1f} ms), "
          f"speed score {max(0.0, 1 - met / 0.3) * 20:.2f} / 20")


if __name__ == "__main__":
    main()
