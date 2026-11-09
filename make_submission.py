"""Build and validate a submission zip.

Usage:
    python make_submission.py --model artifacts/model_B.onnx --name submission_B

Steps:
  1. Copy submission/adm_model.py and the ONNX model (same basename) to build/<name>/.
  2. Check the .py imports only allowed packages (plus the standard library).
  3. Zip and check the size is at most 50 MB.
  4. Unzip into a temporary folder and, in a fresh Python process, load the
     model through get_model(), run it on a recording, validate the output and
     time 50 run() calls.
  5. Print the expected local score from the cross-validation summary.
"""

import argparse
import ast
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent
ALLOWED = {"numpy", "pandas", "pyarrow", "scipy", "sklearn", "onnxruntime", "adm_base_class"}
MAX_BYTES = 50 * 1024 * 1024

CHECK_SCRIPT = r"""
import os, sys, time, importlib.util, pathlib
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd
sub, kit, rec = map(pathlib.Path, sys.argv[1:4])
sys.path.insert(0, str(kit))
py = next(sub.glob("*.py"))
spec = importlib.util.spec_from_file_location(py.stem, py)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
import adm_base_class
model = mod.get_model()
assert isinstance(model, adm_base_class.ADM_Base), "model does not subclass ADM_Base"
df = pd.read_parquet(rec)[["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]]
out = model.run(df)
assert isinstance(out, pd.DataFrame), "output is not a DataFrame"
assert list(out.columns) == ["Activity"], f"bad columns {list(out.columns)}"
assert out.index.equals(df.index), "index differs from input"
vals = set(out["Activity"].unique())
assert vals <= {"Stress", "Activity", "Rest"}, f"bad labels {vals}"
assert out["Activity"].notna().all(), "missing labels"
ref = sys.argv[4] if len(sys.argv) > 4 else ""
if ref:
    r = np.load(ref)
    assert str(r["recording"]).replace("\\", "/").endswith(rec.as_posix().split("data/")[-1]), "reference recording differs"
    names = model.cfg.get("features")
    _, got_names, _ = mod.compute_features(df, model.cfg)
    assert names is None or list(got_names) == list(names), "feature names differ from training"
    _, proba = model.predict_seconds(df)
    d = float(np.abs(proba - r["proba"]).max())
    assert d < 2e-3, f"run() probabilities differ from training path: max |dp| = {d:.2e}"
    print(f"parity OK: run() vs training path max |dp| = {d:.1e}")
short = model.run(df.iloc[:200])
assert short.index.equals(df.index[:200])
ts = []
for _ in range(50):
    t = time.perf_counter(); model.run(df); ts.append(time.perf_counter() - t)
met = sum(ts) / len(ts)
print(f"OK rows={len(df)} labels={sorted(vals)} MET={met*1000:.1f}ms speed={max(0, 1 - met/0.3)*20:.2f}/20")
"""


def check_imports(py):
    import sys as _s

    tree = ast.parse(py.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            continue
        for m in mods:
            top = m.split(".")[0]
            if top not in ALLOWED and top not in _s.stdlib_module_names:
                bad.append(m)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--py", default=str(ROOT / "submission" / "adm_model.py"))
    ap.add_argument("--recording", default=str(ROOT / "data" / "STRESS" / "f01.parquet"))
    ap.add_argument("--cv", default=None, help="cv summary json for the expected score")
    ap.add_argument("--ref", default=None, help="reference probabilities saved by train.py --export")
    a = ap.parse_args()

    py, onx = pathlib.Path(a.py), pathlib.Path(a.model)
    build = ROOT / "build" / a.name
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    shutil.copy(py, build / py.name)
    shutil.copy(onx, build / (py.stem + ".onnx"))

    bad = check_imports(build / py.name)
    assert not bad, f"disallowed imports: {bad}"
    print("imports OK")

    zpath = ROOT / f"{a.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(build.iterdir()):
            z.write(f, f.name)
    size = zpath.stat().st_size
    assert size <= MAX_BYTES, f"zip too large: {size} bytes"
    print(f"{zpath.name}: {size/1024:.0f} KB, files {zipfile.ZipFile(zpath).namelist()}")

    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zpath).extractall(tmp)
        r = subprocess.run([sys.executable, "-c", CHECK_SCRIPT, tmp, str(ROOT), a.recording] + ([str(pathlib.Path(a.ref).resolve())] if a.ref else []),
                           capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr)
            sys.exit("fresh-process check FAILED")

    cv = pathlib.Path(a.cv) if a.cv else None
    if cv and cv.exists():
        s = json.loads(cv.read_text())
        f1 = s["full"][0]
        print(f"expected local quality: macroF1 {f1:.4f} +- {s['full'][1]:.4f} -> {f1*80:.1f} / 80")


if __name__ == "__main__":
    main()
