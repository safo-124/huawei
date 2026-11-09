#!/usr/bin/env bash
cd "C:/Users/fcb363/Downloads/huwaei"
run() { OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 .venv/Scripts/python -u training/model_zoo.py "$1" --jobs 4 > "artifacts/zoo/logs/$1.log" 2>&1; echo "finished $1 $(date +%H:%M)"; }
for m in lgbm xgb catboost rf et; do run $m & done; wait
for m in hgb logreg enet qda mlp; do run $m & done; wait
echo ALL DONE
