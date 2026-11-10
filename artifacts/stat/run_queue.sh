#!/usr/bin/env bash
# Runs the statistical-extension experiments strictly one after another.
# Never run two at once: two XGBoost processes with 4 OpenMP threads each on
# 4 cores slow down by about 20x.
cd "$(dirname "$0")/../.."
run() { n=$1; shift; [ -f "artifacts/stat/$n.json" ] && { echo "skip $n"; return; }
  OMP_NUM_THREADS=4 python3 -u training/stat_ext.py --name "$n" "$@" > "artifacts/stat/logs/$n.log" 2>&1; echo "finished $n $(date +%H:%M)"; }
run calm --calm
run shrink --shrink
run calm_shrink --calm --shrink
run vkf --vkf
run qda --qda
run lgbF --model lgbF
echo ALL DONE
