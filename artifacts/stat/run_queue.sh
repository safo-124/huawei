#!/usr/bin/env bash
# Runs the statistical-extension experiments one after another (4 threads each).
cd "$(dirname "$0")/../.."
run() { n=$1; shift; OMP_NUM_THREADS=4 python3 -u training/stat_ext.py --name "$n" "$@" > "artifacts/stat/logs/$n.log" 2>&1; echo "finished $n $(date +%H:%M)"; }
run base --cost
run shrink --shrink
run calm --calm
run calm_shrink --calm --shrink
run vkf --vkf
run qda --qda
run lgbF --model lgbF
echo ALL DONE
