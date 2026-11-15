#!/usr/bin/env bash
# Runs format-suite experiments strictly one after another: run.sh name1 "args1" name2 "args2" ...
cd "$(dirname "$0")/../.."
while [ $# -gt 1 ]; do
  n=$1; args=$2; shift 2
  [ -f "artifacts/robust/$n.json" ] && { echo "skip $n"; continue; }
  OMP_NUM_THREADS=4 python3 -u training/eval_formats.py --name "$n" $args > "artifacts/robust/logs/$n.log" 2>&1
  grep -h "DONE\|Error" "artifacts/robust/logs/$n.log" | tail -2
done
