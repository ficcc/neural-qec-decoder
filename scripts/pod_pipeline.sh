#!/usr/bin/env bash
# Unattended chain so the pod never idles: wait for current training -> train remaining jobs
# (incl. E6) -> all evaluation shards -> latency (on a quiet machine) -> DONE marker.
set -uo pipefail
cd /workspace/neural-qec-decoder
source /workspace/venv/bin/activate
running() { ps -eo args | grep -c "^python -m experiments.$1"; }
while [ "$(running train_jobs)" -gt 0 ]; do sleep 60; done
echo "$(date) training round 1 done" >> pipeline.log
MIN_D=0 WORKERS=2 bash scripts/pod_run.sh train
sleep 30
while [ "$(running train_jobs)" -gt 0 ]; do sleep 60; done
echo "$(date) all training done; pending: $(python -m experiments.train_jobs --list | head -1)" >> pipeline.log
bash scripts/pod_run.sh eval
sleep 30
while [ "$(running e[1-7]_)" -gt 0 ]; do sleep 60; done
echo "$(date) evaluation done; tracebacks: $(grep -l Traceback logs/*.out | wc -l)" >> pipeline.log
bash scripts/pod_run.sh latency >> pipeline.log 2>&1
for m in e1_control e2_mismatch e3_generalise e4_budget e6_crosstalk e7_drift_adapt; do
  python -m experiments.$m --plot > logs/plot_$m.txt 2>&1
done
echo "$(date) DONE" >> pipeline.log
