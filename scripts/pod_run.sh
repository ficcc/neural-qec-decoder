#!/usr/bin/env bash
# Run on a RunPod PyTorch pod after the repo bundle has been copied to /workspace.
#   bash scripts/pod_run.sh timing   # ~2 min: d=7 100-step timing run, prints s/step
#   bash scripts/pod_run.sh train    # all d>=5 jobs in the background, resumable
#   bash scripts/pod_run.sh status   # progress so far
set -euo pipefail
cd /workspace/neural-qec-decoder
# venv that reuses the image's CUDA torch (the system Python is PEP 668 externally managed)
[ -d /workspace/venv ] || python3 -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
if [ ! -f .deps_ok ]; then
  pip install -q stim pymatching sinter numpy matplotlib pytest
  python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"
  python -m pytest -q -m "not slow" && touch .deps_ok
fi
case "${1:-status}" in
  timing)
    python -m experiments.train_jobs --only e1/d7_s0 --min-d 7 --device cuda --steps 100
    python -c "import json;l=json.load(open('checkpoints/_timing/e1/d7_s0/log.json'));print('d=7 s/step:', round(l['wall_time']/100,3))" ;;
  train)
    N=${WORKERS:-2}   # several small-model workers keep one GPU busy
    for i in $(seq 0 $((N-1))); do
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 nohup python -m experiments.train_jobs --min-d ${MIN_D:-5} --device cuda --shard $i/$N > train_$i.out 2>&1 &
    done
    echo "started $N workers; tail -f train_*.out" ;;
  eval)
    # One single-threaded process per evaluation point: E1 27, E2 54, E3 11, E4 12 (104 of 128 cores).
    mkdir -p logs
    for spec in e1_control:27 e2_mismatch:54 e3_generalise:11 e4_budget:12 e6_crosstalk:6; do
      m=${spec%%:*}; n=${spec##*:}
      for i in $(seq 0 $((n-1))); do
        OMP_NUM_THREADS=1 nohup python -m experiments.$m --shard $i/$n > logs/${m}_$i.out 2>&1 &
      done
    done
    for i in $(seq 0 8); do   # E7 fine-tunes a GNN per point, so give it a few threads
      OMP_NUM_THREADS=4 nohup python -m experiments.e7_drift_adapt --shard $i/9 > logs/e7_drift_adapt_$i.out 2>&1 &
    done
    echo "started evaluation shards" ;;
  latency)
    python -m experiments.e5_latency --device cuda
    OMP_NUM_THREADS=8 python -m experiments.e5_latency --device cpu ;;
  evalstatus)
    echo "running: $(ps -eo args | grep -c '^python -m experiments.e[1-4]_')"; grep -l Traceback logs/*.out 2>/dev/null | head ;;
  status)
    python -m experiments.train_jobs --min-d 5 --list | head -1; cat checkpoints/progress_*.log 2>/dev/null | wc -l; tail -qn1 checkpoints/progress_*.log 2>/dev/null || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader ;;
esac
