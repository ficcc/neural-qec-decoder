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
    N=${WORKERS:-3}   # several small-model workers keep one GPU busy
    for i in $(seq 0 $((N-1))); do
      OMP_NUM_THREADS=8 nohup python -m experiments.train_jobs --min-d 5 --device cuda --shard $i/$N > train_$i.out 2>&1 &
    done
    echo "started $N workers; tail -f train_*.out" ;;
  status)
    python -m experiments.train_jobs --min-d 5 --list | head -1; cat checkpoints/progress_*.log 2>/dev/null | wc -l; tail -qn1 checkpoints/progress_*.log 2>/dev/null || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader ;;
esac
