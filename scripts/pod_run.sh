#!/usr/bin/env bash
# Run on a RunPod PyTorch pod after the repo bundle has been copied to /workspace.
#   bash scripts/pod_run.sh timing   # ~2 min: d=7 100-step timing run, prints s/step
#   bash scripts/pod_run.sh train    # all d>=5 jobs in the background, resumable
#   bash scripts/pod_run.sh status   # progress so far
set -euo pipefail
cd /workspace/neural-qec-decoder
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
    nohup python -m experiments.train_jobs --min-d 5 --device cuda > train.out 2>&1 &
    echo "started; tail -f train.out" ;;
  status)
    python -m experiments.train_jobs --min-d 5 --list | head -1; tail -3 checkpoints/progress.log 2>/dev/null || true ;;
esac
