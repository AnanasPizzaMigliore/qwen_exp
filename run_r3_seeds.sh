#!/usr/bin/env bash
# R3: two further seeds for the matched-budget pair behind the +8.1 pp result (A0' vs A1'),
# with no test-split evaluation during training (scored once after the final epoch).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/penghao/qwen
for SEED in 1 2; do
  for S in a0p a1p; do
    echo "=== $S seed $SEED :: start $(date +%F\ %H:%M:%S) ==="
    conda run --no-capture-output -n llm python finetune_groupA.py --strategy $S --seed $SEED --no-test-eval
    echo "=== $S seed $SEED :: end   $(date +%F\ %H:%M:%S)  exit=$? ==="
  done
done
echo "=== R3 complete :: $(date +%F\ %H:%M:%S) ==="
