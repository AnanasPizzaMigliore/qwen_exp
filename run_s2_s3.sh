#!/usr/bin/env bash
# S3 (matched budgets) + S2 (encoder-only arm).
#
#   a0p  LLM LoRA r=100                 ~40.0M  == A3's budget, LLM-only
#   a1p  LLM LoRA r=95 + ViT LoRA r=32  ~39.8M  == A3's budget, LLM+encoder
#   a4   ViT LoRA r=32, LLM frozen       ~1.8M  encoder isolated (R3)
#
# A3 itself is ~39.9M, so a0p/a1p answer "is A3 just spending more?".
# a4 runs last because it is the novel code path (frozen decoder).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for S in a0p a1p a4; do
  echo "=== $S :: start $(date +%F\ %H:%M:%S) ==="
  conda run -n llm python /home/penghao/qwen/finetune_groupA.py --strategy "$S"
  echo "=== $S :: end   $(date +%F\ %H:%M:%S)  exit=$? ==="
done

echo "=== S2+S3 complete :: $(date +%F\ %H:%M:%S) ==="
