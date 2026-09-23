#!/usr/bin/env bash
# S1 — encoder/merger decomposition. Decoder held at F16 throughout, CPU backend
# so the rows slot into the same table as the existing f16 / mmQ3 / mmQ4 results.
set -u

G=/home/penghao/qwen/gguf
R=/home/penghao/qwen/gguf_results
LLM=$G/qwen35_groupA_a3_f16.gguf

run () {  # $1 = mmproj file, $2 = output tag
  echo "=== $2 :: $(date +%H:%M:%S) ==="
  conda run -n llm python /home/penghao/qwen/eval_gguf.py \
    --model   "$LLM" \
    --mmproj  "$G/$1" \
    --backend cpu \
    --output  "$R/qwen35_groupA_a3_f16_${2}_cpu.json"
}

run qwen35_groupA_a3_mmproj_encQ3_mrgF16.gguf mmEncQ3MrgF16
run qwen35_groupA_a3_mmproj_encF16_mrgQ3.gguf mmEncF16MrgQ3

echo "=== S1 done :: $(date +%H:%M:%S) ==="
