#!/usr/bin/env bash
# Zero-shot base Qwen3.5-0.8B in the GGUF pipeline, so Table 3's zero-shot row is measured the same
# way as the rest of Table 3 (llama-server, full-resolution images, eval_gguf.py scoring).
set -u
G=/home/penghao/qwen/gguf; R=/home/penghao/qwen/gguf_results
cd /home/penghao/qwen
for B in cpu vulkan; do
  python eval_gguf.py --model $G/Qwen3.5-0.8B-f16.gguf --mmproj $G/Qwen3.5-0.8B-mmproj-f16.gguf \
    --backend $B --port 8766 --output $R/qwen35_base_f16_${B}.json
done
