#!/usr/bin/env bash
# A3 on the Products-Real evaluation split (item 1a): full precision and whole-model Q4.
set -u
G=/home/penghao/qwen/gguf; R=/home/penghao/qwen/gguf_results/products_real
cd /home/penghao/qwen
python eval_products_real.py --model $G/qwen35_groupA_a3_f16.gguf    --mmproj $G/qwen35_groupA_a3_mmproj_f16.gguf    --backend vulkan --output $R/a3_f16_vulkan.json
python eval_products_real.py --model $G/qwen35_groupA_a3_Q4_K_M.gguf --mmproj $G/qwen35_groupA_a3_mmproj_Q4_K_M.gguf --backend vulkan --output $R/a3_q4q4_vulkan.json
