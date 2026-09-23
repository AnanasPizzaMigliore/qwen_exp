#!/system/bin/sh
# Run one image on the phone and print only the answer and the timing lines.
#
#   sh run_on_phone.sh <image> [budget] [threads] [llm.gguf] [mmproj.gguf] [binary]
#   sh run_on_phone.sh img/0003.jpg 2048 4
#   sh run_on_phone.sh img/0003.jpg 2048 4 qwen35_groupA_a3_Q4_K_M.gguf qwen35_groupA_a3_mmproj_Q4_K_M.gguf ./llama-mtmd-cli-vulkan
cd "$(dirname "$0")" || exit 1
IMG=${1:?usage: sh run_on_phone.sh img/0003.jpg [budget] [threads]}
BUDGET=${2:-2048}
T=${3:-4}
LLM=${4:-qwen35_groupA_a3_Q4_K_M.gguf}
MM=${5:-qwen35_groupA_a3_mmproj_Q4_K_M.gguf}
BIN=${6:-./llama-mtmd-cli}

# The default layer offload is "auto", which can leave layers on the CPU when a phone
# GPU reports memory conservatively. Ask for all of it on the Vulkan build; override
# with NGL=<n> to test partial offload.
case "$BIN" in *vulkan*) NGL=${NGL:-all} ;; esac

# -c 4096 matters: the default is the model's 262k context, which needs ~5 GB of RAM.
"$BIN" ${NGL:+-ngl "$NGL"} -m "$LLM" --mmproj "$MM" --image "$IMG" -p "$(cat prompt.txt)" \
  -n 256 --temp 0 -t "$T" -c 4096 --image-min-tokens 64 --image-max-tokens "$BUDGET" 2>&1 \
  | grep -E 'ggml_vulkan|using device|image slice encoded|n_tokens_batch|prompt eval time| eval time|total time|\{"year"|error|failed'
