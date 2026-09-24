#!/usr/bin/env bash
# Vision-tower quantization: the projector sweep (Table 3b) and the encoder/merger
# decomposition (Table 2), then evaluation of the two decomposition arms.
#
# Every quantized projector below was re-created with these exact commands at llama.cpp
# commit d05fe1d and compared byte-for-byte with the files used in the paper: all identical.
# Quantization steps are skipped when their output already exists.
set -u

LLAMA_CPP=${LLAMA_CPP:-/home/penghao/llama.cpp}
Q=$LLAMA_CPP/build/bin/llama-quantize
G=/home/penghao/qwen/gguf
R=/home/penghao/qwen/gguf_results
MMF16=$G/qwen35_groupA_a3_mmproj_f16.gguf
LLM=$G/qwen35_groupA_a3_f16.gguf

# ── F16 projector from the merged A3 model ────────────────────────────────────
# convert_hf_to_gguf.py needs patches/llama.cpp-convert-qwen35-0.8b.patch at d05fe1d
# (it registers the Qwen3.5-0.8B tokenizer); the --mmproj export itself does not use it.
[ -f "$MMF16" ] || python "$LLAMA_CPP/convert_hf_to_gguf.py" /home/penghao/qwen/groupA_a3_merged \
  --mmproj --outtype f16 --outfile "$MMF16"

# ── Projector sweep, whole vision tower (Table 3b) ────────────────────────────
for T in Q8_0 Q5_K_M Q4_K_M Q3_K_M; do
  [ -f "$G/qwen35_groupA_a3_mmproj_$T.gguf" ] || \
    "$Q" "$MMF16" "$G/qwen35_groupA_a3_mmproj_$T.gguf" "$T"
done

# ── Encoder/merger decomposition (Table 2) ────────────────────────────────────
# The projector holds 150 encoder tensors (v.*) and 4 merger tensors (mm.*).
# Encoder Q3, merger F16: Q3_K_M recipe with every mm.* tensor forced back to F16.
[ -f "$G/qwen35_groupA_a3_mmproj_encQ3_mrgF16.gguf" ] || \
  "$Q" --tensor-type mm=f16 "$MMF16" "$G/qwen35_groupA_a3_mmproj_encQ3_mrgF16.gguf" q3_k_m 24
# Encoder F16, merger Q3: the mirror. An f16 base type skips the override logic entirely,
# so this also starts from Q3_K_M and forces every v.* tensor back to F16 instead.
[ -f "$G/qwen35_groupA_a3_mmproj_encF16_mrgQ3.gguf" ] || \
  "$Q" --tensor-type v=f16  "$MMF16" "$G/qwen35_groupA_a3_mmproj_encF16_mrgQ3.gguf" q3_k_m 24

# ── Evaluate the decomposition arms ───────────────────────────────────────────
# Decoder held at F16; --backend cpu so the rows match the other Table 3 rows.
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
