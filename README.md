# Expiration Date Recognition — Fine-tuning Code

Code for fine-tuning and evaluating Qwen3.5-0.8B on expiration date recognition from food packaging images.

## Dataset

The image dataset will be released upon paper acceptance.

## Repository Structure

| File | Description |
|------|-------------|
| `finetune_qwen35.py` | Baseline fine-tuning (LLM only, A0 strategy) |
| `finetune_groupA.py` | Group A curriculum fine-tuning (A1–A3 strategies) |
| `prepare_finetune_data.py` | Build fine-tuning JSONL from annotated dataset |
| `merge_lora.py` | Merge LoRA adapter into base model |
| `merge_groupA.py` | Merge Group A LoRA adapter |
| `eval_native.py` | Evaluate HuggingFace model (full precision) |
| `eval_gguf.py` | Evaluate GGUF quantized model via llama-server |
| `convert_to_gguf.py` | Convert fine-tuned model to GGUF (f16 + quantized) |
| `merge_gguf.py` | Merge LLM GGUF + mmproj GGUF into a single unified file |
| `compare_all.py` | Aggregate and compare accuracy across all models/backends |
| `bootstrap_ci.py` | Bootstrap 95% confidence intervals on accuracy |
| `qwen35_test_eval.py` | Zero-shot evaluation of base Qwen3.5-0.8B |

## Requirements

```bash
conda create -n llm python=3.13
conda activate llm
pip install torch transformers peft accelerate pillow numpy
# For GGUF evaluation:
# Build llama.cpp with Vulkan support — see https://github.com/ggerganov/llama.cpp
```

## Usage

### Fine-tune (A3 strategy)
```bash
python finetune_groupA.py
```

### Convert to GGUF
```bash
python convert_to_gguf.py
```

### Evaluate GGUF (Vulkan backend)
```bash
python eval_gguf.py \
  --model /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --backend vulkan \
  --output results.json
```

### Merge LLM + mmproj into single GGUF (for mobile deployment)
```bash
python merge_gguf.py \
  --llm /path/to/llm.gguf \
  --mmproj /path/to/mmproj.gguf \
  --output /path/to/unified.gguf
```

### Compute bootstrap CIs
```bash
python bootstrap_ci.py
```

## Results

See `compare_all.py` for full accuracy tables across quantization levels and backends.
Best configuration: A3 strategy, F16 mmproj + Q4_K_M LLM — **75.8% full-date accuracy** on the 545-image test set.

## Citation

> Paper under review. Citation will be added upon acceptance.
