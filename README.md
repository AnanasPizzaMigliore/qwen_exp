# Expiration Date Recognition — Fine-tuning Code

Code for fine-tuning and evaluating Qwen3.5-0.8B on expiration date recognition from food packaging images.

## Dataset

The image dataset will be released upon paper acceptance.

## Repository Structure

### Training
| File | Description |
|------|-------------|
| `finetune_qwen35.py` | Baseline fine-tuning (LLM LoRA only) |
| `finetune_groupA.py` | Adaptation ablation. `--strategy`: `a0` LLM LoRA, `a1` + vision-encoder LoRA, `a2` + merger, `a3` + both, `a4` vision-encoder LoRA only (LLM frozen), `a0p`/`a1p` A0/A1 re-run at A3's parameter budget |
| `run_s2_s3.sh` | Runs the matched-budget and encoder-only arms (`a0p`, `a1p`, `a4`) |
| `prepare_finetune_data.py` | Build fine-tuning JSONL from annotated dataset |
| `merge_lora.py`, `merge_groupA.py` | Merge LoRA adapters into the base model |

### Quantization and deployment
| File | Description |
|------|-------------|
| `convert_to_gguf.py` | Convert a fine-tuned model to GGUF (F16 + quantized) |
| `merge_gguf.py` | Merge LLM GGUF + mmproj GGUF into a single file (for apps that accept one file) |
| `run_s1.sh` | Vision-tower quantization: builds the projector sweep (Q8_0–Q3_K_M) and the encoder/merger decomposition (only `v.*` or only `mm.*` at Q3), then evaluates the decomposition arms. Its commands reproduce the paper's projector files byte-for-byte |
| `eval_products_real.py` | Cross-dataset evaluation on the Products-Real evaluation split, same prompt and scoring |
| `patches/` | Patch llama.cpp `d05fe1d` needs to convert Qwen3.5-0.8B (registers its tokenizer) |
| `phone_bench/` | On-device benchmark over adb (see below) |

### Evaluation and analysis
| File | Description |
|------|-------------|
| `qwen35_test_eval.py` | Zero-shot evaluation of base Qwen3.5-0.8B |
| `eval_native.py` | Evaluate a Hugging Face model (full precision) |
| `eval_gguf.py` | Evaluate a GGUF model via llama-server (CPU or Vulkan) |
| `eval_token_ablation.py` | Accuracy and latency vs. image-token budget |
| `eval_tome.py` | Token merging (ToMe) on visual tokens, native pipeline |
| `compare_all.py` | Aggregate accuracy across models and backends |
| `bootstrap_ci.py` | Paired-bootstrap 95% CIs for the quantization comparisons |
| `error_analysis.py` | Adaptation-ablation CIs, false-abstention rate, later-than-truth error rate |
| `tag_mechanism.py` | Per-condition accuracy, F16 vs. quantized vision tower |
| `tier_reconcile.py` | Per-tier (hard / non-hard) accuracy for every adaptation arm |

All statistics use seed 42 and 1,000 bootstrap resamples.

## Software versions

| Component | Version |
|-----------|---------|
| Python | 3.13.13 |
| PyTorch | 2.13.0.dev20260510+cu132 (nightly) |
| Transformers | 5.8.0 |
| PEFT | 0.19.1 |
| Accelerate | 1.13.0 |
| CUDA / cuDNN | 13.2 / 9.20 |
| llama.cpp | commit `d05fe1d` (build 9010) |
| Base model | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B), Apache-2.0 |

## Usage

### Fine-tune
```bash
python finetune_groupA.py --strategy a3
```

### Convert to GGUF
At llama.cpp `d05fe1d`, `convert_hf_to_gguf.py` does not recognise the Qwen3.5-0.8B tokenizer. Apply the patch first:
```bash
git -C /path/to/llama.cpp apply /path/to/this/repo/patches/llama.cpp-convert-qwen35-0.8b.patch
python convert_to_gguf.py
```

### Evaluate GGUF
```bash
python eval_gguf.py \
  --model /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --backend vulkan \
  --output results.json
```

**Note on `--backend cpu`:** it keeps the LLM layers on the CPU (`--n-gpu-layers 0`), but llama.cpp's
defaults still run the vision encoder and large prompt batches on a GPU if one is present
(`--mmproj-offload` and `--op-offload` both default to on). For CPU-only timing, add
`--no-mmproj-offload --no-op-offload` to the llama-server command.

### Merge LLM + mmproj into a single GGUF
```bash
python merge_gguf.py \
  --llm /path/to/llm.gguf \
  --mmproj /path/to/mmproj.gguf \
  --output /path/to/unified.gguf
```

### Statistics
```bash
python bootstrap_ci.py       # quantization comparisons
python error_analysis.py     # adaptation ablation + error analysis
python tag_mechanism.py      # per-condition analysis
```

### On-device benchmark (Android, adb)
`phone_bench/build_android.sh` cross-compiles llama.cpp for Android arm64 with the NDK (CPU builds with and
without `i8mm`, plus a Vulkan build). Then either:
```bash
python3 phone_bench/bench_phone.py --info     # device report
python3 phone_bench/bench_phone.py            # full run: configs x token budgets, per-stage timings
```
or push the files and run single images by hand with `phone_bench/run_on_phone.sh` inside `adb shell`.
The harness detects the Android emulator and labels its output as a functional test only: an emulator
measures the host machine, not a phone.

## Results

See `compare_all.py` for full accuracy tables across quantization levels and backends.
Highest observed GGUF accuracy: A3 strategy, F16 mmproj + Q4_K_M LLM — **75.8% full-date accuracy** on the 545-image test set. This is not significantly different from full precision (75.2%, p = 0.218); the paper reports every configuration rather than selecting one.

## Citation

> Paper under review. Citation will be added upon acceptance.
