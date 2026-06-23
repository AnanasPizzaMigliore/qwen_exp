#!/usr/bin/env python3
"""
Merge a Group A LoRA adapter (+ optional merger weights) into the base model
and save the full merged model ready for GGUF conversion.

Usage:
    # A1 (LLM + vision LoRA only)
    conda run -n llm python /home/penghao/qwen/merge_groupA.py --strategy a1

    # A3 (LLM + vision LoRA + merger) — requires merger_weights.safetensors
    conda run -n llm python /home/penghao/qwen/merge_groupA.py --strategy a3
"""
import argparse
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--strategy", required=True, choices=["a0","a1","a2","a3"])
args = ap.parse_args()

BASE_PATH    = "/home/penghao/qwen/Qwen/Qwen3.5-0.8B"
ADAPTER_PATH = f"/home/penghao/qwen/groupA_{args.strategy}"
MERGED_PATH  = f"/home/penghao/qwen/groupA_{args.strategy}_merged"

print(f"Loading base model from {BASE_PATH} ...")
model = AutoModelForImageTextToText.from_pretrained(
    BASE_PATH, torch_dtype=torch.float16, device_map="cpu")

print(f"Loading LoRA adapter from {ADAPTER_PATH} ...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging LoRA into base weights ...")
model = model.merge_and_unload()

# For A2/A3: apply saved merger weights on top
merger_file = Path(ADAPTER_PATH) / "merger_weights.safetensors"
if args.strategy in ("a2", "a3"):
    if not merger_file.exists():
        print(f"WARNING: merger_weights.safetensors not found in {ADAPTER_PATH}")
        print("  The merger was not saved correctly (old bug). Merged model = LoRA only.")
    else:
        import safetensors.torch as st
        merger_state = st.load_file(str(merger_file))
        state_dict = model.state_dict()
        for name, tensor in merger_state.items():
            if name in state_dict:
                state_dict[name] = tensor.to(state_dict[name].dtype)
        model.load_state_dict(state_dict)
        print(f"  Applied {len(merger_state)} merger weight tensors.")

print(f"Saving merged model to {MERGED_PATH} ...")
Path(MERGED_PATH).mkdir(parents=True, exist_ok=True)
model.save_pretrained(MERGED_PATH)

processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
processor.save_pretrained(MERGED_PATH)

print(f"Done. Merged model saved → {MERGED_PATH}")
print(f"Next: run quantize script pointing to {MERGED_PATH}")
