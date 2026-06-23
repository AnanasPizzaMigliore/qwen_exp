"""
Merges LoRA adapter weights into the base model for fast inference.
Run this after fine-tuning completes.

Usage:
  python merge_lora.py
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

ADAPTER_PATH = "/home/penghao/qwen/finetuned/final"
MERGED_PATH  = "/home/penghao/qwen/finetuned/final_merged"
BASE_PATH    = "/home/penghao/qwen/Qwen/Qwen3.5-0.8B"

print("Loading base model...")
model = AutoModelForImageTextToText.from_pretrained(
    BASE_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging adapters into base weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_PATH} ...")
model.save_pretrained(MERGED_PATH)

print("Saving processor...")
processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
processor.save_pretrained(MERGED_PATH)

print(f"Done. Use '{MERGED_PATH}' for inference.")
