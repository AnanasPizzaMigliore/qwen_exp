"""
Fine-tunes Qwen3.5-0.8B VLM on the expiration date extraction task
using LoRA via HuggingFace PEFT + TRL SFTTrainer.

Usage:
  python finetune.py              # LoRA fine-tuning (default)
  python finetune.py --full       # Full fine-tuning (uses more VRAM)
  python finetune.py --resume     # Resume from last checkpoint
"""

import sys
import json
import re
from pathlib import Path
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH  = "/home/penghao/qwen/Qwen/Qwen3.5-0.8B"
TRAIN_DATA  = "/home/penghao/qwen/finetune_data/train.jsonl"
VAL_DATA    = "/home/penghao/qwen/finetune_data/val.jsonl"
OUTPUT_DIR  = "/home/penghao/qwen/finetuned"

FULL_FINETUNE = "--full"   in sys.argv
RESUME        = "--resume" in sys.argv

# LoRA config — targets the language model attention layers
LORA_CONFIG = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 64,
    lora_alpha     = 128,
    lora_dropout   = 0.05,
    bias           = "none",
    # Target the LLM attention layers; skip vision encoder
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)

TRAINING_ARGS = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 5,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = 16,      # effective batch = 16
    gradient_checkpointing      = True,
    learning_rate               = 2e-4,
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.05,
    bf16                        = True,    # RTX 5000 Ada supports bf16
    tf32                        = True,
    logging_steps               = 10,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    report_to                   = "none",
    dataloader_num_workers      = 4,
    remove_unused_columns       = False,
    max_length                  = 2048,
    dataset_text_field          = None,
    resume_from_checkpoint      = RESUME,
)

# ---------------------------------------------------------------------------
# Data collator for vision-language inputs
# ---------------------------------------------------------------------------

class VLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def _build_msgs(self, sample):
        """Convert JSONL sample to Qwen VL message format (content as list)."""
        img_path = sample["images"][0]
        user_text = re.sub(r"<image>\n?", "", sample["messages"][0]["content"]).strip()
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text",  "text":  user_text},
                ],
            },
            sample["messages"][-1],  # assistant message unchanged
        ]

    def __call__(self, samples):
        texts  = []
        images = []
        all_msgs = [self._build_msgs(s) for s in samples]

        for msgs, sample in zip(all_msgs, samples):
            text = self.processor.apply_chat_template(
                msgs,
                add_generation_prompt=False,
                tokenize=False,
            )
            texts.append(text)
            images.append(Image.open(sample["images"][0]).convert("RGB"))

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # Labels = input_ids with prompt tokens masked (-100)
        labels = batch["input_ids"].clone()

        for i, (msgs, sample) in enumerate(zip(all_msgs, samples)):
            prompt_only = self.processor.apply_chat_template(
                msgs[:-1],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_tokens = self.processor(
                text=[prompt_only],
                images=[Image.open(sample["images"][0]).convert("RGB")],
                return_tensors="pt",
                padding=False,
            )
            prompt_len = prompt_tokens["input_ids"].shape[-1]
            labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Mode: {'Full fine-tuning' if FULL_FINETUNE else 'LoRA'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load processor and model
    print("\nLoading model...")
    # max_pixels caps image resolution → limits visual token count per image
    # 512×512 = 262144 → ~1024 visual tokens max (vs 3072 for full-res images)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, max_pixels=512 * 512)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Preserve the original EOS token so generation_config isn't overwritten
    # by tokenizer alignment during training (which breaks inference stopping)
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    model.generation_config.eos_token_id = im_end_id

    # Freeze vision encoder — only fine-tune the language backbone
    for name, param in model.named_parameters():
        if "visual" in name or "vision" in name:
            param.requires_grad = False

    if not FULL_FINETUNE:
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()

    # Load datasets
    print("\nLoading datasets...")
    train_ds = load_dataset("json", data_files=TRAIN_DATA, split="train")
    val_ds   = load_dataset("json", data_files=VAL_DATA,   split="train")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Train
    collator = VLMDataCollator(processor)
    trainer  = SFTTrainer(
        model           = model,
        args            = TRAINING_ARGS,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        processing_class= processor,
    )

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=RESUME)

    # Save final model
    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR + "/final")
    processor.save_pretrained(OUTPUT_DIR + "/final")
    print(f"Saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
