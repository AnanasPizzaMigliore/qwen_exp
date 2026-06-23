#!/usr/bin/env python3
"""
LoRA finetuning of Qwen3.5-0.8B for expiration-date extraction.

Target schema (cleaned dataset): predict {year, month, day} only.
Features:
  - Lightweight chain-of-thought (restate the date, then emit JSON)
  - Data augmentation (color jitter + rotation)
  - Curriculum learning (non_hard first, then all data)
  - Per-epoch test inference
  - OOM-safe training step

Run with:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n llm python /home/penghao/qwen/finetune_qwen35.py
"""
import re
import os
import json
import time
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/home/penghao/qwen/Qwen/Qwen3.5-0.8B"
TRAIN_FOLDER = "/home/penghao/Dataset/train/"
TRAIN_JSON   = "/home/penghao/Dataset/train.json"
TEST_FOLDER  = "/home/penghao/Dataset/test/"
TEST_JSON    = "/home/penghao/Dataset/test.json"
OUTPUT_DIR   = "/home/penghao/qwen/qwen35_finetuned_v3/"
EVAL_OUTPUT  = "/home/penghao/qwen/epoch_results_v3/"

CURRICULUM_EPOCHS = 3   # Phase 1: non_hard only
TOTAL_EPOCHS      = 10  # Phase 2 runs for TOTAL_EPOCHS - CURRICULUM_EPOCHS
GRAD_ACCUM        = 16
LR                = 2e-4
LORA_RANK         = 64
LORA_ALPHA        = 64
MAX_PX            = 1024 * 1024  # matches qwen3.6 preprocessing (thumbnail 1024×1024)
GEN_MAX_TOKENS    = 2000         # matched across all in-scope evals (baseline + finetuned)

PROMPT = """You are reading the expiration date on a food package. Find the
expiration date (also called "best before," "use by," "BB," "EXP," "BBE,"
"consume before," or equivalent). If both a production date and an
expiration date appear, use the EXPIRATION date.

Step 1: In one short sentence, state the expiration date you see.
Step 2: On a new line, output a JSON object with exactly these fields:
{"year": <4-digit integer or null>, "month": <integer 1-12 or null>, "day": <integer 1-31 or null>}

If no expiration date is visible, say so in Step 1 and output:
{"year": null, "month": null, "day": null}

Output only the sentence and the JSON. No markdown fences."""

# Augmentation pipeline (PIL in → PIL out, no flip to preserve text readability)
AUGMENT = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=10, fill=128),
])


def resize_image(image):
    if MAX_PX is None:
        return image
    w, h = image.size
    if w * h > MAX_PX:
        scale = (MAX_PX / (w * h)) ** 0.5
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def extract_json(text):
    """Pull the {year, month, day} JSON object out of the model output."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def make_target(entry):
    """Lightweight CoT (restate the date) + {year, month, day} JSON."""
    year  = entry.get("year")
    month = entry.get("month")
    day   = entry.get("day")
    has_date = any(v is not None for v in (year, month, day))

    if has_date:
        y = str(year) if year is not None else "????"
        m = f"{month:02d}" if isinstance(month, int) else "??"
        d = f"{day:02d}"   if isinstance(day, int)   else "??"
        cot = f"The expiration date on the package reads {y}-{m}-{d}."
    else:
        cot = "No expiration date is visible on this package."

    json_part = json.dumps({"year": year, "month": month, "day": day},
                           ensure_ascii=False)
    return f"{cot}\n{json_part}"


# ── Dataset ───────────────────────────────────────────────────────────────────
class DateDataset(Dataset):
    def __init__(self, entries, folder, processor, augment=False):
        self.processor = processor
        self.folder    = Path(folder)
        self.augment   = augment
        self.entries   = [e for e in entries if (self.folder / e["filename"]).exists()]
        skipped = len(entries) - len(self.entries)
        if skipped:
            print(f"  Skipped {skipped} entries (image not found on disk)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry  = self.entries[idx]
        image  = Image.open(self.folder / entry["filename"]).convert("RGB")
        image  = resize_image(image)
        if self.augment:
            image = AUGMENT(image)
        target = make_target(entry)

        full_msgs = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": PROMPT},
            ]},
            {"role": "assistant", "content": target},
        ]
        prompt_msgs = [full_msgs[0]]

        full_text   = self.processor.apply_chat_template(
            full_msgs,   tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)

        full_enc   = self.processor(text=[full_text],   images=[image], return_tensors="pt")
        prompt_enc = self.processor(text=[prompt_text], images=[image], return_tensors="pt")

        input_ids  = full_enc["input_ids"][0]
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        SEQ_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
        item = {"labels": labels}
        for k, v in full_enc.items():
            if k == "input_ids":
                item[k] = input_ids
            elif k in SEQ_KEYS:
                item[k] = v[0]
            else:
                item[k] = v
        return item


def collate_fn(batch):
    assert len(batch) == 1, "per_device_train_batch_size must be 1"
    item = batch[0]
    SEQ_KEYS = {"input_ids", "attention_mask", "labels", "mm_token_type_ids", "token_type_ids"}
    return {k: v.unsqueeze(0) if k in SEQ_KEYS else v for k, v in item.items()}


# ── Per-epoch eval callback ───────────────────────────────────────────────────
class TestEvalCallback(TrainerCallback):
    def __init__(self, model, processor, test_folder, test_json, eval_output, epoch_offset=0):
        self.model        = model
        self.processor    = processor
        self.test_folder  = Path(test_folder)
        self.eval_output  = Path(eval_output)
        self.epoch_offset = epoch_offset
        self.eval_output.mkdir(parents=True, exist_ok=True)

        if Path(test_json).exists():
            with open(test_json, encoding="utf-8") as f:
                self.test_files = [e["filename"] for e in json.load(f)]
        else:
            exts = (".jpg", ".jpeg", ".png", ".webp")
            self.test_files = sorted(
                f for f in os.listdir(test_folder) if f.lower().endswith(exts))

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) + self.epoch_offset
        print(f"\n--- Epoch {epoch} test inference ---")
        model = kwargs.get("model", self.model)
        model.eval()

        results = {}
        for filename in self.test_files:
            image_path = self.test_folder / filename
            if not image_path.exists():
                continue
            raw = None
            try:
                image = Image.open(image_path).convert("RGB")
                image = resize_image(image)
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": PROMPT},
                ]}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[text], images=[image], return_tensors="pt"
                ).to(next(model.parameters()).device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                g0 = time.time()
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=GEN_MAX_TOKENS)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_s = time.time() - g0

                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                n_new = int(generated.shape[1])
                raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
                parsed = extract_json(raw)
                results[filename] = {"filename": filename, "raw_response": raw,
                                     "latency_s": round(gen_s, 3), "gen_tokens": n_new, **parsed}
            except Exception as e:
                results[filename] = {"filename": filename, "error": str(e), "raw_response": raw}

        out_file = self.eval_output / f"test_results_epoch_{epoch:02d}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        errors = sum(1 for v in results.values() if "error" in v)
        lat = [v["latency_s"] for v in results.values() if "latency_s" in v]
        mean_lat = round(sum(lat) / len(lat), 3) if lat else None
        print(f"  Saved {len(results)} results ({errors} errors) → {out_file}")
        if lat:
            print(f"  Gen latency: mean {mean_lat}s/img  ({len(lat)/sum(lat):.2f} img/s)")
        model.train()


# ── OOM-safe Trainer ─────────────────────────────────────────────────────────
class RobustTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("  [OOM] skipping oversized image — cache cleared")
            return torch.tensor(0.0, device=next(model.parameters()).device,
                                requires_grad=True)


def make_training_args(output_dir, epochs, warmup_steps):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )


# ── Load processor & model ────────────────────────────────────────────────────
print(f"Loading processor from {MODEL_PATH} ...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Loading model ...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False  # incompatible with gradient checkpointing

# ── Apply LoRA ────────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ── Load data ─────────────────────────────────────────────────────────────────
with open(TRAIN_JSON, encoding="utf-8") as f:
    all_entries = json.load(f)

non_hard = [e for e in all_entries if e.get("difficulty_tier") != "hard"]
print(f"\nTotal entries : {len(all_entries)}")
print(f"  non_hard    : {len(non_hard)}")
print(f"  hard        : {len(all_entries) - len(non_hard)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_OUTPUT, exist_ok=True)

# ── Phase 1: curriculum — non_hard only ──────────────────────────────────────
print(f"\n{'='*55}")
print(f"  PHASE 1: non_hard only ({CURRICULUM_EPOCHS} epochs)")
print(f"{'='*55}")

dataset_p1 = DateDataset(non_hard, TRAIN_FOLDER, processor, augment=True)
print(f"  Dataset size: {len(dataset_p1)} images")

callback_p1 = TestEvalCallback(
    model, processor, TEST_FOLDER, TEST_JSON, EVAL_OUTPUT, epoch_offset=0)

trainer_p1 = RobustTrainer(
    model=model,
    args=make_training_args(OUTPUT_DIR + "/phase1", CURRICULUM_EPOCHS, warmup_steps=20),
    train_dataset=dataset_p1,
    data_collator=collate_fn,
    callbacks=[callback_p1],
)
trainer_p1.train()

# ── Phase 2: all data ─────────────────────────────────────────────────────────
phase2_epochs = TOTAL_EPOCHS - CURRICULUM_EPOCHS
print(f"\n{'='*55}")
print(f"  PHASE 2: all data ({phase2_epochs} epochs)")
print(f"{'='*55}")

dataset_p2 = DateDataset(all_entries, TRAIN_FOLDER, processor, augment=True)
print(f"  Dataset size: {len(dataset_p2)} images")

callback_p2 = TestEvalCallback(
    model, processor, TEST_FOLDER, TEST_JSON, EVAL_OUTPUT,
    epoch_offset=CURRICULUM_EPOCHS)

trainer_p2 = RobustTrainer(
    model=model,
    args=make_training_args(OUTPUT_DIR, phase2_epochs, warmup_steps=50),
    train_dataset=dataset_p2,
    data_collator=collate_fn,
    callbacks=[callback_p2],
)
trainer_p2.train()

# ── Save final adapter ────────────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"\nLoRA adapter saved to {OUTPUT_DIR}")
print(f"Epoch results saved to {EVAL_OUTPUT}")
