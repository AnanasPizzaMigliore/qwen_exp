#!/usr/bin/env python3
"""
Group A ablation: what components to finetune.

  --strategy a0  LLM LoRA only                        (baseline, same as finetune_nocurriculum.py)
  --strategy a1  LLM LoRA + vision encoder LoRA       (attn.qkv + attn.proj, r=32)
  --strategy a2  LLM LoRA + visual merger (full grad)
  --strategy a3  LLM LoRA + vision encoder LoRA + merger (full grad)

Settings matched exactly to finetune_nocurriculum.py (the agreed baseline):
  MAX_PX=1024*1024, LR=2e-4, GRAD_ACCUM=16, LORA_RANK=64/ALPHA=64,
  warmup_steps=50, 10 epochs all-data, same AUGMENT, same PROMPT.

Architecture (Qwen3.5-0.8B):
  Vision encoder blocks : 85.1M params  (12 blocks × attn.qkv/attn.proj/mlp.fc1/fc2)
  Visual merger         : 12.6M params  (linear_fc1 3072→3072, linear_fc2 3072→1024)
  LLM                   : 752.4M params

Vision encoder LoRA uses r=32/alpha=32 (half of LLM) via rank_pattern so each
component gets the right capacity without bloating the vision adapter.

Run:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n llm python /home/penghao/qwen/finetune_groupA.py --strategy a1
"""
import re, os, json, time, argparse
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    AutoProcessor, AutoModelForImageTextToText,
    TrainingArguments, Trainer, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--strategy", required=True,
                choices=["a0", "a1", "a2", "a3", "a0p", "a1p", "a4"],
                help="a0=LLM-LoRA only  a1=+vision-LoRA  a2=+merger-full  a3=+both  "
                     "a0p/a1p=A0/A1 re-run at A3's parameter budget  "
                     "a4=vision-LoRA only, LLM frozen")
ap.add_argument("--epochs", type=float, default=10)
ap.add_argument("--seed", type=int, default=None,
                help="seed everything before the adapters are created; outputs get an _s<seed> "
                     "suffix. Without it the script behaves as in the original runs (Trainer's "
                     "default seed 42, which is applied only after adapter initialisation)")
ap.add_argument("--no-test-eval", action="store_true",
                help="do not touch the test split during training; score it once after the "
                     "final epoch instead")
args = ap.parse_args()

STRATEGY = args.strategy
SUFFIX = f"_s{args.seed}" if args.seed is not None else ""
if args.seed is not None:
    from transformers import set_seed
    set_seed(args.seed)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/home/penghao/qwen/Qwen/Qwen3.5-0.8B"
TRAIN_FOLDER = "/home/penghao/Dataset/train/"
TRAIN_JSON   = "/home/penghao/Dataset/train.json"
TEST_FOLDER  = "/home/penghao/Dataset/test/"
TEST_JSON    = "/home/penghao/Dataset/test.json"
OUTPUT_DIR   = f"/home/penghao/qwen/groupA_{STRATEGY}{SUFFIX}/"
EVAL_OUTPUT  = f"/home/penghao/qwen/epoch_results_groupA_{STRATEGY}{SUFFIX}/"

TOTAL_EPOCHS   = args.epochs
GRAD_ACCUM     = 16          # identical to baseline
LR             = 2e-4        # identical to baseline
LORA_RANK      = 64          # identical to baseline
LORA_ALPHA     = 64          # identical to baseline
LORA_RANK_VIS  = 32          # half rank for vision encoder (smaller component)
LORA_ALPHA_VIS = 32
# Matched-budget arms. Measured trainable counts at the stock ranks are
# A0 25.6M, A1 27.3M, A2 38.1M, A3 39.9M — the LLM adapter is 25.6M at r=64
# (0.40M per rank) and the ViT adapter only 1.8M at r=32. A3 wins and is also
# the largest, so a0p and a1p re-run A0 and A1 with the LLM rank raised until
# each matches A3's ~39.9M. Same budget, different placement.
LORA_RANK_MATCHED_A0 = 100   # a0p: LLM-only        -> ~40.0M
LORA_RANK_MATCHED_A1 = 95    # a1p: LLM + ViT r=32  -> ~39.8M
MAX_PX         = 1024 * 1024 # identical to baseline — 1 MP cap
GEN_MAX_TOKENS = 2000        # identical to baseline

# LLM target modules — identical to baseline finetune_qwen35.py
LLM_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# Vision encoder attention layers (suffix-matched by PEFT)
# attn.qkv = fused QKV projection (768→2304), attn.proj = output projection (768→768)
VIS_TARGET_MODULES = ["attn.qkv", "attn.proj"]

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

AUGMENT = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=10, fill=128),
])


def resize_image(image):
    w, h = image.size
    if w * h > MAX_PX:
        scale = (MAX_PX / (w * h)) ** 0.5
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def extract_json(text):
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
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Cannot parse JSON from: {text[:100]}")


def make_target(entry):
    year = entry.get("year"); month = entry.get("month"); day = entry.get("day")
    has_date = any(v is not None for v in (year, month, day))
    if has_date:
        y = str(year) if year is not None else "????"
        m = f"{month:02d}" if isinstance(month, int) else "??"
        d = f"{day:02d}"   if isinstance(day, int)   else "??"
        cot = f"The expiration date on the package reads {y}-{m}-{d}."
    else:
        cot = "No expiration date is visible on this package."
    return f"{cot}\n{json.dumps({'year': year, 'month': month, 'day': day}, ensure_ascii=False)}"


# ── Dataset ───────────────────────────────────────────────────────────────────
class DateDataset(Dataset):
    def __init__(self, entries, folder, processor, augment=False):
        self.processor = processor
        self.folder    = Path(folder)
        self.augment   = augment
        self.entries   = [e for e in entries if (self.folder / e["filename"]).exists()]
        skipped = len(entries) - len(self.entries)
        if skipped:
            print(f"  Skipped {skipped} entries (image not found)")

    def __len__(self): return len(self.entries)

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
        labels     = input_ids.clone()
        labels[:prompt_len] = -100

        SEQ_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
        item = {"labels": labels}
        for k, v in full_enc.items():
            if k == "input_ids":      item[k] = input_ids
            elif k in SEQ_KEYS:       item[k] = v[0]
            else:                     item[k] = v
        return item


def collate_fn(batch):
    assert len(batch) == 1
    item = batch[0]
    SEQ_KEYS = {"input_ids", "attention_mask", "labels", "mm_token_type_ids", "token_type_ids"}
    return {k: v.unsqueeze(0) if k in SEQ_KEYS else v for k, v in item.items()}


# ── Per-epoch eval callback ───────────────────────────────────────────────────
class TestEvalCallback(TrainerCallback):
    def __init__(self, model, processor):
        self.model     = model
        self.processor = processor
        Path(EVAL_OUTPUT).mkdir(parents=True, exist_ok=True)
        with open(TEST_JSON, encoding="utf-8") as f:
            self.test_files = [e["filename"] for e in json.load(f)]

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        print(f"\n--- Epoch {epoch} test inference ---")
        model = kwargs.get("model", self.model)
        model.eval()
        results = {}
        for filename in self.test_files:
            image_path = Path(TEST_FOLDER) / filename
            if not image_path.exists(): continue
            raw = None
            try:
                image = resize_image(Image.open(image_path).convert("RGB"))
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": PROMPT},
                ]}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[text], images=[image], return_tensors="pt"
                ).to(next(model.parameters()).device)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                g0 = time.time()
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=GEN_MAX_TOKENS)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                gen_s = time.time() - g0
                gen = out[:, inputs["input_ids"].shape[1]:]
                raw = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
                parsed = extract_json(raw)
                results[filename] = {"filename": filename, "raw_response": raw,
                                     "latency_s": round(gen_s, 3),
                                     "gen_tokens": int(gen.shape[1]), **parsed}
            except Exception as e:
                results[filename] = {"filename": filename, "error": str(e), "raw_response": raw}

        out_file = Path(EVAL_OUTPUT) / f"test_results_epoch_{epoch:02d}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        errors = sum(1 for v in results.values() if "error" in v)
        lat = [v["latency_s"] for v in results.values() if "latency_s" in v]
        print(f"  Saved {len(results)} results ({errors} errors) → {out_file}")
        if lat: print(f"  Mean latency: {sum(lat)/len(lat):.2f}s/img")
        model.train()


# ── OOM-safe Trainer ──────────────────────────────────────────────────────────
class RobustTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("  [OOM] skipping oversized image")
            return torch.tensor(0.0, device=next(model.parameters()).device,
                                requires_grad=True)


# ── Load processor & base model ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Strategy : {STRATEGY.upper()}")
print(f"  Output   : {OUTPUT_DIR}")
print(f"{'='*60}\n")

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
model.config.use_cache = False

# ── Apply LoRA ────────────────────────────────────────────────────────────────
if STRATEGY in ("a0", "a2", "a0p"):
    # A0 / A2: LLM LoRA only — identical config to finetune_nocurriculum.py baseline.
    # A0': same shape, rank raised to sit at A3's parameter budget.
    llm_r = LORA_RANK_MATCHED_A0 if STRATEGY == "a0p" else LORA_RANK
    lora_cfg = LoraConfig(
        r=llm_r, lora_alpha=llm_r, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=LLM_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)

elif STRATEGY == "a4":
    # A4: vision encoder LoRA ONLY — the LLM stays frozen, so nothing in the
    # decoder adapts. Gradients still reach the ViT because the image embeddings
    # are scattered into inputs_embeds and carry requires_grad through the LLM.
    lora_cfg = LoraConfig(
        r=LORA_RANK_VIS, lora_alpha=LORA_ALPHA_VIS, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=VIS_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)

else:
    # A1 / A3 / A1': LLM LoRA (r=64) + vision encoder LoRA.
    # rank_pattern applies a different r to layers whose name contains the key.
    # "attn.qkv" and "attn.proj" only exist in the vision encoder; LLM layers
    # (q_proj, k_proj, …) are not affected by these patterns.
    # A1' keeps the ViT adapter at r=32 and raises the LLM rank to reach A3's budget.
    llm_r = LORA_RANK_MATCHED_A1 if STRATEGY == "a1p" else LORA_RANK
    lora_cfg = LoraConfig(
        r=llm_r, lora_alpha=llm_r, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=LLM_TARGET_MODULES + VIS_TARGET_MODULES,
        rank_pattern={
            "attn.qkv": LORA_RANK_VIS,
            "attn.proj": LORA_RANK_VIS,
        },
        alpha_pattern={
            "attn.qkv": LORA_ALPHA_VIS,
            "attn.proj": LORA_ALPHA_VIS,
        },
    )
    model = get_peft_model(model, lora_cfg)

# ── Unfreeze visual merger for a2 / a3 ───────────────────────────────────────
if STRATEGY in ("a2", "a3"):
    merger_params = 0
    for name, param in model.named_parameters():
        if "visual.merger" in name:
            param.requires_grad = True
            merger_params += param.numel()
    print(f"  Visual merger unfrozen: {merger_params/1e6:.1f}M additional trainable params")

model.print_trainable_parameters()

# ── Load data (all data, no curriculum — matches finetune_nocurriculum.py) ────
with open(TRAIN_JSON, encoding="utf-8") as f:
    all_entries = json.load(f)
print(f"\nTraining on {len(all_entries)} images (no curriculum)")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_OUTPUT, exist_ok=True)

dataset = DateDataset(all_entries, TRAIN_FOLDER, processor, augment=True)
print(f"  Dataset size: {len(dataset)} images")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=TOTAL_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    report_to="none",
    **({"seed": args.seed} if args.seed is not None else {}),
)

test_eval = TestEvalCallback(model, processor)
trainer = RobustTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
    callbacks=[] if args.no_test_eval else [test_eval],
)
trainer.train()
if args.no_test_eval:
    # The only look at the test split: once, after the last epoch, with the same code path.
    test_eval.on_epoch_end(training_args, trainer.state, None, model=model)

model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# For A2/A3: PEFT's save_pretrained only saves LoRA adapter weights and silently
# drops non-LoRA trainable parameters (the visual merger). Save them separately.
if STRATEGY in ("a2", "a3"):
    import safetensors.torch as st
    merger_state = {n: p.data.cpu()
                    for n, p in model.named_parameters()
                    if "visual.merger" in n}
    st.save_file(merger_state, Path(OUTPUT_DIR) / "merger_weights.safetensors")
    print(f"  Merger weights saved ({len(merger_state)} tensors) → {OUTPUT_DIR}/merger_weights.safetensors")

print(f"\nLoRA adapter saved → {OUTPUT_DIR}")
print(f"Epoch results      → {EVAL_OUTPUT}")
