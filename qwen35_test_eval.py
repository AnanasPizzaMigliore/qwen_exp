#!/usr/bin/env python3
"""
Zero-shot (no finetuning) baseline for Qwen3.5-0.8B on the test set.

Uses the SAME prompt / output schema ({year, month, day}) and image
preprocessing (1024×1024 cap) as finetune_qwen35.py, so the result is a
fair apples-to-apples baseline row for Table 4.

Run with:
    conda run -n llm python /home/penghao/qwen/qwen35_test_eval.py
Then evaluate:
    python /home/penghao/Dataset/evaluate_qwen35.py
"""
import torch
import json
import os
import re
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH  = "/home/penghao/qwen/Qwen/Qwen3.5-0.8B"
TEST_FOLDER = "/home/penghao/Dataset/test/"
TEST_JSON   = "/home/penghao/Dataset/test.json"
OUTPUT_JSON = "/home/penghao/Dataset/qwen35_test_results.json"
SPEED_JSON  = "/home/penghao/Dataset/qwen35_test_speed.json"

MAX_PX         = 1024 * 1024   # match finetune preprocessing
GEN_MAX_TOKENS = 2000          # MUST match finetune_qwen35.py GEN_MAX_TOKENS (fair comparison)

# Identical prompt to finetune_qwen35.py
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
            return json.loads(match.group(0))
        raise


print(f"Loading model from {MODEL_PATH} ...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded.\n")

# Use the test split definition (falls back to scanning the folder)
if os.path.exists(TEST_JSON):
    with open(TEST_JSON, encoding="utf-8") as f:
        files = [e["filename"] for e in json.load(f)]
else:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = sorted(f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(exts))
total = len(files)

# Resume support: keep only successful entries
results = {}
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, encoding="utf-8") as f:
        for fn, entry in json.load(f).items():
            if "error" not in entry:
                results[fn] = entry
    print(f"Resuming: {len(results)} already done.")

pending = [f for f in files if f not in results]
print(f"Processing {len(pending)}/{total} images...\n")

start = time.time()
for i, filename in enumerate(pending, 1):
    image_path = os.path.join(TEST_FOLDER, filename)
    print(f"  [{i:>4}/{len(pending)}] {filename} ...", end=" ", flush=True)
    raw, gen_s, n_new = None, None, None
    try:
        image = Image.open(image_path).convert("RGB")
        image = resize_image(image)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

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
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
        parsed = extract_json(raw)
        results[filename] = {"filename": filename, "raw_response": raw,
                             "latency_s": round(gen_s, 3), "gen_tokens": n_new, **parsed}
        print(f"✓  ({gen_s:.2f}s, {n_new} tok)")
    except Exception as e:
        # Append as a COUNTED miss (year/month/day=null), not a skipped error.
        # Use 'parse_error' (not 'error') so evaluation includes it in the denominator.
        results[filename] = {"filename": filename, "raw_response": raw,
                             "latency_s": round(gen_s, 3) if gen_s is not None else None,
                             "gen_tokens": n_new, "parse_error": str(e),
                             "year": None, "month": None, "day": None}
        print(f"✗  counted as miss — {str(e)[:50]}")

    if i % 10 == 0:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

elapsed = time.time() - start
errors = sum(1 for v in results.values() if "error" in v or "parse_error" in v)

# ── Inference-speed summary (generation latency only, per image) ───────────────
lat = sorted(v["latency_s"] for v in results.values()
             if v.get("latency_s") is not None)
tok = [v["gen_tokens"] for v in results.values() if v.get("gen_tokens") is not None]
def pctl(xs, q):
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None
speed = {
    "model": MODEL_PATH,
    "finetuned": False,
    "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    "dtype": "bfloat16",
    "max_new_tokens": GEN_MAX_TOKENS,
    "max_px": MAX_PX,
    "n_images": len(lat),
    "gen_latency_s": {
        "mean":   round(sum(lat) / len(lat), 3) if lat else None,
        "median": pctl(lat, 0.50),
        "p90":    pctl(lat, 0.90),
        "min":    lat[0]  if lat else None,
        "max":    lat[-1] if lat else None,
    },
    "mean_gen_tokens":   round(sum(tok) / len(tok), 1) if tok else None,
    "throughput_img_per_s": round(len(lat) / sum(lat), 3) if lat else None,
}
with open(SPEED_JSON, "w", encoding="utf-8") as f:
    json.dump(speed, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
print(f"  Total:      {total}")
print(f"  Successful: {total - errors}")
print(f"  Failed:     {errors}")
print(f"  Wall time:  {elapsed:.1f}s")
if lat:
    print(f"  Gen latency: mean {speed['gen_latency_s']['mean']}s / "
          f"median {speed['gen_latency_s']['median']}s / p90 {speed['gen_latency_s']['p90']}s")
    print(f"  Throughput:  {speed['throughput_img_per_s']} img/s  "
          f"(mean {speed['mean_gen_tokens']} gen tokens)")
print(f"  Results ->   {OUTPUT_JSON}")
print(f"  Speed    ->  {SPEED_JSON}")
print(f"{'='*50}")
