#!/usr/bin/env python3
"""
Evaluate the NATIVE merged finetuned model (transformers) on the test set,
using settings matched EXACTLY to eval_gguf.py so the result is directly
comparable to the GGUF numbers:

    - model    : /home/penghao/qwen/qwen35_merged  (the exact source of the GGUF)
    - decoding : greedy  (do_sample=False)        ↔  temperature=0 in eval_gguf
    - thinking : disabled (enable_thinking=False) ↔  chat_template_kwargs in eval_gguf
    - prompt   : identical PROMPT
    - resize   : identical MAX_PX (1024*1024)
    - max tok  : 2000

This isolates the model from the GGUF runtime: if this lands on ~73%, then the
GGUF conversion is lossless and the lower per-epoch numbers in epoch_results_v3/
were simply the in-training LoRA/bf16 callback measuring a different artifact,
NOT quantization "improving" accuracy.

Run:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n llm python /home/penghao/qwen/eval_native.py

    # match GGUF F16 precision exactly:
    ... python /home/penghao/qwen/eval_native.py --dtype float16
"""
import re
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ── Config (matched to finetune_qwen35.py / eval_gguf.py) ──────────────────────
MERGED_PATH  = "/home/penghao/qwen/qwen35_merged"
TEST_FOLDER  = "/home/penghao/Dataset/test/"
TEST_JSON    = "/home/penghao/Dataset/test.json"
GT_FILE      = "/home/penghao/Dataset/expiration_dates_details_true.json"
OUTPUT       = "/home/penghao/qwen/gguf_results/qwen35_finetuned_native.json"

# Default 1 MP matches how the model was TRAINED (resize_image in finetune_qwen35.py).
# llama.cpp/eval_gguf.py feeds the full ~12 MP originals, which is why the GGUF
# scores higher. Raise --max-px to feed higher resolution and close the gap.
MAX_PX         = 1024 * 1024
GEN_MAX_TOKENS = 2000

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


def resize_image(image, max_px):
    if max_px is None or max_px <= 0:
        return image                      # no downscale → full original resolution
    w, h = image.size
    if w * h > max_px:
        scale = (max_px / (w * h)) ** 0.5
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
    return {"year": None, "month": None, "day": None, "parse_error": True}


# ── Scoring (identical to compare_all.py, incl. null/hallucination handling) ───
def norm(v):
    if v is None or v in ("null", ""):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return v


def norm_year(v):
    x = norm(v)
    return x + 2000 if isinstance(x, int) and 0 < x < 100 else x


def pct(c, n):
    return f"{c / n * 100:.1f}" if n else "N/A"


def score(results, test_files, gt_all):
    stats = {t: defaultdict(int) for t in ("hard", "non_hard", "ALL")}
    for fn in test_files:
        p = results.get(fn)
        g = gt_all.get(fn)
        if p is None or g is None:
            continue
        rt = g.get("difficulty_tier", "unknown")
        tier = "non_hard" if rt in ("easy", "medium", "non_hard") else rt
        for grp in (tier, "ALL"):
            stats[grp]["total"] += 1
        if "error" in p or "parse_error" in p:
            continue
        g_has = any(norm(g.get(k)) is not None for k in ("year", "month", "day"))
        if not g_has:
            p_null = all(norm(p.get(k)) is None for k in ("year", "month", "day"))
            for grp in (tier, "ALL"):
                stats[grp]["null_total"] += 1
                if p_null:
                    stats[grp]["null_correct"] += 1
                    stats[grp]["full"] += 1
            continue
        for grp in (tier, "ALL"):
            stats[grp]["gt"] += 1
        y_ok = norm_year(p.get("year")) == norm_year(g.get("year"))
        m_ok = norm(p.get("month")) == norm(g.get("month"))
        g_d = norm(g.get("day"))
        d_ok = (norm(p.get("day")) == g_d) if g_d is not None else None
        for grp in (tier, "ALL"):
            if y_ok:
                stats[grp]["year"] += 1
            if m_ok:
                stats[grp]["month"] += 1
            if d_ok is not None:
                stats[grp]["day_total"] += 1
                if d_ok:
                    stats[grp]["day"] += 1
            if y_ok and m_ok and (d_ok or d_ok is None):
                stats[grp]["full"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MERGED_PATH)
    ap.add_argument("--output", default=None,
                    help="default: gguf_results/qwen35_finetuned_native_<res>.json")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                    help="float16 matches the F16 GGUF; bfloat16 matches training")
    ap.add_argument("--max-px", type=float, default=1.0,
                    help="megapixel cap before the vision encoder. "
                         "1.0 = training-faithful (default); 0 = full ~12 MP original "
                         "(matches llama.cpp/eval_gguf and should close the GGUF gap)")
    args = ap.parse_args()
    max_px = int(args.max_px * 1_000_000) if args.max_px and args.max_px > 0 else None
    if args.output is None:
        tag = "fullres" if max_px is None else f"{args.max_px:g}mp"
        args.output = f"/home/penghao/qwen/gguf_results/qwen35_finetuned_native_{tag}.json"

    with open(TEST_JSON, encoding="utf-8") as f:
        test_files = [e["filename"] for e in json.load(f)]
    with open(GT_FILE, encoding="utf-8") as f:
        gt_all = {e["filename"]: e for e in json.load(f)}

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    res_str = "full (~12 MP, no downscale)" if max_px is None else f"{args.max_px:g} MP cap"
    print(f"Loading {args.model}  (dtype={args.dtype}, image res = {res_str}) ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cuda")
    model.eval()

    results = {}
    wall0 = time.time()
    for i, filename in enumerate(test_files, 1):
        image_path = Path(TEST_FOLDER) / filename
        if not image_path.exists():
            continue
        raw = None
        try:
            image = resize_image(Image.open(image_path).convert("RGB"), max_px)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": PROMPT},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)              # ← matched to eval_gguf.py
            inputs = processor(text=[text], images=[image],
                               return_tensors="pt").to(model.device)

            torch.cuda.synchronize()
            g0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=GEN_MAX_TOKENS,
                                     do_sample=False)   # ← greedy, == temperature 0
            torch.cuda.synchronize()
            gen_s = time.time() - g0

            gen_ids = out[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            parsed = extract_json(raw)
            results[filename] = {"filename": filename, "raw_response": raw,
                                 "latency_s": round(gen_s, 3),
                                 "gen_tokens": int(gen_ids.shape[1]), **parsed}
        except Exception as e:
            results[filename] = {"filename": filename, "error": str(e),
                                 "raw_response": raw}

        if i % 25 == 0 or i == len(test_files):
            elapsed = time.time() - wall0
            print(f"  {i}/{len(test_files)}  ({elapsed:.0f}s, "
                  f"{elapsed / i:.2f}s/img)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results → {args.output}")

    # ── Print score ────────────────────────────────────────────────────────────
    s = score(results, test_files, gt_all)
    a, h, n = s["ALL"], s["hard"], s["non_hard"]
    errs = sum(1 for v in results.values() if "error" in v or "parse_error" in v)
    print(f"\n{'='*70}")
    print(f"  NATIVE merged model — matched greedy / no-thinking  (dtype={args.dtype})")
    print(f"  n={a['total']}  errors/parse_fail={errs}")
    print(f"{'='*70}")
    print(f"  {'tier':<10} {'n_total':>8} {'n_gt':>6} {'no-hal':>7} "
          f"{'year%':>6} {'month%':>6} {'day%':>6} {'full%':>6}")
    for name, ts in (("ALL", a), ("hard", h), ("non_hard", n)):
        print(f"  {name:<10} {ts['total']:>8} {ts['gt']:>6} "
              f"{pct(ts['null_correct'], ts['null_total']):>7} "
              f"{pct(ts['year'], ts['gt']):>6} "
              f"{pct(ts['month'], ts['gt']):>6} "
              f"{pct(ts['day'], ts['day_total']):>6} "
              f"{pct(ts['full'], ts['total']):>6}")
    print(f"{'='*70}")
    print("  Compare ALL full% against the F16 GGUF row (73.0 cpu / 73.4 vulkan).")
    print("  If it matches → GGUF conversion is lossless; the 66.6% per-epoch")
    print("  numbers were the in-training LoRA/bf16 callback, a different artifact.")


if __name__ == "__main__":
    main()
