#!/usr/bin/env python3
"""
Image token budget ablation — find optimal resolution for edge inference.

Sweeps TOKEN_BUDGETS for a given model+mmproj pair on a given backend.

Usage:
    python eval_token_ablation.py                          # defaults: Q3 LLM + F16 mmproj, vulkan
    python eval_token_ablation.py \
        --model  /path/to/llm.gguf \
        --mmproj /path/to/mmproj.gguf \
        --backend vulkan \
        --out-dir /path/to/output/dir
"""
import argparse
import subprocess
import time
import json
import os
import re
import base64
import math
import io
import requests
from pathlib import Path
from collections import defaultdict
from PIL import Image

# ── CLI args ──────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser()
_ap.add_argument("--model",    default="/home/penghao/qwen/gguf/qwen35_groupA_a3_Q3_K_M.gguf")
_ap.add_argument("--mmproj",   default="/home/penghao/qwen/gguf/qwen35_groupA_a3_mmproj_f16.gguf")
_ap.add_argument("--backend",  default="vulkan", choices=["cpu", "vulkan"])
_ap.add_argument("--out-dir",  default="/home/penghao/qwen/gguf_results/token_ablation")
_cli = _ap.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL    = _cli.model
MMPROJ   = _cli.mmproj
BACKEND  = _cli.backend
OUT_DIR  = Path(_cli.out_dir)

TEST_JSON    = "/home/penghao/Dataset/test.json"
TEST_FOLDER  = "/home/penghao/Dataset/test/"
GT_FILE      = "/home/penghao/Dataset/expiration_dates_details_true.json"
LLAMA_SERVER = "/home/penghao/llama.cpp/build/bin/llama-server"

TOKEN_BUDGETS = [256, 512, 768, 1024, 1344, 2048, 3072, 4096, 6144, 8192]

PATCH      = 14   # vision encoder patch size
MERGE      = 2    # spatial merge factor
CELL       = PATCH * MERGE   # 28 px per token (each axis)
PX_PER_TOK = CELL * CELL     # 784 px² per token

GEN_MAX_TOKENS   = 2000
CTX_SIZE         = 16384
CPU_THREADS      = 24
RESTART_INTERVAL = 100
PORT             = 8765

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


# ── Resolution helpers ────────────────────────────────────────────────────────

def tokens_for_image(w, h):
    """Actual token count after rounding dimensions to multiples of CELL."""
    w_r = round(w / CELL) * CELL
    h_r = round(h / CELL) * CELL
    return (w_r // CELL) * (h_r // CELL)


def resize_to_budget(image: Image.Image, token_budget: int) -> tuple[Image.Image, int]:
    """Resize image so token count <= token_budget. Returns (image, actual_tokens)."""
    w, h = image.size
    max_pixels = token_budget * PX_PER_TOK
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        w = round(w * scale / CELL) * CELL
        h = round(h * scale / CELL) * CELL
        image = image.resize((w, h), Image.LANCZOS)
    actual = tokens_for_image(*image.size)
    return image, actual


def encode_image_resized(image_path: str, token_budget: int) -> tuple[str, int]:
    """Load, resize to budget, return (base64 data URL, actual_tokens)."""
    img = Image.open(image_path).convert("RGB")
    img, actual_tokens = resize_to_budget(img, token_budget)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{data}", actual_tokens


# ── Server lifecycle ──────────────────────────────────────────────────────────

def start_server(port=PORT):
    ngl = 0 if BACKEND == "cpu" else 999
    cmd = [
        LLAMA_SERVER,
        "-m", MODEL, "--mmproj", MMPROJ,
        "--n-gpu-layers", str(ngl),
        "--port", str(port), "--host", "127.0.0.1",
        "--ctx-size", str(CTX_SIZE),
        "--threads", str(CPU_THREADS),
        "--log-disable",
    ]
    if BACKEND == "vulkan":
        cmd += ["--mmproj-offload"]
    log = open(f"/tmp/llama_server_{port}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{port}"
    for i in range(120):
        time.sleep(2)
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited early")
        try:
            if requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                print(f"  Server ready ({(i+1)*2}s)")
                return proc, base_url
        except Exception:
            pass
    proc.terminate()
    raise RuntimeError("llama-server did not become ready")


def stop_server(proc):
    proc.terminate()
    try: proc.wait(timeout=20)
    except subprocess.TimeoutExpired: proc.kill()


# ── Inference ─────────────────────────────────────────────────────────────────

def extract_json(text):
    text = text.strip()
    for delim in ("```json", "```"):
        if delim in text:
            text = text.split(delim)[1].split("```")[0]
            break
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m: return json.loads(m.group(0))
        raise


def query(base_url, image_path, token_budget):
    data_url, actual_tokens = encode_image_resized(image_path, token_budget)
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": PROMPT},
        ]}],
        "max_tokens": GEN_MAX_TOKENS,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=180)
    latency = time.time() - t0
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    gen_tokens = data.get("usage", {}).get("completion_tokens")
    return content, latency, gen_tokens, actual_tokens


# ── Accuracy ──────────────────────────────────────────────────────────────────

def norm(v):
    if v is None or v in ("null", ""): return None
    try: return int(v)
    except: return v

def norm_year(v):
    v = norm(v)
    return v + 2000 if isinstance(v, int) and 0 < v < 100 else v

def score(results, gt_all, test_files):
    s = defaultdict(int)
    for fn in test_files:
        p = results.get(fn); g = gt_all.get(fn)
        if not p or not g: continue
        s["total"] += 1
        if "error" in p or "parse_error" in p: continue
        g_has = any(norm(g.get(k)) is not None for k in ("year","month","day"))
        if not g_has:
            s["null_total"] += 1
            if all(norm(p.get(k)) is None for k in ("year","month","day")):
                s["null_ok"] += 1; s["full"] += 1
            continue
        s["gt"] += 1
        py=norm_year(p.get("year")); gy=norm_year(g.get("year"))
        pm=norm(p.get("month"));     gm=norm(g.get("month"))
        pd=norm(p.get("day"));       gd=norm(g.get("day"))
        yok=py==gy; mok=pm==gm
        dok=(pd==gd) if gd is not None else None
        if yok and mok and (dok or dok is None): s["full"] += 1
    return s


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_budget(token_budget, test_files, gt_all):
    llm_stem = Path(MODEL).stem      # e.g. qwen35_groupA_a3_Q3_K_M
    mm_stem  = Path(MMPROJ).stem     # e.g. qwen35_groupA_a3_mmproj_Q4_K_M
    tag = f"{llm_stem}__{mm_stem}__{BACKEND}"
    out_path = OUT_DIR / f"tok{token_budget}_{tag}.json"
    max_px = token_budget * PX_PER_TOK

    print(f"\n{'='*60}")
    print(f"  Token budget: {token_budget}  (max_pixels={max_px/1e6:.2f}MP)")
    print(f"  Output: {out_path.name}")
    print(f"{'='*60}")

    # Resume
    results = {}
    if out_path.exists():
        for fn, entry in json.load(open(out_path)).items():
            if "error" not in entry and "parse_error" not in entry:
                results[fn] = entry
        print(f"  Resuming: {len(results)} already done.")

    pending = [f for f in test_files if f not in results]
    if pending:
        proc, base_url = start_server()
        try:
            wall0 = time.time()
            for i, fn in enumerate(pending, 1):
                if RESTART_INTERVAL and i > 1 and (i-1) % RESTART_INTERVAL == 0:
                    print(f"\n  [restart at {i}]")
                    stop_server(proc)
                    proc, base_url = start_server()

                img_path = os.path.join(TEST_FOLDER, fn)
                print(f"  [{i:>4}/{len(pending)}] {fn} ...", end=" ", flush=True)
                try:
                    raw, lat, gen_tok, actual_tok = query(base_url, img_path, token_budget)
                    parsed = extract_json(raw)
                    results[fn] = {"filename": fn, "raw_response": raw,
                                   "latency_s": round(lat, 3), "gen_tokens": gen_tok,
                                   "image_tokens": actual_tok, **parsed}
                    print(f"✓  ({lat:.2f}s, {actual_tok} img_tok)")
                except Exception as e:
                    results[fn] = {"filename": fn, "parse_error": str(e),
                                   "year": None, "month": None, "day": None}
                    print(f"✗  {str(e)[:80]}")

                if i % 20 == 0:
                    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
                    elapsed = time.time() - wall0
                    print(f"    ETA {(len(pending)-i)*(elapsed/i)/60:.1f} min")
        finally:
            stop_server(proc)

    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)

    s = score(results, gt_all, test_files)
    lats = [v["latency_s"] for v in results.values() if isinstance(v.get("latency_s"), float)]
    img_toks = [v["image_tokens"] for v in results.values() if v.get("image_tokens")]
    mean_lat = sum(lats)/len(lats) if lats else 0
    mean_tok = sum(img_toks)/len(img_toks) if img_toks else token_budget
    all_pct = s["full"]/s["total"]*100 if s["total"] else 0

    return {
        "token_budget": token_budget,
        "mean_img_tokens": round(mean_tok, 1),
        "max_pixels_mp": round(max_px/1e6, 2),
        "ALL_pct": round(all_pct, 1),
        "mean_lat_s": round(mean_lat, 2),
        "n": s["total"],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEST_JSON) as f:
        test_files = [e["filename"] for e in json.load(f)]
    with open(GT_FILE) as f:
        gt_all = {e["filename"]: e for e in json.load(f)}

    summary = []
    for budget in TOKEN_BUDGETS:
        row = run_budget(budget, test_files, gt_all)
        summary.append(row)
        print(f"\n  >> tok={row['token_budget']}  ALL%={row['ALL_pct']}  "
              f"lat={row['mean_lat_s']}s  mean_img_tok={row['mean_img_tokens']}")

    llm_stem = Path(MODEL).stem
    mm_stem  = Path(MMPROJ).stem
    tag = f"{llm_stem}__{mm_stem}__{BACKEND}"
    json.dump(summary, open(OUT_DIR / f"summary_{tag}.json", "w"), indent=2)

    print(f"\n\n{'='*65}")
    print(f"  Token Budget Ablation — {Path(MODEL).stem} / {Path(MMPROJ).stem} / {BACKEND}")
    print(f"{'='*65}")
    print(f"  {'Budget':>8}  {'Avg tokens':>10}  {'Max MP':>7}  {'ALL%':>6}  {'Lat(s)':>7}  {'Est phone(s)':>13}")
    print(f"  {'-'*63}")
    for r in summary:
        est_phone = r['mean_lat_s'] * 10
        print(f"  {r['token_budget']:>8}  {r['mean_img_tokens']:>10.0f}  "
              f"{r['max_pixels_mp']:>7.2f}  {r['ALL_pct']:>6.1f}  "
              f"{r['mean_lat_s']:>7.2f}  {est_phone:>12.1f}s")
    print(f"{'='*65}")
    print(f"  Est phone = desktop Vulkan × 10 (rough mobile proxy)")

    print(f"\nSummary saved: {OUT_DIR / f'summary_{tag}.json'}")


if __name__ == "__main__":
    main()
