#!/usr/bin/env python3
"""
Evaluate a GGUF model on the expiration-date test dataset via llama-server.
Supports CPU and Vulkan backends.

Usage:
    python eval_gguf.py \
        --model /home/penghao/qwen/gguf/qwen35_finetuned_Q4_K_M.gguf \
        --mmproj /home/penghao/qwen/gguf/Qwen3.5-0.8B-mmproj-f16.gguf \
        --backend cpu \
        --output /home/penghao/qwen/gguf_results/finetuned_Q4_K_M_cpu.json
"""
import argparse
import subprocess
import time
import json
import os
import re
import base64
import requests
from pathlib import Path
from collections import defaultdict

TEST_JSON   = "/home/penghao/Dataset/test.json"
TEST_FOLDER = "/home/penghao/Dataset/test/"
GT_FILE     = "/home/penghao/Dataset/expiration_dates_details_true.json"
LLAMA_SERVER = "/home/penghao/llama.cpp/build/bin/llama-server"

GEN_MAX_TOKENS = 2000
CTX_SIZE       = 16384  # enough for large images + prompt + response
CPU_THREADS    = 24

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


# ── Server lifecycle ──────────────────────────────────────────────────────────

def start_server(model_path, mmproj_path, backend, port):
    ngl = 0 if backend == "cpu" else 999
    cmd = [
        LLAMA_SERVER,
        "-m",    model_path,
        "--mmproj", mmproj_path,
        "--n-gpu-layers", str(ngl),
        "--port", str(port),
        "--host", "127.0.0.1",
        "--ctx-size", str(CTX_SIZE),
        "--threads", str(CPU_THREADS),
        "--log-disable",
    ]
    if backend == "vulkan":
        cmd += ["--mmproj-offload"]

    print(f"  Starting llama-server [{backend}] ...")
    print(f"  CMD: {' '.join(cmd)}\n")
    log_file = open(f"/tmp/llama_server_{port}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    base_url = f"http://127.0.0.1:{port}"
    for i in range(120):  # up to 240s
        time.sleep(2)
        if proc.poll() is not None:
            log_file.flush()
            with open(f"/tmp/llama_server_{port}.log") as lf:
                print(f"  Server exited early. Log:\n{lf.read()[-2000:]}")
            raise RuntimeError("llama-server exited before becoming ready")
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                print(f"  Server ready (waited {(i+1)*2}s)")
                return proc, base_url
        except Exception:
            pass

    proc.terminate()
    log_file.flush()
    with open(f"/tmp/llama_server_{port}.log") as lf:
        print(f"  Server log:\n{lf.read()[-2000:]}")
    raise RuntimeError("llama-server did not become ready within 240s")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── Image + inference ─────────────────────────────────────────────────────────

def encode_image(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    return f"data:{mime};base64,{data}"


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
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def query_server(base_url, image_path, timeout=180):
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "max_tokens": GEN_MAX_TOKENS,
        "temperature": 0,
        "stream": False,
        # Match the HuggingFace chat template behaviour: close the <think> block
        # before generation so the model outputs the CoT directly (no thinking loop).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    latency = time.time() - t0
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    gen_tokens = data.get("usage", {}).get("completion_tokens")
    return content, latency, gen_tokens


# ── Accuracy helpers ──────────────────────────────────────────────────────────

def norm(val):
    if val is None or val == "null" or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return val


def norm_year(val):
    v = norm(val)
    if isinstance(v, int) and 0 < v < 100:
        return v + 2000
    return v


def compute_accuracy(results, gt_all, test_files):
    stats = {t: defaultdict(int) for t in ("hard", "non_hard", "ALL")}

    for fn in test_files:
        p = results.get(fn)
        g = gt_all.get(fn)
        if p is None or g is None:
            continue

        raw_tier = g.get("difficulty_tier", "unknown")
        tier = "non_hard" if raw_tier in ("easy", "medium", "non_hard") else raw_tier
        for grp in (tier, "ALL"):
            stats[grp]["total"] += 1

        if "error" in p or "parse_error" in p:
            continue

        g_has = any(norm(g.get(k)) is not None for k in ("year", "month", "day"))
        if not g_has:
            continue

        for grp in (tier, "ALL"):
            stats[grp]["gt_has_date"] += 1

        p_y = norm_year(p.get("year"));  g_y = norm_year(g.get("year"))
        p_m = norm(p.get("month"));      g_m = norm(g.get("month"))
        p_d = norm(p.get("day"));        g_d = norm(g.get("day"))

        y_ok = (p_y == g_y)
        m_ok = (p_m == g_m)
        d_ok = (p_d == g_d) if g_d is not None else None

        for grp in (tier, "ALL"):
            if y_ok: stats[grp]["year_correct"] += 1
            if m_ok: stats[grp]["month_correct"] += 1
            if d_ok is not None:
                stats[grp]["day_total"] += 1
                if d_ok: stats[grp]["day_correct"] += 1
            if y_ok and m_ok:
                stats[grp]["ym_correct"] += 1
            if y_ok and m_ok and (d_ok or d_ok is None):
                stats[grp]["full_correct"] += 1

    return stats


def pct(c, n):
    return f"{c/n*100:.1f}%" if n else "N/A"


def print_stats(label, stats):
    print(f"\n{'='*64}")
    print(f"  {label}")
    h, n, a = stats["hard"], stats["non_hard"], stats["ALL"]
    print(f"  n_total : hard={h['total']}  non_hard={n['total']}  ALL={a['total']}")
    print(f"  n_gt    : hard={h['gt_has_date']}  non_hard={n['gt_has_date']}  ALL={a['gt_has_date']}")
    print(f"{'='*64}")
    rows = [
        ("year",       "year_correct",  "gt_has_date"),
        ("month",      "month_correct", "gt_has_date"),
        ("day",        "day_correct",   "day_total"),
        ("year+month", "ym_correct",    "gt_has_date"),
        ("full_exact", "full_correct",  "gt_has_date"),
    ]
    for name, key, denom_key in rows:
        print(f"  {name:12s}  hard={pct(h[key], h[denom_key]):7s}  "
              f"non_hard={pct(n[key], n[denom_key]):7s}  "
              f"ALL={pct(a[key], a[denom_key])}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, help="Path to GGUF model file")
    ap.add_argument("--mmproj",  required=True, help="Path to mmproj GGUF file")
    ap.add_argument("--backend", choices=["cpu", "vulkan"], default="cpu")
    ap.add_argument("--output",  required=True, help="Output JSON path for results")
    ap.add_argument("--port",    type=int, default=8765)
    ap.add_argument("--restart-interval", type=int, default=100,
                    help="Restart llama-server every N images (Vulkan only, 0=never)")
    args = ap.parse_args()

    with open(TEST_JSON, encoding="utf-8") as f:
        test_files = [e["filename"] for e in json.load(f)]
    with open(GT_FILE, encoding="utf-8") as f:
        gt_all = {e["filename"]: e for e in json.load(f)}

    # Resume support: keep successful entries; retry any with errors
    results = {}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for fn, entry in json.load(f).items():
                if "error" not in entry and "parse_error" not in entry:
                    results[fn] = entry
        print(f"Resuming: {len(results)} already done.")

    pending = [f for f in test_files if f not in results]
    print(f"Processing {len(pending)}/{len(test_files)} images  "
          f"[{args.backend}  {Path(args.model).name}]\n")

    if pending:
        restart_every = args.restart_interval if args.backend == "vulkan" else 0
        proc, base_url = start_server(args.model, args.mmproj, args.backend, args.port)
        try:
            wall0 = time.time()
            for i, filename in enumerate(pending, 1):
                # Restart server periodically on Vulkan to clear accumulated GPU state
                if restart_every and i > 1 and (i - 1) % restart_every == 0:
                    print(f"\n  [restart server at image {i}]")
                    stop_server(proc)
                    proc, base_url = start_server(args.model, args.mmproj, args.backend, args.port)

                image_path = os.path.join(TEST_FOLDER, filename)
                print(f"  [{i:>4}/{len(pending)}] {filename} ...", end=" ", flush=True)
                try:
                    raw, latency, gen_tokens = query_server(base_url, image_path)
                    parsed = extract_json(raw)
                    results[filename] = {
                        "filename": filename, "raw_response": raw,
                        "latency_s": round(latency, 3), "gen_tokens": gen_tokens,
                        **parsed,
                    }
                    print(f"✓  ({latency:.2f}s, {gen_tokens} tok)")
                except Exception as e:
                    results[filename] = {
                        "filename": filename, "parse_error": str(e),
                        "year": None, "month": None, "day": None,
                    }
                    print(f"✗  {str(e)[:80]}")

                if i % 20 == 0:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    elapsed = time.time() - wall0
                    remaining = (len(pending) - i) * (elapsed / i)
                    print(f"    [{i}/{len(pending)}] ETA {remaining/60:.1f} min")

        finally:
            stop_server(proc)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Speed summary
    lats = sorted(v["latency_s"] for v in results.values() if v.get("latency_s") is not None)
    errors = sum(1 for v in results.values() if "error" in v or "parse_error" in v)
    if lats:
        print(f"\nSpeed  : mean={sum(lats)/len(lats):.2f}s  "
              f"median={lats[len(lats)//2]:.2f}s  "
              f"p90={lats[int(0.9*len(lats))]:.2f}s  "
              f"throughput={len(lats)/sum(lats):.3f} img/s")
    print(f"Errors : {errors}/{len(results)}")

    stats = compute_accuracy(results, gt_all, test_files)
    label = f"{Path(args.model).stem}  [{args.backend}]"
    print_stats(label, stats)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
