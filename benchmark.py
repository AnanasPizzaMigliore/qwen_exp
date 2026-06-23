"""
Benchmarks all GGUF quantizations × CPU/Vulkan backends × thinking on/off
against the product image dataset using the same prompt as qwen.py.

Uses llama-server so the model is loaded ONCE per config, then all images
are processed as HTTP requests — avoids 16s model reload per image.

Per-config time:  ~model_load  +  N_images × inference_time
Full run estimate: 32 configs × (load + 665 × ~2s) ≈ 12–20 hours
                   with --limit 50: ≈ 1–2 hours

Usage:
  python benchmark.py                              # all 32 configs, all 665 images
  python benchmark.py --limit 50                  # quick run (50 images per config)
  python benchmark.py --quants q4_k_m q8_0        # specific quants
  python benchmark.py --backends vulkan            # one backend
  python benchmark.py --thinking on               # one thinking mode
  python benchmark.py --quants q4_k_m --backends vulkan --thinking on --limit 50
"""

import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLAMA_DIR   = Path("/home/penghao/llama.cpp")
GGUF_DIR    = Path("/home/penghao/qwen/gguf")
MODEL_NAME  = "Qwen3.5-0.8B"
MMPROJ_GGUF = GGUF_DIR / f"{MODEL_NAME}-mmproj-f16.gguf"

IMAGE_DIR   = Path("/home/penghao/qwen/Products-Real/evaluation/images")
GT_JSON     = Path("/home/penghao/qwen/Products-Real/evaluation/annotations.json")
RESULTS_DIR = Path("/home/penghao/qwen/benchmark_results")

LLAMA_SERVER = LLAMA_DIR / "build/bin/llama-server"

QUANT_TYPES  = ["f16", "bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]
BACKENDS     = {"cpu": 0, "vulkan": -1}
THINKING     = {"on": True, "off": False}

SERVER_PORT     = 8088
SERVER_HOST     = "127.0.0.1"
SERVER_READY_TIMEOUT = 120   # seconds to wait for server to start
MAX_NEW_TOKENS  = 512

PROMPT = (
    "You are a strict OCR bot. Find the EXPIRATION date on this product label and return it as JSON.\n\n"
    "STEP 1 — FIND THE RIGHT DATE:\n"
    "- Expiration labels (use this date): EXP, EXPIRY, EXPIRATION, USE BY, USE BEFORE, BEST BY, BEST BEFORE, BB, BBE, SELL BY, BFD, BEST IF USED BY.\n"
    "- Manufacturing labels (IGNORE this date): MFG, MFGD, MFD, MANUFACTURED, PROD, PRD, DOM, PKD, PACKED, MAN.\n"
    "- If there is only ONE date and no label, assume it is the expiration date.\n"
    "- If there are TWO unlabeled dates, the LATER date is typically the expiration date.\n\n"
    "STEP 2 — TRANSCRIBE EXACTLY (never guess or convert):\n"
    "- 'year': Copy digits exactly. '21' stays '21', '2024' stays '2024'. If absent → null.\n"
    "- 'month': Copy exactly (number or text). '08' stays '08', 'MAR' stays 'MAR'. If absent → null.\n"
    "- 'day': The day of the month (a number between 1–31). If no day is printed (e.g. only month+year visible) → null. NEVER copy the year or month into this field.\n"
    "- 'label': The expiration label text (e.g. 'EXP', 'Best By'). If none → null.\n\n"
    "STEP 3 — OUTPUT:\n"
    "Return ONLY a raw JSON object. No markdown, no explanation.\n"
    '{"year": ..., "month": ..., "day": ..., "label": ...}'
)

# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def start_server(gguf: Path, gpu_layers: int, thinking: bool) -> subprocess.Popen:
    cmd = [
        str(LLAMA_SERVER),
        "--model",        str(gguf),
        "--mmproj",       str(MMPROJ_GGUF),
        "--n-gpu-layers", str(gpu_layers),
        "--port",         str(SERVER_PORT),
        "--host",         SERVER_HOST,
        "--jinja",
        "--reasoning",    "on" if thinking else "off",
        "--n-predict",    str(MAX_NEW_TOKENS),
        "--log-disable",
        "--no-warmup",
    ]
    if gpu_layers != 0:
        cmd.append("--mmproj-offload")
    else:
        cmd.append("--no-mmproj-offload")

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def wait_for_server(timeout: int = SERVER_READY_TIMEOUT) -> bool:
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                data = json.loads(r.read())
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def stop_server(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_image(image_path: Path) -> dict:
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    ext     = image_path.suffix.lstrip(".")
    data_url = f"data:image/{ext};base64,{img_b64}"

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",      "text": PROMPT},
                ],
            }
        ],
        "max_tokens":   MAX_NEW_TOKENS,
        "temperature":  0,
        "stream":       False,
    }).encode()

    url = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions"
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp    = json.loads(r.read())
            elapsed = time.perf_counter() - t0
    except Exception as e:
        return {"elapsed": time.perf_counter() - t0, "tps": None,
                "parsed": {"error": str(e)}}

    content = resp["choices"][0]["message"].get("content", "")
    usage   = resp.get("usage", {})
    tps     = None
    if usage.get("eval_count") and usage.get("eval_duration"):
        tps = round(usage["eval_count"] / (usage["eval_duration"] / 1e9), 1)

    return {"elapsed": round(time.perf_counter() - t0, 3),
            "tps":     tps,
            "parsed":  parse_output(content)}

# ---------------------------------------------------------------------------
# Output parsing  (mirrors qwen.py)
# ---------------------------------------------------------------------------

def parse_output(raw: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = text.replace("```json", "").replace("```", "").strip()

    matches = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
    if not matches:
        return {"error": "No JSON block found", "raw": raw[:200]}

    content = matches[-1]
    content = re.sub(r":\s*(\d+)(?=\s*[,}])", r': "\1"', content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw": raw[:200]}

    if isinstance(parsed.get("month"), str) and not parsed["month"].isdigit():
        parsed["month"] = parsed["month"].upper()

    yr = str(parsed.get("year") or "").strip()
    dy = str(parsed.get("day")  or "").strip()
    if dy not in ("None", "null", "") and yr not in ("None", "null", ""):
        if dy == yr or (len(yr) == 4 and dy == yr[-2:]):
            parsed["day"] = None

    return parsed

# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def _norm(val):
    if val is None:
        return None
    s = str(val).strip()
    return None if s.lower() in ("none", "null", "") else s

def extract_gt(gt_info, cls_type):
    result = {"year": None, "month": None, "day": None}
    for ann in gt_info.get("ann", []):
        if ann.get("cls") == cls_type:
            for dmy in ann.get("dmy_ann", []):
                k = dmy.get("cls")
                if k in result:
                    result[k] = str(dmy.get("transcription", "")).strip()
            break
    return result

def evaluate(predictions: dict, gt_data: dict) -> dict:
    total = correct = parse_errors = prod_confusions = 0
    for fname, pred in predictions.items():
        if fname not in gt_data:
            continue
        total += 1
        if "error" in pred:
            parse_errors += 1
            continue
        gt_exp  = extract_gt(gt_data[fname], "exp")
        gt_prod = extract_gt(gt_data[fname], "date")
        if _norm(pred.get("year")) == _norm(gt_exp["year"]) and \
           _norm(pred.get("month")) == _norm(gt_exp["month"]) and \
           _norm(pred.get("day")) == _norm(gt_exp["day"]):
            correct += 1
        elif any(_norm(gt_prod[k]) for k in gt_prod) and \
             _norm(pred.get("year")) == _norm(gt_prod["year"]) and \
             _norm(pred.get("month")) == _norm(gt_prod["month"]) and \
             _norm(pred.get("day")) == _norm(gt_prod["day"]):
            prod_confusions += 1
    accuracy = correct / total * 100 if total else 0
    return {"total": total, "correct": correct, "accuracy": round(accuracy, 2),
            "parse_errors": parse_errors, "prod_confusions": prod_confusions,
            "other_failures": total - correct - parse_errors - prod_confusions}

# ---------------------------------------------------------------------------
# Run one config
# ---------------------------------------------------------------------------

def run_config(quant, backend, thinking, images, gt_data, out_dir):
    tag      = f"{quant}_{backend}_think{'on' if thinking else 'off'}"
    gguf     = GGUF_DIR / f"{MODEL_NAME}-{quant}.gguf"
    gpu_layers = BACKENDS[backend]

    if not gguf.exists():
        print(f"[{tag}] SKIP — GGUF not found: {gguf.name}")
        return None

    pred_file    = out_dir / f"{tag}_predictions.json"
    metrics_file = out_dir / f"{tag}_metrics.json"

    if metrics_file.exists():
        print(f"[{tag}] SKIP — already completed")
        return json.loads(metrics_file.read_text())

    print(f"\n{'='*65}")
    print(f"  {tag}  ({len(images)} images)")
    print(f"  GGUF: {gguf.name}  |  GPU layers: {gpu_layers}  |  Thinking: {thinking}")
    print(f"{'='*65}")

    print("  Starting server...", end=" ", flush=True)
    t_load = time.time()
    proc   = start_server(gguf, gpu_layers, thinking)

    if not wait_for_server():
        print("FAILED (timeout)")
        stop_server(proc)
        return None
    load_time = round(time.time() - t_load, 1)
    print(f"ready in {load_time}s")

    predictions = {}
    latencies   = []
    tps_list    = []

    try:
        for i, img_path in enumerate(images, 1):
            fname = img_path.name
            print(f"  [{i:>4}/{len(images)}] {fname}", end="  ", flush=True)
            res = infer_image(img_path)
            predictions[fname] = res["parsed"]
            latencies.append(res["elapsed"])
            if res["tps"]:
                tps_list.append(res["tps"])
            status = "ERR" if "error" in res["parsed"] else "OK "
            print(f"{status}  {res['elapsed']:.2f}s"
                  + (f"  {res['tps']:>6.1f} tok/s" if res["tps"] else ""), flush=True)
    finally:
        stop_server(proc)

    acc = evaluate(predictions, gt_data)
    metrics = {
        "config": {
            "quant": quant, "backend": backend, "thinking": thinking,
            "gpu_layers": gpu_layers, "gguf": gguf.name, "n_images": len(images),
        },
        "performance": {
            "model_load_s":  load_time,
            "avg_latency_s": round(sum(latencies) / len(latencies), 3),
            "total_time_s":  round(sum(latencies), 1),
            "avg_tps":       round(sum(tps_list) / len(tps_list), 1) if tps_list else None,
        },
        "accuracy": acc,
    }

    pred_file.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
    metrics_file.write_text(json.dumps(metrics, indent=2))
    print(f"\n  Accuracy: {acc['accuracy']:.2f}%  |"
          f"  Avg: {metrics['performance']['avg_latency_s']:.2f}s/img  |"
          f"  Load: {load_time}s")
    return metrics

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_metrics):
    print(f"\n{'='*95}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*95}")
    print(f"  {'Config':<38} {'Accuracy':>9} {'Avg s/img':>10} {'tok/s':>8} {'Load s':>7} {'Err':>5}")
    print(f"  {'-'*93}")
    for m in sorted(all_metrics, key=lambda x: -x["accuracy"]["accuracy"]):
        c    = m["config"]
        perf = m["performance"]
        acc  = m["accuracy"]
        name = f"{c['quant']}_{c['backend']}_think{'on' if c['thinking'] else 'off'}"
        tps  = f"{perf['avg_tps']:.1f}" if perf["avg_tps"] else "—"
        print(f"  {name:<38} {acc['accuracy']:>8.2f}%  {perf['avg_latency_s']:>9.3f}s"
              f"  {tps:>7}  {perf['model_load_s']:>6.1f}  {acc['parse_errors']:>5}")
    print(f"{'='*95}\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    def get_flag(flag, default, multi=False):
        if flag not in args:
            return default
        idx = args.index(flag) + 1
        if multi:
            vals = []
            while idx < len(args) and not args[idx].startswith("--"):
                vals.append(args[idx]); idx += 1
            return vals or default
        return args[idx] if idx < len(args) else default

    quants   = get_flag("--quants",   QUANT_TYPES,    multi=True)
    backends = get_flag("--backends", list(BACKENDS), multi=True)
    thinking = get_flag("--thinking", list(THINKING), multi=True)
    limit    = int(get_flag("--limit", 0))

    if not MMPROJ_GGUF.exists():
        print(f"ERROR: mmproj not found: {MMPROJ_GGUF}")
        print("Run: conda run -n llama python convert_to_gguf.py")
        sys.exit(1)

    images = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.png"))
    if limit:
        images = images[:limit]

    gt_data = json.loads(GT_JSON.read_text())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_configs = len(quants) * len(backends) * len(thinking)
    print(f"\nBenchmark plan: {len(quants)} quants × {len(backends)} backends"
          f" × {len(thinking)} thinking modes = {n_configs} configs")
    print(f"Images per config: {len(images)}")
    print(f"Results dir: {RESULTS_DIR}\n")

    all_metrics = []
    for quant in quants:
        for backend in backends:
            for think_mode in thinking:
                think_bool = THINKING.get(think_mode, bool(think_mode))
                metrics = run_config(quant, backend, think_bool, images, gt_data, RESULTS_DIR)
                if metrics:
                    all_metrics.append(metrics)

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2))

    if all_metrics:
        print_summary(all_metrics)
    print(f"Full results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
