"""
Converts the Qwen HuggingFace model to GGUF and produces the same set of
quantized variants for both CPU and Vulkan backends.

Since GGUF is backend-agnostic, the same file is used for both — the backend
is selected at runtime via --n-gpu-layers (0 = CPU, -1 = all layers on Vulkan).

Workflow:
  1. convert_hf_to_gguf.py  →  base F16 GGUF  (sequential)
  2. Parallel threads        →  one GGUF per quant type

Usage:
  python convert_to_gguf.py                    # all quant types
  python convert_to_gguf.py q4_k_m q8_0       # selected types only
"""

import sys
import subprocess
import threading
from pathlib import Path

# --- CONFIGURATION ---
MODEL_PATH   = Path("/home/penghao/qwen/Qwen/Qwen3.5-0.8B")
LLAMA_DIR    = Path("/home/penghao/llama.cpp")
OUTPUT_DIR   = Path("/home/penghao/qwen/gguf")
MODEL_NAME   = "Qwen3.5-0.8B"

CONVERT_SCRIPT = LLAMA_DIR / "convert_hf_to_gguf.py"
LLAMA_QUANTIZE = LLAMA_DIR / "build/bin/llama-quantize"
LLAMA_CLI      = LLAMA_DIR / "build/bin/llama-cli"

BASE_GGUF  = OUTPUT_DIR / f"{MODEL_NAME}-f16.gguf"
MMPROJ_GGUF = OUTPUT_DIR / f"{MODEL_NAME}-mmproj-f16.gguf"

# Same quant types applied to both CPU and Vulkan backends
QUANT_TYPES = {
    "f16":    "f16",     # ~1.6 GB  full fp16
    "bf16":   "bf16",    # ~1.6 GB  bfloat16
    "q8_0":   "q8_0",    # ~0.9 GB  8-bit, near-lossless
    "q6_k":   "q6_k",    # ~0.7 GB  6-bit K-quant
    "q5_k_m": "q5_k_m", # ~0.6 GB  5-bit K-quant medium
    "q4_k_m": "q4_k_m", # ~0.5 GB  4-bit K-quant medium
    "q3_k_m": "q3_k_m", # ~0.4 GB  3-bit K-quant medium
    "q2_k":   "q2_k",   # ~0.3 GB  2-bit K-quant (smallest, most lossy)
}


def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def run_cmd(cmd, tag):
    log(tag, "$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in result.stdout.splitlines():
        log(tag, line)
    return result.returncode


def convert_to_f16():
    if BASE_GGUF.exists():
        print(f"\n[skip] Base GGUF already exists: {BASE_GGUF.name}\n")
    else:
        print(f"\n{'='*60}")
        print(f"  Converting HF model → {BASE_GGUF.name}")
        print(f"{'='*60}\n")
        rc = run_cmd(
            [sys.executable, str(CONVERT_SCRIPT),
             str(MODEL_PATH), "--outtype", "f16", "--outfile", str(BASE_GGUF)],
            "convert",
        )
        if rc != 0:
            print(f"ERROR: LLM conversion failed (exit {rc})")
            sys.exit(rc)

    if MMPROJ_GGUF.exists():
        print(f"[skip] mmproj GGUF already exists: {MMPROJ_GGUF.name}\n")
    else:
        print(f"\n{'='*60}")
        print(f"  Extracting vision encoder → {MMPROJ_GGUF.name}")
        print(f"{'='*60}\n")
        rc = run_cmd(
            [sys.executable, str(CONVERT_SCRIPT),
             str(MODEL_PATH), "--mmproj", "--outtype", "f16", "--outfile", str(MMPROJ_GGUF)],
            "mmproj",
        )
        if rc != 0:
            print(f"ERROR: mmproj conversion failed (exit {rc})")
            sys.exit(rc)


def build_quant(name, token, results: dict, lock: threading.Lock):
    out_path = OUTPUT_DIR / f"{MODEL_NAME}-{name}.gguf"
    tag = name.upper()

    if out_path.exists():
        log(tag, f"[skip] already exists: {out_path.name}")
        with lock:
            results[name] = {"path": out_path, "ok": True}
        return

    if name in ("f16", "bf16"):
        rc = run_cmd(
            [sys.executable, str(CONVERT_SCRIPT),
             str(MODEL_PATH), "--outtype", token, "--outfile", str(out_path)],
            tag,
        )
    else:
        rc = run_cmd(
            [str(LLAMA_QUANTIZE), str(BASE_GGUF), str(out_path), token],
            tag,
        )

    with lock:
        results[name] = {"path": out_path, "ok": rc == 0}
    if rc != 0:
        log(tag, f"ERROR: failed (exit {rc})")


def print_summary(results: dict):
    print(f"\n{'='*65}")
    print("  OUTPUT SUMMARY  (usable for both CPU and Vulkan backends)")
    print(f"{'='*65}")
    print(f"  {'Quant':<10}  {'File':<36}  {'Size (MB)':>9}  {'Build':>5}")
    print(f"  {'-'*63}")
    for name, info in results.items():
        p    = info["path"]
        size = f"{p.stat().st_size / 1024**2:.1f}" if p.exists() else "—"
        status = "OK" if info["ok"] else "FAIL"
        print(f"  {name:<10}  {p.name:<36}  {size:>9}  {status:>5}")
    print(f"{'='*65}")
    print()
    print("  Run on CPU:    --n-gpu-layers 0")
    print("  Run on Vulkan: --n-gpu-layers -1")
    print(f"\n  Example:")
    print(f"    {LLAMA_CLI} \\")
    print(f"      --model {OUTPUT_DIR}/{MODEL_NAME}-q4_k_m.gguf \\")
    print(f"      --n-gpu-layers -1 \\")
    print(f"      --prompt \"Hello\"")
    print()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    quant_args = [a for a in sys.argv[1:]]
    if quant_args:
        unknown = set(quant_args) - set(QUANT_TYPES)
        if unknown:
            print(f"Unknown quant types: {unknown}")
            print(f"Available: {list(QUANT_TYPES.keys())}")
            sys.exit(1)
        types = {k: v for k, v in QUANT_TYPES.items() if k in quant_args}
    else:
        types = QUANT_TYPES

    # Base F16 must exist before any llama-quantize jobs start
    needs_base = any(k not in ("f16", "bf16") for k in types)
    if needs_base:
        convert_to_f16()

    # Build all quant types in parallel
    print(f"\n{'='*60}")
    print(f"  Building {len(types)} GGUF(s) in parallel")
    print(f"{'='*60}\n")

    results = {}
    lock    = threading.Lock()
    threads = [
        threading.Thread(target=build_quant, args=(name, token, results, lock), daemon=True)
        for name, token in types.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print_summary(results)


if __name__ == "__main__":
    main()
