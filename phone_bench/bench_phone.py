#!/usr/bin/env python3
"""
On-device latency (and accuracy check) for the expiration-date VLM, over adb.

Runs llama-mtmd-cli on the phone's CPU -- the Android build has no GPU backend,
so there is no hidden offload -- over a fixed, seeded subset of test images,
and records the per-phase split that llama.cpp reports:

    vit_ms      vision encoder, image -> visual tokens
    prefill_ms  LLM prompt eval, INCLUDING the vision encoder
    decode_ms   token generation
    total_ms    prefill + decode (model load excluded)

Works with any adb transport: phone on this machine's USB, an adb server
reverse-tunnelled over SSH (ssh -R 5037:127.0.0.1:5037 ...), or wireless adb.

Usage:
    python3 bench_phone.py --info            # device report only, pushes nothing
    python3 bench_phone.py                   # full run, default configs
    python3 bench_phone.py --n-images 5 --configs q4q4 --budgets 2048   # smoke test
"""
import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ADB = "/home/penghao/android-sdk/platform-tools/adb"
HERE = Path(__file__).resolve().parent
QWEN = Path("/home/penghao/qwen")
GGUF = QWEN / "gguf"
TEST_JSON = "/home/penghao/Dataset/test.json"
TEST_DIR = Path("/home/penghao/Dataset/test")
GT_FILE = "/home/penghao/Dataset/expiration_dates_details_true.json"
REMOTE = "/data/local/tmp/edb"

CONFIGS = {
    # id: (LLM gguf, mmproj gguf, total MB as reported in the paper)
    "q4q4": ("qwen35_groupA_a3_Q4_K_M.gguf", "qwen35_groupA_a3_mmproj_Q4_K_M.gguf", 577),
    "q3q4": ("qwen35_groupA_a3_Q3_K_M.gguf", "qwen35_groupA_a3_mmproj_Q4_K_M.gguf", 517),
    "f16":  ("qwen35_groupA_a3_f16.gguf",    "qwen35_groupA_a3_mmproj_f16.gguf",   1837),
}

PROMPT = re.search(r'PROMPT = """(.*?)"""', (QWEN / "eval_gguf.py").read_text(), re.S).group(1)


def adb(*args, check=True, capture=True, timeout=None):
    r = subprocess.run([ADB, *args], capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode != 0:
        sys.exit(f"adb {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout if capture else ""


def sh(cmd, timeout=None):
    return adb("shell", cmd, timeout=timeout)


def device_info():
    prop = lambda k: sh(f"getprop {k}").strip()
    feats = sh("grep -m1 -i Features /proc/cpuinfo || true").strip()   # absent on x86
    freqs = {}
    for line in sh("for c in /sys/devices/system/cpu/cpu[0-9]*; do "
                   "echo ${c##*cpu} $(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null); done").split("\n"):
        p = line.split()
        if len(p) == 2 and p[1].isdigit():
            freqs[int(p[0])] = int(p[1])
    mem_kb = int(re.search(r"MemTotal:\s+(\d+)", sh("cat /proc/meminfo")).group(1))
    return {
        "emulator": "1" in (prop("ro.kernel.qemu"), prop("ro.boot.qemu")),
        "n_cpus": int(sh("nproc").strip() or 0),
        "model": prop("ro.product.model"),
        "brand": prop("ro.product.brand"),
        "soc": prop("ro.soc.model") or prop("ro.board.platform") or prop("ro.hardware"),
        "android": prop("ro.build.version.release"),
        "abi": prop("ro.product.cpu.abi"),
        "ram_gb": round(mem_kb / 1024 / 1024, 1),
        "i8mm": "i8mm" in feats.split(),
        "dotprod": "asimddp" in feats.split(),
        "cpu_max_khz": freqs,
    }


def big_cores(freqs, want=4, n_cpus=0):
    """Fastest cores first; returns (core ids, taskset hex mask).
    Emulated CPUs expose no cpufreq, so fall back to the first `want` cores."""
    if not freqs:
        freqs = {c: 0 for c in range(max(n_cpus, 1))}
    order = sorted(freqs, key=lambda c: (-freqs[c], -c))[:want]
    mask = sum(1 << c for c in order)
    return sorted(order), format(mask, "x")


def pick_images(n, seed=42):
    files = [e["filename"] for e in json.load(open(TEST_JSON))]
    gt = {e["filename"]: e for e in json.load(open(GT_FILE))}
    hard = [f for f in files if gt[f].get("difficulty_tier") == "hard"]
    easy = [f for f in files if f not in set(hard)]
    rng = random.Random(seed)
    n_hard = round(n * len(hard) / len(files))            # keep the test-set tier ratio
    return sorted(rng.sample(hard, n_hard) + rng.sample(easy, n - n_hard)), gt


def norm(v):
    if v in (None, "null", ""):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def correct(pred, g):
    ny = lambda v: (norm(v) + 2000) if isinstance(norm(v), int) and 0 < norm(v) < 100 else norm(v)
    g_has = any(norm(g.get(k)) is not None for k in ("year", "month", "day"))
    if pred is None:
        return False
    if not g_has:
        return all(norm(pred.get(k)) is None for k in ("year", "month", "day"))
    gd = norm(g.get("day"))
    d_ok = (norm(pred.get("day")) == gd) if gd is not None else True
    return ny(pred.get("year")) == ny(g.get("year")) and norm(pred.get("month")) == norm(g.get("month")) and d_ok


def parse(log):
    f = lambda pat: (float(m.group(1)) if (m := re.search(pat, log)) else None)
    # The prompt contains an example null JSON; only trust text the model generated.
    # Generation starts at <think>, which is sometimes left unclosed.
    for marker in ("</think>", "<think>", "image decoded"):
        if marker in log:
            gen = log.rsplit(marker, 1)[1]
            break
    else:
        gen = log
    m = re.findall(r'\{"year".*?\}', gen)
    pred = None
    if m:
        try:
            pred = json.loads(m[0])
        except json.JSONDecodeError:
            pass
    return {
        "vit_ms": f(r"image slice encoded in (\d+) ms"),
        "img_tokens": f(r"n_tokens_batch = (\d+)"),
        "prefill_ms": f(r"prompt eval time =\s+([\d.]+) ms"),
        "prefill_tok": f(r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens"),
        "decode_ms": f(r"\s eval time =\s+([\d.]+) ms"),
        "gen_tok": f(r"\s eval time =\s+[\d.]+ ms /\s+(\d+) runs"),
        "total_ms": f(r"total time =\s+([\d.]+) ms"),
        "load_ms": f(r"load time =\s+([\d.]+) ms"),
        "pred": pred,
    }


def battery_temp():
    m = re.search(r"temperature: (\d+)", sh("dumpsys battery"))
    return int(m.group(1)) / 10 if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", action="store_true", help="print device report and exit")
    ap.add_argument("--configs", nargs="+", default=["q4q4", "q3q4", "f16"], choices=list(CONFIGS))
    ap.add_argument("--budgets", nargs="+", type=int, default=[2048],
                    help="image token budgets for every config")
    ap.add_argument("--sweep-config", default="q4q4",
                    help="config that additionally gets the budget sweep below")
    ap.add_argument("--sweep", nargs="+", type=int, default=[512, 1024])
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0, help="0 = number of big cores used")
    ap.add_argument("--cooldown", type=int, default=60, help="seconds idle between configs")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    devs = [l for l in adb("devices").splitlines()[1:] if l.strip().endswith("device")]
    if not devs:
        sys.exit("No device. Check `adb devices` -- USB debugging on, and the RSA prompt accepted.")
    info = device_info()
    cores, mask = big_cores(info["cpu_max_khz"], n_cpus=info["n_cpus"])
    threads = args.threads or len(cores)
    info.update(bench_cores=cores, taskset_mask=mask, threads=threads)
    print(json.dumps(info, indent=2))
    if info["emulator"]:
        print("\n" + "!" * 72 +
              "\n  EMULATOR DETECTED - functional test only. Timings measure the host"
              "\n  machine through a virtual CPU, NOT phone latency. Do not report them."
              "\n" + "!" * 72)
    if args.info:
        return

    if info["abi"] == "arm64-v8a":
        variant = "-i8mm" if info["i8mm"] else ""
    elif info["abi"] == "x86_64":
        variant = "-x86_64"
    else:
        sys.exit(f"Unsupported ABI {info['abi']}")
    print(f"\nbinary: llama-mtmd-cli{variant}   cores {cores} (mask 0x{mask})   threads {threads}")

    images, gt = pick_images(args.n_images)
    out = Path(args.out)
    if info["emulator"] and args.out == str(HERE / "results"):
        out = HERE / "results_emulator"
    out.mkdir(parents=True, exist_ok=True)
    (out / "device.json").write_text(json.dumps(info, indent=2))
    (out / "images.txt").write_text("\n".join(images) + "\n")

    # ── push ────────────────────────────────────────────────────────────────
    sh(f"mkdir -p {REMOTE}/img")
    adb("push", str(HERE / "bin" / f"llama-mtmd-cli{variant}"), f"{REMOTE}/llama-mtmd-cli")
    sh(f"chmod 755 {REMOTE}/llama-mtmd-cli")
    for c in args.configs:
        for fn in CONFIGS[c][:2]:
            if sh(f"[ -f {REMOTE}/{fn} ] && echo y || echo n").strip() != "y":
                print(f"pushing {fn} ...")
                adb("push", str(GGUF / fn), f"{REMOTE}/{fn}", capture=False)
    for fn in images:
        if sh(f"[ -f {REMOTE}/img/{fn} ] && echo y || echo n").strip() != "y":
            adb("push", str(TEST_DIR / fn), f"{REMOTE}/img/{fn}")
    (out / "prompt.txt").write_text(PROMPT)
    adb("push", str(out / "prompt.txt"), f"{REMOTE}/prompt.txt")

    # ── run ─────────────────────────────────────────────────────────────────
    plan = [(c, b) for c in args.configs for b in args.budgets]
    plan += [(args.sweep_config, b) for b in args.sweep
             if args.sweep_config in args.configs and b not in args.budgets]

    csv_path = out / "phone_runs.csv"
    fields = ["config", "budget", "image", "tier", "img_tokens", "vit_ms", "prefill_ms",
              "prefill_tok", "decode_ms", "gen_tok", "total_ms", "load_ms",
              "batt_c", "correct", "answer"]
    new = not csv_path.exists()
    fh = open(csv_path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()

    for i, (cfg, budget) in enumerate(plan):
        llm, mm, _ = CONFIGS[cfg]
        if i:
            print(f"\ncooldown {args.cooldown}s ...")
            time.sleep(args.cooldown)
        print(f"\n=== {cfg}  budget {budget} ===")
        for j, fn in enumerate(images, 1):
            cmd = (f"cd {REMOTE} && taskset {mask} ./llama-mtmd-cli -m {llm} --mmproj {mm} "
                   f"--image img/{fn} -p \"$(cat prompt.txt)\" -n 256 --temp 0 -t {threads} -c 4096 "
                   f"--image-min-tokens 64 --image-max-tokens {budget} 2>&1")
            t0 = time.time()
            log = sh(cmd, timeout=1800)
            r = parse(log)
            ok = correct(r["pred"], gt[fn])
            row = {"config": cfg, "budget": budget, "image": fn,
                   "tier": gt[fn].get("difficulty_tier"),
                   **{k: r[k] for k in ("img_tokens", "vit_ms", "prefill_ms", "prefill_tok",
                                        "decode_ms", "gen_tok", "total_ms", "load_ms")},
                   "batt_c": battery_temp(), "correct": int(ok),
                   "answer": json.dumps(r["pred"])}
            w.writerow(row)
            fh.flush()
            tot = r["total_ms"]
            print(f"  [{j:>2}/{len(images)}] {fn}  "
                  f"{(tot or 0)/1000:6.1f}s  (vit {(r['vit_ms'] or 0)/1000:5.1f}s, "
                  f"decode {(r['decode_ms'] or 0)/1000:4.1f}s)  {'ok' if ok else 'WRONG'}  "
                  f"wall {time.time()-t0:5.1f}s")
            if tot is None:
                (out / f"fail_{cfg}_{budget}_{fn}.log").write_text(log)
    fh.close()
    print(f"\nrows -> {csv_path}")
    print("summarise with:  python3 summarise.py")


if __name__ == "__main__":
    main()
