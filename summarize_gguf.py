#!/usr/bin/env python3
"""
Print a summary table of all GGUF eval results.
Run after run_gguf_eval.sh completes.

Usage:
    python3 /home/penghao/qwen/summarize_gguf.py
"""
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("/home/penghao/qwen/gguf_results")
GT_FILE     = Path("/home/penghao/Dataset/expiration_dates_details_true.json")
TEST_JSON   = Path("/home/penghao/Dataset/test.json")

with open(GT_FILE, encoding="utf-8") as f:
    gt_all = {e["filename"]: e for e in json.load(f)}
with open(TEST_JSON, encoding="utf-8") as f:
    test_files = [e["filename"] for e in json.load(f)]


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


def pct(c, n):
    return f"{c/n*100:.1f}" if n else "N/A"


def score(results):
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
            stats[grp]["gt"] += 1
        p_y = norm_year(p.get("year"));  g_y = norm_year(g.get("year"))
        p_m = norm(p.get("month"));      g_m = norm(g.get("month"))
        p_d = norm(p.get("day"));        g_d = norm(g.get("day"))
        y_ok = (p_y == g_y)
        m_ok = (p_m == g_m)
        d_ok = (p_d == g_d) if g_d is not None else None
        for grp in (tier, "ALL"):
            if y_ok and m_ok and (d_ok or d_ok is None):
                stats[grp]["full"] += 1
            if y_ok and m_ok:
                stats[grp]["ym"] += 1
    return stats


def mean_latency(results):
    lats = [v["latency_s"] for v in results.values() if v.get("latency_s") is not None]
    return sum(lats) / len(lats) if lats else None


# ── Collect all result files ──────────────────────────────────────────────────
MODEL_ORDER = ["f16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
BACKEND_ORDER = ["cpu", "vulkan"]

print(f"\n{'='*90}")
print(f"  GGUF Evaluation Summary — finetuned Qwen3.5-0.8B")
print(f"{'='*90}")
print(f"  {'Model':<22} {'Backend':>8}  {'ALL full%':>10}  {'hard full%':>10}  {'nonhrd full%':>12}  {'lat(s)':>7}  {'n':>5}")
print(f"  {'-'*22} {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*7}  {'-'*5}")

for quant in MODEL_ORDER:
    for backend in BACKEND_ORDER:
        fname = RESULTS_DIR / f"qwen35_finetuned_{quant}_{backend}.json"
        if not fname.exists():
            print(f"  {'qwen35_finetuned_'+quant:<22} {backend:>8}  {'(missing)':>10}")
            continue
        with open(fname, encoding="utf-8") as f:
            results = json.load(f)
        s = score(results)
        lat = mean_latency(results)
        n = s["ALL"]["total"]
        print(f"  {'qwen35_finetuned_'+quant:<22} {backend:>8}"
              f"  {pct(s['ALL']['full'], s['ALL']['gt']):>10}"
              f"  {pct(s['hard']['full'], s['hard']['gt']):>10}"
              f"  {pct(s['non_hard']['full'], s['non_hard']['gt']):>12}"
              f"  {lat:>7.2f}"
              f"  {n:>5}")
    print()

print(f"{'='*90}")
print(f"  Note: full_exact = year+month correct, day correct where GT has day")
