#!/usr/bin/env python3
"""
Compare test-set accuracy across all models:
  - Qwen3.6          (zero-shot, cloud API results)
  - Qwen3.5-0.8B     (zero-shot, HuggingFace inference)
  - Finetuned + GGUF (F16 / Q8_0 / Q5_K_M / Q4_K_M / Q3_K_M × CPU / Vulkan)

Usage:
    python3 /home/penghao/qwen/compare_all.py
"""
import json
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
GT_FILE      = Path("/home/penghao/Dataset/expiration_dates_details_true.json")
TEST_JSON    = Path("/home/penghao/Dataset/test.json")
QWEN36_FILE  = Path("/home/penghao/Dataset/expiration_dates_details_qwen3.6.json")
QWEN35_FILE  = Path("/home/penghao/Dataset/qwen35_test_results.json")
GGUF_DIR     = Path("/home/penghao/qwen/gguf_results")

# ── Ground truth & test split ─────────────────────────────────────────────────
with open(GT_FILE) as f:
    gt_all = {e["filename"]: e for e in json.load(f)}
with open(TEST_JSON) as f:
    test_files = [e["filename"] for e in json.load(f)]


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def score(results):
    """
    Compute accuracy stats over the test split.
    results: dict  filename -> entry  (must have year/month/day keys)
    Returns nested dict: tier -> counter_name -> int
    """
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

        # entries with hard errors are counted as misses, not excluded
        if "error" in p or "parse_error" in p:
            continue

        g_has = any(norm(g.get(k)) is not None for k in ("year", "month", "day"))

        if not g_has:
            # GT is null — correct if model also outputs null (no hallucination)
            p_null = all(norm(p.get(k)) is None for k in ("year", "month", "day"))
            for grp in (tier, "ALL"):
                stats[grp]["null_total"] += 1
                if p_null:
                    stats[grp]["null_correct"] += 1
                    stats[grp]["full"] += 1   # counts toward overall accuracy
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
            if y_ok:
                stats[grp]["year"] += 1
            if m_ok:
                stats[grp]["month"] += 1
            if d_ok is not None:
                stats[grp]["day_total"] += 1
                if d_ok:
                    stats[grp]["day"] += 1
            if y_ok and m_ok:
                stats[grp]["ym"] += 1
            if y_ok and m_ok and (d_ok or d_ok is None):
                stats[grp]["full"] += 1

    return stats


def mean_latency(results):
    lats = [v["latency_s"] for v in results.values() if isinstance(v.get("latency_s"), (int, float))]
    return sum(lats) / len(lats) if lats else None


def pct(c, n):
    return f"{c / n * 100:.1f}" if n else "N/A"


# ── Load all result sets ──────────────────────────────────────────────────────
def load_json_dict(path):
    """Load a JSON file that is either a list of entries or a filename->entry dict."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {e["filename"]: e for e in raw}
    return raw  # already a dict


rows = []  # (label, results_dict, backend_tag)

# Qwen3.6 zero-shot
if QWEN36_FILE.exists():
    rows.append(("Qwen3.6 (zero-shot)", load_json_dict(QWEN36_FILE), "—"))

# Qwen3.5-0.8B zero-shot
if QWEN35_FILE.exists():
    rows.append(("Qwen3.5-0.8B (zero-shot)", load_json_dict(QWEN35_FILE), "—"))

# Finetuned GGUF: CPU then Vulkan
QUANTS = ["f16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
for backend in ("cpu", "vulkan"):
    for quant in QUANTS:
        path = GGUF_DIR / f"qwen35_finetuned_{quant}_{backend}.json"
        if path.exists():
            label = f"Finetuned {quant}"
            rows.append((label, load_json_dict(path), backend))

# Group A3 GGUF: CPU then Vulkan
for backend in ("cpu", "vulkan"):
    for quant in QUANTS:
        path = GGUF_DIR / f"qwen35_groupA_a3_{quant}_{backend}.json"
        if path.exists():
            label = f"A3 {quant}"
            rows.append((label, load_json_dict(path), backend))

# ── Print table ───────────────────────────────────────────────────────────────
W_LABEL = 26
HDR = (f"  {'Model':<{W_LABEL}} {'Backend':>8}  "
       f"{'ALL%':>6}  {'hard%':>6}  {'non_hard%':>9}  "
       f"{'no-hallu%':>9}  {'year%':>6}  {'month%':>6}  {'day%':>6}  {'lat(s)':>7}")
SEP = "  " + "-" * (len(HDR) - 2)

print(f"\n{'=' * len(HDR)}")
print(f"  Test-set evaluation — all models")
print(f"  n=545 total  (520 with GT date + 25 null-date for hallucination test)")
print(f"{'=' * len(HDR)}")
print(HDR)
print(SEP)

prev_backend = None
for label, results, backend in rows:
    if prev_backend is not None and backend != prev_backend:
        print()  # blank line between backend groups
    prev_backend = backend

    s = score(results)
    lat = mean_latency(results)
    a, h, n = s["ALL"], s["hard"], s["non_hard"]

    lat_str = f"{lat:.2f}" if lat else "  —  "

    # full% over all 545 (gt-positive correct + null-correct)
    full_total = a["full"]
    denom_all  = a["total"]  # 545

    print(f"  {label:<{W_LABEL}} {backend:>8}  "
          f"{pct(full_total, denom_all):>6}  "
          f"{pct(h['full'], h['total']):>6}  "
          f"{pct(n['full'], n['total']):>9}  "
          f"{pct(a['null_correct'], a['null_total']):>9}  "
          f"{pct(a['year'], a['gt']):>6}  "
          f"{pct(a['month'], a['gt']):>6}  "
          f"{pct(a['day'], a['day_total']):>6}  "
          f"{lat_str:>7}")

print(f"{'=' * len(HDR)}")
print(f"  ALL%/hard%/non_hard% = full_exact / n_total (545/439/106)")
print(f"  no-hallu% = model outputs null when GT is null (over 25 null images)")
print(f"  full_exact = year+month correct, day correct where GT has day")
print()

# ── Per-tier detail for each model ────────────────────────────────────────────
print(f"{'=' * len(HDR)}")
print(f"  Detailed breakdown by tier")
print(f"{'=' * len(HDR)}")
for label, results, backend in rows:
    s = score(results)
    a, h, n = s["ALL"], s["hard"], s["non_hard"]
    tag = f"{label} [{backend}]"
    print(f"\n  {tag}")
    print(f"  {'':4s} {'tier':<10} {'n_total':>8} {'n_gt':>6} {'no-hal':>6} "
          f"{'year%':>6} {'month%':>6} {'day%':>6} {'ym%':>6} {'full%':>6}")
    for tier_name, ts in (("ALL", a), ("hard", h), ("non_hard", n)):
        print(f"  {'':4s} {tier_name:<10} "
              f"{ts['total']:>8} {ts['gt']:>6} "
              f"{pct(ts['null_correct'], ts['null_total']):>6} "
              f"{pct(ts['year'], ts['gt']):>6} "
              f"{pct(ts['month'], ts['gt']):>6} "
              f"{pct(ts['day'], ts['day_total']):>6} "
              f"{pct(ts['ym'], ts['gt']):>6} "
              f"{pct(ts['full'], ts['total']):>6}")
