#!/usr/bin/env python3
"""
Bootstrap 95% CIs on full-date accuracy for key A3 configurations.
Resamples the 545 test images 1000x and reports mean ± CI.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

GT_FILE   = Path("/home/penghao/Dataset/expiration_dates_details_true.json")
TEST_JSON = Path("/home/penghao/Dataset/test.json")
GGUF_DIR  = Path("/home/penghao/qwen/gguf_results")

gt_all     = {e["filename"]: e for e in json.load(open(GT_FILE))}
test_files = [e["filename"] for e in json.load(open(TEST_JSON))]
N = len(test_files)

def norm(v):
    if v is None or v in ("null", ""): return None
    try: return int(v)
    except: return v

def norm_year(v):
    v = norm(v)
    return v + 2000 if isinstance(v, int) and 0 < v < 100 else v

def is_correct(p, g):
    if p is None or g is None: return None
    if "error" in p or "parse_error" in p: return False
    g_has = any(norm(g.get(k)) is not None for k in ("year","month","day"))
    if not g_has:
        return all(norm(p.get(k)) is None for k in ("year","month","day"))
    py = norm_year(p.get("year")); gy = norm_year(g.get("year"))
    pm = norm(p.get("month"));     gm = norm(g.get("month"))
    pd = norm(p.get("day"));       gd = norm(g.get("day"))
    dok = (pd == gd) if gd is not None else None
    return (py == gy) and (pm == gm) and (dok or dok is None)

def load(path):
    r = json.load(open(path))
    return {e["filename"]: e for e in r} if isinstance(r, list) else r

def per_image_scores(results):
    scores = []
    for fn in test_files:
        p = results.get(fn)
        g = gt_all.get(fn)
        c = is_correct(p, g)
        scores.append(1 if c else 0)
    return np.array(scores, dtype=float)

def bootstrap_ci(scores, n_boot=1000, alpha=0.05, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    mean = scores.mean() * 100
    boot = np.array([rng.choice(scores, size=len(scores), replace=True).mean()
                     for _ in range(n_boot)]) * 100
    lo, hi = np.percentile(boot, [alpha/2*100, (1-alpha/2)*100])
    return mean, lo, hi

def delta_ci(scores_a, scores_b, n_boot=1000, alpha=0.05, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    diff = (scores_a - scores_b)
    observed = diff.mean() * 100
    idx = np.arange(len(scores_a))
    boot_diffs = []
    for _ in range(n_boot):
        i = rng.choice(idx, size=len(idx), replace=True)
        boot_diffs.append(diff[i].mean() * 100)
    boot_diffs = np.array(boot_diffs)
    lo, hi = np.percentile(boot_diffs, [alpha/2*100, (1-alpha/2)*100])
    p_val = (np.sum(boot_diffs <= 0) / n_boot) if observed > 0 else (np.sum(boot_diffs >= 0) / n_boot)
    return observed, lo, hi, p_val

rng = np.random.default_rng(42)
N_BOOT = 1000

CONFIGS = [
    # label, path
    ("A3 F16+F16 (baseline)",          "qwen35_groupA_a3_f16_cpu.json"),
    ("A3 F16+Q8_0",                    "qwen35_groupA_a3_Q8_0_cpu.json"),
    ("A3 F16+Q5_K_M",                  "qwen35_groupA_a3_Q5_K_M_cpu.json"),
    ("A3 F16+Q4_K_M  ← best",         "qwen35_groupA_a3_Q4_K_M_cpu.json"),
    ("A3 F16+Q3_K_M",                  "qwen35_groupA_a3_Q3_K_M_cpu.json"),
    ("A3 Q8_0+Q8_0",                   "qwen35_groupA_a3_Q8_0_mmQ8_0_cpu.json"),
    ("A3 Q5_K_M+Q5_K_M",               "qwen35_groupA_a3_Q5_K_M_mmQ5_K_M_cpu.json"),
    ("A3 Q4_K_M+Q4_K_M",               "qwen35_groupA_a3_Q4_K_M_mmQ4_K_M_cpu.json"),
    ("A3 Q3_K_M+Q3_K_M  ← collapse",  "qwen35_groupA_a3_Q3_K_M_mmQ3_K_M_cpu.json"),
    ("A3 Q3_K_M mmproj + F16 LLM",    "qwen35_groupA_a3_f16_mmQ3_K_M_cpu.json"),
    ("A3 Q4_K_M mmproj + F16 LLM",    "qwen35_groupA_a3_f16_mmQ4_K_M_cpu.json"),
]

print(f"\n{'='*78}")
print(f"  Bootstrap 95% CIs  (n={N} images, {N_BOOT} resamples)")
print(f"{'='*78}")
print(f"  {'Config':<38} {'ALL%':>6}  {'95% CI':>15}  {'±':>5}")
print(f"  {'-'*76}")

scores_map = {}
for label, fname in CONFIGS:
    path = GGUF_DIR / fname
    if not path.exists():
        print(f"  {label:<38}  MISSING: {fname}")
        continue
    s = per_image_scores(load(path))
    scores_map[label] = s
    mean, lo, hi = bootstrap_ci(s, N_BOOT, rng=rng)
    half = (hi - lo) / 2
    print(f"  {label:<38} {mean:>6.1f}  [{lo:>5.1f}, {hi:>5.1f}]  ±{half:.1f}")

print(f"\n{'='*78}")
print(f"  Pairwise delta tests (A vs B, observed diff, 95% CI of diff, p-value)")
print(f"{'='*78}")

PAIRS = [
    ("A3 F16+Q4_K_M  ← best",    "A3 Q4_K_M+Q4_K_M",
     "F16 mmproj vs Q4 mmproj  (LLM fixed at Q4)"),
    ("A3 F16+F16 (baseline)",      "A3 F16+Q4_K_M  ← best",
     "F16 LLM vs Q4 LLM  (mmproj fixed at F16)"),
    ("A3 F16+Q4_K_M  ← best",    "A3 Q3_K_M+Q3_K_M  ← collapse",
     "Best Q4 vs Q3 collapse"),
    ("A3 Q3_K_M mmproj + F16 LLM","A3 F16+F16 (baseline)",
     "Q3 mmproj alone vs F16 baseline"),
    ("A3 Q4_K_M mmproj + F16 LLM","A3 F16+F16 (baseline)",
     "Q4 mmproj alone vs F16 baseline"),
]

print(f"\n  {'Comparison':<48} {'Δ%':>6}  {'95% CI':>15}  {'p':>6}")
print(f"  {'-'*76}")
for a_label, b_label, desc in PAIRS:
    if a_label not in scores_map or b_label not in scores_map:
        print(f"  {desc:<48}  (missing data)")
        continue
    obs, lo, hi, pv = delta_ci(scores_map[a_label], scores_map[b_label], N_BOOT, rng=rng)
    sig = "**" if pv < 0.05 else "  "
    print(f"  {desc:<48} {obs:>+6.1f}  [{lo:>+5.1f}, {hi:>+5.1f}]  {pv:.3f}{sig}")

print(f"\n  ** p < 0.05 (one-sided bootstrap)")
print()
