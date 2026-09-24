#!/usr/bin/env python3
"""
R5 — is the fine-tuned reader reading the year or guessing it?

1. Products-Real by printed year (<= 2025 vs >= 2026), as requested. Products-Real has
   almost no 2026+ labels, so the >= 2026 cell is reported with its n and cannot carry
   a conclusion on its own.
2. In-domain test split by printed year: 2026-2027, where 946 of 1,268 dated training
   labels sit, vs 2028-2031. A reader that guesses from the prior should do worse on
   2028+ and should pull wrong years back toward 2026-2027.
3. Products-Real images where A3 got only the year wrong: how often the zero-shot base
   model read that year correctly, i.e. whether the year was legible.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/penghao/qwen")
import eval_products_real as pr  # noqa: E402

ROOT = Path("/home/penghao/qwen")
GT = {e["filename"]: e for e in json.load(open("/home/penghao/Dataset/expiration_dates_details_true.json"))}
TEST = [e["filename"] for e in json.load(open("/home/penghao/Dataset/test.json"))]
TRAIN = {e["filename"] for e in json.load(open("/home/penghao/Dataset/train.json"))}


def load(p):
    r = json.load(open(p))
    return {e["filename"]: e for e in r} if isinstance(r, list) else r


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, den = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / den, z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (c - h) * 100, (c + h) * 100


def line(label, k, n, extra=""):
    lo, hi = wilson(k, n)
    return f"  {label:<34} {k:>3}/{n:<3} {k/max(n,1)*100:5.1f}%  [{lo:4.1f}, {hi:4.1f}]{extra}"


# ── 1. Products-Real by printed year ─────────────────────────────────────────
ann = json.load(open(pr.PR_DIR / "annotations.json"))
print("1. PRODUCTS-REAL, full-date accuracy by printed year")
for name in ("a3_f16_vulkan", "a3_q4q4_vulkan"):
    res = json.load(open(ROOT / f"gguf_results/products_real/{name}.json"))
    print(f" {name}")
    for label, keep in (("printed year <= 2025", lambda y: y is not None and y <= 2025),
                        ("printed year >= 2026", lambda y: y is not None and y >= 2026),
                        ("no printed year", lambda y: y is None)):
        fs = [f for f in ann if keep(pr.gt_of(ann[f])["year"])]
        k = sum(pr.correct(res[f], pr.gt_of(ann[f])) for f in fs)
        print(line(label, k, len(fs)))

# ── 2. In-domain test split by printed year ───────────────────────────────────
tr_years = collections.Counter(pr.year(GT[f].get("year")) for f in TRAIN if pr.year(GT[f].get("year")))
print(f"\n2. IN-DOMAIN TEST SPLIT by printed year   (training label years: {dict(sorted(tr_years.items()))})")
CONFIGS = [("A3 F16 (GGUF CPU)", "gguf_results/qwen35_groupA_a3_f16_cpu.json"),
           ("A3 whole-model Q4 (GGUF CPU)", "gguf_results/qwen35_groupA_a3_Q4_K_M_mmQ4_K_M_cpu.json"),
           ("A1' (native, epoch 10)", "epoch_results_groupA_a1p/test_results_epoch_10.json")]
for cname, path in CONFIGS:
    res = load(ROOT / path)
    print(f" {cname}")
    for label, lo_y, hi_y in (("printed 2026-2027", 2026, 2027), ("printed 2028-2031", 2028, 2031)):
        fs = [f for f in TEST if lo_y <= (pr.year(GT[f].get("year")) or 0) <= hi_y]
        full = sum(pr.correct(res.get(f, {}), {"year": pr.year(GT[f].get("year")),
                                               "month": pr.to_int(GT[f].get("month")),
                                               "day": pr.to_int(GT[f].get("day"))}) for f in fs)
        yr = sum(pr.year(res.get(f, {}).get("year")) == pr.year(GT[f].get("year")) for f in fs)
        pulled = sum(1 for f in fs if pr.year(res.get(f, {}).get("year")) in (2026, 2027)
                     and pr.year(res.get(f, {}).get("year")) != pr.year(GT[f].get("year")))
        print(line(label + ", full date", full, len(fs)))
        print(line(label + ", year alone", yr, len(fs),
                   f"   wrong year pulled to 2026-27: {pulled}"))

# ── 3. Was the year legible where A3 got only the year wrong? ────────────────
a3 = json.load(open(ROOT / "gguf_results/products_real/a3_f16_vulkan.json"))
base = json.load(open(ROOT / "benchmark_results/bf16_vulkan_thinkoff_predictions.json"))
year_only = [f for f in ann
             if not pr.correct(a3[f], pr.gt_of(ann[f]))
             and pr.to_int(a3[f].get("month")) == pr.gt_of(ann[f])["month"]
             and (pr.gt_of(ann[f])["day"] is None or pr.to_int(a3[f].get("day")) == pr.gt_of(ann[f])["day"])
             and pr.gt_of(ann[f])["year"] is not None]
base_ok = sum(pr.year(base.get(f, {}).get("year")) == pr.gt_of(ann[f])["year"] for f in year_only)
print(f"\n3. PRODUCTS-REAL images where A3 F16 got only the year wrong: {len(year_only)}")
print(line("zero-shot base read that year right", base_ok, len(year_only),
           "   (base run used an older prompt; a legibility check, not a comparison)"))
