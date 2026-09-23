#!/usr/bin/env python3
"""
S5 — error analysis, and S6 — bootstrap CIs on the adaptation ablation.

S5 reports the two metrics R2 asked for, on the deployed configuration:
  * false-abstention rate: date IS present, model returns all-null
  * later-date share: among wrong date-present predictions, how many name a
    date LATER than the truth. This is the safety-relevant direction — an
    assistive reader that reads a date as later than actual can lead someone
    to eat expired food. Feeds the ethics statement (G5).

S6 puts paired-bootstrap CIs on the A0-A3 adaptation gaps, which currently
carry none while every precision claim in the paper does.

Seed 42, 1000 resamples, matching bootstrap_ci.py.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np

SEED, N_BOOT = 42, 1000
GT_FILE = Path("/home/penghao/Dataset/expiration_dates_details_true.json")
TEST_JSON = Path("/home/penghao/Dataset/test.json")

# Adaptation arms: per-epoch dumps from the training callback (1024px, epoch 10).
ARMS = [
    ("A5  zero-shot (frozen)",    "/home/penghao/Dataset/qwen35_test_results.json"),
    ("A4  ViT LoRA only",         "/home/penghao/qwen/epoch_results_groupA_a4/test_results_epoch_10.json"),
    ("A0  LLM LoRA",              "/home/penghao/qwen/epoch_results_nocurriculum/test_results_epoch_10.json"),
    ("A1  + ViT LoRA",            "/home/penghao/qwen/epoch_results_groupA_a1/test_results_epoch_10.json"),
    ("A2  + merger",              "/home/penghao/qwen/epoch_results_groupA_a2/test_results_epoch_10.json"),
    ("A3  + ViT LoRA + merger",   "/home/penghao/qwen/epoch_results_groupA_a3/test_results_epoch_10.json"),
    ("A0' LLM LoRA @ A3 budget",  "/home/penghao/qwen/epoch_results_groupA_a0p/test_results_epoch_10.json"),
    ("A1' + ViT LoRA @ A3 budget","/home/penghao/qwen/epoch_results_groupA_a1p/test_results_epoch_10.json"),
]

# Trainable parameters, measured (see check of LoRA budgets).
PARAMS = {
    "A5  zero-shot (frozen)":      0.0,
    "A4  ViT LoRA only":           1.8,
    "A0  LLM LoRA":               25.6,
    "A1  + ViT LoRA":             27.3,
    "A2  + merger":               38.1,
    "A3  + ViT LoRA + merger":    39.9,
    "A0' LLM LoRA @ A3 budget":   40.0,
    "A1' + ViT LoRA @ A3 budget": 39.8,
}

# Deployed configuration, for the S5 error analysis.
DEPLOYED = ("A3 F16 mmproj + F16 decoder",
            "/home/penghao/qwen/gguf_results/qwen35_groupA_a3_f16_cpu.json")

gt_all = {e["filename"]: e for e in json.load(open(GT_FILE))}
test_files = [e["filename"] for e in json.load(open(TEST_JSON))]


def norm(v):
    if v is None or v in ("null", ""):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def norm_year(v):
    v = norm(v)
    return v + 2000 if isinstance(v, int) and 0 < v < 100 else v


def ymd(e):
    return norm_year(e.get("year")), norm(e.get("month")), norm(e.get("day"))


def has_date(e):
    return any(norm(e.get(k)) is not None for k in ("year", "month", "day"))


def is_correct(p, g):
    if p is None or g is None:
        return False
    if "error" in p or "parse_error" in p:
        return False
    if not has_date(g):
        return not has_date(p)
    py, pm, pd = ymd(p)
    gy, gm, gd = ymd(g)
    dok = (pd == gd) if gd is not None else None
    return py == gy and pm == gm and (dok or dok is None)


def load(path):
    r = json.load(open(path))
    return {e["filename"]: e for e in r} if isinstance(r, list) else r


def scores(res):
    return np.array([1.0 if is_correct(res.get(f), gt_all.get(f)) else 0.0 for f in test_files])


def ci(vec, rng, n_boot=N_BOOT):
    idx = np.arange(len(vec))
    boot = np.array([vec[rng.choice(idx, size=len(idx), replace=True)].mean()
                     for _ in range(n_boot)]) * 100
    return np.percentile(boot, [2.5, 97.5])


def paired(a, b, rng, n_boot=N_BOOT):
    d = a - b
    idx = np.arange(len(d))
    boot = np.array([d[rng.choice(idx, size=len(idx), replace=True)].mean()
                     for _ in range(n_boot)]) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = (boot <= 0).mean() if d.mean() > 0 else (boot >= 0).mean()
    return d.mean() * 100, lo, hi, p


def as_ordinal(y, m, dd):
    """Coarse comparable key; missing month/day float to the start of the period."""
    if y is None:
        return None
    try:
        return date(int(y), int(m) if m else 1, int(dd) if dd else 1).toordinal()
    except (ValueError, TypeError):
        return None


# ────────────────────────────── S5 ──────────────────────────────
label, path = DEPLOYED
res = load(path)

n_present = n_abstain = 0
n_wrong = n_later = n_earlier = n_incomparable = 0

for fn in test_files:
    g, p = gt_all.get(fn), res.get(fn)
    if not g or not p or not has_date(g):
        continue
    n_present += 1
    if "error" in p or "parse_error" in p:
        continue
    if not has_date(p):
        n_abstain += 1
        continue
    if is_correct(p, g):
        continue
    n_wrong += 1
    po, go = as_ordinal(*ymd(p)), as_ordinal(*ymd(g))
    if po is None or go is None:
        n_incomparable += 1
    elif po > go:
        n_later += 1
    elif po < go:
        n_earlier += 1
    else:
        n_incomparable += 1   # same coarse date, differed on a null field

print(f"\n{'='*72}")
print(f"  S5 — error analysis   ({label}, n={len(test_files)})")
print(f"{'='*72}")
print(f"  date-present images                        {n_present}")
print(f"  false abstentions (returned all-null)      {n_abstain}"
      f"   = {n_abstain/n_present*100:.1f}% of date-present")
print(f"  wrong date-present predictions             {n_wrong}")
if n_wrong:
    print(f"    predicted LATER than truth               {n_later}"
          f"   = {n_later/n_wrong*100:.1f}% of errors   <-- safety-relevant")
    print(f"    predicted EARLIER than truth             {n_earlier}"
          f"   = {n_earlier/n_wrong*100:.1f}% of errors")
    print(f"    not orderable (missing field)            {n_incomparable}")
    print(f"\n  later-date errors as a share of all date-present images:"
          f"  {n_later/n_present*100:.1f}%")

# ────────────────────────────── S6 ──────────────────────────────
print(f"\n{'='*72}")
print(f"  S6 — adaptation ablation with CIs   (1024px eval, seed {SEED}, {N_BOOT} resamples)")
print(f"{'='*72}")
print(f"  {'arm':<28} {'params':>8} {'ALL%':>7}  {'95% CI':>16}")
print(f"  {'-'*66}")

S = {}
for name, p in ARMS:
    if not Path(p).exists():
        print(f"  {name:<28}  MISSING {p}")
        continue
    s = scores(load(p))
    S[name] = s
    rng = np.random.default_rng(SEED)
    lo, hi = ci(s, rng)
    print(f"  {name:<28} {PARAMS.get(name, 0):>7.1f}M {s.mean()*100:>6.1f}%  [{lo:>5.1f}, {hi:>5.1f}]")

PAIRS = [
    # S3 — the matched-budget tests. A3, A0' and A1' all sit at ~40M.
    ("A3  + ViT LoRA + merger", "A0' LLM LoRA @ A3 budget",   "MATCHED  A3 vs A0'  (placement, budget held)"),
    ("A3  + ViT LoRA + merger", "A1' + ViT LoRA @ A3 budget", "MATCHED  A3 vs A1'  (merger, budget held)"),
    ("A1' + ViT LoRA @ A3 budget", "A0' LLM LoRA @ A3 budget", "MATCHED  A1' vs A0' (encoder, budget held)"),
    # unmatched originals, for reference
    ("A3  + ViT LoRA + merger", "A0  LLM LoRA",            "A3 vs A0   (placement + budget)"),
    ("A3  + ViT LoRA + merger", "A1  + ViT LoRA",          "A3 vs A1   (merger adds?)"),
    ("A1  + ViT LoRA",          "A0  LLM LoRA",            "A1 vs A0   (ViT LoRA alone)"),
    ("A2  + merger",            "A0  LLM LoRA",            "A2 vs A0   (merger alone)"),
    ("A1  + ViT LoRA",          "A2  + merger",            "A1 vs A2   (encoder vs merger)"),
    # S2 — encoder isolated
    ("A4  ViT LoRA only",       "A5  zero-shot (frozen)",  "A4 vs A5   (encoder alone vs frozen)"),
    ("A0  LLM LoRA",            "A4  ViT LoRA only",       "A0 vs A4   (decoder vs encoder)"),
    ("A0  LLM LoRA",            "A5  zero-shot (frozen)",  "A0 vs A5   (adaptation gap)"),
]

print(f"\n  {'comparison':<36} {'delta':>7}  {'95% CI':>16}  {'p':>6}")
print(f"  {'-'*70}")
for a, b, desc in PAIRS:
    if a not in S or b not in S:
        print(f"  {desc:<36}  (missing)")
        continue
    rng = np.random.default_rng(SEED)
    d, lo, hi, p = paired(S[a], S[b], rng)
    star = " *" if (lo > 0 or hi < 0) else ""
    print(f"  {desc:<36} {d:>+6.1f}  [{lo:>+5.1f}, {hi:>+5.1f}]  {p:.3f}{star}")

print(f"\n  * = 95% CI excludes zero\n")
