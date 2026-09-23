#!/usr/bin/env python3
"""
S7 — measure the quantization mechanism instead of asserting it.

Section 6 argues that low-bit encoder quantization converts borderline glyphs
into misreads on signal-poor images. The evidence currently offered is a
*different, larger* model's zero-shot per-tag accuracy over all 1,815 images,
which shows the task is bimodal in visual difficulty but says nothing about
what quantization does.

The direct test: per-tag accuracy of F16-mmproj vs Q3-mmproj, decoder held at
F16, over the 545-image test set (the same population as every other number in
the paper). If the ~10 pp loss concentrates on signal-poor tags, the mechanism
is demonstrated rather than argued.

Q4-mmproj is included as a control: it should show a small, flat delta with no
tag concentration.

Paired bootstrap over images, seed 42 (matches bootstrap_ci.py).

Run:
    conda run -n llm python /home/penghao/qwen/tag_mechanism.py
"""
import json
from pathlib import Path

import numpy as np

SEED      = 42
N_BOOT    = 1000
MIN_N     = 20        # don't report tags rarer than this — CIs are meaningless

GT_FILE   = Path("/home/penghao/Dataset/expiration_dates_details_true.json")
TEST_JSON = Path("/home/penghao/Dataset/test.json")
GGUF_DIR  = Path("/home/penghao/qwen/gguf_results")

BASELINE  = "qwen35_groupA_a3_f16_cpu.json"            # F16 mmproj + F16 decoder
ARMS = [
    ("Q3 mmproj", "qwen35_groupA_a3_f16_mmQ3_K_M_cpu.json"),
    ("Q4 mmproj", "qwen35_groupA_a3_f16_mmQ4_K_M_cpu.json"),
]

# Tags the mechanism claim is about: the date signal itself is degraded.
SIGNAL_POOR = {"embossed", "low_contrast", "glare", "blurry", "faded",
               "curved_surface", "small", "angled", "occluded", "shadow"}


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


def is_correct(p, g):
    if p is None or g is None:
        return False
    if "error" in p or "parse_error" in p:
        return False
    g_has = any(norm(g.get(k)) is not None for k in ("year", "month", "day"))
    if not g_has:
        return all(norm(p.get(k)) is None for k in ("year", "month", "day"))
    py, gy = norm_year(p.get("year")), norm_year(g.get("year"))
    pm, gm = norm(p.get("month")), norm(g.get("month"))
    pd, gd = norm(p.get("day")), norm(g.get("day"))
    dok = (pd == gd) if gd is not None else None
    return (py == gy) and (pm == gm) and (dok or dok is None)


def load(path):
    r = json.load(open(path))
    return {e["filename"]: e for e in r} if isinstance(r, list) else r


def tags_of(entry):
    """All tags on an image, from every tag family."""
    out = set()
    for fam in ("visual_tags", "layout_tags", "format_tags"):
        for t in (entry.get(fam) or []):
            out.add(t)
    return out


def paired_delta_ci(a, b, rng, n_boot=N_BOOT):
    """a, b: 0/1 arrays over the same images. Returns (delta_pp, lo, hi)."""
    diff = a - b
    idx = np.arange(len(diff))
    boot = np.array([diff[rng.choice(idx, size=len(idx), replace=True)].mean()
                     for _ in range(n_boot)]) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return diff.mean() * 100, lo, hi


def main():
    gt_all = load(GT_FILE)
    test_files = [e["filename"] for e in json.load(open(TEST_JSON))]

    base = load(GGUF_DIR / BASELINE)

    # per-image correctness vectors, aligned to test_files
    def scores(results):
        return np.array([1.0 if is_correct(results.get(fn), gt_all.get(fn)) else 0.0
                         for fn in test_files])

    s_base = scores(base)

    # tag -> boolean mask over test_files
    tag_masks = {}
    for i, fn in enumerate(test_files):
        g = gt_all.get(fn)
        if not g:
            continue
        for t in tags_of(g):
            tag_masks.setdefault(t, np.zeros(len(test_files), dtype=bool))[i] = True

    for arm_label, arm_file in ARMS:
        path = GGUF_DIR / arm_file
        if not path.exists():
            print(f"  MISSING: {arm_file}")
            continue
        s_arm = scores(load(path))
        rng = np.random.default_rng(SEED)

        overall, o_lo, o_hi = paired_delta_ci(s_base, s_arm, rng)

        print(f"\n{'='*84}")
        print(f"  F16 mmproj  vs  {arm_label}    (decoder F16, n={len(test_files)} test images)")
        print(f"  Overall: F16 {s_base.mean()*100:.1f}%  ->  {arm_label} {s_arm.mean()*100:.1f}%"
              f"   delta {overall:+.1f} pp  [{o_lo:+.1f}, {o_hi:+.1f}]")
        print(f"{'='*84}")
        print(f"  {'tag':<32} {'n':>4}  {'F16':>6} {'quant':>6}  {'delta':>7}  {'95% CI':>16}")
        print(f"  {'-'*82}")

        rows = []
        for tag, mask in tag_masks.items():
            n = int(mask.sum())
            if n < MIN_N:
                continue
            a, b = s_base[mask], s_arm[mask]
            rng_t = np.random.default_rng(SEED)
            d, lo, hi = paired_delta_ci(a, b, rng_t)
            rows.append((d, tag, n, a.mean()*100, b.mean()*100, lo, hi))

        # biggest drop first
        rows.sort(key=lambda r: -r[0])
        for d, tag, n, fa, qa, lo, hi in rows:
            star = " *" if lo > 0 else ""
            poor = "  [signal-poor]" if tag in SIGNAL_POOR else ""
            print(f"  {tag:<32} {n:>4}  {fa:>5.1f}% {qa:>5.1f}%  {d:>+6.1f}  "
                  f"[{lo:>+5.1f}, {hi:>+5.1f}]{star}{poor}")

        # grouped: signal-poor vs everything else
        poor_mask = np.zeros(len(test_files), dtype=bool)
        for t in SIGNAL_POOR:
            if t in tag_masks:
                poor_mask |= tag_masks[t]
        clean_mask = ~poor_mask

        print(f"  {'-'*82}")
        for label, m in (("signal-poor (any)", poor_mask), ("no signal-poor tag", clean_mask)):
            n = int(m.sum())
            if n == 0:
                continue
            rng_g = np.random.default_rng(SEED)
            d, lo, hi = paired_delta_ci(s_base[m], s_arm[m], rng_g)
            print(f"  {label:<32} {n:>4}  {s_base[m].mean()*100:>5.1f}% "
                  f"{s_arm[m].mean()*100:>5.1f}%  {d:>+6.1f}  [{lo:>+5.1f}, {hi:>+5.1f}]")

    print(f"\n  * = 95% CI excludes zero (this tag is significantly hurt)")
    print(f"  Population: 545-image test split (not the 1,815-image corpus used in Fig. 7)\n")


if __name__ == "__main__":
    main()
