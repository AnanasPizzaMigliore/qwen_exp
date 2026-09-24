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
    return diff.mean() * 100, lo, hi, one_sided_p(diff.mean(), boot)


def one_sided_p(observed, boot):
    """Share of resamples on the far side of zero, as in bootstrap_ci.py.
    With 1,000 resamples the resolution is 0.001, so 0 means p < 0.001."""
    return float((boot <= 0).mean() if observed > 0 else (boot >= 0).mean())


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def contrast(d, a_idx, b_idx, rng, n_boot=N_BOOT):
    """(mean penalty in group a) - (mean penalty in group b), groups resampled independently."""
    boot = np.array([d[rng.choice(a_idx, size=len(a_idx), replace=True)].mean()
                     - d[rng.choice(b_idx, size=len(b_idx), replace=True)].mean()
                     for _ in range(n_boot)]) * 100
    obs = d[a_idx].mean() - d[b_idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return obs * 100, lo, hi, one_sided_p(obs, boot)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


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

        overall, o_lo, o_hi, o_p = paired_delta_ci(s_base, s_arm, rng)

        print(f"\n{'='*104}")
        print(f"  F16 mmproj  vs  {arm_label}    (decoder F16, n={len(test_files)} test images)")
        print(f"  Overall: F16 {s_base.mean()*100:.1f}%  ->  {arm_label} {s_arm.mean()*100:.1f}%"
              f"   delta {overall:+.1f} pp  [{o_lo:+.1f}, {o_hi:+.1f}]  p {fmt_p(o_p)}")
        print(f"{'='*104}")
        print(f"  {'tag':<32} {'n':>4}  {'F16':>6} {'quant':>6}  {'delta':>7}  {'95% CI':>16}"
              f"  {'p':>6}  {'p Holm':>6}")
        print(f"  {'-'*102}")

        rows = []
        for tag, mask in tag_masks.items():
            n = int(mask.sum())
            if n < MIN_N:
                continue
            a, b = s_base[mask], s_arm[mask]
            rng_t = np.random.default_rng(SEED)
            d, lo, hi, p = paired_delta_ci(a, b, rng_t)
            rows.append([d, tag, n, a.mean()*100, b.mean()*100, lo, hi, p])

        # Holm across every tag with n >= MIN_N in this comparison (one family per comparison)
        for r, p_adj in zip(rows, holm([r[7] for r in rows])):
            r.append(p_adj)

        # biggest drop first
        rows.sort(key=lambda r: -r[0])
        for d, tag, n, fa, qa, lo, hi, p, p_adj in rows:
            star = " *" if p_adj < 0.05 else ""
            poor = "  [signal-poor]" if tag in SIGNAL_POOR else ""
            print(f"  {tag:<32} {n:>4}  {fa:>5.1f}% {qa:>5.1f}%  {d:>+6.1f}  "
                  f"[{lo:>+5.1f}, {hi:>+5.1f}]  {fmt_p(p):>6}  {fmt_p(p_adj):>6}{star}{poor}")

        # derived union row, reported for context only: it overlaps its components, so it is
        # not part of the Holm family
        u = tag_masks.get("nonstandard_format", 0) | tag_masks.get("ambiguous_format", 0)
        if isinstance(u, np.ndarray) and u.sum():
            rng_u = np.random.default_rng(SEED)
            d, lo, hi, p = paired_delta_ci(s_base[u], s_arm[u], rng_u)
            print(f"  {'nonstandard ∪ ambiguous_format':<32} {int(u.sum()):>4}  "
                  f"{s_base[u].mean()*100:>5.1f}% {s_arm[u].mean()*100:>5.1f}%  {d:>+6.1f}  "
                  f"[{lo:>+5.1f}, {hi:>+5.1f}]  {fmt_p(p):>6}     n/a   (union, outside Holm family)")

        # grouped: signal-poor vs everything else
        poor_mask = np.zeros(len(test_files), dtype=bool)
        for t in SIGNAL_POOR:
            if t in tag_masks:
                poor_mask |= tag_masks[t]
        clean_mask = ~poor_mask

        print(f"  {'-'*102}")
        for label, m in (("signal-poor (any)", poor_mask), ("no signal-poor tag", clean_mask)):
            n = int(m.sum())
            if n == 0:
                continue
            rng_g = np.random.default_rng(SEED)
            d, lo, hi, p = paired_delta_ci(s_base[m], s_arm[m], rng_g)
            print(f"  {label:<32} {n:>4}  {s_base[m].mean()*100:>5.1f}% "
                  f"{s_arm[m].mean()*100:>5.1f}%  {d:>+6.1f}  [{lo:>+5.1f}, {hi:>+5.1f}]  {fmt_p(p):>6}")

        # Does the penalty concentrate on signal-poor images? Difference of the two group
        # penalties, on all images and on date-present images only (date-absent images are
        # scored as abstention, a different task).
        d_vec = s_base - s_arm
        present = np.array([any(norm(gt_all[fn].get(k)) is not None for k in ("year", "month", "day"))
                            for fn in test_files])
        print(f"  contrast (signal-poor penalty - clean penalty):")
        for label, sel in (("all images", np.ones(len(test_files), dtype=bool)),
                           ("date-present only", present)):
            ip, ic = np.where(poor_mask & sel)[0], np.where(clean_mask & sel)[0]
            rng_c = np.random.default_rng(SEED)
            obs, lo, hi, p = contrast(d_vec, ip, ic, rng_c)
            print(f"    {label:<19} n={len(ip)} vs {len(ic):<3}  {obs:>+6.1f} pp  "
                  f"[{lo:>+5.1f}, {hi:>+5.1f}]  one-sided p {fmt_p(p)}")

    print(f"\n  p: one-sided paired bootstrap ({N_BOOT} resamples, seed {SEED}); resolution 0.001")
    print(f"  p Holm: Holm step-down across the tags with n >= {MIN_N} within each comparison")
    print(f"  * = significant after Holm correction (p Holm < 0.05)")
    print(f"  Population: 545-image test split (not the 1,815-image corpus used in Fig. 7)\n")


if __name__ == "__main__":
    main()
