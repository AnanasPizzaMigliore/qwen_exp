#!/usr/bin/env python3
"""Score the R2 OCR baseline exactly like the VLM rows (tier_reconcile.okf), plus the
any-candidate oracle, the safety metrics and a paired comparison with A3 F16."""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/penghao/qwen")
from tier_reconcile import gt, hard, nonh, okf, tf, load, SEED, N_BOOT  # noqa: E402
import error_analysis as ea  # noqa: E402

D = "/home/penghao/qwen/gguf_results/ocr_baseline/"
pred = load(D + "ocr_baseline_predictions.json")
a3 = load("/home/penghao/qwen/gguf_results/qwen35_groupA_a3_f16_cpu.json")
NULL = {"year": None, "month": None, "day": None}


def oracle_ok(f):
    cands = pred[f]["candidates"]
    if not cands:
        return okf(NULL, gt[f])
    return any(okf({"year": y, "month": m, "day": d}, gt[f]) for y, m, d in cands)


def ci(s):
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(s))
    b = np.array([s[rng.choice(idx, size=len(idx), replace=True)].mean() for _ in range(N_BOOT)]) * 100
    return np.percentile(b, [2.5, 97.5])


def row(label, s, lat=None):
    lo, hi = ci(s)
    print(f"  {label:<34} hard {s[hard].mean()*100:5.1f}  non-hard {s[nonh].mean()*100:5.1f}  "
          f"ALL {s.mean()*100:5.1f} [{lo:4.1f}, {hi:4.1f}]  ({int(s.sum())}/{len(s)})"
          + (f"  {lat:.2f} s/img" if lat else ""))


s_ocr = np.array([1.0 if okf(pred.get(f), gt[f]) else 0.0 for f in tf])
s_or = np.array([1.0 if oracle_ok(f) else 0.0 for f in tf])
s_a3 = np.array([1.0 if okf(a3.get(f), gt[f]) else 0.0 for f in tf])
lat = np.mean([pred[f]["latency_s"] for f in tf])

print("R2 — OCR BASELINE on the 545-image test split (same scoring as Table 3)")
row("PaddleOCR + date rules + keyword pick", s_ocr, lat)
row("  oracle: any parsed date correct", s_or)
row("A3 F16 (reference)", s_a3)

d = s_a3 - s_ocr
rng = np.random.default_rng(SEED)
idx = np.arange(len(d))
b = np.array([d[rng.choice(idx, size=len(idx), replace=True)].mean() for _ in range(N_BOOT)]) * 100
lo, hi = np.percentile(b, [2.5, 97.5])
print(f"\n  A3 F16 minus OCR baseline: {d.mean()*100:+.1f} pp [{lo:+.1f}, {hi:+.1f}], "
      f"one-sided p {'<0.001' if (b <= 0).mean() == 0 else round((b <= 0).mean(), 3)}")
print(f"  A3 F16 minus OCR oracle:   {(s_a3 - s_or).mean()*100:+.1f} pp")

# where the baseline loses: no date parsed, wrong date chosen, or never read correctly
present = [f for f in tf if ea.has_date(gt[f])]
none = sum(1 for f in present if not pred[f]["candidates"])
wrong_pick = sum(1 for f in present if pred[f]["candidates"] and oracle_ok(f) and not okf(pred[f], gt[f]))
never = sum(1 for f in present if pred[f]["candidates"] and not oracle_ok(f))
print(f"\n  dated images ({len(present)}): no date parsed {none}; right date found but wrong one "
      f"picked {wrong_pick}; no parsed date correct {never}")

c = ea.safety(pred)
print("\n  safety (Wilson 95%)")
print("    abstains on date-absent images         ", ea.rate(c["abstain_absent"], c["absent"]))
print("    false abstention on dated images       ", ea.rate(c["false_abstain"], c["present"]))
print("    later than printed, among wrong answers", ea.rate(c["later"], c["wrong"]))
print("    later than printed, among dated images ", ea.rate(c["later"], c["present"]))
