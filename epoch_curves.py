#!/usr/bin/env python3
"""
R4 — per-epoch test-split accuracy for every adaptation arm.

TestEvalCallback scored the test split after every epoch of every reported run. This
tabulates those curves so the disclosure can state what was visible during training:
the epoch-10 score used in the paper, the best epoch in hindsight, and the gap between
them (an upper bound on what selecting an epoch by test score could have gained).
Scoring is identical to Table 1 (tier_reconcile.py).
"""
import glob
import json
from pathlib import Path

import numpy as np

from tier_reconcile import gt, load, okf, tf

ARMS = [
    ("A0",  "epoch_results_nocurriculum"),
    ("A1",  "epoch_results_groupA_a1"),
    ("A2",  "epoch_results_groupA_a2"),
    ("A3",  "epoch_results_groupA_a3"),
    ("A0'", "epoch_results_groupA_a0p"),
    ("A1'", "epoch_results_groupA_a1p"),
    ("A4",  "epoch_results_groupA_a4"),
]
ROOT = Path("/home/penghao/qwen")


def acc(path):
    r = load(path)
    return float(np.mean([okf(r.get(f), gt.get(f)) for f in tf]) * 100)


print(f"{'arm':<5}" + "".join(f"{'e'+str(e):>6}" for e in range(1, 11))
      + f"{'final':>8}{'best':>7}{'@ep':>5}{'best-final':>12}")
rows = []
for arm, d in ARMS:
    curve = {}
    for f in glob.glob(str(ROOT / d / "test_results_epoch_*.json")):
        curve[int(Path(f).stem.split("_")[-1])] = acc(f)
    if not curve:
        print(f"{arm:<5}  (no per-epoch files)")
        continue
    final = curve.get(10, float("nan"))
    best_ep = max(curve, key=lambda e: (curve[e], -e))
    rows.append((arm, curve, final, best_ep))
    print(f"{arm:<5}" + "".join(f"{curve.get(e, float('nan')):>6.1f}" for e in range(1, 11))
          + f"{final:>8.1f}{curve[best_ep]:>7.1f}{best_ep:>5}{curve[best_ep]-final:>+12.1f}")

gaps = [c[b] - f for _, c, f, b in rows]
print(f"\nbest-in-hindsight minus epoch 10: mean {np.mean(gaps):+.2f} pp, max {max(gaps):+.2f} pp")
print("epoch 10 is the best epoch for:", ", ".join(a for a, c, f, b in rows if c[b] == f) or "none")
