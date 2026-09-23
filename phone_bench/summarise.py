#!/usr/bin/env python3
"""Median / p90 per (config, budget) from phone_runs.csv."""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results" / "phone_runs.csv")
rows = list(csv.DictReader(open(path)))
groups = defaultdict(list)
for r in rows:
    if r["total_ms"]:
        groups[(r["config"], int(r["budget"]))].append(r)


def med(rs, k):
    v = [float(r[k]) for r in rs if r[k]]
    return st.median(v) / 1000 if v else float("nan")


def p90(rs, k):
    v = sorted(float(r[k]) for r in rs if r[k])
    return v[min(len(v) - 1, int(0.9 * len(v)))] / 1000 if v else float("nan")


print(f"{'config':<6} {'budget':>6} {'n':>3}  {'total':>7} {'p90':>6}  {'vit':>6} {'prefill':>7} "
      f"{'decode':>6}  {'acc':>5}")
print("-" * 66)
for (c, b), rs in sorted(groups.items()):
    acc = sum(int(r["correct"]) for r in rs) / len(rs) * 100
    print(f"{c:<6} {b:>6} {len(rs):>3}  {med(rs,'total_ms'):>6.1f}s {p90(rs,'total_ms'):>5.1f}s  "
          f"{med(rs,'vit_ms'):>5.1f}s {med(rs,'prefill_ms'):>6.1f}s {med(rs,'decode_ms'):>5.1f}s  "
          f"{acc:>4.0f}%")
print("\nmedians in seconds; prefill includes the vision encoder; model load excluded")
