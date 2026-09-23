"""N1: per-tier reconciliation for every arm.  N2: A4's CI."""
import json
from pathlib import Path
import numpy as np

SEED, N_BOOT = 42, 1000
gt = {e["filename"]: e for e in json.load(open("/home/penghao/Dataset/expiration_dates_details_true.json"))}
tf = [e["filename"] for e in json.load(open("/home/penghao/Dataset/test.json"))]

ARMS = [
    ("A5  zero-shot",      "/home/penghao/Dataset/qwen35_test_results.json"),
    ("A4  ViT only",       "/home/penghao/qwen/epoch_results_groupA_a4/test_results_epoch_10.json"),
    ("A0  LLM LoRA",       "/home/penghao/qwen/epoch_results_nocurriculum/test_results_epoch_10.json"),
    ("A1  + ViT",          "/home/penghao/qwen/epoch_results_groupA_a1/test_results_epoch_10.json"),
    ("A2  + merger",       "/home/penghao/qwen/epoch_results_groupA_a2/test_results_epoch_10.json"),
    ("A3  + ViT + merger", "/home/penghao/qwen/epoch_results_groupA_a3/test_results_epoch_10.json"),
    ("A0' LLM @40M",       "/home/penghao/qwen/epoch_results_groupA_a0p/test_results_epoch_10.json"),
    ("A1' + ViT @40M",     "/home/penghao/qwen/epoch_results_groupA_a1p/test_results_epoch_10.json"),
]


def norm(v):
    if v is None or v in ("null", ""):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def ny(v):
    v = norm(v)
    return v + 2000 if isinstance(v, int) and 0 < v < 100 else v


def okf(p, g):
    if p is None or g is None or "error" in p or "parse_error" in p:
        return False
    gh = any(norm(g.get(k)) is not None for k in ("year", "month", "day"))
    if not gh:
        return all(norm(p.get(k)) is None for k in ("year", "month", "day"))
    dok = (norm(p.get("day")) == norm(g.get("day"))) if norm(g.get("day")) is not None else None
    return ny(p.get("year")) == ny(g.get("year")) and norm(p.get("month")) == norm(g.get("month")) and (dok or dok is None)


def load(p):
    r = json.load(open(p))
    return {e["filename"]: e for e in r} if isinstance(r, list) else r


def tier(fn):
    t = gt[fn].get("difficulty_tier", "unknown")
    return "non_hard" if t in ("easy", "medium", "non_hard") else t


hard = np.array([tier(f) == "hard" for f in tf])
nonh = ~hard

print(f"tiers: hard={hard.sum()}  non_hard={nonh.sum()}  total={len(tf)}\n")
print(f"  {'arm':<20} {'hard%':>7} {'non-hard%':>10} {'ALL%':>7} {'implied':>8} {'95% CI (ALL)':>17}")
print(f"  {'-'*76}")

for name, path in ARMS:
    if not Path(path).exists():
        print(f"  {name:<20}  MISSING")
        continue
    res = load(path)
    s = np.array([1.0 if okf(res.get(f), gt.get(f)) else 0.0 for f in tf])
    h, nh, allp = s[hard].mean()*100, s[nonh].mean()*100, s.mean()*100
    implied = (h*hard.sum() + nh*nonh.sum()) / len(tf)
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(s))
    boot = np.array([s[rng.choice(idx, size=len(idx), replace=True)].mean()
                     for _ in range(N_BOOT)]) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    flag = "" if abs(implied - allp) < 0.05 else "  <-- MISMATCH"
    print(f"  {name:<20} {h:>6.1f}% {nh:>9.1f}% {allp:>6.1f}% {implied:>7.1f}% "
          f"  [{lo:>5.1f}, {hi:>5.1f}]{flag}")

print()
res = load("/home/penghao/qwen/epoch_results_groupA_a3/test_results_epoch_10.json")
s = np.array([1.0 if okf(res.get(f), gt.get(f)) else 0.0 for f in tf])
print(f"A3 correct count: {int(s.sum())}/{len(tf)} = {s.mean()*100:.2f}%")
print(f"  paper's tier figures 69.7 hard / 90.6 non-hard would need "
      f"{round(0.697*hard.sum())} + {round(0.906*nonh.sum())} = "
      f"{round(0.697*hard.sum())+round(0.906*nonh.sum())} correct")
