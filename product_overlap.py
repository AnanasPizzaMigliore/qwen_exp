#!/usr/bin/env python3
"""
R1 — recover product identity across the train/test split with DINOv2, without retraining.

  embed      DINOv2-base CLS embedding of every image (whole photo, no centre crop).
  pairs      each test image's most similar training image; the top N pairs by cosine
             similarity go to a CSV and side-by-side review sheets for checking by eye.
  report     accuracy on test images with no same-product image in train, at several
             similarity thresholds (preliminary) or from the checked CSV (final).

    python product_overlap.py embed
    python product_overlap.py pairs --n 150
    python product_overlap.py report                      # thresholds only
    python product_overlap.py report --checked pairs.csv  # after the by-eye check
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

DS = Path("/home/penghao/Dataset")
OUT = DS / "r1_product_overlap"
EMB = OUT / "dinov2_base_cls.npz"
QWEN = Path("/home/penghao/qwen")

CONFIGS = [("A3 F16", QWEN / "gguf_results/qwen35_groupA_a3_f16_cpu.json"),
           ("A3 whole-model Q4", QWEN / "gguf_results/qwen35_groupA_a3_Q4_K_M_mmQ4_K_M_cpu.json"),
           ("A1' native", QWEN / "epoch_results_groupA_a1p/test_results_epoch_10.json"),
           ("A0' native", QWEN / "epoch_results_groupA_a0p/test_results_epoch_10.json")]


def splits():
    tr = [e["filename"] for e in json.load(open(DS / "train.json"))]
    te = [e["filename"] for e in json.load(open(DS / "test.json"))]
    return tr, te


def open_upright(path):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def embed():
    import torch
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval().cuda()
    tr, te = splits()
    out = {}
    for split, files in (("train", tr), ("test", te)):
        vecs = []
        for i, f in enumerate(files, 1):
            # one image per pass: whole photos differ in aspect ratio, and DINOv2
            # interpolates its position embeddings to any size
            x = proc(images=open_upright(DS / split / f), do_center_crop=False,
                     size={"shortest_edge": 224}, return_tensors="pt").to("cuda")
            with torch.no_grad():
                v = model(**x).pooler_output
            vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
            if i % 100 == 0:
                print(f"  {split} {i}/{len(files)}")
        out[split] = np.concatenate(vecs)
        out[split + "_files"] = np.array(files)
        print()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(EMB, **out)
    print(f"saved {EMB}")


def nearest():
    z = np.load(EMB)
    sim = z["test"] @ z["train"].T
    nn = sim.argmax(1)
    return list(z["test_files"]), list(z["train_files"]), sim, nn, sim.max(1)


def pairs(n):
    from PIL import Image, ImageDraw
    te, tr, sim, nn, best = nearest()
    order = np.argsort(-best)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"pair": k + 1, "test": te[i], "train": tr[nn[i]], "similarity": f"{best[i]:.4f}",
             "same_product": ""} for k, i in enumerate(order[:n])]
    with open(OUT / "pairs_to_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # side-by-side sheets, 10 pairs each; freshly encoded, so no EXIF / GPS
    H, per = 340, 10
    sheets = OUT / "sheets"
    sheets.mkdir(exist_ok=True)
    for s in range(0, n, per):
        chunk = rows[s:s + per]
        canvas = Image.new("RGB", (2 * 260 + 40, len(chunk) * (H + 30)), "white")
        d = ImageDraw.Draw(canvas)
        for j, r in enumerate(chunk):
            y = j * (H + 30)
            d.text((6, y + 4), f"pair {r['pair']}   sim {r['similarity']}   "
                               f"test {r['test']}  |  train {r['train']}", fill="black")
            for c, (split, f) in enumerate((("test", r["test"]), ("train", r["train"]))):
                im = open_upright(DS / split / f)
                im.thumbnail((260, H))
                canvas.paste(im, (10 + c * 270, y + 22))
        canvas.save(sheets / f"pairs_{s + 1:03d}-{s + len(chunk):03d}.jpg", quality=85)
    print(f"{n} pairs -> {OUT/'pairs_to_check.csv'}  and  {len(range(0, n, per))} sheets in {sheets}")
    print(f"similarity of pair 1: {best[order[0]]:.3f}   pair {n}: {best[order[n-1]]:.3f}   "
          f"median over all test images: {np.median(best):.3f}")


def scores(path, files):
    import sys
    sys.path.insert(0, str(QWEN))
    from tier_reconcile import gt, load, okf
    r = load(path)
    return np.array([1.0 if okf(r.get(f), gt.get(f)) else 0.0 for f in files])


def report(checked):
    te, tr, sim, nn, best = nearest()
    te = np.array(te)
    if checked:
        rows = list(csv.DictReader(open(checked)))
        same = {r["test"] for r in rows if r["same_product"].strip().lower() in ("y", "yes", "1", "same")}
        checked_min = min(float(r["similarity"]) for r in rows)
        tail = [r for r in rows if float(r["similarity"]) <= np.quantile([float(x["similarity"]) for x in rows], 0.2)]
        tail_same = sum(r["test"] in same for r in tail)
        overlap = np.array([f in same for f in te])
        print(f"checked {len(rows)} pairs down to similarity {checked_min:.3f}: {len(same)} same-product")
        print(f"same-product among the lowest-similarity fifth of checked pairs: {tail_same}/{len(tail)}"
              + ("  <- still finding matches: check further down the list" if tail_same else ""))
        subsets = [("no same-product image in train", ~overlap), ("has a same-product image", overlap)]
    else:
        subsets = []
        for t in (0.95, 0.90, 0.85, 0.80, 0.75):
            subsets.append((f"nearest-train similarity < {t:.2f}", best < t))
    print(f"\n{'subset':<36}{'n':>5}" + "".join(f"{c:>20}" for c, _ in CONFIGS))
    all_scores = {c: scores(p, list(te)) for c, p in CONFIGS}
    print(f"{'all test images':<36}{len(te):>5}" + "".join(f"{s.mean()*100:>19.1f}%" for s in all_scores.values()))
    for label, mask in subsets:
        print(f"{label:<36}{int(mask.sum()):>5}" +
              "".join(f"{s[mask].mean()*100 if mask.sum() else float('nan'):>19.1f}%" for s in all_scores.values()))
    a1, a0 = all_scores["A1' native"], all_scores["A0' native"]
    for label, mask in subsets[:1] if checked else subsets:
        if mask.sum():
            print(f"  A1' - A0' on [{label}]: {(a1[mask] - a0[mask]).mean()*100:+.1f} pp (n={int(mask.sum())})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["embed", "pairs", "report"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--checked", default=None)
    a = ap.parse_args()
    {"embed": embed, "pairs": lambda: pairs(a.n), "report": lambda: report(a.checked)}[a.step]()
