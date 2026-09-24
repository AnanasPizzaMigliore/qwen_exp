#!/usr/bin/env python3
"""
Cross-dataset check: evaluate a GGUF model on the Products-Real evaluation split
(Seker & Ahn, 665 images) with the same prompt, server settings and scoring rule
as eval_gguf.py.

Ground truth is the day/month/year breakdown of each image's single 'exp' box.
Scoring matches the paper: years normalised (two-digit -> 20xx), text months
mapped to numbers, and the day only required when the label has one.

    python eval_products_real.py --model m.gguf --mmproj mm.gguf --backend vulkan --output out.json
"""
import argparse
import json
import os
import time
from pathlib import Path

from eval_gguf import PROMPT, extract_json, query_server, start_server, stop_server  # noqa: F401

PR_DIR = Path("/home/penghao/qwen/Products-Real/evaluation")
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def to_int(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    if s in ("", "NULL", "NONE"):
        return None
    if s[:3] in MONTHS:
        return MONTHS[s[:3]]
    try:
        return int(s)
    except ValueError:
        return s


def year(v):
    v = to_int(v)
    return v + 2000 if isinstance(v, int) and 0 < v < 100 else v


def gt_of(entry):
    exp = next(a for a in entry["ann"] if a["cls"] == "exp")
    d = {x["cls"]: x["transcription"] for x in exp.get("dmy_ann", [])}
    return {"year": year(d.get("year")), "month": to_int(d.get("month")), "day": to_int(d.get("day"))}


def correct(p, g):
    if p is None or "error" in p or "parse_error" in p:
        return False
    d_ok = (to_int(p.get("day")) == g["day"]) if g["day"] is not None else True
    return year(p.get("year")) == g["year"] and to_int(p.get("month")) == g["month"] and d_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--backend", choices=["cpu", "vulkan"], default="vulkan")
    ap.add_argument("--output", required=True)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    ann = json.load(open(PR_DIR / "annotations.json"))
    files = sorted(ann)
    out = Path(args.output)
    res = {}
    if out.exists():
        res = {k: v for k, v in json.load(open(out)).items() if "parse_error" not in v}
    pending = [f for f in files if f not in res]
    print(f"{len(pending)}/{len(files)} to run  [{args.backend}  {Path(args.model).name}]")

    if pending:
        proc, url = start_server(args.model, args.mmproj, args.backend, args.port)
        try:
            for i, fn in enumerate(pending, 1):
                if args.backend == "vulkan" and i > 1 and (i - 1) % 100 == 0:
                    stop_server(proc)
                    proc, url = start_server(args.model, args.mmproj, args.backend, args.port)
                try:
                    raw, lat, ntok = query_server(url, str(PR_DIR / "images" / fn))
                    res[fn] = {"raw_response": raw, "latency_s": round(lat, 3), "gen_tokens": ntok,
                               **extract_json(raw)}
                except Exception as e:
                    res[fn] = {"parse_error": str(e), "year": None, "month": None, "day": None}
                if i % 50 == 0:
                    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
                    print(f"  {i}/{len(pending)}")
        finally:
            stop_server(proc)
        json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)

    gts = {f: gt_of(ann[f]) for f in files}
    ok = sum(correct(res.get(f), gts[f]) for f in files)
    errs = sum(1 for f in files if "parse_error" in res.get(f, {}))
    print(f"\nProducts-Real evaluation: {ok}/{len(files)} = {ok/len(files)*100:.1f}% full-date exact"
          f"   (parse errors {errs})")
    for name, keep in (("full date labelled", lambda g: None not in g.values()),
                       ("month+year only", lambda g: g["day"] is None),
                       ("day+month only", lambda g: g["year"] is None)):
        fs = [f for f in files if keep(gts[f])]
        if fs:
            k = sum(correct(res.get(f), gts[f]) for f in fs)
            print(f"  {name:<20} {k:>4}/{len(fs):<4} {k/len(fs)*100:5.1f}%")


if __name__ == "__main__":
    main()
