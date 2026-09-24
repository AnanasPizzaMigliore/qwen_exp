#!/usr/bin/env python3
"""
R2 — conventional OCR baseline on the same 545 test images, scored like every VLM row.

  ocr    PaddleOCR (PP-OCRv5 detection + Latin recognition) on each full-resolution image,
         long side capped at 1920 px. Raw lines, scores and boxes -> ocr_raw.json.
         Run in the separate `ocr` conda env.
  parse  multilingual date patterns + a keyword rule that picks the expiration date:
           * a date on the same line as, or the line after, an expiry keyword wins;
           * a production keyword on the same line counts against it;
           * otherwise the latest date; no date found -> abstain (all null).
         Writes predictions in eval_gguf.py's format, plus an oracle file that is
         correct whenever ANY parsed date matches the label (abstains only if none parsed).
         Pure Python; runs in either env.

    conda run -n ocr python ocr_baseline.py ocr
    python ocr_baseline.py parse
"""
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

DS = Path("/home/penghao/Dataset")
OUT = Path("/home/penghao/qwen/gguf_results/ocr_baseline")
RAW = OUT / "ocr_raw.json"

MONTHS = {}
for i, names in enumerate([
        "JAN ENE GEN JAN", "FEB FEV FEB", "MAR MAR MRZ MAE", "APR ABR AVR APR",
        "MAY MAI MAG MEI", "JUN JUIN GIU", "JUL JUIL LUG", "AUG AGO AOU AGO",
        "SEP SET SEPT", "OCT OUT OTT OKT", "NOV", "DEC DIC DEZ DES"], 1):
    for n in names.split():
        MONTHS[n] = i

EXPIRY = ["CAD", "CADUC", "CONSUMIR", "CONS PREF", "CONS.PREF", "PREFERENTEMENTE", "PREFERENCIA",
          "ANTES", "BEST BEFORE", "BEST BY", "BB", "BBE", "EXP", "USE BY", "VAL", "VENC", "VTO",
          "DLC", "DLUO", "DDM", "CONSOMMER", "CONSUMARSI", "SCAD", "MHD", "MINDESTENS", "HALTBAR",
          "FC", "F.C", "C.P", "CP"]
PRODUCTION = ["ENV", "ENVASADO", "FAB", "FABRIC", "ELAB", "PROD", "LOTE", "LOT", "PACKED", "PKD",
              "MFG", "MFD", "EMB"]

SEP = r"[\s./\-]{1,2}"   # up to two characters, e.g. "02. 10. 2026"
Y4, Y2, NUM = r"(20\d{2})", r"(\d{2})", r"(\d{1,2})"
MON = r"([A-Z]{3,5})\.?"
PATTERNS = [  # (regex, field order) — longest / least ambiguous first
    (rf"\b{Y4}{SEP}{NUM}{SEP}{NUM}\b", "ymd"),
    (rf"\b{NUM}{SEP}{NUM}{SEP}{Y4}\b", "dmy"),
    (rf"\b{NUM}{SEP}{NUM}{SEP}{Y2}\b", "dmy"),
    (rf"\b{NUM}{SEP}?{MON}{SEP}?{Y4}\b", "dMy"),
    (rf"\b{NUM}{SEP}?{MON}{SEP}?{Y2}\b", "dMy"),
    (rf"\b{MON}{SEP}?{Y4}\b", "My"),
    (rf"\b{Y4}{SEP}{NUM}\b", "ym"),
    (rf"\b{NUM}{SEP}{Y4}\b", "my"),
    (rf"\b{NUM}[./\-]{Y2}\b", "my2"),  # MM/YY, accepted only with a plausible year below
]


def norm_text(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    return s.replace("O", "O")


def to_year(y):
    y = int(y)
    return y + 2000 if y < 100 else y


def valid(y, m, d):
    return 2020 <= y <= 2040 and 1 <= m <= 12 and (d is None or 1 <= d <= 31)


# Compact DDMMYY / DDMMYYYY (e.g. "150926"). Lot codes and barcodes look the same, so these
# are only accepted on a line that carries, or follows, an expiry keyword.
COMPACT = (rf"\b(\d{{2}})(\d{{2}})(20\d{{2}}|\d{{2}})\b", "dmy")


def parse_dates(text, compact=False):
    """All plausible (year, month, day|None) in a line."""
    t = norm_text(text)
    found, taken = [], []
    for rx, order in PATTERNS + ([COMPACT] if compact else []):
        for mt in re.finditer(rx, t):
            if any(a < mt.end() and mt.start() < b for a, b in taken):
                continue
            g = mt.groups()
            try:
                if order == "ymd":
                    y, m, d = to_year(g[0]), int(g[1]), int(g[2])
                elif order == "dmy":
                    d, m, y = int(g[0]), int(g[1]), to_year(g[2])
                elif order == "dMy":
                    if g[1][:3] not in MONTHS:
                        continue
                    d, m, y = int(g[0]), MONTHS[g[1][:3]], to_year(g[2])
                elif order == "My":
                    if g[0][:3] not in MONTHS:
                        continue
                    d, m, y = None, MONTHS[g[0][:3]], to_year(g[1])
                elif order == "ym":
                    d, m, y = None, int(g[1]), to_year(g[0])
                elif order == "my":
                    d, m, y = None, int(g[0]), to_year(g[1])
                else:  # my2: two-digit year only if plausible (24-35), else it is likely DD/MM
                    if not 24 <= int(g[1]) <= 35:
                        continue
                    d, m, y = None, int(g[0]), to_year(g[1])
            except ValueError:
                continue
            if valid(y, m, d):
                found.append((y, m, d))
                taken.append((mt.start(), mt.end()))
    return found


def has_kw(text, kws):
    t = " " + re.sub(r"[^A-Z. ]", " ", norm_text(text)) + " "
    return any(re.search(rf"(?<![A-Z]){re.escape(k)}(?![A-Z])", t) for k in kws)


def select(lines):
    """lines: OCR text lines in reading order. Returns (chosen date | None, all candidates)."""
    cands = []
    for i, line in enumerate(lines):
        near_kw = has_kw(line, EXPIRY) or (i > 0 and has_kw(lines[i - 1], EXPIRY))
        for dt in parse_dates(line, compact=near_kw):
            score = 0
            if near_kw:
                score += 2
            if has_kw(line, PRODUCTION):
                score -= 2
            cands.append((score, dt))
    if not cands:
        return None, []
    best = max(cands, key=lambda c: (c[0], c[1][0], c[1][1], c[1][2] or 0))
    return best[1], [c[1] for c in cands]


def run_ocr():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="es", use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=True, text_det_limit_side_len=1920,
                    text_det_limit_type="max")
    files = [e["filename"] for e in json.load(open(DS / "test.json"))]
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.load(open(RAW)) if RAW.exists() else {}
    for i, f in enumerate(files, 1):
        if f in raw:
            continue
        t0 = time.time()
        res = ocr.predict(str(DS / "test" / f))[0]
        polys = res["rec_polys"]
        raw[f] = {"texts": list(res["rec_texts"]), "scores": [float(s) for s in res["rec_scores"]],
                  "boxes": [[[float(x), float(y)] for x, y in p] for p in polys],
                  "latency_s": round(time.time() - t0, 3)}
        print(f"  {i}/{len(files)}  {f}  {len(raw[f]['texts'])} lines  {raw[f]['latency_s']:.1f}s", flush=True)
        if i % 10 == 0:
            json.dump(raw, open(RAW, "w"))
    json.dump(raw, open(RAW, "w"))
    print(f"OCR done -> {RAW}")


def reading_order(entry):
    """Top-to-bottom, then left-to-right, by box centre."""
    items = []
    for text, box in zip(entry["texts"], entry["boxes"]):
        ys, xs = [p[1] for p in box], [p[0] for p in box]
        items.append((sum(ys) / len(ys), sum(xs) / len(xs), text))
    h = sorted(items)
    return [t for _, _, t in h]


def run_parse():
    raw = json.load(open(RAW))
    pred, oracle = {}, {}
    for f, e in raw.items():
        lines = reading_order(e)
        chosen, cands = select(lines)
        y, m, d = chosen if chosen else (None, None, None)
        pred[f] = {"filename": f, "year": y, "month": m, "day": d,
                   "latency_s": e["latency_s"], "candidates": cands, "ocr_lines": lines}
        oracle[f] = {"filename": f, "candidates": cands}
    json.dump(pred, open(OUT / "ocr_baseline_predictions.json", "w"), indent=1, ensure_ascii=False)
    json.dump(oracle, open(OUT / "ocr_baseline_candidates.json", "w"), indent=1, ensure_ascii=False)
    n_any = sum(1 for v in pred.values() if v["candidates"])
    print(f"parsed {len(pred)} images; a date was found on {n_any}; "
          f"mean candidates per image {sum(len(v['candidates']) for v in pred.values())/len(pred):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["ocr", "parse"])
    {"ocr": run_ocr, "parse": run_parse}[ap.parse_args().step]()
