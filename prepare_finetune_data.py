"""
Converts annotations.json + images into JSONL for fine-tuning.

Sources:
  Products-Real/train/      → train.jsonl  (1102 images, raw transcription format)
  Products-Real/evaluation/ → val.jsonl    (665 images,  dmy_ann structured format)

Train annotations use:
  cls="date"  — expiration or production date with raw transcription string
  cls="due"   — bbox of the expiration label (no text)
  cls="prod"  — bbox of the production label (no text)

Eval annotations use:
  cls="exp"   — expiration date, nested dmy_ann with year/month/day/label fields
"""

import json
import re
from pathlib import Path

TRAIN_IMAGE_DIR = Path("/home/penghao/qwen/Products-Real/train/images")
TRAIN_GT_JSON   = Path("/home/penghao/qwen/Products-Real/train/annotations.json")
EVAL_IMAGE_DIR  = Path("/home/penghao/qwen/Products-Real/evaluation/images")
EVAL_GT_JSON    = Path("/home/penghao/qwen/Products-Real/evaluation/annotations.json")
OUTPUT_DIR      = Path("/home/penghao/qwen/finetune_data")

PROMPT = (
    "You are a strict OCR bot. Find the EXPIRATION date on this product label and return it as JSON.\n\n"
    "STEP 1 — FIND THE RIGHT DATE:\n"
    "- Expiration labels (use this date): EXP, EXPIRY, EXPIRATION, USE BY, USE BEFORE, BEST BY, BEST BEFORE, BB, BBE, SELL BY, BFD, BEST IF USED BY.\n"
    "- Manufacturing labels (IGNORE this date): MFG, MFGD, MFD, MANUFACTURED, PROD, PRD, DOM, PKD, PACKED, MAN.\n"
    "- If there is only ONE date and no label, assume it is the expiration date.\n"
    "- If there are TWO unlabeled dates, the LATER date is typically the expiration date.\n\n"
    "STEP 2 — TRANSCRIBE EXACTLY (never guess or convert):\n"
    "- 'year': Copy digits exactly. '21' stays '21', '2024' stays '2024'. If absent → null.\n"
    "- 'month': Copy exactly (number or text). '08' stays '08', 'MAR' stays 'MAR'. If absent → null.\n"
    "- 'day': The day of the month (a number between 1-31). If no day is printed → null. NEVER copy the year or month into this field.\n"
    "- 'label': The expiration label text (e.g. 'EXP', 'Best By'). If none → null.\n\n"
    "STEP 3 — OUTPUT:\n"
    "Return ONLY a raw JSON object. No markdown, no explanation.\n"
    '{"year": ..., "month": ..., "day": ..., "label": ...}'
)

# ---------------------------------------------------------------------------
# Eval data parsing (structured dmy_ann format)
# ---------------------------------------------------------------------------

def extract_gt_eval(gt_info):
    result = {"year": None, "month": None, "day": None, "label": None}
    for ann in gt_info.get("ann", []):
        if ann.get("cls") == "exp":
            for dmy in ann.get("dmy_ann", []):
                k = dmy.get("cls")
                if k in result:
                    val = str(dmy.get("transcription", "")).strip()
                    result[k] = val if val else None
            break
    return result

# ---------------------------------------------------------------------------
# Train data parsing (raw transcription string format)
# ---------------------------------------------------------------------------

MONTH_NAMES = {
    "JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC",
    "ENE","ABR","AGO","DIC",  # Spanish abbreviations
}

def parse_transcription(s):
    """
    Parse a raw date string into (year_str, month_str, day_str).
    Preserves the original digit strings (e.g. '21' stays '21', not '2021').
    Returns (None, None, None) if unparseable.
    """
    s = s.strip()

    # Find month name abbreviation
    month_name = None
    for mn in MONTH_NAMES:
        if re.search(r'(?<![A-Za-z])' + mn + r'(?![A-Za-z])', s.upper()):
            month_name = mn
            break

    nums = re.findall(r'\d+', s)
    if not nums:
        return None, None, None

    year = month = day = None

    if month_name:
        month = month_name
        four_d = [n for n in nums if len(n) == 4]
        if four_d:
            year = four_d[0]
            others = [n for n in nums if n != year]
            day = others[0] if others else None
        else:
            # e.g. "MAR 22 21" or "FEB/26/21"
            if len(nums) >= 2:
                # Assume first remaining = day, second = year
                day, year = nums[0], nums[1]
            elif len(nums) == 1:
                year = nums[0]
        return year, month, day

    four_d = [n for n in nums if len(n) == 4]

    if four_d:
        year = four_d[0]
        others = [n for n in nums if n != year and len(n) <= 2]
        if len(others) >= 2:
            a, b = int(others[0]), int(others[1])
            yr_idx = next(i for i, n in enumerate(nums) if len(n) == 4)
            if a > 12:
                day, month = others[0], others[1]
            elif b > 12:
                month, day = others[0], others[1]
            elif yr_idx == 0:
                # YYYY.MM.DD
                month, day = others[0], others[1]
            else:
                # DD.MM.YYYY (year at end, European format)
                day, month = others[0], others[1]
        elif len(others) == 1:
            month = others[0]
        return year, month, day

    # All short numbers — 3 components
    if len(nums) == 3:
        a, b, c = int(nums[0]), int(nums[1]), int(nums[2])
        if b > 12:
            # Middle can't be month → MM.DD.YY
            month, day, year = nums[0], nums[1], nums[2]
        elif a in range(18, 35):
            # First component is a plausible 2-digit year (2018–2034) → YY.MM.DD
            year, month, day = nums[0], nums[1], nums[2]
        elif c in range(18, 35) and a > 12:
            # First > 12 so it's a day; last is year → DD.MM.YY
            day, month, year = nums[0], nums[1], nums[2]
        elif a > 12:
            # First > 12 can't be month → day first → DD.MM.YY
            day, month, year = nums[0], nums[1], nums[2]
        else:
            # Default: YY.MM.DD
            year, month, day = nums[0], nums[1], nums[2]
        return year, month, day

    # 2 components — partial date (MM/YYYY or YYYY/MM)
    if len(nums) == 2:
        a, b = int(nums[0]), int(nums[1])
        if len(nums[0]) == 4 or a > 31:
            year, month = nums[0], nums[1]
        elif len(nums[1]) == 4 or b > 31:
            month, year = nums[0], nums[1]
        elif a in range(18, 35):
            year, month = nums[0], nums[1]
        else:
            month, year = nums[0], nums[1]
        return year, month, None

    if len(nums) == 1:
        return nums[0], None, None

    return None, None, None


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def find_exp_transcription(info):
    """
    From a train-format annotation, return the transcription of the
    expiration date annotation.  Returns None if not determinable.
    """
    dates = [a for a in info.get("ann", [])
             if a.get("cls") == "date" and a.get("transcription") and a.get("bbox")]
    if not dates:
        return None
    if len(dates) == 1:
        return dates[0]["transcription"]

    due_bboxes  = [a["bbox"] for a in info.get("ann", []) if a.get("cls") == "due"  and a.get("bbox")]
    prod_bboxes = [a["bbox"] for a in info.get("ann", []) if a.get("cls") == "prod" and a.get("bbox")]

    if due_bboxes:
        # Date closest to any 'due' label → expiration
        due_center = _bbox_center(due_bboxes[0])
        return min(dates, key=lambda a: _dist(_bbox_center(a["bbox"]), due_center))["transcription"]

    if prod_bboxes:
        # Date closest to 'prod' label → production; the OTHER one is expiration
        prod_center = _bbox_center(prod_bboxes[0])
        prod_ann = min(dates, key=lambda a: _dist(_bbox_center(a["bbox"]), prod_center))
        exp_dates = [a for a in dates if a is not prod_ann]
        return exp_dates[0]["transcription"] if exp_dates else None

    # No labels: take the last (usually latest) date
    return dates[-1]["transcription"]


def extract_gt_train(gt_info):
    transcription = find_exp_transcription(gt_info)
    if transcription is None:
        return None
    year, month, day = parse_transcription(transcription)
    return {"year": year, "month": month, "day": day, "label": None}

# ---------------------------------------------------------------------------
# Sample builder
# ---------------------------------------------------------------------------

def build_sample(image_path, gt):
    return {
        "messages": [
            {"role": "user",      "content": f"<image>\n{PROMPT}"},
            {"role": "assistant", "content": json.dumps(gt, ensure_ascii=False)},
        ],
        "images": [str(image_path)],
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Training set: Products-Real/train ---
    train_gt = json.loads(TRAIN_GT_JSON.read_text())
    train_samples = []
    train_skipped = 0
    for fname, info in train_gt.items():
        img_path = TRAIN_IMAGE_DIR / fname
        if not img_path.exists():
            train_skipped += 1
            continue
        gt = extract_gt_train(info)
        if gt is None:
            train_skipped += 1
            continue
        train_samples.append(build_sample(img_path, gt))

    # --- Validation set: Products-Real/evaluation ---
    eval_gt = json.loads(EVAL_GT_JSON.read_text())
    val_samples = []
    val_skipped = 0
    for fname, info in eval_gt.items():
        img_path = EVAL_IMAGE_DIR / fname
        if not img_path.exists():
            val_skipped += 1
            continue
        gt = extract_gt_eval(info)
        val_samples.append(build_sample(img_path, gt))

    def write_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path   = OUTPUT_DIR / "val.jsonl"
    write_jsonl(train_path, train_samples)
    write_jsonl(val_path,   val_samples)

    print(f"Train: {len(train_samples)} samples  (skipped: {train_skipped})  → {train_path}")
    print(f"Val:   {len(val_samples)} samples  (skipped: {val_skipped})  → {val_path}")
    print(f"\nSample train entry:")
    print(json.dumps(train_samples[0], indent=2, ensure_ascii=False)[:400])
    print(f"\nSample val entry:")
    print(json.dumps(val_samples[0], indent=2, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
