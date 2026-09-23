#!/usr/bin/env python3
"""
ToMe (Token Merging) evaluation for the fine-tuned A3 Qwen3.5-0.8B model.

Applies training-free bipartite soft matching (Bolya et al., 2022) to the
visual tokens the vision tower outputs (post 2x2 spatial merger, before the
LLM). True in-block ToMe is incompatible with this architecture: the merger
needs the intact patch grid to group 2x2 neighbours, so the merge point is
the ViT output, where token count is free to change. The LLM prefill then
runs on keep_ratio * N visual tokens.

Merged tokens are size-weighted averages; each surviving token keeps the
M-RoPE position of its destination token. Prefill and greedy decode are done
manually so position_ids stay exact.

Run:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n llm python /home/penghao/qwen/eval_tome.py \
        --ratios 1.0 0.75 0.5 0.25
"""
import re
import json
import time
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH  = "/home/penghao/qwen/groupA_a3_merged"
TEST_FOLDER = "/home/penghao/Dataset/test/"
TEST_JSON   = "/home/penghao/Dataset/test.json"
OUT_DIR     = Path("/home/penghao/qwen/gguf_results/tome")

MAX_PX         = 1024 * 1024   # match training / eval_native.py
GEN_MAX_TOKENS = 2000

PROMPT = """You are reading the expiration date on a food package. Find the
expiration date (also called "best before," "use by," "BB," "EXP," "BBE,"
"consume before," or equivalent). If both a production date and an
expiration date appear, use the EXPIRATION date.

Step 1: In one short sentence, state the expiration date you see.
Step 2: On a new line, output a JSON object with exactly these fields:
{"year": <4-digit integer or null>, "month": <integer 1-12 or null>, "day": <integer 1-31 or null>}

If no expiration date is visible, say so in Step 1 and output:
{"year": null, "month": null, "day": null}

Output only the sentence and the JSON. No markdown fences."""


def resize_image(image):
    w, h = image.size
    if w * h > MAX_PX:
        scale = (MAX_PX / (w * h)) ** 0.5
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def extract_json(text):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


# ── ToMe: iterative bipartite soft matching ──────────────────────────────────

@torch.no_grad()
def tome_merge(feats: torch.Tensor, keep_ratio: float):
    """
    feats: (N, D) visual tokens. Returns (merged (M, D), kept original indices (M,)).
    Size-weighted averaging; each bipartite round merges at most half the tokens,
    so low ratios take several rounds.
    """
    N = feats.shape[0]
    target = max(1, int(round(N * keep_ratio)))
    if target >= N:
        return feats, torch.arange(N, device=feats.device)

    x = feats.float()
    sizes = torch.ones(N, device=feats.device)
    orig = torch.arange(N, device=feats.device)

    while x.shape[0] > target:
        n = x.shape[0]
        r = min(n - target, (n - 1) // 2)
        if r <= 0:
            break
        nrm = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a_idx = torch.arange(0, n, 2, device=x.device)   # sources
        b_idx = torch.arange(1, n, 2, device=x.device)   # destinations
        sim = nrm[a_idx] @ nrm[b_idx].T                  # (nA, nB)
        best_sim, best_dst = sim.max(dim=-1)
        merge_order = best_sim.argsort(descending=True)
        src_sel = merge_order[:r]                        # A rows to merge away
        dst_sel = best_dst[src_sel]                      # B rows receiving them

        # weighted sums into destinations
        wsum = x[b_idx] * sizes[b_idx, None]
        wsz = sizes[b_idx].clone()
        wsum.index_add_(0, dst_sel, x[a_idx[src_sel]] * sizes[a_idx[src_sel], None])
        wsz.index_add_(0, dst_sel, sizes[a_idx[src_sel]])
        new_b = wsum / wsz[:, None]

        keep_a_mask = torch.ones(len(a_idx), dtype=torch.bool, device=x.device)
        keep_a_mask[src_sel] = False
        keep_a = a_idx[keep_a_mask]

        cur_idx = torch.cat([keep_a, b_idx])
        cur_x = torch.cat([x[keep_a], new_b])
        cur_sz = torch.cat([sizes[keep_a], wsz])
        cur_orig = torch.cat([orig[keep_a], orig[b_idx]])

        order = cur_orig.argsort()                       # preserve spatial order
        x, sizes, orig = cur_x[order], cur_sz[order], cur_orig[order]

    return x.to(feats.dtype), orig


# ── Manual prefill + greedy decode with exact M-RoPE positions ───────────────

@torch.no_grad()
def generate_tome(model, processor, image, keep_ratio, eos_id, image_token_id):
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]                       # (1, L)
    L = input_ids.shape[1]

    # visual features (post spatial-merger): (N, hidden)
    vis = model.model.get_image_features(
        inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
    image_embeds = torch.cat(list(vis.pooler_output), dim=0)
    N = image_embeds.shape[0]

    # full-sequence 3D positions, before any reduction
    position_ids, _ = model.model.get_rope_index(
        input_ids,
        mm_token_type_ids=inputs["mm_token_type_ids"],
        image_grid_thw=inputs["image_grid_thw"],
        attention_mask=inputs["attention_mask"],
    )                                                     # (3, 1, L)

    merged, kept = tome_merge(image_embeds, keep_ratio)
    M = merged.shape[0]

    embeds = model.get_input_embeddings()(input_ids)      # (1, L, D)
    img_pos = (input_ids[0] == image_token_id).nonzero().squeeze(-1)
    assert img_pos.numel() == N, f"placeholders {img_pos.numel()} != visual tokens {N}"

    embeds[0, img_pos[kept]] = merged.to(embeds.dtype)
    keep_mask = torch.ones(L, dtype=torch.bool, device=device)
    keep_mask[img_pos] = False
    keep_mask[img_pos[kept]] = True

    embeds_red = embeds[:, keep_mask]
    pos_red = position_ids[:, :, keep_mask]

    # prefill
    out = model.model(inputs_embeds=embeds_red, position_ids=pos_red, use_cache=True)
    logits = model.lm_head(out.last_hidden_state[:, -1:])

    # greedy decode; text positions are identical across the 3 M-RoPE planes
    next_pos = int(position_ids.max().item()) + 1
    generated = []
    for step in range(GEN_MAX_TOKENS):
        tok = int(logits[0, -1].argmax().item())
        if tok == eos_id:
            break
        generated.append(tok)
        emb = model.get_input_embeddings()(
            torch.tensor([[tok]], device=device))
        pos = torch.full((3, 1, 1), next_pos + step, device=device, dtype=pos_red.dtype)
        out = model.model(inputs_embeds=emb, position_ids=pos,
                          past_key_values=out.past_key_values, use_cache=True)
        logits = model.lm_head(out.last_hidden_state[:, -1:])

    raw = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return raw, N, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    ap.add_argument("--limit", type=int, default=0, help="only first K images (0 = all)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="cuda")
    model.eval()
    image_token_id = model.config.image_token_id
    eos_id = model.generation_config.eos_token_id

    with open(TEST_JSON, encoding="utf-8") as f:
        test_files = [e["filename"] for e in json.load(f)]
    if args.limit:
        test_files = test_files[:args.limit]

    for ratio in args.ratios:
        out_path = OUT_DIR / f"tome_r{int(ratio*100):03d}.json"
        results = {}
        if out_path.exists():
            for fn, entry in json.load(open(out_path)).items():
                if "error" not in entry and "parse_error" not in entry:
                    results[fn] = entry
            print(f"[r={ratio}] resuming: {len(results)} done")

        pending = [f for f in test_files if f not in results]
        print(f"\n=== keep_ratio={ratio}  ({len(pending)} images) ===")
        wall0 = time.time()
        for i, fn in enumerate(pending, 1):
            path = Path(TEST_FOLDER) / fn
            if not path.exists():
                continue
            raw = None
            try:
                image = resize_image(Image.open(path).convert("RGB"))
                t0 = time.time()
                raw, n_vis, n_kept = generate_tome(
                    model, processor, image, ratio, eos_id, image_token_id)
                lat = time.time() - t0
                parsed = extract_json(raw)
                results[fn] = {"filename": fn, "raw_response": raw,
                               "latency_s": round(lat, 3),
                               "visual_tokens": n_vis, "kept_tokens": n_kept,
                               **parsed}
                print(f"  [{i:>4}/{len(pending)}] {fn} ok ({lat:.2f}s, {n_vis}->{n_kept} tok)")
            except Exception as e:
                results[fn] = {"filename": fn, "parse_error": str(e),
                               "raw_response": raw,
                               "year": None, "month": None, "day": None}
                print(f"  [{i:>4}/{len(pending)}] {fn} FAIL {str(e)[:80]}")
            if i % 20 == 0:
                json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
                el = time.time() - wall0
                print(f"    ETA {(len(pending)-i)*(el/i)/60:.1f} min")

        json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
        print(f"[r={ratio}] saved -> {out_path}")


if __name__ == "__main__":
    main()
