#!/usr/bin/env python3
"""
Merge a llama.cpp LLM GGUF and mmproj GGUF into a single unified GGUF file.
The merged file contains all LLM metadata + all clip.* metadata from mmproj,
plus tensors from both files.

Usage:
    python merge_gguf.py --llm <llm.gguf> --mmproj <mmproj.gguf> --output <out.gguf>
"""
import argparse
import numpy as np
from pathlib import Path
import gguf

SKIP_FROM_MMPROJ = {
    "general.architecture", "general.type", "general.name", "general.size_label",
    "general.quantization_version", "general.file_type",
}


def merge(llm_path: Path, mmproj_path: Path, out_path: Path):
    llm    = gguf.GGUFReader(str(llm_path),    mode="r")
    mmproj = gguf.GGUFReader(str(mmproj_path), mode="r")

    writer = gguf.GGUFWriter(str(out_path), arch="qwen35")

    # ── KV metadata ──────────────────────────────────────────────────────────
    skip_auto = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                 "general.architecture"}

    for name, field in llm.fields.items():
        if name in skip_auto:
            continue
        _copy_field(writer, name, field)

    for name, field in mmproj.fields.items():
        if name in skip_auto or name in SKIP_FROM_MMPROJ:
            continue
        _copy_field(writer, name, field)

    # ── Tensors ───────────────────────────────────────────────────────────────
    for tensor in llm.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)

    for tensor in mmproj.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Merged → {out_path}  ({out_path.stat().st_size / 1024**2:.0f} MB)")


def _copy_field(writer: gguf.GGUFWriter, name: str, field):
    t = field.types[0]
    GGUFValueType = gguf.GGUFValueType

    if t == GGUFValueType.STRING:
        writer.add_string(name, str(bytes(field.parts[-1]), "utf-8"))
    elif t == GGUFValueType.BOOL:
        writer.add_bool(name, bool(field.parts[-1][0]))
    elif t == GGUFValueType.UINT8:
        writer.add_uint8(name, int(field.parts[-1][0]))
    elif t == GGUFValueType.INT8:
        writer.add_int8(name, int(field.parts[-1][0]))
    elif t == GGUFValueType.UINT16:
        writer.add_uint16(name, int(field.parts[-1][0]))
    elif t == GGUFValueType.INT16:
        writer.add_int16(name, int(field.parts[-1][0]))
    elif t == GGUFValueType.UINT32:
        writer.add_uint32(name, int(field.parts[-1][0]))
    elif t == GGUFValueType.INT32:
        writer.add_int32(name, int(field.parts[-1][0]))
    elif t == GGUFValueType.FLOAT32:
        writer.add_float32(name, float(field.parts[-1][0]))
    elif t == GGUFValueType.UINT64:
        writer.add_uint64(name, int(field.parts[-1][0]))
    elif t == GGUFValueType.INT64:
        writer.add_int64(name, int(field.parts[-1][0]))
    elif t == GGUFValueType.FLOAT64:
        writer.add_float64(name, float(field.parts[-1][0]))
    elif t == GGUFValueType.ARRAY:
        elem_type = field.types[1]
        if elem_type == GGUFValueType.STRING:
            arr = []
            for idx in field.data:
                raw = bytes(field.parts[idx])
                try:
                    arr.append(raw.decode("utf-8"))
                except UnicodeDecodeError:
                    arr.append(raw)
            writer.add_array(name, arr)
        else:
            writer.add_array(name, field.parts[-1].tolist())
    else:
        print(f"  [skip] unsupported type {t} for field {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm",    required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    merge(Path(args.llm), Path(args.mmproj), Path(args.output))
