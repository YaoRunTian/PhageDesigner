#!/usr/bin/env python3
"""Build a bounded ESM2-650M reference-protein library for conditioning.

The full Microviridae manifest is intentionally not embedded at this stage.
This script selects a reproducible, de-duplicated reference panel from the
quality-controlled annotations, with extra priority for spike/G-protein
products.  ESM2 remains frozen; this is an embedding-cache step only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_INPUT = PROJECT / "results/10_microviridae_function_labels/microviridae_protein_function_manifest.tsv"
DEFAULT_OUT = PROJECT / "results/12_esm2_microviridae_reference_library"
DEFAULT_MODEL = PROJECT / "models/esm2/esm2_t33_650M_UR50D"
SPIKE_RE = re.compile(r"spike|major\s+spike|\bG\s+protein\b", re.I)


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gpu", default="1")
    p.add_argument("--max-per-label", type=int, default=32)
    p.add_argument("--max-spike", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-protein-length", type=int, default=1022)
    return p.parse_args()


def stable(row):
    return hashlib.sha256((row.get("protein_sequence", "") + "\0" + row.get("protein_id", "")).encode()).hexdigest()


def main() -> int:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with a.input.open(encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h, delimiter="\t"):
            seq = (row.get("protein_sequence") or "").strip().upper().rstrip("*")
            if not seq or len(seq) > a.max_protein_length:
                continue
            row["protein_sequence"] = seq
            rows.append(row)

    # De-duplicate by amino-acid sequence while retaining the most informative
    # annotation row.  Spike/G rows are selected first and marked explicitly.
    unique = {}
    for row in sorted(rows, key=stable):
        unique.setdefault(row["protein_sequence"], row)
    rows = list(unique.values())
    selected = []
    selected_seq = set()
    spike_rows = [r for r in rows if SPIKE_RE.search(r.get("product", ""))]
    for row in sorted(spike_rows, key=stable)[: a.max_spike]:
        row = dict(row); row["reference_role"] = "spike_or_G_product"
        selected.append(row); selected_seq.add(row["protein_sequence"])
    by_label = {}
    for row in rows:
        label = row.get("normalized_function", "unknown")
        if row.get("label_scope") != "single" or label == "unknown":
            continue
        by_label.setdefault(label, []).append(row)
    for label in sorted(by_label):
        n = 0
        for row in sorted(by_label[label], key=stable):
            if row["protein_sequence"] in selected_seq:
                continue
            row = dict(row); row["reference_role"] = "single_function_reference"
            selected.append(row); selected_seq.add(row["protein_sequence"])
            n += 1
            if n >= a.max_per_label:
                break
    if not selected:
        raise RuntimeError("No eligible reference proteins found")

    # Use physical GPU selection only after argument parsing; with CUDA_VISIBLE_DEVICES
    # set, the requested physical GPU appears as cuda:0 inside this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer, EsmModel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = EsmModel.from_pretrained(a.model, local_files_only=True, add_pooling_layer=False, torch_dtype=torch.float32).to(device).eval()
    model.requires_grad_(False)
    tensors = []
    with torch.no_grad():
        for start in range(0, len(selected), a.batch_size):
            batch = selected[start : start + a.batch_size]
            encoded = tokenizer([r["protein_sequence"] for r in batch], return_tensors="pt", padding=True, truncation=False, return_special_tokens_mask=True)
            special = encoded.pop("special_tokens_mask").bool()
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].bool() & ~special.to(device)
            for i, row in enumerate(batch):
                valid = mask[i]
                tensors.append((hidden[i][valid].mean(0).float().cpu()))
    embeddings = torch.stack(tensors)
    model_info = {"model": str(a.model), "device": str(device), "embedding_dim": int(embeddings.shape[1]), "dtype": str(embeddings.dtype)}
    torch.save(embeddings, a.output_dir / "reference_embeddings.pt")

    fields = list(selected[0]) + ["reference_index"]
    with (a.output_dir / "reference_proteins.tsv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(selected):
            row = dict(row); row["reference_index"] = str(i); w.writerow(row)
    with (a.output_dir / "reference_proteins.fasta").open("w", encoding="utf-8") as h:
        for i, row in enumerate(selected):
            h.write(f">ref_{i}|{row.get('phage_id','')}|{row.get('protein_id','')}|{row.get('reference_role','')}\n{row['protein_sequence']}\n")

    summary = {
        "input_rows_after_length_filter": len(rows),
        "unique_protein_sequences": len(unique),
        "selected_references": len(selected),
        "spike_or_G_references": sum(r.get("reference_role") == "spike_or_G_product" for r in selected),
        "reference_function_counts": dict(Counter(r.get("normalized_function", "unknown") for r in selected)),
        "model": model_info,
        "outputs": {
            "embeddings": str(a.output_dir / "reference_embeddings.pt"),
            "proteins": str(a.output_dir / "reference_proteins.tsv"),
            "fasta": str(a.output_dir / "reference_proteins.fasta"),
        },
    }
    (a.output_dir / "reference_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
