#!/usr/bin/env python3
"""Prepare paired protein-CDS examples and frozen ESM2 embeddings.

Each row is a true pair from the same phage/protein annotation.  The first
interface-training pass uses the protein CDS as the DNA target (bounded to a
local 256-nt window); this is deliberately smaller than whole-genome training
so that the adapter experiment is measurable on one RTX 5090.
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
DEFAULT_INPUT = PROJECT / "results/11_microviridae_cluster_splits/train.tsv"
DEFAULT_MODEL = PROJECT / "models/esm2/esm2_t33_650M_UR50D"
DEFAULT_OUT = PROJECT / "results/14_microviridae_paired_embeddings"
DNA_RE = re.compile(r"^[ACGT]+$")


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gpu", default="1")
    p.add_argument("--max-pairs", type=int, default=2048)
    p.add_argument("--max-per-label", type=int, default=0)
    p.add_argument("--max-dna-length", type=int, default=256)
    p.add_argument("--min-dna-length", type=int, default=48)
    p.add_argument("--max-protein-length", type=int, default=1022)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260907)
    return p.parse_args()


def stable(row: dict[str, str], seed: int) -> str:
    return hashlib.sha256((f"{seed}\0" + row.get("phage_id", "") + "\0" + row.get("protein_id", "")).encode()).hexdigest()


def main() -> int:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer, EsmModel
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible CUDA device")

    rows = []
    with a.input.open(encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h, delimiter="\t"):
            if row.get("label_scope") != "single":
                continue
            protein = (row.get("protein_sequence") or "").strip().upper().rstrip("*")
            dna = (row.get("cds_sequence") or "").strip().upper()
            if len(protein) < 20 or len(protein) > a.max_protein_length:
                continue
            if len(dna) < a.min_dna_length or not DNA_RE.fullmatch(dna):
                continue
            row["protein_sequence"] = protein
            row["dna_sequence"] = dna[: a.max_dna_length]
            row["label"] = row.get("normalized_function", "unknown")
            rows.append(row)
    if a.max_per_label:
        selected = []
        for label in sorted({r["label"] for r in rows}):
            group = sorted((r for r in rows if r["label"] == label), key=lambda r: stable(r, a.seed))
            selected.extend(group[: a.max_per_label])
        rows = selected
    else:
        rows = sorted(rows, key=lambda r: stable(r, a.seed))
    rows = rows[: a.max_pairs]
    if not rows:
        raise RuntimeError("No valid paired protein/CDS rows")

    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = EsmModel.from_pretrained(a.model, local_files_only=True, add_pooling_layer=False, torch_dtype=torch.float32).cuda().eval()
    model.requires_grad_(False)
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(rows), a.batch_size):
            batch = rows[start : start + a.batch_size]
            encoded = tokenizer([r["protein_sequence"] for r in batch], return_tensors="pt", padding=True, truncation=False, return_special_tokens_mask=True)
            special = encoded.pop("special_tokens_mask").bool().cuda()
            encoded = {k: v.cuda() for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].bool() & ~special
            embeddings.append(((hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)).float().cpu())
    embedding_tensor = torch.cat(embeddings, dim=0).half()
    torch.save(embedding_tensor, a.output_dir / "protein_embeddings.pt")

    fields = ["pair_index", "phage_id", "protein_id", "genome_cluster", "label", "protein_length_aa", "dna_length_nt", "protein_sequence", "dna_sequence"]
    with (a.output_dir / "paired_examples.tsv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for i, row in enumerate(rows):
            w.writerow({"pair_index": i, "phage_id": row.get("phage_id", ""), "protein_id": row.get("protein_id", ""), "genome_cluster": row.get("genome_cluster", ""), "label": row["label"], "protein_length_aa": len(row["protein_sequence"]), "dna_length_nt": len(row["dna_sequence"]), "protein_sequence": row["protein_sequence"], "dna_sequence": row["dna_sequence"]})

    summary = {
        "pairs": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "max_dna_length": a.max_dna_length,
        "embedding_shape": list(embedding_tensor.shape),
        "embedding_dtype": str(embedding_tensor.dtype),
        "model": str(a.model),
        "outputs": {"pairs": str(a.output_dir / "paired_examples.tsv"), "embeddings": str(a.output_dir / "protein_embeddings.pt")},
    }
    (a.output_dir / "pair_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
