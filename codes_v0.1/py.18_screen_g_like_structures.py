#!/usr/bin/env python3
"""Predict and compare structures for pre-screened PhiX174 G-like proteins.

ESMFold predicts the reference and candidate structures under identical
settings.  Candidates are compared at corresponding C-alpha positions after
Kabsch superposition.  The reported TM-like score is an inexpensive screening
metric, not a replacement for experimental structural validation or TM-align.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
PHAGE_ROOT = Path("/public8/lilab/student/rtyao/phage")
DEFAULT_INPUT = PROJECT / "results/17_g_like_sequence_screen"
DEFAULT_REFERENCE = PHAGE_ROOT / "test_king/04b_synteny/phix174_reference_proteins.fasta"
DEFAULT_MODEL = PROJECT / "models/esmfold/esmfold_v1"
DEFAULT_OUT = PROJECT / "results/18_g_like_final_screen"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--reference-fasta", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--reference-id", default="NC_001422.1_ORF.3")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gpu", default="1")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-recycles", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--min-mean-plddt", type=float, default=40.0)
    p.add_argument("--min-tm-like", type=float, default=0.50)
    p.add_argument("--max-ca-rmsd", type=float, default=5.0)
    return p.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    header = None
    parts = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header, parts = line[1:].split()[0], []
            elif line:
                parts.append(line.upper())
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def structural_metrics(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    x = candidate - candidate.mean(axis=0, keepdims=True)
    y = reference - reference.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    aligned = x @ rotation
    distances = np.linalg.norm(aligned - y, axis=1)
    rmsd = float(np.sqrt(np.mean(distances**2)))
    length = len(distances)
    d0 = max(0.5, 1.24 * np.cbrt(max(length - 15, 1)) - 1.8)
    tm_like = float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))
    return rmsd, tm_like


def main() -> int:
    a = parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    records = read_fasta(a.input_dir / "prestructure_pass.fasta")
    if not records:
        raise RuntimeError("No candidates passed the pre-structure screen")
    references = dict(read_fasta(a.reference_fasta))
    reference = references[a.reference_id]

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = EsmForProteinFolding.from_pretrained(
        a.model, local_files_only=True, low_cpu_mem_usage=True
    ).cuda().eval()
    model.esm = model.esm.half()
    model.trunk.set_chunk_size(a.chunk_size)
    model.requires_grad_(False)

    def infer(sequences: list[str]):
        tokens = tokenizer(
            sequences, return_tensors="pt", add_special_tokens=False, padding=True
        )
        tokens = {k: v.cuda() for k, v in tokens.items()}
        with torch.no_grad():
            output = model(**tokens, num_recycles=a.num_recycles)
        ca = output.positions[-1][:, :, 1, :].float().cpu().numpy()
        plddt = output.plddt[:, :, 1].float().cpu().numpy()
        if float(np.nanmax(plddt)) <= 1.5:
            plddt = plddt * 100.0
        lengths = tokens["attention_mask"].sum(dim=1).cpu().tolist()
        return [(ca[i, : int(n)], plddt[i, : int(n)]) for i, n in enumerate(lengths)]

    reference_ca, reference_plddt = infer([reference])[0]
    metrics = []
    for start in range(0, len(records), a.batch_size):
        batch = records[start : start + a.batch_size]
        predictions = infer([sequence for _, sequence in batch])
        for (candidate_id, sequence), (ca, plddt) in zip(batch, predictions):
            rmsd, tm_like = structural_metrics(ca, reference_ca)
            mean_plddt = float(np.mean(plddt))
            metrics.append(
                {
                    "candidate_id": candidate_id,
                    "mean_plddt": mean_plddt,
                    "ca_rmsd": rmsd,
                    "tm_like_score": tm_like,
                    "structure_pass": mean_plddt >= a.min_mean_plddt
                    and rmsd <= a.max_ca_rmsd
                    and tm_like >= a.min_tm_like,
                    "sequence": sequence,
                }
            )
        print(f"structures {min(start + a.batch_size, len(records))}/{len(records)}", flush=True)

    earlier = {}
    with (a.input_dir / "sequence_screening.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            earlier[row["candidate_id"]] = row
    for row in metrics:
        row["mask_rate"] = float(earlier[row["candidate_id"]]["mask_rate"])
    fields = ["candidate_id", "mask_rate", "mean_plddt", "ca_rmsd", "tm_like_score", "structure_pass", "sequence"]
    with (a.output_dir / "structure_metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(metrics)
    with (a.output_dir / "final_pass.fasta").open("w", encoding="utf-8") as handle:
        for row in metrics:
            if row["structure_pass"]:
                handle.write(f">{row['candidate_id']}\n{row['sequence']}\n")

    by_rate = Counter(f"{row['mask_rate']:.2f}" for row in metrics if row["structure_pass"])
    summary = {
        "status": "passed",
        "structure_model": str(a.model),
        "reference_id": a.reference_id,
        "reference_mean_plddt": float(np.mean(reference_plddt)),
        "screened_after_previous_stages": len(metrics),
        "thresholds": {
            "min_mean_plddt": a.min_mean_plddt,
            "min_tm_like": a.min_tm_like,
            "max_ca_rmsd": a.max_ca_rmsd,
            "num_recycles": a.num_recycles,
        },
        "final_pass": sum(row["structure_pass"] for row in metrics),
        "final_pass_by_mask_rate": dict(by_rate),
        "limitations": [
            "ESMFold is a computational structure screen and shares ESM-family representations.",
            "TM-like score uses corresponding C-alpha atoms after Kabsch fitting; it is not TM-align.",
        ],
        "outputs": {
            "metrics": str(a.output_dir / "structure_metrics.tsv"),
            "final_fasta": str(a.output_dir / "final_pass.fasta"),
        },
    }
    (a.output_dir / "final_screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
