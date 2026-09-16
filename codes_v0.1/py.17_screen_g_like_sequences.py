#!/usr/bin/env python3
"""Screen ESM2-generated PhiX174 G-like proteins before structure prediction.

Stages are cumulative and auditable: sequence integrity, direct MMseqs2 match
to PhiX174 G, PHROG-1483 assignment, conservation of naturally supported
positions, and the independently fine-tuned ESM2 functional classifier.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
PHAGE_ROOT = Path("/public8/lilab/student/rtyao/phage")
DEFAULT_INPUT = PROJECT / "results/16_g_like_esm2_3000"
DEFAULT_REFERENCE = PHAGE_ROOT / "test_king/04b_synteny/phix174_reference_proteins.fasta"
DEFAULT_PHROG_FASTA = PHAGE_ROOT / "tools/phrogs/FAA_phrog/phrog_1483.faa"
DEFAULT_PHROG_DB = PHAGE_ROOT / "tools/phrogs/phrogs_mmseqs_db"
DEFAULT_MMSEQS = PHAGE_ROOT / "tools/mmseqs/bin/mmseqs"
DEFAULT_CLASSIFIER = PROJECT / "results/05c_esm2_650m_100k_frozen_head/final_model"
DEFAULT_OUT = PROJECT / "results/17_g_like_sequence_screen"
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--reference-fasta", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--reference-id", default="NC_001422.1_ORF.3")
    p.add_argument("--phrog-fasta", type=Path, default=DEFAULT_PHROG_FASTA)
    p.add_argument("--phrog-db", type=Path, default=DEFAULT_PHROG_DB)
    p.add_argument("--mmseqs", type=Path, default=DEFAULT_MMSEQS)
    p.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gpu", default="1")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--min-identity", type=float, default=0.40)
    p.add_argument("--min-coverage", type=float, default=0.80)
    p.add_argument("--max-evalue", type=float, default=1e-5)
    p.add_argument("--conserved-frequency", type=float, default=0.80)
    p.add_argument("--min-conserved-retention", type=float, default=0.80)
    p.add_argument("--target-class", default="infection")
    p.add_argument("--min-class-probability", type=float, default=0.50)
    return p.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    parts: list[str] = []
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


def run(command: list[str], log_path: Path) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")


def parse_m8(path: Path) -> dict[str, dict]:
    best: dict[str, dict] = {}
    if not path.exists():
        return best
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) != 8:
                continue
            query, target = fields[:2]
            row = {
                "target": target,
                "pident": float(fields[2]) / 100.0,
                "alnlen": int(fields[3]),
                "qcov": float(fields[4]),
                "tcov": float(fields[5]),
                "evalue": float(fields[6]),
                "bits": float(fields[7]),
            }
            if query not in best or row["bits"] > best[query]["bits"]:
                best[query] = row
    return best


def conservation_profile(reference: str, homologs: list[str], frequency: float) -> dict:
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    counts = [Counter() for _ in reference]
    accepted = 0
    for homolog in homologs:
        homolog = homolog.rstrip("*")
        if not 120 <= len(homolog) <= 230 or set(homolog) - CANONICAL_AA:
            continue
        alignment = aligner.align(reference, homolog)[0]
        mapping: dict[int, str] = {}
        for (r0, r1), (h0, h1) in zip(alignment.aligned[0], alignment.aligned[1]):
            for offset in range(min(r1 - r0, h1 - h0)):
                mapping[int(r0 + offset)] = homolog[int(h0 + offset)]
        coverage = len(mapping) / len(reference)
        identity = sum(reference[i] == aa for i, aa in mapping.items()) / max(len(mapping), 1)
        if coverage < 0.70 or identity < 0.25:
            continue
        accepted += 1
        for i, aa in mapping.items():
            counts[i][aa] += 1
    if accepted < 10:
        raise RuntimeError(f"Too few natural G homologs for conservation profile: {accepted}")
    positions = []
    consensus = {}
    details = []
    for i, counter in enumerate(counts):
        coverage = sum(counter.values()) / accepted
        aa, n = counter.most_common(1)[0] if counter else ("-", 0)
        modal_frequency = n / max(sum(counter.values()), 1)
        conserved = coverage >= 0.80 and modal_frequency >= frequency
        if conserved:
            positions.append(i)
            consensus[i] = aa
        details.append(
            {
                "position_1based": i + 1,
                "reference_aa": reference[i],
                "consensus_aa": aa,
                "coverage": coverage,
                "consensus_frequency": modal_frequency,
                "conserved": conserved,
            }
        )
    return {
        "accepted_homologs": accepted,
        "positions": positions,
        "consensus": consensus,
        "details": details,
    }


def cumulative_counts(rows: list[dict], stages: list[str]) -> dict:
    output = {}
    active = [True] * len(rows)
    for stage in stages:
        active = [old and bool(row[stage]) for old, row in zip(active, rows)]
        output[stage] = {
            "total": sum(active),
            "by_mask_rate": dict(
                Counter(f"{row['mask_rate']:.2f}" for row, keep in zip(rows, active) if keep)
            ),
        }
    return output


def main() -> int:
    a = parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    records = [(h.split("|")[0], s) for h, s in read_fasta(a.input_dir / "g_like_3000.fasta")]
    metadata = {}
    with (a.input_dir / "generation_metadata.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            metadata[row["candidate_id"]] = row
    refs = dict(read_fasta(a.reference_fasta))
    reference = refs[a.reference_id]
    rows = []
    for candidate_id, sequence in records:
        meta = metadata[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "mask_rate": float(meta["mask_rate"]),
                "mutation_count": int(meta["mutation_count"]),
                "sequence": sequence,
                "integrity_pass": len(sequence) == 175
                and sequence.startswith("M")
                and not (set(sequence) - CANONICAL_AA),
            }
        )

    query_fasta = a.output_dir / "integrity_pass.fasta"
    with query_fasta.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["integrity_pass"]:
                handle.write(f">{row['candidate_id']}\n{row['sequence']}\n")

    direct_m8 = a.output_dir / "candidate_vs_phix174_G.m8"
    run(
        [
            str(a.mmseqs), "easy-search", str(query_fasta), str(a.reference_fasta),
            str(direct_m8), str(a.output_dir / "mmseqs_tmp_direct"), "--threads", str(a.threads),
            "-s", "7", "--max-seqs", "1", "--format-output",
            "query,target,pident,alnlen,qcov,tcov,evalue,bits",
        ],
        a.output_dir / "mmseqs_direct.log",
    )
    direct = parse_m8(direct_m8)
    for row in rows:
        hit = direct.get(row["candidate_id"], {})
        row["mmseqs_target"] = hit.get("target", "")
        for key in ["pident", "qcov", "tcov", "evalue", "bits"]:
            row[f"mmseqs_{key}"] = hit.get(key, 0.0 if key != "evalue" else 1.0)
        row["mmseqs_pass"] = (
            row["mmseqs_target"] == a.reference_id
            and row["mmseqs_pident"] >= a.min_identity
            and row["mmseqs_qcov"] >= a.min_coverage
            and row["mmseqs_tcov"] >= a.min_coverage
            and row["mmseqs_evalue"] <= a.max_evalue
        )

    phrog_m8 = a.output_dir / "candidate_vs_phrogs.m8"
    run(
        [
            str(a.mmseqs), "easy-search", str(query_fasta), str(a.phrog_db),
            str(phrog_m8), str(a.output_dir / "mmseqs_tmp_phrog"), "--threads", str(a.threads),
            "-s", "7", "--max-seqs", "10", "--format-output",
            "query,target,pident,alnlen,qcov,tcov,evalue,bits",
        ],
        a.output_dir / "mmseqs_phrog.log",
    )
    phrog = parse_m8(phrog_m8)
    for row in rows:
        hit = phrog.get(row["candidate_id"], {})
        row["phrog_target"] = hit.get("target", "")
        row["phrog_qcov"] = hit.get("qcov", 0.0)
        row["phrog_tcov"] = hit.get("tcov", 0.0)
        row["phrog_evalue"] = hit.get("evalue", 1.0)
        row["phrog_pass"] = (
            row["phrog_target"] == "phrog_1483"
            and row["phrog_qcov"] >= a.min_coverage
            and row["phrog_tcov"] >= a.min_coverage
            and row["phrog_evalue"] <= a.max_evalue
        )

    homologs = [s for _, s in read_fasta(a.phrog_fasta)]
    profile = conservation_profile(reference, homologs, a.conserved_frequency)
    conserved_positions = profile["positions"]
    consensus = profile["consensus"]
    with (a.output_dir / "conservation_profile.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile["details"][0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(profile["details"])
    for row in rows:
        retained = sum(row["sequence"][i] == consensus[i] for i in conserved_positions)
        row["conserved_sites"] = len(conserved_positions)
        row["conserved_retained"] = retained
        row["conserved_retention"] = retained / max(len(conserved_positions), 1)
        row["conservation_pass"] = row["conserved_retention"] >= a.min_conserved_retention

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer, EsmForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(a.classifier, local_files_only=True)
    model = EsmForSequenceClassification.from_pretrained(
        a.classifier, local_files_only=True, dtype=torch.float16
    ).cuda().eval()
    model.requires_grad_(False)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    target_id = next((i for i, label in id2label.items() if label == a.target_class), None)
    if target_id is None:
        raise ValueError(f"Target class not found in classifier: {a.target_class}")
    reference_tokens = tokenizer(reference, return_tensors="pt")
    reference_tokens = {k: v.cuda() for k, v in reference_tokens.items()}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        reference_probabilities = model(**reference_tokens).logits.float().softmax(-1)[0].cpu()
    reference_predicted_id = int(reference_probabilities.argmax())
    classifier_positive_control_pass = (
        reference_predicted_id == target_id
        and float(reference_probabilities[target_id]) >= a.min_class_probability
    )
    with torch.no_grad():
        for start in range(0, len(rows), a.batch_size):
            batch = rows[start : start + a.batch_size]
            tokens = tokenizer(
                [x["sequence"] for x in batch], return_tensors="pt", padding=True, truncation=True,
                max_length=1022,
            )
            tokens = {k: v.cuda() for k, v in tokens.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probabilities = model(**tokens).logits.float().softmax(-1).cpu()
            for item, probs in zip(batch, probabilities):
                predicted_id = int(probs.argmax())
                item["esm2_predicted_class"] = id2label[predicted_id]
                item["esm2_target_probability"] = float(probs[target_id])
                item["esm2_classification_pass"] = (
                    predicted_id == target_id
                    and item["esm2_target_probability"] >= a.min_class_probability
                )

    diagnostic_stages = [
        "integrity_pass", "mmseqs_pass", "phrog_pass", "conservation_pass",
        "esm2_classification_pass",
    ]
    gating_stages = ["integrity_pass", "mmseqs_pass", "phrog_pass", "conservation_pass"]
    if classifier_positive_control_pass:
        gating_stages.append("esm2_classification_pass")
    for row in rows:
        row["prestructure_pass"] = all(bool(row[s]) for s in gating_stages)
    fieldnames = [k for k in rows[0] if k != "sequence"] + ["sequence"]
    screening = a.output_dir / "sequence_screening.tsv"
    with screening.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    prestructure = a.output_dir / "prestructure_pass.fasta"
    with prestructure.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["prestructure_pass"]:
                handle.write(f">{row['candidate_id']}\n{row['sequence']}\n")

    summary = {
        "status": "passed",
        "input_sequences": len(rows),
        "thresholds": {
            "min_identity": a.min_identity,
            "min_coverage": a.min_coverage,
            "max_evalue": a.max_evalue,
            "phrog_required": "phrog_1483 major spike protein",
            "conserved_frequency": a.conserved_frequency,
            "min_conserved_retention": a.min_conserved_retention,
            "target_class": a.target_class,
            "min_class_probability": a.min_class_probability,
        },
        "conservation_reference": {
            "natural_homologs": profile["accepted_homologs"],
            "conserved_sites": len(conserved_positions),
        },
        "classifier_positive_control": {
            "reference_predicted_class": id2label[reference_predicted_id],
            "reference_target_probability": float(reference_probabilities[target_id]),
            "pass": classifier_positive_control_pass,
            "used_as_hard_gate": classifier_positive_control_pass,
            "reason_if_not_used": "The PhiX174 G positive control fails this classifier gate; candidate predictions are retained as diagnostic values only."
            if not classifier_positive_control_pass else "",
        },
        "cumulative_counts": cumulative_counts(rows, diagnostic_stages),
        "gating_stages_for_structure": gating_stages,
        "prestructure_pass": sum(x["prestructure_pass"] for x in rows),
        "outputs": {
            "screening": str(screening),
            "prestructure_fasta": str(prestructure),
            "conservation_profile": str(a.output_dir / "conservation_profile.tsv"),
        },
    }
    (a.output_dir / "sequence_screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
