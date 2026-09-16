#!/usr/bin/env python3
"""GPD 数据集基础质量控制、完全去重和 cluster-aware 划分。

保持原始 pickle 只读，默认输出 genome/protein manifest；不复制大型 pickle。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_INPUT = PROJECT / "results/02_phagescope_v2/gpd_phage_data.pkl"
DEFAULT_OUT = PROJECT / "results/03_gpd_qc"
DNA_RE = re.compile(r"^[ACGTN]+$")
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY*]+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-genome-length", type=int, default=1000)
    p.add_argument("--min-protein-length", type=int, default=20)
    p.add_argument("--max-n-fraction", type=float, default=0.05)
    p.add_argument("--write-pickle", action="store_true", help="写出筛选后的 pickle（需要额外磁盘空间）")
    p.add_argument("--dataset", default="GPD", help="数据集名称，用于输出文件命名")
    return p.parse_args()


def setup_logger(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("py03_gpd")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def stable_split(cluster_id: str) -> str:
    """Stable hash split; cluster IDs never cross splits."""
    value = int(hashlib.sha256(cluster_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "validation"
    return "test"


def sequence_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("ascii", errors="ignore")).hexdigest()


def protein_is_valid(prot: dict, min_len: int) -> bool:
    aa = (prot.get("seq") or "").strip().upper()
    dna = (prot.get("dna_seq") or "").strip().upper()
    return bool(aa and len(aa) >= min_len and AA_RE.fullmatch(aa) and dna and DNA_RE.fullmatch(dna)
                and prot.get("dna_translation_verified") is True)


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.output_dir)
    started = time.time()
    if not args.input.exists():
        logger.error("input does not exist: %s", args.input)
        return 2
    logger.info("loading GPD pickle: %s", args.input)
    with args.input.open("rb") as handle:
        data = pickle.load(handle)
    logger.info("loaded %d records", len(data))

    counts = Counter()
    reasons = Counter()
    seen_hash: dict[str, str] = {}
    retained: dict[str, dict] = {}
    protein_rows: list[dict] = []
    for index, (phage_id, record) in enumerate(data.items(), start=1):
        counts["input_records"] += 1
        genome = (record.get("genome_seq") or "").strip().upper()
        if len(genome) < args.min_genome_length:
            reasons["short_or_empty_genome"] += 1
            continue
        if not DNA_RE.fullmatch(genome):
            reasons["illegal_dna_characters"] += 1
            continue
        n_fraction = genome.count("N") / len(genome)
        if n_fraction > args.max_n_fraction:
            reasons["excessive_N_fraction"] += 1
            continue
        digest = sequence_hash(genome)
        if digest in seen_hash:
            reasons["exact_duplicate_genome"] += 1
            continue
        proteins = record.get("proteins") or {}
        valid_proteins = []
        for protein_id, prot in proteins.items():
            if protein_is_valid(prot, args.min_protein_length):
                valid_proteins.append((protein_id, prot))
        if not valid_proteins:
            reasons["no_valid_protein_dna_pair"] += 1
            continue
        seen_hash[digest] = phage_id
        metadata = record.get("metadata") or {}
        cluster = str(metadata.get("cluster") or metadata.get("subcluster") or f"singleton_{digest[:16]}").strip()
        retained[phage_id] = record
        counts["retained_genomes"] += 1
        counts["retained_proteins"] += len(valid_proteins)
        counts["genome_bp"] += len(genome)
        for protein_id, prot in valid_proteins:
            labels = prot.get("classification") or []
            protein_rows.append({
                "phage_id": phage_id, "protein_id": protein_id, "genome_cluster": cluster,
                "split": "", "protein_length_aa": len((prot.get("seq") or "")),
                "dna_length_nt": len((prot.get("dna_seq") or "")),
                "product": prot.get("product") or "unknown",
                "classification": ";".join(str(x) for x in labels),
                "dna_translation_verified": "1",
            })
        if index % 50000 == 0:
            logger.info("scanned %d/%d records", index, len(data))

    cluster_to_phages: dict[str, list[str]] = defaultdict(list)
    genome_rows = []
    for phage_id, record in retained.items():
        genome = (record.get("genome_seq") or "").strip().upper()
        metadata = record.get("metadata") or {}
        digest = sequence_hash(genome)
        cluster = str(metadata.get("cluster") or metadata.get("subcluster") or f"singleton_{digest[:16]}").strip()
        cluster_to_phages[cluster].append(phage_id)
    cluster_split = {cluster: stable_split(cluster) for cluster in cluster_to_phages}
    split_counts = Counter()
    for phage_id, record in retained.items():
        genome = (record.get("genome_seq") or "").strip().upper()
        metadata = record.get("metadata") or {}
        digest = sequence_hash(genome)
        cluster = str(metadata.get("cluster") or metadata.get("subcluster") or f"singleton_{digest[:16]}").strip()
        split = cluster_split[cluster]
        split_counts[split] += 1
        genome_rows.append({
            "phage_id": phage_id, "genome_length_bp": len(genome),
            "gc_fraction": round((genome.count("G") + genome.count("C")) / len(genome), 6),
            "n_fraction": round(genome.count("N") / len(genome), 6),
            "sequence_sha256": digest, "genome_cluster": cluster, "split": split,
            "taxonomy": metadata.get("taxonomy", "Unknown"), "host": metadata.get("host", "Unknown"),
            "completeness": metadata.get("completeness", "Not-determined"),
            "valid_protein_count": sum(1 for p in (record.get("proteins") or {}).values() if protein_is_valid(p, args.min_protein_length)),
        })
    phage_split = {row["phage_id"]: row["split"] for row in genome_rows}
    for row in protein_rows:
        row["split"] = phage_split[row["phage_id"]]
    genome_fields = list(genome_rows[0].keys()) if genome_rows else ["phage_id"]
    protein_fields = list(protein_rows[0].keys()) if protein_rows else ["phage_id"]
    prefix = args.dataset.lower()
    write_tsv(args.output_dir / f"{prefix}_genome_manifest.tsv", genome_rows, genome_fields)
    write_tsv(args.output_dir / f"{prefix}_protein_manifest.tsv", protein_rows, protein_fields)
    summary = {
        "dataset": args.dataset, "input": str(args.input), "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 2), "parameters": {"input": str(args.input), "output_dir": str(args.output_dir), "min_genome_length": args.min_genome_length, "min_protein_length": args.min_protein_length, "max_n_fraction": args.max_n_fraction, "write_pickle": args.write_pickle},
        "counts": dict(counts), "filter_reasons": dict(reasons), "clusters": len(cluster_to_phages),
        "split_genomes": dict(split_counts), "split_clusters": Counter(cluster_split.values()),
        "outputs": {"genome_manifest": str(args.output_dir / f"{prefix}_genome_manifest.tsv"), "protein_manifest": str(args.output_dir / f"{prefix}_protein_manifest.tsv")},
    }
    summary["split_clusters"] = dict(summary["split_clusters"])
    if args.write_pickle:
        path = args.output_dir / f"{prefix.lower()}_filtered_phage_data.pkl"
        with path.open("wb") as handle:
            pickle.dump(retained, handle, protocol=pickle.HIGHEST_PROTOCOL)
        summary["outputs"]["filtered_pickle"] = str(path)
    (args.output_dir / f"{prefix.lower()}_qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [f"# {args.dataset} dataset QC and cluster-aware split", "", f"- Input records: {counts['input_records']:,}", f"- Retained genomes: {counts['retained_genomes']:,}", f"- Retained valid protein–DNA pairs: {counts['retained_proteins']:,}", f"- Exact duplicate genomes removed: {reasons['exact_duplicate_genome']:,}", f"- Genome clusters: {len(cluster_to_phages):,}", "", "| Split | Genomes |", "|---|---:|"]
    report += [f"| {split} | {split_counts.get(split, 0):,} |" for split in ("train", "validation", "test")]
    report += ["", "Filtering requires genome length ≥ 1,000 bp, legal DNA characters, N fraction ≤ configured threshold, and at least one protein with a verified DNA coding sequence. Splits are assigned by genome cluster; clusters do not cross splits."]
    (args.output_dir / f"{prefix.lower()}_qc_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("completed in %.1f minutes", (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
