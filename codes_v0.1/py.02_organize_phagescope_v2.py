#!/usr/bin/env python3
"""将 PhageScope v2.0 全部数据库整理为 py.01 兼容结构并生成简明统计。

默认逐个处理全部数据库；也可用 ``--dataset Genbank`` 指定单库：
1. genome/protein tar.gz 与 metadata 的 ID 能否一致对应；
2. 按 Start/Stop/Strand 从 genome_seq 提取每个蛋白的 dna_seq；
3. dna_seq 翻译后与 AA 序列的一致性；
4. 附加注释能否按 Phage_ID 合并。

原始数据只读；输出写入 phage_designer/results/02_phagescope_v2。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import pickle
import re
import sys
import tarfile
import time
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote


PROJECT_DIR = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
SOURCE_DIR = PROJECT_DIR / "data/phagescope_v2/source"
DEFAULT_OUT = PROJECT_DIR / "results/02_phagescope_v2"
DATASETS = ["Genbank", "RefSeq", "DDBJ", "EMBL", "PhagesDB", "GVD", "GPD", "MGV", "TemPhD", "CHVD", "IGVD", "IMG_VR", "GOV2", "STV", "GSV", "UHGV", "HPGC", "ELGV", "OVD", "OPD", "SVD", "VMGC", "URPC", "BAPS", "TYMEFLIES_Viral", "MetaVR"]
CODON_TABLE = {
    "TTT":"F", "TTC":"F", "TTA":"L", "TTG":"L", "TCT":"S", "TCC":"S", "TCA":"S", "TCG":"S",
    "TAT":"Y", "TAC":"Y", "TAA":"*", "TAG":"*", "TGT":"C", "TGC":"C", "TGA":"*", "TGG":"W",
    "CTT":"L", "CTC":"L", "CTA":"L", "CTG":"L", "CCT":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAT":"H", "CAC":"H", "CAA":"Q", "CAG":"Q", "CGT":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "ATT":"I", "ATC":"I", "ATA":"I", "ATG":"M", "ACT":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAT":"N", "AAC":"N", "AAA":"K", "AAG":"K", "AGT":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GTT":"V", "GTC":"V", "GTA":"V", "GTG":"V", "GCT":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAT":"D", "GAC":"D", "GAA":"E", "GAG":"E", "GGT":"G", "GGC":"G", "GGA":"G", "GGG":"G",
}
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY*]+$")
DNA_RE = re.compile(r"^[ACGTN]+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dataset", default="all", help="all=全部 v2 数据库；也可指定单个数据库")
    p.add_argument("--max-genomes", type=int, default=0, help="0=all; use a small number for a smoke test")
    p.add_argument("--min-genome-length", type=int, default=1000)
    p.add_argument("--min-protein-length", type=int, default=1)
    p.add_argument("--keep-unmatched", action="store_true", help="保留没有 metadata 的 genome/protein ID")
    p.add_argument("--no-pickle", action="store_true", help="只输出统计，不保存逐数据库 pickle")
    return p.parse_args()


def setup_logging(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("organize_genbank_v2")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logfile = logging.FileHandler(out / "run.log", mode="w", encoding="utf-8")
    logfile.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(logfile)
    return logger


def clean_seq(value: Any) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def translate(dna: str) -> str:
    return "".join(CODON_TABLE.get(dna[i:i + 3], "X") for i in range(0, len(dna) - 2, 3))


def reverse_complement(dna: str) -> str:
    return dna.translate(str.maketrans("ATCGNatcgn", "TAGCNtagcn"))[::-1]


def extract_dna(genome: str, start: int, stop: int, strand: str) -> str:
    if not genome or start <= 0 or stop <= 0 or stop < start:
        return ""
    dna = genome[max(0, start - 1):min(len(genome), stop)]
    return reverse_complement(dna) if strand == "-" else dna


def verify_translation(aa: str, dna: str, tolerance: float = 0.05) -> bool:
    aa, dna = clean_seq(aa), clean_seq(dna)
    if not aa or not dna or len(dna) < 3 or len(dna) % 3:
        return False
    observed = translate(dna).rstrip("*")
    expected = aa.rstrip("*")
    if not observed or not expected:
        return False
    n = min(len(observed), len(expected))
    matches = sum(a == b for a, b in zip(observed[:n], expected[:n]))
    return matches / max(len(observed), len(expected)) >= (1.0 - tolerance)


def fasta_text(raw: bytes) -> tuple[str, str]:
    text = raw.decode("utf-8", errors="replace")
    header, *lines = text.strip().splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    return (header[1:].split()[0] if header.startswith(">") else "", clean_seq(seq))


def read_genome_tar(path: Path, logger: logging.Logger, min_len: int) -> dict[str, str]:
    genomes: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as tar:
        for index, member in enumerate(tar):
            if not member.isfile() or not member.name.lower().endswith((".fasta", ".fa", ".fna")):
                continue
            stream = tar.extractfile(member)
            if stream is None:
                continue
            filename_id = Path(member.name).stem
            header_id, seq = fasta_text(stream.read())
            genome_id = filename_id or header_id
            if len(seq) >= min_len:
                genomes[genome_id] = seq
            if index and index % 10000 == 0:
                logger.info("genomes extracted: %d", len(genomes))
    logger.info("genomes extracted total: %d", len(genomes))
    return genomes


def read_protein_tar(path: Path, logger: logging.Logger, min_len: int) -> dict[str, dict[str, str]]:
    proteins: dict[str, dict[str, str]] = defaultdict(dict)
    with tarfile.open(path, "r:gz") as tar:
        for index, member in enumerate(tar):
            if not member.isfile() or not member.name.lower().endswith((".fasta", ".fa", ".faa")):
                continue
            stream = tar.extractfile(member)
            if stream is None:
                continue
            parts = Path(member.name).parts
            filename_id = Path(member.name).stem
            if len(parts) >= 3:
                phage_id = parts[-2]
            else:
                phage_id = ""
            header_id, seq = fasta_text(stream.read())
            protein_id = filename_id or header_id
            if len(seq) >= min_len and phage_id and protein_id:
                proteins[phage_id][protein_id] = seq
            if index and index % 50000 == 0:
                logger.info("proteins extracted: %d", sum(len(x) for x in proteins.values()))
    logger.info("proteins extracted total: %d from %d phages", sum(len(x) for x in proteins.values()), len(proteins))
    return dict(proteins)


def read_tsv(path: Path, logger: logging.Logger) -> list[dict[str, str]]:
    if not path.exists():
        logger.warning("missing annotation file: %s", path)
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    logger.info("read %d rows: %s", len(rows), path.name)
    return rows


def phage_meta(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        gid = (row.get("Phage_ID") or row.get("phage_id") or "").strip()
        if not gid:
            continue
        result[gid] = {
            "length": safe_int(row.get("Length")),
            "gc_content": safe_float(row.get("GC_content")),
            "taxonomy": row.get("Taxonomy") or "Unknown",
            "completeness": row.get("Completeness") or "Not-determined",
            "host": row.get("Host") or "Unknown",
            "lifestyle": row.get("Lifestyle") or "Unknown",
            "cluster": row.get("Cluster") or "",
            "subcluster": row.get("Subcluster") or "",
            "phage_source": row.get("Phage_source") or row.get("Data_source") or "Genbank",
        }
    return result


def protein_meta(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        gid = (row.get("Phage_ID") or row.get("phage_id") or "").strip()
        pid = (row.get("Protein_ID") or row.get("protein_id") or "").strip()
        if not gid or not pid:
            continue
        classification = row.get("Protein_classification") or row.get("Classification") or ""
        item = {
            "start": safe_int(row.get("Start") or row.get("start")),
            "stop": safe_int(row.get("Stop") or row.get("stop")),
            "strand": row.get("Strand") if row.get("Strand") in ("+", "-") else "+",
            "product": row.get("Product") or "unknown",
            "classification": [x.strip() for x in classification.split(";") if x.strip()],
            "molecular_weight": safe_float(row.get("Molecular_weight")),
            "aromaticity": safe_float(row.get("Aromaticity")),
            "instability_index": safe_float(row.get("Instability_index")),
            "isoelectric_point": safe_float(row.get("Isoelectric_point")),
            "helix_fraction": safe_float(row.get("Helix_fraction")),
            "turn_fraction": safe_float(row.get("Turn_fraction")),
            "sheet_fraction": safe_float(row.get("Sheet_fraction")),
            "reduced_coefficient": safe_float(row.get("Reduced_coefficient")),
            "oxidized_coefficient": safe_float(row.get("Oxidized_coefficient")),
        }
        item["coding_segments"] = [{"start": item["start"], "stop": item["stop"], "strand": item["strand"]}]
        if pid in result[gid]:
            previous = result[gid][pid]
            previous.setdefault("coding_segments", []).extend(item["coding_segments"])
            # Keep the first complete annotation, while retaining all coordinate segments.
        else:
            result[gid][pid] = item
    return dict(result)


def extract_cds(genome: str, annotation: dict[str, Any]) -> str:
    segments = annotation.get("coding_segments") or []
    if not segments:
        return extract_dna(genome, annotation.get("start", 0), annotation.get("stop", 0), annotation.get("strand", "+"))
    strand = segments[0].get("strand", "+")
    ordered = sorted(segments, key=lambda x: (safe_int(x.get("start")), safe_int(x.get("stop"))), reverse=(strand == "-"))
    return "".join(extract_dna(genome, safe_int(seg.get("start")), safe_int(seg.get("stop")), strand) for seg in ordered)


def row_phage_id(row: dict[str, str]) -> str:
    for key in ("Phage_ID", "phage_id", "seqid", "Sequence_ID", "Genome_ID"):
        if row.get(key):
            return row[key].strip()
    return ""


def load_extra_annotations(source: Path, slug: str, logger: logging.Logger) -> dict[str, dict[str, list[dict[str, str]]]]:
    root = source / "annotations"
    specs = {
        "trna_tmRNA": root / "trna_crispr" / f"{slug}_phage_trna_gene_meta_data.tsv",
        "crispr_array": root / "trna_crispr" / f"{slug}_phage_crispr_array_meta_data.tsv",
        "anticrispr": root / "anticrispr" / f"{slug}_phage_anticrispr_protein_meta_data.tsv",
        "amr": root / "amr" / f"{slug}_antimicrobial_resistance_gene_data.tsv",
        "virulent_factor": root / "virulent" / f"{slug}_virulent_factor_data.tsv",
        "terminator": root / "terminator" / f"{slug}_phage_transcription_terminator_meta_data.tsv",
        "transmembrane": root / "transmembrane" / f"{slug}_phage_transmembrane_protein_meta_data.tsv",
    }
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for category, path in specs.items():
        for row in read_tsv(path, logger):
            gid = row_phage_id(row)
            if gid:
                grouped[gid][category].append(row)
    return {gid: dict(categories) for gid, categories in grouped.items()}


def load_gff3_counts(path: Path, logger: logging.Logger) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    if not path.exists():
        logger.warning("missing GFF3: %s", path)
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                counts[fields[0]][fields[2]] += 1
    return {gid: dict(counter) for gid, counter in counts.items()}


def build(genomes: dict[str, str], proteins: dict[str, dict[str, str]], metadata: dict[str, dict[str, Any]],
          protein_annotations: dict[str, dict[str, dict[str, Any]]], extras: dict[str, dict[str, list[dict[str, str]]]],
          gff3_counts: dict[str, dict[str, int]], args: argparse.Namespace, logger: logging.Logger) -> tuple[OrderedDict, dict[str, Any]]:
    phage_ids = set(metadata) | set(genomes) | set(proteins) | set(protein_annotations)
    if not args.keep_unmatched:
        phage_ids &= set(metadata) | set(genomes)
    ordered_phage_ids = sorted(phage_ids)
    if args.max_genomes > 0:
        ordered_phage_ids = ordered_phage_ids[:args.max_genomes]
    output: OrderedDict[str, dict[str, Any]] = OrderedDict()
    stats = Counter()
    for index, gid in enumerate(ordered_phage_ids, start=1):
        genome = genomes.get(gid, "")
        if genome:
            stats["phages_with_genome"] += 1
        entry: dict[str, Any] = {
            "genome_seq": genome,
            "metadata": metadata.get(gid, {"length": len(genome), "gc_content": 0.0, "taxonomy": "Unknown",
                                             "completeness": "Not-determined", "host": "Unknown", "lifestyle": "Unknown",
                                             "cluster": "", "subcluster": "", "phage_source": args.dataset}),
            "proteins": OrderedDict(),
            "annotations": extras.get(gid, {}),
            "gff3_feature_counts": gff3_counts.get(gid, {}),
        }
        prot_ids = set(proteins.get(gid, {})) | set(protein_annotations.get(gid, {}))
        for pid in sorted(prot_ids):
            seq = proteins.get(gid, {}).get(pid, "")
            annot = dict(protein_annotations.get(gid, {}).get(pid, {}))
            if seq:
                stats["proteins_with_aa"] += 1
            if seq and not AA_RE.fullmatch(seq):
                stats["proteins_aa_invalid"] += 1
            prot = {"seq": seq, **annot}
            dna = extract_cds(genome, prot)
            if dna:
                prot["dna_seq"] = dna
                stats["proteins_with_dna"] += 1
                if DNA_RE.fullmatch(dna):
                    stats["proteins_dna_valid_chars"] += 1
                if seq and verify_translation(seq, dna):
                    prot["dna_translation_verified"] = True
                    stats["proteins_dna_verified"] += 1
                else:
                    prot["dna_translation_verified"] = False
                    stats["proteins_dna_unverified"] += 1
            else:
                stats["proteins_without_dna"] += 1
                prot["dna_seq"] = ""
                prot["dna_translation_verified"] = False
            defaults: dict[str, Any] = {"start": 0, "stop": 0, "strand": "+", "product": "unknown",
                                        "classification": [], "molecular_weight": 0.0, "aromaticity": 0.0,
                                        "instability_index": 0.0, "isoelectric_point": 0.0, "helix_fraction": 0.0,
                                        "turn_fraction": 0.0, "sheet_fraction": 0.0, "reduced_coefficient": 0.0,
                                        "oxidized_coefficient": 0.0}
            for key, value in defaults.items():
                prot.setdefault(key, value)
            entry["proteins"][pid] = prot
        stats["phages_with_proteins"] += bool(entry["proteins"])
        stats["total_proteins"] += len(entry["proteins"])
        output[gid] = entry
        if index % 500 == 0:
            logger.info("built %d/%d phages", index, len(ordered_phage_ids))
    stats["phages_total"] = len(output)
    stats["annotation_phages"] = sum(bool(x.get("annotations")) for x in output.values())
    stats["gff3_phages"] = sum(bool(x.get("gff3_feature_counts")) for x in output.values())
    return output, dict(stats)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.output_dir)
    source = args.source_dir
    started = time.time()
    datasets = DATASETS if args.dataset.lower() == "all" else [args.dataset]
    all_stats = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        slug = dataset.lower()
        files = {
            "genome": source / "genome_fasta" / f"{slug}.tar.gz",
            "protein": source / "protein_fasta" / f"{slug}.tar.gz",
            "phage_meta": source / "metadata" / f"{slug}_phage_meta_data.tsv",
            "protein_meta": source / "metadata" / f"{slug}_phage_annotated_protein_meta_data.tsv",
            "gff3": source / "annotations" / "gff3" / f"{slug}.gff3",
        }
        required = [files[key] for key in ("genome", "protein", "phage_meta", "protein_meta")]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            logger.warning("[%s] missing required files; skipped: %s", dataset, missing)
            all_stats[dataset] = {"error": "missing required files", "missing": missing}
            continue
        logger.info("[%s] organizing from %s", dataset, source)
        genomes = read_genome_tar(files["genome"], logger, args.min_genome_length)
        proteins = read_protein_tar(files["protein"], logger, args.min_protein_length)
        metadata = phage_meta(read_tsv(files["phage_meta"], logger))
        protein_annotations = protein_meta(read_tsv(files["protein_meta"], logger))
        extras = load_extra_annotations(source, slug, logger)
        gff3_counts = load_gff3_counts(files["gff3"], logger)
        output, stats = build(genomes, proteins, metadata, protein_annotations, extras, gff3_counts, args, logger)
        if not args.no_pickle:
            out_pkl = args.output_dir / f"{slug}_phage_data.pkl"
            with out_pkl.open("wb") as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("[%s] wrote %s", dataset, out_pkl)
            stats["pickle"] = str(out_pkl)
        stats["extra_annotation_categories"] = sorted({cat for value in extras.values() for cat in value})
        stats["input_files"] = {key: str(value) for key, value in files.items()}
        all_stats[dataset] = stats
    totals = Counter()
    for stats in all_stats.values():
        for key in ("phages_total", "phages_with_genome", "total_proteins", "proteins_with_dna", "proteins_dna_verified", "proteins_dna_unverified", "annotation_phages", "gff3_phages"):
            totals[key] += safe_int(stats.get(key, 0))
    summary = {"dataset": args.dataset, "datasets": datasets, "source_dir": str(source),
               "generated_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.time() - started, 2),
               "stats_by_dataset": all_stats, "totals": dict(totals),
               "schema": {"top_level": "OrderedDict[Phage_ID, record]", "record_keys": ["genome_seq", "metadata", "proteins", "annotations", "gff3_feature_counts"],
                          "protein_keys_include": ["seq", "dna_seq", "coding_segments", "start", "stop", "strand", "product", "classification", "dna_translation_verified"]}}
    write_json(args.output_dir / "phagescope_v2_summary.json", summary)
    report = ["# PhageScope v2 full preprocessing summary", "", f"- generated: `{summary['generated_at']}`", f"- datasets requested: {len(datasets)}", "", "| Dataset | Phages | Genomes | Proteins | DNA seq | DNA verified | Extra annotation phages | GFF3 phages |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in datasets:
        stats = all_stats.get(dataset, {})
        report.append(f"| {dataset} | {stats.get('phages_total', 0):,} | {stats.get('phages_with_genome', 0):,} | {stats.get('total_proteins', 0):,} | {stats.get('proteins_with_dna', 0):,} | {stats.get('proteins_dna_verified', 0):,} | {stats.get('annotation_phages', 0):,} | {stats.get('gff3_phages', 0):,} |")
    report += [f"| **Total** | **{totals['phages_total']:,}** | **{totals['phages_with_genome']:,}** | **{totals['total_proteins']:,}** | **{totals['proteins_with_dna']:,}** | **{totals['proteins_dna_verified']:,}** | **{totals['annotation_phages']:,}** | **{totals['gff3_phages']:,}** |", "", "This report follows the concise py.01 preprocessing style. It summarizes sequence counts, DNA extraction, translation verification, and annotation linkage without a separate complex audit layer."]
    (args.output_dir / "phagescope_v2_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("completed %d datasets in %.1f minutes", len(datasets), (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
