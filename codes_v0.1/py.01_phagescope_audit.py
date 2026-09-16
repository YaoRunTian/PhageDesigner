#!/usr/bin/env python3
"""PhageScope 数据结构与质量审计（只读原始 pickle）。

默认执行 inspect 模式，仅加载指定的小型数据库并输出 schema/sample 报告。
full 模式逐个加载数据库 pickle，绝不读取 phage_data_all.pkl；每个数据库
处理完成后释放对象。原始数据不被修改。
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
from typing import Any, Iterable


PROJECT_DIR = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
SOURCE_DIR = Path("/public8/lilab/student/rtyao/phage/results/01_preprocess")
DEFAULT_OUT = PROJECT_DIR / "results/01_phagescope_audit"
DATABASE_ORDER = (
    "genebank", "chvd", "ddbj", "embl", "gov2", "gpd", "gvd", "igvd",
    "img_vr", "mgv", "phagesdb", "refseq", "stv", "temphd",
)
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY*]+$")
DNA_RE = re.compile(r"^[ACGTN]+$")
HYPOTHETICAL_RE = re.compile(r"hypothetical|uncharacterized|unknown|putative", re.I)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("inspect", "full"), default="inspect")
    p.add_argument("--databases", default="embl", help="逗号分隔；full 默认仍可显式指定 all")
    p.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sample-genomes", type=int, default=3)
    p.add_argument("--sample-proteins", type=int, default=5)
    p.add_argument("--max-genomes", type=int, default=0, help="full 模式限制每库基因组数；0=不限制")
    p.add_argument("--translation-samples", type=int, default=200,
                   help="每库抽样验证 dna_seq 翻译；0=不验证")
    return p.parse_args()


def setup_logging(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phagescope_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    log_file = logging.FileHandler(out / "run.log", mode="w", encoding="utf-8")
    log_file.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(log_file)
    return logger


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def clean_seq(value: Any) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def translate(dna: str) -> str:
    return "".join(CODON_TABLE.get(dna[i:i + 3], "X")
                   for i in range(0, len(dna) - 2, 3))


def translation_match(aa: str, dna: str) -> bool | None:
    aa, dna = clean_seq(aa), clean_seq(dna)
    if not aa or not dna or len(dna) < 3 or len(dna) % 3:
        return None
    translated = translate(dna).rstrip("*")
    target = aa.rstrip("*")
    n = min(len(translated), len(target))
    if not n:
        return False
    matches = sum(a == b for a, b in zip(translated[:n], target[:n]))
    return matches / max(len(translated), len(target)) >= 0.95


def parse_databases(arg: str, source: Path) -> list[str]:
    available = {p.name[len("phage_data_"):-4] for p in source.glob("phage_data_*.pkl")
                 if p.name != "phage_data_all.pkl"}
    if arg.strip().lower() == "all":
        # temphd 当前没有成功生成 pickle；all 模式只运行已有数据库，避免因单库缺失中止整个审计。
        names = [name for name in DATABASE_ORDER if name in available]
    else:
        names = [x.strip() for x in arg.split(",") if x.strip()]
    missing = [name for name in names if name not in available]
    if missing:
        raise FileNotFoundError(f"找不到数据库 pickle: {missing}; available={sorted(available)}")
    return names


def load_database(name: str, source: Path) -> dict:
    path = source / f"phage_data_{name}.pkl"
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} 顶层对象不是 dict，而是 {type(data).__name__}")
    return data


def schema_probe(data: dict, genome_limit: int, protein_limit: int) -> dict[str, Any]:
    records = []
    genome_keys, metadata_keys, protein_keys = set(), set(), set()
    protein_examples = []
    for i, (gid, record) in enumerate(data.items()):
        if not isinstance(record, dict):
            records.append({"genome_id_type": type(gid).__name__, "record_type": type(record).__name__})
            continue
        genome_keys.update(record.keys())
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            metadata_keys.update(metadata.keys())
        proteins = record.get("proteins")
        if isinstance(proteins, dict):
            for pid, protein in list(proteins.items())[:protein_limit]:
                if isinstance(protein, dict):
                    protein_keys.update(protein.keys())
                    if len(protein_examples) < protein_limit:
                        protein_examples.append({"protein_id_type": type(pid).__name__,
                                                 "fields": sorted(protein.keys())})
        if len(records) < genome_limit:
            records.append({"genome_id": str(gid), "record_type": type(record).__name__,
                            "genome_keys": sorted(record.keys()),
                            "metadata_keys": sorted(metadata.keys()) if isinstance(metadata, dict) else [],
                            "protein_count": len(proteins) if isinstance(proteins, dict) else None})
        if i + 1 >= genome_limit and records:
            break
    return {"top_level_type": type(data).__name__, "top_level_count": len(data),
            "genome_record_keys": sorted(genome_keys), "metadata_keys": sorted(metadata_keys),
            "protein_record_keys": sorted(protein_keys), "genome_examples": records,
            "protein_examples": protein_examples}


def empty_counter() -> Counter:
    return Counter()


def audit_database(name: str, data: dict, args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
    genome_hashes: Counter[str] = Counter()
    protein_hashes: Counter[str] = Counter()
    lengths: list[int] = []
    gc_values: list[float] = []
    taxonomy = Counter()
    completeness = Counter()
    protein_products = Counter()
    protein_classes = Counter()
    genome_counts = Counter()
    protein_counts = Counter()
    translate_total = translate_ok = 0
    invalid_examples: list[str] = []
    start = time.time()

    for genome_index, (genome_id, record) in enumerate(data.items(), start=1):
        if args.max_genomes and genome_index > args.max_genomes:
            break
        genome_counts["records"] += 1
        if not isinstance(record, dict):
            genome_counts["invalid_record"] += 1
            continue
        genome = clean_seq(record.get("genome_seq"))
        if not genome:
            genome_counts["missing_sequence"] += 1
        else:
            genome_counts["with_sequence"] += 1
            genome_hashes[sha256_text(genome)] += 1
            lengths.append(len(genome))
            invalid = sum(ch not in "ACGTN" for ch in genome)
            genome_counts["invalid_sequence"] += int(invalid > 0)
            genome_counts["n_bases"] += genome.count("N")
            if invalid and len(invalid_examples) < 20:
                invalid_examples.append(str(genome_id))
            gc_values.append(100.0 * (genome.count("G") + genome.count("C")) / len(genome))

        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        taxonomy[str(metadata.get("taxonomy") or "Unknown")] += 1
        completeness[str(metadata.get("completeness") or "Unknown")] += 1
        proteins = record.get("proteins") if isinstance(record.get("proteins"), dict) else {}
        protein_counts["proteins"] += len(proteins)
        for protein in proteins.values():
            if not isinstance(protein, dict):
                protein_counts["invalid_record"] += 1
                continue
            aa = clean_seq(protein.get("seq"))
            dna = clean_seq(protein.get("dna_seq"))
            protein_counts["with_aa"] += int(bool(aa))
            protein_counts["with_dna"] += int(bool(dna))
            protein_counts["aa_invalid"] += int(bool(aa) and not AA_RE.fullmatch(aa))
            protein_counts["dna_invalid"] += int(bool(dna) and not DNA_RE.fullmatch(dna))
            if aa:
                protein_hashes[sha256_text(aa)] += 1
            product = str(protein.get("product") or "").strip()
            classes = protein.get("classification")
            if product:
                protein_counts["with_product"] += 1
                protein_products[product] += 1
                protein_counts["hypothetical"] += int(bool(HYPOTHETICAL_RE.search(product)))
            if classes:
                protein_counts["with_classification"] += 1
                if isinstance(classes, (list, tuple, set)):
                    protein_classes.update(str(x).strip() for x in classes if str(x).strip())
                else:
                    protein_classes[str(classes).strip()] += 1
            if args.translation_samples and dna and aa and translate_total < args.translation_samples:
                translate_total += 1
                translate_ok += int(translation_match(aa, dna) is True)
        if genome_index % 10000 == 0:
            logger.info("[%s] processed %d genomes", name, genome_index)

    def mean(values: Iterable[float]) -> float | None:
        values = list(values)
        return sum(values) / len(values) if values else None

    result = {
        "database": name,
        "records": genome_counts["records"],
        "genome_counts": dict(genome_counts),
        "protein_counts": dict(protein_counts),
        "genome_length_min": min(lengths) if lengths else None,
        "genome_length_median": sorted(lengths)[len(lengths) // 2] if lengths else None,
        "genome_length_max": max(lengths) if lengths else None,
        "gc_mean": mean(gc_values),
        "taxonomy_top": taxonomy.most_common(20),
        "completeness": dict(completeness),
        "product_top": protein_products.most_common(20),
        "classification_top": protein_classes.most_common(20),
        "translation_checked": translate_total,
        "translation_matched": translate_ok,
        "genome_unique_hashes": len(genome_hashes),
        "genome_duplicate_records": sum(n - 1 for n in genome_hashes.values() if n > 1),
        "protein_unique_hashes": len(protein_hashes),
        "protein_duplicate_records": sum(n - 1 for n in protein_hashes.values() if n > 1),
        "invalid_genome_examples": invalid_examples,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    logger.info("[%s] finished: %d genomes, %d proteins, %.1fs", name, result["records"],
                protein_counts["proteins"], result["elapsed_seconds"])
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key)
                         for key in keys} for row in rows)


def write_outputs(out: Path, mode: str, databases: list[str], schema: dict[str, Any], results: list[dict[str, Any]]) -> None:
    (out / "schema_report.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "audit_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    inventory = [{"database": r["database"], "records": r["records"], "genome_with_sequence": r["genome_counts"].get("with_sequence", 0),
                  "proteins": r["protein_counts"].get("proteins", 0), "proteins_with_dna": r["protein_counts"].get("with_dna", 0),
                  "proteins_with_product": r["protein_counts"].get("with_product", 0), "elapsed_seconds": r["elapsed_seconds"]} for r in results]
    quality = [{"database": r["database"], "length_min": r["genome_length_min"], "length_median": r["genome_length_median"],
                "length_max": r["genome_length_max"], "gc_mean": r["gc_mean"],
                "invalid_genome": r["genome_counts"].get("invalid_sequence", 0), "n_bases": r["genome_counts"].get("n_bases", 0)} for r in results]
    proteins = [{"database": r["database"], **r["protein_counts"], "translation_checked": r["translation_checked"],
                 "translation_matched": r["translation_matched"]} for r in results]
    duplicates = [{"database": r["database"], "unique_genome_hashes": r["genome_unique_hashes"],
                   "genome_duplicate_records": r["genome_duplicate_records"], "unique_protein_hashes": r["protein_unique_hashes"],
                   "protein_duplicate_records": r["protein_duplicate_records"]} for r in results]
    functions = [{"database": r["database"], "product_top20": r["product_top"], "classification_top20": r["classification_top"]} for r in results]
    families = [{"database": r["database"], "taxonomy_top20": r["taxonomy_top"], "completeness": r["completeness"]} for r in results]
    write_csv(out / "database_inventory.csv", inventory)
    write_csv(out / "genome_quality_summary.csv", quality)
    write_csv(out / "protein_quality_summary.csv", proteins)
    write_csv(out / "duplicate_summary.csv", duplicates)
    write_csv(out / "function_coverage.csv", functions)
    write_csv(out / "family_summary.csv", families)
    report = [f"# PhageScope audit report\n", f"- mode: `{mode}`", f"- databases: `{', '.join(databases)}`",
              f"- generated: `{datetime.now(timezone.utc).isoformat()}`", "- source: individual `phage_data_<db>.pkl`; merged pickle was not read", "",
              "## Interpretation", "", "该报告是数据审计，不是模型性能结论。完整审计完成前，不应进行随机 split 或 Evo2 条件微调。", ""]
    for r in results:
        pc, gc = r["protein_counts"], r["genome_counts"]
        report += [f"## {r['database']}", "", f"- records: {r['records']}", f"- genomes with sequence: {gc.get('with_sequence', 0)}",
                   f"- proteins: {pc.get('proteins', 0)}", f"- proteins with DNA CDS: {pc.get('with_dna', 0)}",
                   f"- proteins with product: {pc.get('with_product', 0)}", f"- hypothetical-like products: {pc.get('hypothetical', 0)}",
                   f"- translation checks: {r['translation_matched']}/{r['translation_checked']}",
                   f"- duplicate genome records within DB: {r['genome_duplicate_records']}", ""]
    (out / "audit_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    args = parse_args()
    databases = parse_databases(args.databases, args.source_dir)
    out = args.output_dir / args.mode
    logger = setup_logging(out)
    logger.info("Starting PhageScope audit: mode=%s databases=%s", args.mode, databases)
    available = sorted(p.name[len("phage_data_"):-4] for p in args.source_dir.glob("phage_data_*.pkl")
                       if p.name != "phage_data_all.pkl")
    missing_expected = [name for name in DATABASE_ORDER if name not in available]
    if missing_expected:
        logger.warning("Expected databases without individual pickle: %s", missing_expected)
    schema: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    for name in databases:
        path = args.source_dir / f"phage_data_{name}.pkl"
        logger.info("Loading %s (%.2f GB)", path, path.stat().st_size / 1024**3)
        data = load_database(name, args.source_dir)
        if not schema:
            schema = {"source_dir": str(args.source_dir), "sample_database": name,
                      "available_databases": available, "missing_expected_databases": missing_expected,
                      "sample": schema_probe(data, args.sample_genomes, args.sample_proteins)}
        if args.mode == "full":
            results.append(audit_database(name, data, args, logger))
        else:
            results.append({"database": name, "records": len(data), "schema_only": True,
                            "sample": schema["sample"], "genome_counts": {},
                            "protein_counts": {}, "genome_length_min": None,
                            "genome_length_median": None, "genome_length_max": None,
                            "gc_mean": None, "taxonomy_top": [], "completeness": {},
                            "product_top": [], "classification_top": [],
                            "translation_checked": 0, "translation_matched": 0,
                            "genome_unique_hashes": 0, "genome_duplicate_records": 0,
                            "protein_unique_hashes": 0, "protein_duplicate_records": 0,
                            "elapsed_seconds": 0})
        del data
    write_outputs(out, args.mode, databases, schema, results)
    logger.info("Wrote audit outputs to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
