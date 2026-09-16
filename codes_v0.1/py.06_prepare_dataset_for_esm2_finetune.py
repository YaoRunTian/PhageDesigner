#!/usr/bin/env python3
"""Prepare leakage-controlled GPD datasets for formal ESM-2 fine-tuning.

The script uses only the official GPD ``classification`` field. It retains the
nine valid single-function labels, removes exact duplicate amino-acid sequences,
excludes duplicate sequences with conflicting annotations, assigns genome
clusters to train/validation/final-test (70/15/15), and builds nested 100k and
500k training subsets. Multi-function proteins are kept as a disjoint external
test set.
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
DEFAULT_MANIFEST = PROJECT / "results/03_gpd_qc/gpd_protein_manifest.tsv"
DEFAULT_PICKLE = PROJECT / "results/02_phagescope_v2/gpd_phage_data.pkl"
DEFAULT_OUT = PROJECT / "results/06_prepare_dataset_for_esm2_finetune"
LABELS = (
    "assembly", "replication", "infection", "packaging", "integration",
    "regulation", "lysis", "immune", "tRNA_related",
)
LOWER_TO_LABEL = {label.lower(): label for label in LABELS}
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
LABEL_SPLIT_RE = re.compile(r"[;|,]+")
FIELDS = (
    "phage_id", "protein_id", "genome_cluster", "sequence_sha256",
    "protein_length_aa", "classification", "classification_labels", "split",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pickle", dest="pickle_path", type=Path, default=DEFAULT_PICKLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--subset-sizes", type=int, nargs="*", default=[100_000, 500_000])
    parser.add_argument("--max-rows", type=int, default=0,
                        help="0 scans the complete manifest; positive values are for smoke tests")
    return parser.parse_args()


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("py06_esm2_dataset")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(out_dir / "run.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def validate_args(args: argparse.Namespace) -> None:
    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not args.pickle_path.exists():
        raise SystemExit(f"pickle not found: {args.pickle_path}")
    fractions = (args.train_fraction, args.validation_fraction, args.test_fraction)
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise SystemExit("train/validation/test fractions must be positive and sum to 1")
    if args.max_rows < 0:
        raise SystemExit("--max-rows must be >= 0")
    if any(size <= 0 for size in args.subset_sizes):
        raise SystemExit("--subset-sizes must contain positive integers")


def normalize_sequence(sequence: str) -> str:
    return (sequence or "").strip().upper().rstrip("*")


def parse_annotation(raw: str) -> tuple[str | None, tuple[str, ...]]:
    """Return strict annotation kind and canonical labels.

    A single-function record must contain exactly one non-empty raw token and
    that token must be one of the nine official labels. Multi-function records
    require at least two distinct recognized labels. This preserves the
    previously audited total of 2,961,179 valid single-function occurrences.
    """
    raw_tokens = [token.strip() for token in LABEL_SPLIT_RE.split((raw or "").strip()) if token.strip()]
    recognized = {LOWER_TO_LABEL[token.lower()] for token in raw_tokens if token.lower() in LOWER_TO_LABEL}
    labels = tuple(label for label in LABELS if label in recognized)
    if len(raw_tokens) == 1 and len(labels) == 1:
        return "single", labels
    if len(labels) >= 2:
        return "multifunction", labels
    return None, ()


def stable_fraction(value: str, seed: int, namespace: str) -> float:
    digest = hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def assign_split(cluster: str, args: argparse.Namespace) -> str:
    value = stable_fraction(cluster, args.seed, "cluster_split")
    if value < args.train_fraction:
        return "train"
    if value < args.train_fraction + args.validation_fraction:
        return "validation"
    return "final_test"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, records: list[tuple], split: str) -> Counter:
    counts = Counter()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for phage_id, protein_id, cluster, seq_hash, seq_len, raw, labels in records:
            writer.writerow({
                "phage_id": phage_id,
                "protein_id": protein_id,
                "genome_cluster": cluster,
                "sequence_sha256": seq_hash,
                "protein_length_aa": seq_len,
                "classification": raw,
                "classification_labels": ";".join(labels),
                "split": split,
            })
            for label in labels:
                counts[label] += 1
    return counts


def proportional_quotas(counts: Counter, target: int) -> dict[str, int]:
    available = sum(counts.values())
    target = min(target, available)
    exact = {label: target * counts[label] / available for label in LABELS}
    quotas = {label: min(counts[label], int(exact[label])) for label in LABELS}
    remaining = target - sum(quotas.values())
    order = sorted(LABELS, key=lambda label: (exact[label] - int(exact[label]), counts[label]), reverse=True)
    while remaining:
        changed = False
        for label in order:
            if quotas[label] < counts[label] and remaining:
                quotas[label] += 1
                remaining -= 1
                changed = True
        if not changed:
            break
    return quotas


def nested_subsets(train_records: list[tuple], sizes: list[int], seed: int) -> dict[int, list[tuple]]:
    by_label: dict[str, list[tuple]] = defaultdict(list)
    for record in train_records:
        by_label[record[-1][0]].append(record)
    for label, records in by_label.items():
        records.sort(key=lambda record: stable_fraction(record[3], seed, f"nested_{label}"))
    counts = Counter({label: len(by_label[label]) for label in LABELS})
    output = {}
    previous_hashes: set[str] = set()
    for size in sorted(set(sizes)):
        quotas = proportional_quotas(counts, size)
        subset = []
        for label in LABELS:
            subset.extend(by_label[label][:quotas[label]])
        subset.sort(key=lambda record: stable_fraction(record[3], seed, f"subset_{size}"))
        hashes = {record[3] for record in subset}
        if not previous_hashes.issubset(hashes):
            raise RuntimeError(f"nested subset invariant failed at size {size}")
        previous_hashes = hashes
        output[size] = subset
    return output


def subset_name(size: int) -> str:
    return f"train_{size // 1000}k" if size >= 1000 and size % 1000 == 0 else f"train_{size}"


def main() -> int:
    args = parse_args()
    validate_args(args)
    logger = setup_logger(args.output_dir)
    started = time.time()
    logger.info("loading protein source pickle: %s", args.pickle_path)
    with args.pickle_path.open("rb") as handle:
        phage_data = pickle.load(handle)
    logger.info("loaded %d phage records", len(phage_data))

    # digest -> [representative tuple, annotation kind, labels, occurrences, conflict]
    dedup: dict[str, list] = {}
    counts = Counter()
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"phage_id", "protein_id", "genome_cluster", "classification"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"manifest missing columns: {sorted(missing)}")
        for row_index, row in enumerate(reader, start=1):
            if args.max_rows and row_index > args.max_rows:
                break
            counts["manifest_rows_scanned"] += 1
            kind, labels = parse_annotation(row.get("classification", ""))
            if kind is None:
                counts["excluded_no_valid_classification"] += 1
                continue
            phage_id, protein_id = row["phage_id"], row["protein_id"]
            protein = ((phage_data.get(phage_id, {}).get("proteins") or {}).get(protein_id) or {})
            sequence = normalize_sequence(protein.get("seq") or "")
            if not sequence or not AA_RE.fullmatch(sequence):
                counts["excluded_missing_or_invalid_sequence"] += 1
                continue
            seq_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            cluster = (row.get("genome_cluster") or "").strip() or f"singleton_{phage_id}"
            record = (phage_id, protein_id, cluster, seq_hash, len(sequence),
                      row.get("classification", ""), labels)
            state = dedup.get(seq_hash)
            if state is None:
                dedup[seq_hash] = [record, kind, labels, 1, False]
            else:
                state[3] += 1
                if state[1] != kind or state[2] != labels:
                    state[4] = True
            counts[f"eligible_{kind}_occurrences"] += 1
            if row_index % 500_000 == 0:
                logger.info("scanned %d rows; unique eligible sequences=%d", row_index, len(dedup))

    single_records, multifunction_records = [], []
    for record, kind, labels, occurrences, conflict in dedup.values():
        if occurrences > 1:
            counts["duplicate_occurrences_removed"] += occurrences - 1
        if conflict:
            counts["conflicting_sequence_annotations_excluded"] += 1
            continue
        if kind == "single":
            single_records.append(record)
        else:
            multifunction_records.append(record)
    del dedup
    del phage_data

    split_records: dict[str, list[tuple]] = {"train": [], "validation": [], "final_test": []}
    for record in single_records:
        split_records[assign_split(record[2], args)].append(record)
    for split in split_records:
        split_records[split].sort(key=lambda record: (record[-1][0], record[3]))
    multifunction_records.sort(key=lambda record: (record[-1], record[3]))

    outputs = {}
    class_counts = {}
    for split, filename in (
        ("train", "train_full.tsv"),
        ("validation", "validation.tsv"),
        ("final_test", "final_test.tsv"),
    ):
        path = args.output_dir / filename
        class_counts[split] = dict(write_manifest(path, split_records[split], split))
        outputs[split] = str(path)
    multi_path = args.output_dir / "multifunction_external_test.tsv"
    class_counts["multifunction_external_test"] = dict(
        write_manifest(multi_path, multifunction_records, "multifunction_external_test")
    )
    outputs["multifunction_external_test"] = str(multi_path)

    subsets = nested_subsets(split_records["train"], args.subset_sizes, args.seed)
    subset_counts = {}
    for requested_size, records in subsets.items():
        name = subset_name(requested_size)
        path = args.output_dir / f"{name}.tsv"
        subset_counts[str(requested_size)] = dict(write_manifest(path, records, name))
        outputs[name] = str(path)

    split_hashes = {split: {record[3] for record in records} for split, records in split_records.items()}
    split_clusters = {split: {record[2] for record in records} for split, records in split_records.items()}
    leakage = {}
    names = ("train", "validation", "final_test")
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            leakage[f"sequence_overlap_{left}_{right}"] = len(split_hashes[left] & split_hashes[right])
            leakage[f"cluster_overlap_{left}_{right}"] = len(split_clusters[left] & split_clusters[right])
    multi_hashes = {record[3] for record in multifunction_records}
    leakage["multifunction_vs_single_sequence_overlap"] = len(
        multi_hashes & set().union(*split_hashes.values())
    )
    if any(leakage.values()):
        raise RuntimeError(f"leakage audit failed: {leakage}")

    file_hashes = {key: sha256_file(Path(path)) for key, path in outputs.items()}
    summary = {
        "dataset": "GPD",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 2),
        "parameters": {
            "manifest": str(args.manifest), "pickle": str(args.pickle_path),
            "seed": args.seed, "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction, "test_fraction": args.test_fraction,
            "subset_sizes": sorted(set(args.subset_sizes)), "max_rows": args.max_rows,
        },
        "labels": list(LABELS),
        "counts": dict(counts),
        "unique_single_function_proteins": len(single_records),
        "unique_multifunction_external_proteins": len(multifunction_records),
        "split_records": {split: len(records) for split, records in split_records.items()},
        "split_clusters": {split: len(clusters) for split, clusters in split_clusters.items()},
        "class_counts": class_counts,
        "nested_subset_class_counts": subset_counts,
        "leakage_audit": leakage,
        "outputs": outputs,
        "sha256": file_hashes,
    }
    summary_path = args.output_dir / "dataset_split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# GPD dataset preparation for formal ESM-2 fine-tuning", "",
        f"- Unique single-function proteins: {len(single_records):,}",
        f"- Unique multi-function external-test proteins: {len(multifunction_records):,}",
        f"- Duplicate occurrences removed: {counts['duplicate_occurrences_removed']:,}",
        f"- Conflicting sequence annotations excluded: {counts['conflicting_sequence_annotations_excluded']:,}",
        "", "## Formal split", "", "| Split | Proteins | Genome clusters |",
        "|---|---:|---:|",
    ]
    report.extend(
        f"| {split} | {len(split_records[split]):,} | {len(split_clusters[split]):,} |"
        for split in names
    )
    report += ["", "## Leakage audit", "", "| Check | Overlap |", "|---|---:|"]
    report.extend(f"| {key} | {value:,} |" for key, value in leakage.items())
    report += [
        "", "## Notes", "",
        "Splits are assigned by a stable hash of genome_cluster. Exact amino-acid duplicates are removed globally before splitting. Sequences with conflicting single/multi-function or discordant labels are excluded. The validation set is used for model selection; final_test must remain untouched until the configuration is frozen. Multi-function proteins form a disjoint external test set.",
    ]
    (args.output_dir / "dataset_split_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("completed in %.1f minutes", (time.time() - started) / 60)
    logger.info("split sizes: %s", {key: len(value) for key, value in split_records.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
