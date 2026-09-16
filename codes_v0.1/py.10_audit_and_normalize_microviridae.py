#!/usr/bin/env python3
"""Audit Microviridae protein annotations and create reproducible labels.

The input is the quality-controlled protein manifest produced by
``py.09_Microviridae_dataset.py``.  Raw ``product`` and ``classification``
values are retained.  ``classification`` is authoritative when it contains a
known functional token; ``product`` is used only when classification is
unknown.  This prevents a verbose product description from silently
overriding the database's curated class while still rescuing useful product
annotations such as ``Major spike protein (G protein)``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_INPUT = PROJECT / "results/09_microviridae_dataset/microviridae_proteins.tsv"
DEFAULT_OUT = PROJECT / "results/10_microviridae_function_labels"

# These are intentionally broad first-level labels.  They are a rule-based
# normalization for model development, not expert-confirmed functional truth.
CLASS_MAP = {
    "assembly": "structural",
    "structural": "structural",
    "morphogenesis": "structural",
    "replication": "replication",
    "packaging": "packaging",
    "lysis": "lysis",
    "infection": "host_interaction",
    "host_interaction": "host_interaction",
    "integration": "integration",
    "regulation": "regulation",
    "transcription": "transcription",
    "translation": "translation",
    "metabolism": "metabolism",
    "immune": "defense",
    "defense": "defense",
    "tRNA_related": "tRNA_related",
}
IGNORED_CLASS = {"", "unsorted", "hypothetical", "unknown", "uncharacterized"}

PRODUCT_RULES = [
    ("host_interaction", re.compile(
        r"spike|tail\s*fiber|receptor|adsorption|host\s*range|ejection|injection|pilot\s+protein",
        re.I)),
    ("structural", re.compile(
        r"capsid|coat\s+protein|major\s+head|head\s+protein|scaffold|baseplate|portal|virion|structural|morphogenesis|assembly",
        re.I)),
    ("packaging", re.compile(r"packag|terminase", re.I)),
    ("replication", re.compile(
        r"replicat|polymerase|primase|helicase|nuclease|exonuclease|recombin|dna[- ]binding|restriction",
        re.I)),
    ("transcription", re.compile(r"transcription|transcriptase|rna\s+polymerase|sigma\s+factor", re.I)),
    ("translation", re.compile(r"translation|ribosom|trna|aminoacyl", re.I)),
    ("lysis", re.compile(r"lysis|lysin|holin|endolysin|spanin", re.I)),
    ("defense", re.compile(r"anti[- ]?crispr|anti[- ]?restriction|immunity|defen[cs]|toxin", re.I)),
    ("metabolism", re.compile(r"metabol|dehydrogenase|kinase|synthetase|oxidoreductase", re.I)),
    ("regulation", re.compile(r"regulator|regulation|transcriptional\s+regulator", re.I)),
]
UNKNOWN_PRODUCT = re.compile(
    r"(^|\b)(unknown|hypothetical|uncharacterized|unnamed|putative\s+protein|protein\s+of\s+unknown\s+function)(\b|$)",
    re.I,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def classification_labels(text: str) -> list[str]:
    labels = set()
    for token in re.split(r"[;|,]", text.lower()):
        token = token.strip()
        if token in IGNORED_CLASS:
            continue
        if token in CLASS_MAP:
            labels.add(CLASS_MAP[token])
    return sorted(labels)


def product_labels(text: str) -> list[str]:
    if not text or UNKNOWN_PRODUCT.search(text):
        return []
    return sorted({label for label, pattern in PRODUCT_RULES if pattern.search(text)})


def stable_key(row: dict[str, str]) -> str:
    return hashlib.sha256(
        (row.get("phage_id", "") + "\0" + row.get("protein_id", "") + "\0" + row.get("protein_sequence", "")).encode()
    ).hexdigest()


def main() -> int:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = a.output_dir / "microviridae_protein_function_manifest.tsv"
    rows_by_scope: dict[str, list[dict[str, str]]] = {"single": [], "multi": [], "unknown": []}
    counts = Counter()
    class_counts = Counter()
    product_counts = Counter()
    normalized_counts = Counter()
    confidence_counts = Counter()
    source_counts = Counter()
    conflict_counts = Counter()

    with a.input.open(encoding="utf-8", newline="") as src, manifest_path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src, delimiter="\t")
        fields = reader.fieldnames or []
        extra = [
            "classification_function", "product_function", "normalized_function",
            "label_scope", "label_source", "label_confidence", "annotation_conflict",
        ]
        writer = csv.DictWriter(dst, fieldnames=fields + extra, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            counts["proteins"] += 1
            raw_class = clean_text(row.get("classification"))
            raw_product = clean_text(row.get("product"))
            cls = classification_labels(raw_class)
            prod = product_labels(raw_product)
            class_text = ";".join(cls) if cls else "unknown"
            product_text = ";".join(prod) if prod else "unknown"
            if cls:
                labels = cls
                source = "classification"
                confidence = "high"
            elif prod:
                labels = prod
                source = "product"
                confidence = "medium"
            else:
                labels = []
                source = "none"
                confidence = "unknown"
            conflict = bool(cls and prod and set(cls) != set(prod))
            scope = "single" if len(labels) == 1 else "multi" if len(labels) > 1 else "unknown"
            normalized = ";".join(labels) if labels else "unknown"
            row.update(
                classification_function=class_text,
                product_function=product_text,
                normalized_function=normalized,
                label_scope=scope,
                label_source=source,
                label_confidence=confidence,
                annotation_conflict="1" if conflict else "0",
            )
            writer.writerow(row)
            rows_by_scope[scope].append(row)
            normalized_counts[normalized] += 1
            confidence_counts[confidence] += 1
            class_counts[class_text] += 1
            product_counts[product_text] += 1
            source_counts[source] += 1
            conflict_counts["conflict" if conflict else "consistent"] += 1

    # Write subsets for downstream split construction and smoke tests.
    output_paths = {"manifest": str(manifest_path)}
    for scope, rows in rows_by_scope.items():
        path = a.output_dir / f"microviridae_{scope}_function_proteins.tsv"
        if rows:
            fields = list(rows[0])
        else:
            fields = []
        with path.open("w", encoding="utf-8", newline="") as h:
            if fields:
                writer = csv.DictWriter(h, fieldnames=fields, delimiter="\t", extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        output_paths[scope] = str(path)

    summary = {
        "dataset": "Microviridae",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(a.input),
        "counts": dict(counts),
        "label_scope_counts": {k: len(v) for k, v in rows_by_scope.items()},
        "normalized_function_counts": dict(normalized_counts.most_common()),
        "classification_function_counts": dict(class_counts.most_common()),
        "product_function_counts": dict(product_counts.most_common()),
        "label_confidence_counts": dict(confidence_counts),
        "label_source_counts": dict(source_counts),
        "annotation_conflict_counts": dict(conflict_counts),
        "rules": {
            "classification_priority": True,
            "product_fills_unknown_classification": True,
            "unknown_and_unsorted_are_excluded_from_function_labels": True,
            "labels_are_rule_based_not_expert_gold_standard": True,
        },
        "outputs": output_paths,
    }
    summary_path = a.output_dir / "microviridae_function_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_paths["summary"] = str(summary_path)

    report = [
        "# Microviridae function-label audit", "",
        f"- Proteins processed: {counts['proteins']:,}",
        "- Classification is authoritative when a known token is present; product fills unknown classification.",
        "- Labels are rule-based development labels, not manually reviewed ground truth.", "",
        "| Label scope | Count |", "|---|---:|",
    ]
    report.extend(f"| {key} | {len(rows):,} |" for key, rows in rows_by_scope.items())
    report.extend(["", "| Normalized function | Count |", "|---|---:|"])
    report.extend(f"| {key} | {value:,} |" for key, value in normalized_counts.most_common())
    report.extend(["", "| Confidence | Count |", "|---|---:|"])
    report.extend(f"| {key} | {value:,} |" for key, value in confidence_counts.most_common())
    report.extend(["", "| Annotation comparison | Count |", "|---|---:|"])
    report.extend(f"| {key} | {value:,} |" for key, value in conflict_counts.most_common())
    (a.output_dir / "microviridae_function_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
