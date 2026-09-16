#!/usr/bin/env python3
"""Create deterministic genome-cluster-level splits for Microviridae proteins.

Only high-confidence single-function rows are used for train/validation/test.
Multi-function rows are kept as a separate evaluation set, restricted to
clusters assigned to ``final_test`` so no genome cluster leaks between the
single-label training data and the multi-function evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_LABELS = PROJECT / "results/10_microviridae_function_labels/microviridae_protein_function_manifest.tsv"
DEFAULT_GENOMES = PROJECT / "results/09_microviridae_dataset/microviridae_genomes.tsv"
DEFAULT_OUT = PROJECT / "results/11_microviridae_cluster_splits"


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--genomes", type=Path, default=DEFAULT_GENOMES)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--validation-frac", type=float, default=0.1)
    p.add_argument("--min-confidence", choices=["high", "medium"], default="high")
    return p.parse_args()


def stable_hash(text: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}\0{text}".encode()).hexdigest()[:16], 16)


def read_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as h:
        yield from csv.DictReader(h, delimiter="\t")


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    genome_cluster = {}
    for row in read_tsv(a.genomes):
        genome_cluster[row["phage_id"]] = row.get("genome_cluster") or row["phage_id"]

    rows = list(read_tsv(a.labels))
    fields = list(rows[0]) if rows else []
    clusters: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        row["genome_cluster"] = genome_cluster.get(row.get("phage_id", ""), row.get("phage_id", ""))
        clusters[row["genome_cluster"]].append(row)

    # Assign whole clusters greedily to the target split.  Large clusters are
    # placed first; deterministic hashing breaks ties.
    cluster_items = sorted(
        clusters.items(),
        key=lambda kv: (-len(kv[1]), stable_hash(kv[0], a.seed)),
    )
    total_rows = len(rows)
    targets = {
        "train": total_rows * a.train_frac,
        "validation": total_rows * a.validation_frac,
        "final_test": total_rows * (1.0 - a.train_frac - a.validation_frac),
    }
    assigned_counts = Counter()
    cluster_split = {}
    for cluster, members in cluster_items:
        # Minimize normalized deficit.  Do not let validation/test be starved.
        choices = []
        for split, target in targets.items():
            deficit = max(target - assigned_counts[split], 0.0) / max(target, 1.0)
            choices.append((deficit, -assigned_counts[split], split))
        choices.sort(reverse=True)
        split = choices[0][2]
        cluster_split[cluster] = split
        assigned_counts[split] += len(members)

    for row in rows:
        row["split"] = cluster_split[row["genome_cluster"]]

    # High-confidence single labels are the classifier/generative training
    # population.  Medium confidence is retained in a separate optional pool.
    conf_rank = {"high": 2, "medium": 1, "unknown": 0}
    min_rank = conf_rank[a.min_confidence]
    single = [r for r in rows if r.get("label_scope") == "single" and conf_rank.get(r.get("label_confidence", "unknown"), 0) >= min_rank]
    multi_test = [r for r in rows if r.get("label_scope") == "multi" and r.get("split") == "final_test"]
    unknown_test = [r for r in rows if r.get("label_scope") == "unknown" and r.get("split") == "final_test"]
    optional_medium = [r for r in rows if r.get("label_scope") == "single" and r.get("label_confidence") == "medium"]

    outputs = {}
    for name, subset in [
        ("train.tsv", [r for r in single if r["split"] == "train"]),
        ("validation.tsv", [r for r in single if r["split"] == "validation"]),
        ("final_test.tsv", [r for r in single if r["split"] == "final_test"]),
        ("multifunction_test.tsv", multi_test),
        ("unknown_final_test.tsv", unknown_test),
        ("single_function_medium_optional.tsv", optional_medium),
    ]:
        path = a.output_dir / name
        write_rows(path, subset, fields + ["genome_cluster", "split"])
        outputs[name] = str(path)
    assignment_rows = [{"genome_cluster": c, "split": s, "protein_rows": str(len(clusters[c]))} for c, s in sorted(cluster_split.items())]
    assignment_path = a.output_dir / "cluster_assignments.tsv"
    write_rows(assignment_path, assignment_rows, ["genome_cluster", "split", "protein_rows"])
    outputs["cluster_assignments.tsv"] = str(assignment_path)

    def label_counts(subset):
        return dict(Counter(r.get("normalized_function", "unknown") for r in subset).most_common())

    summary = {
        "dataset": "Microviridae",
        "seed": a.seed,
        "split_rule": "whole genome_cluster; deterministic greedy balancing by protein-row count",
        "min_confidence": a.min_confidence,
        "cluster_count": len(clusters),
        "cluster_split_counts": dict(Counter(cluster_split.values())),
        "all_manifest_rows": len(rows),
        "single_function_rows_used": len(single),
        "split_row_counts": {
            "train": sum(r["split"] == "train" for r in single),
            "validation": sum(r["split"] == "validation" for r in single),
            "final_test": sum(r["split"] == "final_test" for r in single),
            "multifunction_test": len(multi_test),
            "unknown_final_test": len(unknown_test),
        },
        "label_counts": {
            "train": label_counts([r for r in single if r["split"] == "train"]),
            "validation": label_counts([r for r in single if r["split"] == "validation"]),
            "final_test": label_counts([r for r in single if r["split"] == "final_test"]),
            "multifunction_test": label_counts(multi_test),
        },
        "outputs": outputs,
    }
    (a.output_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Microviridae cluster-level splits", "",
        f"- Genome clusters: {len(clusters):,}",
        f"- High-confidence single-function rows used: {len(single):,}",
        f"- Train / validation / final test: {summary['split_row_counts']['train']:,} / {summary['split_row_counts']['validation']:,} / {summary['split_row_counts']['final_test']:,}",
        f"- Multi-function final-test rows: {len(multi_test):,}",
        "", "The split unit is genome_cluster; proteins from the same cluster never cross train/validation/final_test.",
    ]
    (a.output_dir / "split_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
