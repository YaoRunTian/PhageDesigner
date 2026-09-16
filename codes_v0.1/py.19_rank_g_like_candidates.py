#!/usr/bin/env python3
"""Rank and diversity-select G-like proteins that passed sequence and structure screens."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_SEQUENCE = PROJECT / "results/17_g_like_sequence_screen/sequence_screening.tsv"
DEFAULT_STRUCTURE = PROJECT / "results/18_g_like_final_screen/structure_metrics.tsv"
DEFAULT_OUT = PROJECT / "results/19_g_like_candidate_ranking"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence-metrics", type=Path, default=DEFAULT_SEQUENCE)
    p.add_argument("--structure-metrics", type=Path, default=DEFAULT_STRUCTURE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--select-n", type=int, default=100)
    p.add_argument("--quality-weight", type=float, default=0.75)
    p.add_argument("--diversity-weight", type=float, default=0.25)
    return p.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def minmax(values: list[float], value: float, reverse: bool = False) -> float:
    lo, hi = min(values), max(values)
    score = 1.0 if hi == lo else (value - lo) / (hi - lo)
    return 1.0 - score if reverse else score


def hamming_fraction(a: str, b: str) -> float:
    if len(a) != len(b):
        return 1.0
    return sum(x != y for x, y in zip(a, b)) / max(len(a), 1)


def allocate_quotas(rows: list[dict], n: int) -> dict[str, int]:
    """Reserve representation for higher-mutation groups, then reallocate shortages."""
    available = Counter(row["mask_rate"] for row in rows)
    rates = sorted(available, key=float)
    if not rates:
        return {}
    base = n // len(rates)
    quotas = {rate: min(base, available[rate]) for rate in rates}
    remaining = n - sum(quotas.values())
    while remaining:
        changed = False
        for rate in rates:
            if quotas[rate] < available[rate]:
                quotas[rate] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return quotas


def main() -> int:
    a = parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {a.output_dir}")
    if a.select_n < 1:
        raise ValueError("--select-n must be positive")
    if abs(a.quality_weight + a.diversity_weight - 1.0) > 1e-9:
        raise ValueError("quality and diversity weights must sum to 1")
    a.output_dir.mkdir(parents=True)

    seq = {row["candidate_id"]: row for row in read_tsv(a.sequence_metrics)}
    structures = [row for row in read_tsv(a.structure_metrics) if as_bool(row["structure_pass"])]
    rows = []
    for structure in structures:
        earlier = seq[structure["candidate_id"]]
        rows.append(
            {
                "candidate_id": structure["candidate_id"],
                "mask_rate": f"{float(structure['mask_rate']):.2f}",
                "mutation_count": int(earlier["mutation_count"]),
                "mmseqs_pident": float(earlier["mmseqs_pident"]),
                "conserved_retention": float(earlier["conserved_retention"]),
                "mean_plddt": float(structure["mean_plddt"]),
                "ca_rmsd": float(structure["ca_rmsd"]),
                "tm_like_score": float(structure["tm_like_score"]),
                "sequence": structure["sequence"],
            }
        )
    if not rows:
        raise RuntimeError("No structure-passing candidates")

    components = {
        "mean_plddt": [r["mean_plddt"] for r in rows],
        "tm_like_score": [r["tm_like_score"] for r in rows],
        "ca_rmsd": [r["ca_rmsd"] for r in rows],
        "conserved_retention": [r["conserved_retention"] for r in rows],
        "mmseqs_pident": [r["mmseqs_pident"] for r in rows],
    }
    for row in rows:
        row["quality_score"] = (
            0.25 * minmax(components["mean_plddt"], row["mean_plddt"])
            + 0.25 * minmax(components["tm_like_score"], row["tm_like_score"])
            + 0.15 * minmax(components["ca_rmsd"], row["ca_rmsd"], reverse=True)
            + 0.20 * minmax(components["conserved_retention"], row["conserved_retention"])
            + 0.15 * minmax(components["mmseqs_pident"], row["mmseqs_pident"])
        )

    quotas = allocate_quotas(rows, min(a.select_n, len(rows)))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["mask_rate"]].append(row)
    selected: list[dict] = []
    for rate in sorted(grouped, key=float, reverse=True):
        pool = sorted(grouped[rate], key=lambda x: (-x["quality_score"], x["candidate_id"]))
        for _ in range(quotas[rate]):
            best = None
            for row in pool:
                if row in selected:
                    continue
                novelty = min((hamming_fraction(row["sequence"], s["sequence"]) for s in selected), default=1.0)
                selection_score = a.quality_weight * row["quality_score"] + a.diversity_weight * novelty
                key = (selection_score, row["quality_score"], row["candidate_id"])
                if best is None or key > best[0]:
                    best = (key, row, novelty, selection_score)
            if best is None:
                break
            best[1]["novelty_at_selection"] = best[2]
            best[1]["selection_score"] = best[3]
            selected.append(best[1])

    selected_ids = {r["candidate_id"] for r in selected}
    for row in rows:
        row["selected"] = row["candidate_id"] in selected_ids
        row.setdefault("novelty_at_selection", "")
        row.setdefault("selection_score", "")
    rows.sort(key=lambda x: (-x["quality_score"], x["candidate_id"]))
    for rank, row in enumerate(rows, 1):
        row["quality_rank"] = rank

    fields = [
        "candidate_id", "mask_rate", "mutation_count", "mmseqs_pident",
        "conserved_retention", "mean_plddt", "ca_rmsd", "tm_like_score",
        "quality_score", "quality_rank", "novelty_at_selection", "selection_score",
        "selected", "sequence",
    ]
    with (a.output_dir / "all_ranked_candidates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    selected.sort(key=lambda x: (-x["selection_score"], -x["quality_score"], x["candidate_id"]))
    with (a.output_dir / "selected_candidates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(selected)
    with (a.output_dir / "selected_candidates.fasta").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(f">{row['candidate_id']} mask_rate={row['mask_rate']} quality={row['quality_score']:.6f}\n{row['sequence']}\n")

    summary = {
        "status": "passed",
        "input_structure_pass": len(rows),
        "selected": len(selected),
        "available_by_mask_rate": dict(Counter(r["mask_rate"] for r in rows)),
        "selection_quota_by_mask_rate": quotas,
        "selected_by_mask_rate": dict(Counter(r["mask_rate"] for r in selected)),
        "quality_formula": {
            "mean_plddt": 0.25, "tm_like_score": 0.25, "inverse_ca_rmsd": 0.15,
            "conserved_retention": 0.20, "mmseqs_pident": 0.15,
        },
        "diverse_selection": {"quality_weight": a.quality_weight, "diversity_weight": a.diversity_weight},
    }
    (a.output_dir / "ranking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
