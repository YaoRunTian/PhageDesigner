#!/usr/bin/env python3
"""Compare RNA folding around the PhiX174 G CDS start for designed genomes.

The fixed window contains 60 nt upstream and the first 180 nt of G by default.
ViennaRNA reports MFE, ensemble free energy, centroid structure, ensemble
diversity and start-region accessibility.  These are local RNA-structure
descriptors, not evidence of infectivity.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_GENOMES = PROJECT / "results/21_phix174_g_replacement_smoke/designed_phix174_genomes.fasta"
DEFAULT_GENBANK = PROJECT / "references/NC_001422.1.gb"
DEFAULT_OUT = PROJECT / "results/25_g_start_mrna_evidence"
G_START0 = 2394


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--genomes", type=Path, default=DEFAULT_GENOMES)
    p.add_argument("--genbank", type=Path, default=DEFAULT_GENBANK)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--upstream", type=int, default=60)
    p.add_argument("--downstream", type=int, default=180)
    p.add_argument("--accessibility-start", type=int, default=-4,
                   help="Offset from G AUG, inclusive")
    p.add_argument("--accessibility-end", type=int, default=30,
                   help="Offset from G AUG, exclusive")
    return p.parse_args()


def circular_slice(seq: str, start: int, end: int) -> str:
    n = len(seq)
    return "".join(seq[i % n] for i in range(start, end))


def fold_metrics(dna: str, access_start: int, access_end: int) -> dict:
    import RNA

    rna = dna.replace("T", "U")
    fc = RNA.fold_compound(rna)
    mfe_structure, mfe = fc.mfe()
    _, ensemble_energy = fc.pf()
    centroid, distance = fc.centroid()
    # bpp(i,j) is 1-based.  Unpaired probability = 1 - sum_j p(i,j).
    bpp = np.asarray(fc.bpp(), dtype=float)
    unpaired = []
    for i in range(1, len(rna) + 1):
        # ViennaRNA returns a 1-based upper-triangular matrix.
        paired = float(bpp[i, :].sum() + bpp[:, i].sum() - bpp[i, i])
        unpaired.append(max(0.0, min(1.0, 1.0 - paired)))
    return {
        "mfe_kcal_mol": float(mfe),
        "ensemble_free_energy_kcal_mol": float(ensemble_energy),
        "ensemble_diversity": float(fc.mean_bp_distance()),
        "mfe_structure": mfe_structure,
        "centroid_structure": centroid,
        "centroid_distance": float(distance),
        "start_accessibility_mean": float(np.mean(unpaired[access_start:access_end])),
        "start_accessibility_min": float(np.min(unpaired[access_start:access_end])),
    }


def main() -> int:
    a = parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    from Bio import SeqIO
    import RNA

    record = SeqIO.read(a.genbank, "genbank")
    reference = str(record.seq).upper()
    genomes = [(r.id, str(r.seq).upper()) for r in SeqIO.parse(a.genomes, "fasta")]
    if not genomes:
        raise RuntimeError(f"No genomes found in {a.genomes}")
    start, end = G_START0 - a.upstream, G_START0 + a.downstream
    access_start = a.upstream + a.accessibility_start
    access_end = a.upstream + a.accessibility_end
    reference_window = circular_slice(reference, start, end)
    ref = fold_metrics(reference_window, access_start, access_end)
    rows = []
    for candidate_id, genome in genomes:
        if len(genome) != len(reference):
            raise ValueError(f"{candidate_id}: genome length {len(genome)} != {len(reference)}")
        window = circular_slice(genome, start, end)
        metrics = fold_metrics(window, access_start, access_end)
        rows.append({
            "candidate_id": candidate_id,
            **metrics,
            "delta_mfe_vs_phix174": metrics["mfe_kcal_mol"] - ref["mfe_kcal_mol"],
            "delta_ensemble_energy_vs_phix174": metrics["ensemble_free_energy_kcal_mol"] - ref["ensemble_free_energy_kcal_mol"],
            "delta_start_accessibility_vs_phix174": metrics["start_accessibility_mean"] - ref["start_accessibility_mean"],
            "mfe_base_pair_distance_vs_phix174": RNA.bp_distance(metrics["mfe_structure"], ref["mfe_structure"]),
            "window_nt_changes_vs_phix174": sum(x != y for x, y in zip(window, reference_window)),
            "window_sequence": window,
        })
    fields = list(rows[0])
    with (a.output_dir / "g_start_mrna_evidence.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": "complete",
        "candidate_count": len(rows),
        "viennarna_version": RNA.__version__,
        "window": {"upstream_nt": a.upstream, "downstream_nt": a.downstream, "length": len(reference_window)},
        "accessibility_offsets_from_G_start": [a.accessibility_start, a.accessibility_end],
        "reference_metrics": ref,
        "interpretation": "Report changes relative to PhiX174; no hard viability threshold is applied.",
    }
    (a.output_dir / "g_start_mrna_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
