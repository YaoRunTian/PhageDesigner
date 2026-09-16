#!/usr/bin/env python3
"""Compile sequence, multimer, RNA and Evo2 evidence for designed genomes.

The final table is descriptive.  It intentionally contains no ``viable`` or
binary feasibility column.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_CDS = PROJECT / "results/20_evo2_reverse_translation_smoke/selected_cds.tsv"
DEFAULT_RANKING = PROJECT / "results/19_g_like_candidate_ranking/all_ranked_candidates.tsv"
DEFAULT_SEQUENCE = PROJECT / "results/17_g_like_sequence_screen/sequence_screening.tsv"
DEFAULT_MONOMER = PROJECT / "results/18_g_like_final_screen/structure_metrics.tsv"
DEFAULT_MULTIMER = PROJECT / "results/24_multimer_structure_evidence/structure_evidence.tsv"
DEFAULT_MRNA = PROJECT / "results/25_g_start_mrna_evidence/g_start_mrna_evidence.tsv"
DEFAULT_EVO2 = PROJECT / "results/26_full_genome_evo2_evidence/full_genome_likelihood_evidence.tsv"
DEFAULT_OUT = PROJECT / "results/27_computational_evidence"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selected-cds", type=Path, default=DEFAULT_CDS)
    p.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    p.add_argument("--sequence-screen", type=Path, default=DEFAULT_SEQUENCE)
    p.add_argument("--monomer-screen", type=Path, default=DEFAULT_MONOMER)
    p.add_argument("--multimer", type=Path, default=DEFAULT_MULTIMER)
    p.add_argument("--mrna", type=Path, default=DEFAULT_MRNA)
    p.add_argument("--evo2", type=Path, default=DEFAULT_EVO2)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def canonical_id(value: str) -> str:
    return value.removeprefix("PHIX174_G_")


def index(path: Path, key: str) -> dict[str, dict[str, str]]:
    return {canonical_id(r[key]): r for r in read_rows(path)}


def add(row: dict, source: dict | None, mapping: dict[str, str], prefix: str = "") -> None:
    source = source or {}
    for old, new in mapping.items():
        row[prefix + new] = source.get(old, "")


def main() -> int:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    selected = read_rows(a.selected_cds)
    ranking = index(a.ranking, "candidate_id")
    sequence = index(a.sequence_screen, "candidate_id")
    monomer = index(a.monomer_screen, "candidate_id")
    mrna = index(a.mrna, "candidate_id")
    evo = {canonical_id(r["sequence_id"]): r for r in read_rows(a.evo2) if r["kind"] == "designed"}
    multimer_rows = read_rows(a.multimer)
    multimer = {(canonical_id(r["candidate_id"]), r["complex_type"]): r for r in multimer_rows}

    rows = []
    for selected_row in selected:
        candidate_id = canonical_id(selected_row["candidate_id"])
        row = {
            "candidate_id": candidate_id,
            "mask_rate": selected_row.get("mask_rate", ""),
            "protein_length_aa": len(selected_row.get("protein_sequence", "")),
            "cds_length_nt_including_stop": len(selected_row.get("cds_sequence", "")) + len(selected_row.get("stop_codon", "")),
            "reverse_translation_evo2_log_likelihood": selected_row.get("evo2_log_likelihood", ""),
        }
        add(row, ranking.get(candidate_id), {
            "mutation_count": "mutation_count",
            "combined_score": "upstream_combined_score",
        })
        add(row, sequence.get(candidate_id), {
            "mmseqs_pident": "G_mmseqs_identity",
            "mmseqs_qcov": "G_mmseqs_query_coverage",
            "mmseqs_tcov": "G_mmseqs_target_coverage",
            "phrog_target": "PHROG_target",
            "phrog_evalue": "PHROG_evalue",
            "conserved_retention": "G_conserved_site_retention",
        })
        add(row, monomer.get(candidate_id), {
            "mean_plddt": "ESMFold_monomer_mean_pLDDT",
            "ca_rmsd": "ESMFold_CA_RMSD_vs_ESMFold_reference",
            "tm_like_score": "ESMFold_TM_like_vs_ESMFold_reference",
        })
        structural_map = {
            "confidence_score": "confidence_score",
            "ptm": "pTM",
            "iptm": "ipTM",
            "protein_iptm": "protein_ipTM",
            "complex_plddt": "complex_pLDDT",
            "pae_interchain": "mean_interchain_PAE_A",
            "pae_FG": "mean_FG_PAE_A",
            "pae_GJ": "mean_GJ_PAE_A",
            "contact_pairs_GG": "GG_heavy_atom_contact_pairs",
            "contact_pairs_FG": "FG_heavy_atom_contact_pairs",
            "contact_pairs_GJ": "GJ_heavy_atom_contact_pairs",
            "g_interface_position_count": "G_interface_position_count",
            "conserved_g_interface_positions": "conserved_G_interface_positions",
            "conserved_g_interface_retention": "conserved_G_interface_retention",
            "tm_align_mean": "TM_align_G_chain_mean_vs_36CQ",
            "tm_align_min": "TM_align_G_chain_min_vs_36CQ",
        }
        add(row, multimer.get((candidate_id, "g5")), structural_map, "G5_")
        add(row, multimer.get((candidate_id, "fgj3")), structural_map, "F1G1J1_")
        add(row, mrna.get(candidate_id), {
            "mfe_kcal_mol": "G_start_window_MFE_kcal_mol",
            "delta_mfe_vs_phix174": "G_start_delta_MFE_vs_PhiX174",
            "ensemble_diversity": "G_start_ensemble_diversity",
            "start_accessibility_mean": "G_start_accessibility_mean",
            "delta_start_accessibility_vs_phix174": "G_start_delta_accessibility_vs_PhiX174",
            "mfe_base_pair_distance_vs_phix174": "G_start_structure_bp_distance_vs_PhiX174",
            "window_nt_changes_vs_phix174": "G_start_window_nt_changes",
        })
        add(row, evo.get(candidate_id), {
            "mean_log_likelihood_per_nt": "full_genome_Evo2_mean_log_likelihood_per_nt",
            "delta_vs_phix174": "full_genome_Evo2_delta_vs_PhiX174",
            "natural_variant_zscore": "full_genome_Evo2_natural_variant_zscore",
            "natural_variant_percentile": "full_genome_Evo2_natural_variant_percentile",
        })
        required = [
            "G5_ipTM", "F1G1J1_ipTM", "G_start_window_MFE_kcal_mol",
            "full_genome_Evo2_mean_log_likelihood_per_nt",
        ]
        missing = [name for name in required if row.get(name, "") == ""]
        row["evidence_data_status"] = "complete" if not missing else "missing: " + ",".join(missing)
        row["interpretation_scope"] = "computational evidence only; no viable designation"
        rows.append(row)

    if not rows:
        raise RuntimeError("No selected candidates")
    fields = list(rows[0])
    table = a.output_dir / "integrated_computational_evidence.tsv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)

    complete = sum(r["evidence_data_status"] == "complete" for r in rows)
    summary = {
        "status": "complete" if complete == len(rows) else "partial",
        "candidate_count": len(rows),
        "complete_evidence_rows": complete,
        "columns": len(fields),
        "viability_label_present": False,
        "table": str(table),
    }
    (a.output_dir / "integrated_evidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    def values(column: str) -> list[float]:
        return [float(r[column]) for r in rows if r.get(column, "") not in {"", None}]

    def span(column: str, digits: int = 3) -> str:
        xs = values(column)
        return f"{min(xs):.{digits}f}–{max(xs):.{digits}f}"

    g5_reference = next(r for r in multimer_rows if r["candidate_id"] == "phix174_reference" and r["complex_type"] == "g5")
    fgj_reference = next(r for r in multimer_rows if r["candidate_id"] == "phix174_reference" and r["complex_type"] == "fgj3")
    by_mask = {}
    for mask in sorted({r["mask_rate"] for r in rows}, key=float):
        group = [r for r in rows if r["mask_rate"] == mask]
        by_mask[mask] = {
            "mean_evo2_logp": statistics.mean(float(r["full_genome_Evo2_mean_log_likelihood_per_nt"]) for r in group),
            "mean_natural_percentile": statistics.mean(float(r["full_genome_Evo2_natural_variant_percentile"]) for r in group),
        }
    best_evo = max(rows, key=lambda r: float(r["full_genome_Evo2_natural_variant_percentile"]))
    best_tm = max(rows, key=lambda r: float(r["G5_TM_align_G_chain_mean_vs_36CQ"]))
    best_g5_iptm = max(rows, key=lambda r: float(r["G5_ipTM"]))
    report = f"""# ΦX174 G-like candidate computational evidence

## Scope

The table integrates sequence similarity and annotation, ESMFold monomer evidence,
Boltz-2 G pentamer and F/G/J asymmetric-unit predictions, TM-align comparisons to
the natural ΦX174 cryo-EM structure (PDB 36CQ), interface contacts and conserved
interface residues, ViennaRNA folding around the G start codon, and full-genome
Evo2 likelihood relative to ΦX174 and natural variants.

## Result

- Candidates: {len(rows)}
- Rows with all requested evidence fields: {complete}/{len(rows)}
- A `viable` label was not generated.

## Main observations

- G5 predictions span ipTM {span('G5_ipTM')} and mean inter-chain PAE {span('G5_mean_interchain_PAE_A', 2)} Å. The no-MSA ΦX174 reference prediction itself has ipTM {float(g5_reference['iptm']):.3f} and PAE {float(g5_reference['pae_interchain']):.2f} Å, so absolute complex-confidence values are exploratory.
- Chainwise TM-align similarity to the natural 36CQ G pentamer spans {span('G5_TM_align_G_chain_mean_vs_36CQ')}. Conserved G-interface retention spans {span('G5_conserved_G_interface_retention')}.
- F1/G1/J1 predictions span ipTM {span('F1G1J1_ipTM')} and mean inter-chain PAE {span('F1G1J1_mean_interchain_PAE_A', 2)} Å. The corresponding ΦX174 reference prediction has ipTM {float(fgj_reference['iptm']):.3f} and PAE {float(fgj_reference['pae_interchain']):.2f} Å.
- G-start-window ΔMFE spans {span('G_start_delta_MFE_vs_PhiX174', 2)} kcal/mol and the change in mean start accessibility spans {span('G_start_delta_accessibility_vs_PhiX174', 3)}.
- Every designed full genome has lower Evo2 likelihood than ΦX174. The best empirical natural-variant percentile is {100*float(best_evo['full_genome_Evo2_natural_variant_percentile']):.1f}% ({best_evo['candidate_id']}). Mean percentiles by mask rate are 5%: {100*by_mask['0.05']['mean_natural_percentile']:.1f}%, 10%: {100*by_mask['0.10']['mean_natural_percentile']:.2f}%, 20%: {100*by_mask['0.20']['mean_natural_percentile']:.1f}%.
- Evidence is not concordant enough for a composite biological claim: {best_tm['candidate_id']} has the highest G-chain TM-align mean ({float(best_tm['G5_TM_align_G_chain_mean_vs_36CQ']):.3f}), whereas {best_g5_iptm['candidate_id']} has the highest G5 ipTM ({float(best_g5_iptm['G5_ipTM']):.3f}) but an Evo2 natural-variant percentile of {100*float(best_g5_iptm['full_genome_Evo2_natural_variant_percentile']):.1f}%.

## Resource-qualified substitution

The requested F5/G5/J5 Boltz-2 inference was attempted on one RTX 5090 (32 GB) and exceeded device memory during the first batch. The evidence table therefore reports an F1/G1/J1 asymmetric-unit complex. This retains local F–G and G–J interface analysis but is not equivalent to King's AlphaFold 3 five-fold-vertex prediction.

## Interpretation limits

ipTM, PAE, pTM, pLDDT and Evo2 likelihood are model-derived quantities. Interface
contacts and conservation are structural descriptors. RNA folding is limited to a
local sequence window. These signals can prioritize candidates but cannot establish
infectivity, assembly, host range or laboratory feasibility.

## Material passport

- Reference genome: NC_001422.1, 5,386 nt; G CDS 2,395–2,922 (1-based).
- Natural structure: ΦX174 cryo-EM PDB 36CQ.
- Structure model: Boltz-2 v2.2.1, bfloat16, 3 recycles, 200 sampling steps, one diffusion sample, single-sequence mode, no template.
- RNA model: ViennaRNA 2.7.2; 240-nt G-start window.
- Genome model: Microviridae-fine-tuned Evo2-7B-8k; identical `+~` prefix; 119 unique legal natural variants.
- Structure comparison: tmtools TM-align 0.3.0; 5 Å heavy-atom contact definition.

## Sources

- https://github.com/evo-design/evo/tree/main/phage_gen/analysis
- https://pubmed.ncbi.nlm.nih.gov/42561074/
- https://www.rcsb.org/structure/36CQ
- https://github.com/jwohlwend/boltz
"""
    (a.output_dir / "computational_evidence_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
