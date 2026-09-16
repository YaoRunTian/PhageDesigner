#!/usr/bin/env python3
"""Prepare PhiX174 G5 and F5/G5/J5 Boltz-2 structure-prediction inputs.

The script also extracts the experimentally resolved local five-fold vertex
from PDB 36CQ.  Boltz inputs deliberately use no structural template so that
agreement with 36CQ is an out-of-sample comparison rather than template
leakage.  Single-sequence mode is explicit (``msa: empty``) and therefore a
documented limitation relative to King's AlphaFold 3 web-server analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_CANDIDATES = PROJECT / "results/20_evo2_reverse_translation_smoke/selected_cds.tsv"
DEFAULT_GENBANK = PROJECT / "references/NC_001422.1.gb"
DEFAULT_ASSEMBLY = PROJECT / "references/36CQ-assembly1.cif"
DEFAULT_OUT = PROJECT / "results/23_multimer_structure_inputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    p.add_argument("--genbank", type=Path, default=DEFAULT_GENBANK)
    p.add_argument("--natural-assembly", type=Path, default=DEFAULT_ASSEMBLY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def protein_features(genbank: Path) -> dict[str, str]:
    from Bio import SeqIO

    record = SeqIO.read(genbank, "genbank")
    by_product = {}
    for feature in record.features:
        if feature.type != "CDS":
            continue
        product = feature.qualifiers.get("product", [""])[0].lower()
        aa = feature.qualifiers.get("translation", [""])[0].rstrip("*")
        if aa:
            by_product[product] = aa
    wanted = {
        "F": "major head protein",
        "G": "major spike protein",
        "J": "dna condensation",
    }
    missing = [label for label, product in wanted.items() if product not in by_product]
    if missing:
        raise RuntimeError(f"Missing reference proteins: {missing}")
    return {label: by_product[product] for label, product in wanted.items()}


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"No candidate proteins in {path}")
    if len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("Candidate IDs are not unique")
    return rows


def suffix(chain_id: str) -> str:
    return "" if "-" not in chain_id else chain_id[chain_id.index("-") :]


def ca_centroid(chain) -> np.ndarray:
    coords = [r["CA"].coord for r in chain if r.id[0] == " " and "CA" in r]
    if not coords:
        raise ValueError(f"No C-alpha coordinates for chain {chain.id}")
    return np.mean(np.asarray(coords), axis=0)


def extract_natural_vertex(assembly: Path, out_dir: Path) -> dict:
    from Bio.PDB import MMCIFIO, MMCIFParser, Select

    structure = MMCIFParser(QUIET=True).get_structure("36CQ", str(assembly))
    model = next(structure.get_models())
    # 36CQ entity lengths identify J/F/G as 31/426/175 resolved residues.
    groups: dict[str, dict[str, object]] = {}
    for chain in model:
        n = sum(1 for r in chain if r.id[0] == " ")
        label = {31: "J", 426: "F", 175: "G"}.get(n)
        if label:
            groups.setdefault(suffix(chain.id), {})[label] = chain
    complete = {k: v for k, v in groups.items() if set(v) == {"F", "G", "J"}}
    if len(complete) != 60:
        raise RuntimeError(f"Expected 60 complete 36CQ asymmetric units, found {len(complete)}")

    # Five G subunits at one spike are each other's nearest neighbors.  Anchor
    # on the deposited first chain to obtain one deterministic natural vertex.
    anchor_key = "" if "" in complete else sorted(complete)[0]
    anchor = ca_centroid(complete[anchor_key]["G"])
    ranked = sorted(complete, key=lambda k: float(np.linalg.norm(ca_centroid(complete[k]["G"]) - anchor)))
    selected_suffixes = ranked[:5]
    selected_g = {complete[k]["G"].id for k in selected_suffixes}
    selected_all = {complete[k][label].id for k in selected_suffixes for label in ("F", "G", "J")}

    class ChainSelect(Select):
        def __init__(self, ids):
            self.ids = set(ids)

        def accept_chain(self, chain):
            return chain.id in self.ids

    g5_path = out_dir / "natural_36CQ_G5.cif"
    fgj_path = out_dir / "natural_36CQ_F5G5J5.cif"
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(g5_path), ChainSelect(selected_g))
    io.save(str(fgj_path), ChainSelect(selected_all))
    return {
        "anchor_suffix": anchor_key,
        "selected_suffixes": selected_suffixes,
        "g_chain_ids": sorted(selected_g),
        "fgj_chain_ids": sorted(selected_all),
        "g5_pdb": str(g5_path),
        "fgj_pdb": str(fgj_path),
    }


def yaml_text(name: str, f: str, g: str, j: str, complex_type: str) -> str:
    if complex_type == "g5":
        sequences = [
            "  - protein:\n"
            "      id: [G1, G2, G3, G4, G5]\n"
            f"      sequence: {g}\n"
            "      msa: empty\n"
        ]
    elif complex_type == "fgj15":
        sequences = [
            "  - protein:\n"
            "      id: [F1, F2, F3, F4, F5]\n"
            f"      sequence: {f}\n"
            "      msa: empty\n",
            "  - protein:\n"
            "      id: [G1, G2, G3, G4, G5]\n"
            f"      sequence: {g}\n"
            "      msa: empty\n",
            "  - protein:\n"
            "      id: [J1, J2, J3, J4, J5]\n"
            f"      sequence: {j}\n"
            "      msa: empty\n",
        ]
    elif complex_type == "fgj3":
        sequences = [
            "  - protein:\n"
            "      id: F1\n"
            f"      sequence: {f}\n"
            "      msa: empty\n",
            "  - protein:\n"
            "      id: G1\n"
            f"      sequence: {g}\n"
            "      msa: empty\n",
            "  - protein:\n"
            "      id: J1\n"
            f"      sequence: {j}\n"
            "      msa: empty\n",
        ]
    else:
        raise ValueError(complex_type)
    return "version: 1\nsequences:\n" + "".join(sequences)


def main() -> int:
    a = parse_args()
    (a.output_dir / "g5_yaml").mkdir(parents=True, exist_ok=True)
    (a.output_dir / "fgj15_yaml").mkdir(parents=True, exist_ok=True)
    (a.output_dir / "fgj3_yaml").mkdir(parents=True, exist_ok=True)
    refs = protein_features(a.genbank)
    candidates = read_candidates(a.candidates)
    natural = extract_natural_vertex(a.natural_assembly, a.output_dir)

    manifest = []
    entries = [("phix174_reference", refs["G"], "reference")]
    entries += [(r["candidate_id"], r["protein_sequence"], "candidate") for r in candidates]
    for candidate_id, g_sequence, kind in entries:
        if len(g_sequence) != 175:
            raise ValueError(f"{candidate_id}: expected 175-aa G protein, got {len(g_sequence)}")
        safe_id = candidate_id.replace("/", "_")
        for complex_type, folder in (("g5", "g5_yaml"), ("fgj15", "fgj15_yaml"), ("fgj3", "fgj3_yaml")):
            path = a.output_dir / folder / f"{safe_id}.yaml"
            path.write_text(yaml_text(safe_id, refs["F"], g_sequence, refs["J"], complex_type), encoding="utf-8")
            manifest.append(
                {
                    "candidate_id": candidate_id,
                    "kind": kind,
                    "complex_type": complex_type,
                    "yaml": str(path),
                    "g_length": len(g_sequence),
                    "f_length": len(refs["F"]),
                    "j_length": len(refs["J"]),
                }
            )
    fields = list(manifest[0])
    with (a.output_dir / "prediction_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(manifest)
    summary = {
        "status": "prepared",
        "candidate_count": len(candidates),
        "prediction_jobs": len(manifest),
        "complexes": {
            "G5": "5 x 175 aa",
            "F5G5J5": "5 x (427 + 175 + 38 aa)",
            "F1G1J1": "427 + 175 + 38 aa fallback for 32-GB GPUs",
        },
        "natural_reference": natural,
        "prediction_model": "Boltz-2 (local open model used because AlphaFold 3 weights/server automation are unavailable)",
        "msa_mode": "single-sequence (msa: empty)",
        "template_used_for_prediction": False,
    }
    (a.output_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
