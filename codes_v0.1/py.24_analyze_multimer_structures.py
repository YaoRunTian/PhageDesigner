#!/usr/bin/env python3
"""Analyze Boltz G5 and F5/G5/J5 predictions against natural PDB 36CQ.

Outputs native Boltz confidence quantities (ipTM, pTM and PAE), chainwise
TM-align scores against the natural G pentamer, explicit heavy-atom interface
contacts, and conservation of G interface residues.  No viability label is
created.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_INPUTS = PROJECT / "results/23_multimer_structure_inputs"
DEFAULT_G5 = PROJECT / "results/23a_boltz_g5"
DEFAULT_FGJ = PROJECT / "results/23b_boltz_fgj15"
DEFAULT_FGJ3 = PROJECT / "results/23c_boltz_fgj3"
DEFAULT_CONSERVATION = PROJECT / "results/17_g_like_sequence_screen/conservation_profile.tsv"
DEFAULT_OUT = PROJECT / "results/24_multimer_structure_evidence"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUTS)
    p.add_argument("--g5-dir", type=Path, default=DEFAULT_G5)
    p.add_argument("--fgj-dir", type=Path, default=DEFAULT_FGJ)
    p.add_argument("--fgj3-dir", type=Path, default=DEFAULT_FGJ3)
    p.add_argument("--conservation", type=Path, default=DEFAULT_CONSERVATION)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--contact-cutoff", type=float, default=5.0)
    return p.parse_args()


def load_structure(path: Path):
    from Bio.PDB import MMCIFParser, PDBParser

    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def residues(chain):
    return [r for r in chain if r.id[0] == " " and "CA" in r]


def chain_type(chain) -> str:
    n = len(residues(chain))
    if 165 <= n <= 180:
        return "G"
    if 410 <= n <= 435:
        return "F"
    if 25 <= n <= 45:
        return "J"
    return "other"


def one_letter(residue) -> str:
    from Bio.PDB.Polypeptide import protein_letters_3to1

    return protein_letters_3to1.get(residue.resname.upper(), "X")


def tm_score(chain_a, chain_b) -> float:
    from tmtools import tm_align

    ra, rb = residues(chain_a), residues(chain_b)
    ca = np.asarray([r["CA"].coord for r in ra], dtype=float)
    cb = np.asarray([r["CA"].coord for r in rb], dtype=float)
    sa = "".join(one_letter(r) for r in ra)
    sb = "".join(one_letter(r) for r in rb)
    result = tm_align(ca, cb, sa, sb)
    return float((result.tm_norm_chain1 + result.tm_norm_chain2) / 2.0)


def matched_g_tm(predicted, natural) -> tuple[float, float, str]:
    pg = [c for c in next(predicted.get_models()) if chain_type(c) == "G"]
    ng = [c for c in next(natural.get_models()) if chain_type(c) == "G"]
    if len(pg) != 5 or len(ng) != 5:
        raise ValueError(f"TM-align requires five G chains; predicted={len(pg)}, natural={len(ng)}")
    scores = np.asarray([[tm_score(a, b) for b in ng] for a in pg])
    rows, cols = linear_sum_assignment(-scores)
    chosen = scores[rows, cols]
    mapping = ";".join(f"{pg[i].id}:{ng[j].id}" for i, j in zip(rows, cols))
    return float(chosen.mean()), float(chosen.min()), mapping


def contact_pairs(chain_a, chain_b, cutoff: float) -> set[tuple[int, int]]:
    ra, rb = residues(chain_a), residues(chain_b)
    aa, atom_to_ra = [], []
    for i, r in enumerate(ra, 1):
        for atom in r:
            if atom.element != "H":
                aa.append(atom.coord); atom_to_ra.append(i)
    ab, atom_to_rb = [], []
    for j, r in enumerate(rb, 1):
        for atom in r:
            if atom.element != "H":
                ab.append(atom.coord); atom_to_rb.append(j)
    if not aa or not ab:
        return set()
    tree = cKDTree(np.asarray(ab))
    pairs = set()
    for i, neighbors in enumerate(tree.query_ball_point(np.asarray(aa), cutoff)):
        for j in neighbors:
            pairs.add((atom_to_ra[i], atom_to_rb[j]))
    return pairs


def interface_metrics(structure, cutoff: float) -> dict:
    chains = list(next(structure.get_models()))
    typed = [(c, chain_type(c)) for c in chains]
    counts = {"GG": 0, "FG": 0, "GJ": 0, "FJ": 0}
    g_positions: dict[str, set[int]] = {c.id: set() for c, t in typed if t == "G"}
    for (ca, ta), (cb, tb) in combinations(typed, 2):
        key = "".join(sorted((ta, tb)))
        key = {"GG": "GG", "FG": "FG", "GJ": "GJ", "FJ": "FJ"}.get(key)
        if not key:
            continue
        pairs = contact_pairs(ca, cb, cutoff)
        counts[key] += len(pairs)
        if ta == "G":
            g_positions[ca.id].update(i for i, _ in pairs)
        if tb == "G":
            g_positions[cb.id].update(j for _, j in pairs)
    all_g_positions = set().union(*g_positions.values()) if g_positions else set()
    return {
        **{f"contact_pairs_{k}": v for k, v in counts.items()},
        "g_interface_position_count": len(all_g_positions),
        "g_interface_positions": ",".join(map(str, sorted(all_g_positions))),
    }


def conserved_positions(path: Path) -> tuple[set[int], dict[int, str]]:
    positions, consensus = set(), {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["conserved"].lower() in {"true", "1", "yes"}:
                pos = int(row["position_1based"])
                positions.add(pos); consensus[pos] = row["consensus_aa"]
    return positions, consensus


def candidate_sequence(path: Path, candidate_id: str) -> str:
    if candidate_id == "phix174_reference":
        # All natural G chains have the same complete 175-aa sequence.
        structure = load_structure(path)
        chain = next(c for c in next(structure.get_models()) if chain_type(c) == "G")
        return "".join(one_letter(r) for r in residues(chain))
    # Prediction sequence is read directly from a G chain, avoiding metadata drift.
    structure = load_structure(path)
    chain = next(c for c in next(structure.get_models()) if chain_type(c) == "G")
    return "".join(one_letter(r) for r in residues(chain))


def prediction_files(root: Path, candidate_id: str) -> tuple[Path, Path, Path | None]:
    predictions_roots = list(root.glob("**/predictions"))
    output_id = candidate_id.removeprefix("PHIX174_G_")
    candidates = [p / candidate_id for p in predictions_roots] + [p / output_id for p in predictions_roots]
    folder = next((p for p in candidates if p.exists()), root / "predictions" / output_id)
    structures = sorted(folder.glob("*_model_0.cif")) + sorted(folder.glob("*_model_0.pdb"))
    confidences = sorted(folder.glob("confidence_*_model_0.json"))
    paes = sorted(folder.glob("pae_*_model_0.npz"))
    if not structures or not confidences:
        raise FileNotFoundError(f"Incomplete Boltz output for {candidate_id} under {root}")
    return structures[0], confidences[0], paes[0] if paes else None


def mean_pae(path: Path | None, complex_type: str) -> dict:
    if path is None:
        return {"pae_all": "", "pae_interchain": "", "pae_FG": "", "pae_GJ": ""}
    data = np.load(path)
    key = "pae" if "pae" in data.files else data.files[0]
    pae = np.asarray(data[key], dtype=float)
    if complex_type == "g5":
        lengths = [175] * 5
    elif complex_type == "fgj15":
        lengths = [427] * 5 + [175] * 5 + [38] * 5
    elif complex_type == "fgj3":
        lengths = [427, 175, 38]
    else:
        raise ValueError(complex_type)
    if pae.shape[0] != sum(lengths):
        raise ValueError(f"PAE shape {pae.shape} disagrees with expected {sum(lengths)} tokens")
    bounds = np.cumsum([0] + lengths)
    if complex_type == "g5":
        types = ["G"] * 5
    elif complex_type == "fgj15":
        types = ["F"] * 5 + ["G"] * 5 + ["J"] * 5
    else:
        types = ["F", "G", "J"]
    inter, fg, gj = [], [], []
    for i in range(len(lengths)):
        for j in range(i + 1, len(lengths)):
            block = np.concatenate((pae[bounds[i]:bounds[i + 1], bounds[j]:bounds[j + 1]].ravel(),
                                    pae[bounds[j]:bounds[j + 1], bounds[i]:bounds[i + 1]].ravel()))
            inter.append(block)
            pair = {types[i], types[j]}
            if pair == {"F", "G"}: fg.append(block)
            if pair == {"G", "J"}: gj.append(block)
    avg = lambda xs: float(np.mean(np.concatenate(xs))) if xs else ""
    return {"pae_all": float(np.mean(pae)), "pae_interchain": avg(inter), "pae_FG": avg(fg), "pae_GJ": avg(gj)}


def analyze_one(candidate_id: str, complex_type: str, root: Path, natural, conserved: set[int], consensus: dict[int, str], cutoff: float) -> dict:
    structure_path, confidence_path, pae_path = prediction_files(root, candidate_id)
    predicted = load_structure(structure_path)
    confidence = json.loads(confidence_path.read_text())
    interfaces = interface_metrics(predicted, cutoff)
    sequence = candidate_sequence(structure_path, candidate_id)
    interface_pos = {int(x) for x in interfaces["g_interface_positions"].split(",") if x}
    conserved_interface = interface_pos & conserved
    retained = sum(sequence[p - 1] == consensus[p] for p in conserved_interface if p <= len(sequence))
    row = {
        "candidate_id": candidate_id,
        "complex_type": complex_type,
        "structure_file": str(structure_path),
        "confidence_score": confidence.get("confidence_score", ""),
        "ptm": confidence.get("ptm", ""),
        "iptm": confidence.get("iptm", ""),
        "protein_iptm": confidence.get("protein_iptm", ""),
        "complex_plddt": confidence.get("complex_plddt", ""),
        **mean_pae(pae_path, complex_type),
        **interfaces,
        "conserved_g_interface_positions": len(conserved_interface),
        "conserved_g_interface_retained": retained,
        "conserved_g_interface_retention": retained / len(conserved_interface) if conserved_interface else "",
    }
    if complex_type == "g5":
        mean_tm, min_tm, mapping = matched_g_tm(predicted, natural)
        row.update(tm_align_mean=mean_tm, tm_align_min=min_tm, tm_chain_mapping=mapping)
    else:
        row.update(tm_align_mean="", tm_align_min="", tm_chain_mapping="")
    return row


def main() -> int:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    natural_g5 = load_structure(a.input_dir / "natural_36CQ_G5.cif")
    natural_fgj = load_structure(PROJECT / "references/36CQ.cif")
    conserved, consensus = conserved_positions(a.conservation)
    with (a.input_dir / "prediction_manifest.tsv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    rows, errors = [], []
    manifest = [item for item in manifest if item["complex_type"] in {"g5", "fgj3"}]
    for item in manifest:
        candidate_id, complex_type = item["candidate_id"], item["complex_type"]
        try:
            natural = natural_g5 if complex_type == "g5" else natural_fgj
            root = a.g5_dir if complex_type == "g5" else a.fgj3_dir
            rows.append(analyze_one(candidate_id, complex_type, root, natural, conserved, consensus, a.contact_cutoff))
        except Exception as exc:
            errors.append({"candidate_id": candidate_id, "complex_type": complex_type, "error": f"{type(exc).__name__}: {exc}"})
    if rows:
        fields = list(rows[0])
        with (a.output_dir / "structure_evidence.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    if errors:
        fields = list(errors[0])
        with (a.output_dir / "structure_errors.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader(); writer.writerows(errors)
    else:
        (a.output_dir / "structure_errors.tsv").unlink(missing_ok=True)
    natural_metrics = {
        "G5": interface_metrics(natural_g5, a.contact_cutoff),
        "F1G1J1": interface_metrics(natural_fgj, a.contact_cutoff),
    }
    summary = {
        "status": "complete" if not errors else "partial",
        "rows": len(rows), "errors": len(errors),
        "natural_reference": "PDB 36CQ cryo-EM assembly 1",
        "contact_cutoff_angstrom": a.contact_cutoff,
        "natural_interface_metrics": natural_metrics,
        "notes": [
            "ipTM/pTM/PAE are native Boltz-2 confidence outputs, not experimental observables.",
            "TM-align is computed per G chain with optimal one-to-one pentamer-chain assignment.",
            "Heavy-atom contacts and conserved interface retention are descriptive evidence only.",
            "Full F5G5J5 Boltz-2 inference exceeded one RTX 5090 (32 GB); F1G1J1 is reported instead.",
        ],
    }
    (a.output_dir / "structure_evidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
