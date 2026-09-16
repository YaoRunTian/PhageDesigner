#!/usr/bin/env python3
"""Reverse-translate selected G-like proteins and rank legal CDSs with Evo2.

All proposed CDSs are hard constrained to translate exactly to the requested
protein.  Evo2 does not invent amino acids here: it ranks synonymous/codon
choices by autoregressive likelihood in the PhiX174 genomic context.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
PHAGE_ROOT = Path("/public8/lilab/student/rtyao/phage")
DEFAULT_INPUT = PROJECT / "results/19_g_like_candidate_ranking/selected_candidates.tsv"
DEFAULT_GENBANK = PROJECT / "references/NC_001422.1.gb"
DEFAULT_CHECKPOINT = PHAGE_ROOT / "results/02_microviridae_preprocess/evo2_7b_microviridae.pt"
DEFAULT_OUT = PROJECT / "results/20_evo2_reverse_translation_smoke"
G_START0, G_AA_END0, G_CDS_END0 = 2394, 2919, 2922


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--genbank", type=Path, default=DEFAULT_GENBANK)
    p.add_argument("--evo-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gpu", default="1")
    p.add_argument("--proteins", type=int, default=10)
    p.add_argument("--dna-candidates-per-protein", type=int, default=10)
    p.add_argument("--left-context", type=int, default=1024)
    p.add_argument("--right-context", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--skip-evo2", action="store_true", help="Debug only: rank by codon-use likelihood")
    return p.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def stratified_proteins(rows: list[dict[str, str]], n: int) -> list[dict[str, str]]:
    """Take the best available rows while representing every mask-rate group."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[f"{float(row['mask_rate']):.2f}"].append(row)
    rates = sorted(grouped, key=float)
    quotas = {rate: min(n // len(rates), len(grouped[rate])) for rate in rates}
    remaining = n - sum(quotas.values())
    while remaining:
        changed = False
        for rate in rates:
            if quotas[rate] < len(grouped[rate]):
                quotas[rate] += 1; remaining -= 1; changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    chosen = []
    for rate in rates:
        chosen.extend(grouped[rate][:quotas[rate]])
    return chosen


def codon_model(record) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    from Bio.Data import CodonTable
    table = CodonTable.unambiguous_dna_by_id[11]
    synonymous: dict[str, list[str]] = defaultdict(list)
    for codon, aa in table.forward_table.items():
        synonymous[aa].append(codon)
    counts: dict[str, Counter] = defaultdict(Counter)
    for feature in record.features:
        if feature.type != "CDS":
            continue
        dna = str(feature.extract(record.seq)).upper()
        for i in range(0, len(dna) - 2, 3):
            codon = dna[i:i + 3]
            aa = table.forward_table.get(codon)
            if aa:
                counts[aa][codon] += 1
    probabilities = {}
    for aa, codons in synonymous.items():
        denom = sum(counts[aa][c] + 0.5 for c in codons)
        probabilities[aa] = {c: (counts[aa][c] + 0.5) / denom for c in codons}
    return dict(synonymous), probabilities


def sample_cds(protein: str, reference_protein: str, native_codons: list[str], synonymous, probabilities, rng, deterministic=False) -> tuple[str, float]:
    codons, logp = [], 0.0
    for i, aa in enumerate(protein):
        if i < len(reference_protein) and aa == reference_protein[i]:
            codon = native_codons[i]
        else:
            choices = synonymous[aa]
            weights = [probabilities[aa][c] for c in choices]
            codon = max(choices, key=lambda c: probabilities[aa][c]) if deterministic else rng.choices(choices, weights=weights, k=1)[0]
        codons.append(codon)
        logp += math.log(probabilities[aa][codon])
    return "".join(codons), logp / len(protein)


class Evo2Scorer:
    def __init__(self, checkpoint: Path, gpu: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        import torch
        import yaml
        from torch import nn
        spec = importlib.util.find_spec("evo2")
        if spec is None:
            raise RuntimeError("evo2 package is unavailable")
        config_path = Path(spec.origin).parent / "configs/evo2-7b-8k.yml"
        cfg = yaml.safe_load(config_path.read_text())
        cfg.update(inference_mode=False, use_fp8_input_projections=False, use_flash_attn=False,
                   use_flashfft=False, use_hcs_kernel=False, use_hcm_kernel=False, use_hcl_kernel=False)
        from vortex.model.model import StripedHyena
        from vortex.model.utils import dotdict, load_checkpoint
        from vortex.model.tokenizer import CharLevelTokenizer
        self.torch = torch
        self.model = StripedHyena(dotdict(cfg))
        load_checkpoint(self.model, str(checkpoint))
        with torch.inference_mode(False), torch.no_grad():
            for module in self.model.modules():
                for name, value in list(module._parameters.items()):
                    if value is not None and torch.is_inference(value):
                        module._parameters[name] = nn.Parameter(value.clone(), requires_grad=False)
                for name, value in list(module._buffers.items()):
                    if value is not None and torch.is_inference(value):
                        module._buffers[name] = value.clone()
        self.model.requires_grad_(False); self.model.eval()
        self.tokenizer = CharLevelTokenizer(512)

    def score(self, dna: str, score_start: int, score_end: int) -> float:
        torch = self.torch
        ids = torch.tensor([self.tokenizer.tokenize("+~" + dna)], device="cuda", dtype=torch.long)
        with torch.no_grad():
            logits, _ = self.model(ids[:, :-1])
            log_probs = torch.log_softmax(logits[:, 1:].float(), dim=-1)
            targets = ids[:, 2:]
            base_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        return float(base_logp[score_start:score_end].mean().item())


def main() -> int:
    a = parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    from Bio import SeqIO
    from Bio.Seq import Seq
    record = SeqIO.read(a.genbank, "genbank")
    genome = str(record.seq).upper()
    original_cds = genome[G_START0:G_AA_END0]
    stop = genome[G_AA_END0:G_CDS_END0]
    reference_protein = str(Seq(original_cds).translate())
    native_codons = [original_cds[i:i + 3] for i in range(0, len(original_cds), 3)]
    synonymous, probabilities = codon_model(record)
    proteins = stratified_proteins(read_tsv(a.input), a.proteins)
    if not proteins:
        raise RuntimeError("No selected proteins")
    if any(len(row["sequence"]) != 175 for row in proteins):
        raise ValueError("Every selected protein must be 175 aa")

    rng = random.Random(a.seed)
    proposals = []
    for row in proteins:
        seen = set()
        attempts = 0
        while len(seen) < a.dna_candidates_per_protein and attempts < a.dna_candidates_per_protein * 200:
            cds, codon_score = sample_cds(row["sequence"], reference_protein, native_codons, synonymous,
                                           probabilities, rng, deterministic=(attempts == 0))
            attempts += 1
            if cds in seen:
                continue
            if str(Seq(cds).translate()) != row["sequence"]:
                raise AssertionError("Hard translation constraint failed")
            seen.add(cds)
            proposals.append({
                "candidate_id": row["candidate_id"], "mask_rate": row["mask_rate"],
                "protein_sequence": row["sequence"], "dna_variant": len(seen),
                "cds_sequence": cds, "stop_codon": stop, "codon_log_likelihood": codon_score,
            })
        if len(seen) < a.dna_candidates_per_protein:
            raise RuntimeError(f"Could create only {len(seen)} unique CDSs for {row['candidate_id']}")

    window_start = max(0, G_START0 - a.left_context)
    window_end = min(len(genome), G_CDS_END0 + a.right_context)
    scorer = None if a.skip_evo2 else Evo2Scorer(a.evo_checkpoint, a.gpu)
    for i, row in enumerate(proposals, 1):
        designed = genome[:G_START0] + row["cds_sequence"] + stop + genome[G_CDS_END0:]
        window = designed[window_start:window_end]
        score_start = G_START0 - window_start
        score_end = G_CDS_END0 - window_start + a.right_context
        row["evo2_log_likelihood"] = "" if scorer is None else scorer.score(window, score_start, min(score_end, len(window)))
        print(f"Evo2 scoring {i}/{len(proposals)}", flush=True)

    for row in proposals:
        row["ranking_score"] = float(row["evo2_log_likelihood"]) if row["evo2_log_likelihood"] != "" else row["codon_log_likelihood"]
        row["selected_best"] = False
    by_protein = defaultdict(list)
    for row in proposals:
        by_protein[row["candidate_id"]].append(row)
    selected = []
    for candidate_id, rows in by_protein.items():
        best = max(rows, key=lambda x: (x["ranking_score"], x["codon_log_likelihood"]))
        best["selected_best"] = True; selected.append(best)

    fields = ["candidate_id", "mask_rate", "dna_variant", "codon_log_likelihood", "evo2_log_likelihood",
              "ranking_score", "selected_best", "protein_sequence", "cds_sequence", "stop_codon"]
    with (a.output_dir / "reverse_translation_candidates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(proposals)
    with (a.output_dir / "selected_cds.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(selected)
    with (a.output_dir / "selected_cds.fasta").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(f">{row['candidate_id']} evo2_logp={row['evo2_log_likelihood']}\n{row['cds_sequence']}{row['stop_codon']}\n")
    summary = {
        "status": "passed", "proteins": len(proteins), "legal_cds_candidates": len(proposals),
        "proteins_by_mask_rate": dict(Counter(f"{float(r['mask_rate']):.2f}" for r in proteins)),
        "selected_cds": len(selected), "translation_constraint_pass": len(proposals),
        "evo2_scoring_used": scorer is not None, "evo2_checkpoint": str(a.evo_checkpoint),
        "reference": {"accession": record.id, "genome_length": len(genome), "g_cds_1based": "2395..2922", "stop_codon": stop},
        "method": "Generate only codons encoding the requested protein; preserve native codons at unchanged residues; rank variants by Evo2 log-likelihood over G CDS plus downstream context.",
    }
    (a.output_dir / "reverse_translation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
