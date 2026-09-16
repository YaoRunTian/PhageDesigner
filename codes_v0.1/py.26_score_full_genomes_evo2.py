#!/usr/bin/env python3
"""Score designed genomes, PhiX174 and natural variants with fine-tuned Evo2.

Mean autoregressive log-likelihood per nucleotide is computed with the same
``+~`` prefix for every sequence.  Designed scores are contextualized by the
PhiX174 reference and the empirical distribution of natural PhiX174 variants.
No viability threshold or label is produced.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
PHAGE_ROOT = Path("/public8/lilab/student/rtyao/phage")
DEFAULT_GENOMES = PROJECT / "results/21_phix174_g_replacement_smoke/designed_phix174_genomes.fasta"
DEFAULT_GENBANK = PROJECT / "references/NC_001422.1.gb"
DEFAULT_VARIANTS = PHAGE_ROOT / "raw_data/microviridae/phage_sft_genomes_phix174_variants.fna"
DEFAULT_CHECKPOINT = PHAGE_ROOT / "results/02_microviridae_preprocess/evo2_7b_microviridae.pt"
DEFAULT_OUT = PROJECT / "results/26_full_genome_evo2_evidence"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--designed-genomes", type=Path, default=DEFAULT_GENOMES)
    p.add_argument("--genbank", type=Path, default=DEFAULT_GENBANK)
    p.add_argument("--natural-variants", type=Path, default=DEFAULT_VARIANTS)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gpu", default="1")
    p.add_argument("--natural-limit", type=int, default=0, help="0 means all legal natural variants")
    return p.parse_args()


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
        cfg.update(
            inference_mode=False,
            use_fp8_input_projections=False,
            use_flash_attn=False,
            use_flashfft=False,
            use_hcs_kernel=False,
            use_hcm_kernel=False,
            use_hcl_kernel=False,
        )
        from vortex.model.model import StripedHyena
        from vortex.model.tokenizer import CharLevelTokenizer
        from vortex.model.utils import dotdict, load_checkpoint

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

    def score(self, dna: str) -> float:
        torch = self.torch
        ids = torch.tensor([self.tokenizer.tokenize("+~" + dna)], device="cuda", dtype=torch.long)
        with torch.no_grad():
            logits, _ = self.model(ids[:, :-1])
            log_probs = torch.log_softmax(logits[:, 1:].float(), dim=-1)
            targets = ids[:, 2:]
            base_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        return float(base_logp.mean().item())


def main() -> int:
    a = parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    from Bio import SeqIO

    reference_record = SeqIO.read(a.genbank, "genbank")
    entries = [("phix174_reference", "reference", str(reference_record.seq).upper())]
    entries += [(r.id, "designed", str(r.seq).upper()) for r in SeqIO.parse(a.designed_genomes, "fasta")]
    natural = []
    rejected = []
    seen = set()
    for record in SeqIO.parse(a.natural_variants, "fasta"):
        seq = str(record.seq).upper()
        if set(seq) - set("ACGT"):
            rejected.append({"id": record.id, "reason": "non-ACGT"}); continue
        if seq in seen:
            rejected.append({"id": record.id, "reason": "exact_duplicate"}); continue
        seen.add(seq); natural.append((record.id, "natural_variant", seq))
    if a.natural_limit:
        natural = natural[:a.natural_limit]
    entries += natural
    if any(set(seq) - set("ACGT") for _, _, seq in entries):
        raise ValueError("Reference or designed sequences contain non-canonical nucleotides")
    if any(len(seq) + 2 > 8192 for _, _, seq in entries):
        raise ValueError("A sequence exceeds the Evo2-7B-8k context window")

    scorer = Evo2Scorer(a.checkpoint, a.gpu)
    rows = []
    partial = a.output_dir / "likelihood_scores.partial.tsv"
    for i, (sequence_id, kind, seq) in enumerate(entries, 1):
        score = scorer.score(seq)
        row = {
            "sequence_id": sequence_id,
            "kind": kind,
            "length_nt": len(seq),
            "mean_log_likelihood_per_nt": score,
            "perplexity": math.exp(-score),
        }
        rows.append(row)
        with partial.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
            writer.writeheader(); writer.writerows(rows)
        print(f"Evo2 full-genome likelihood {i}/{len(entries)} {kind} {sequence_id} {score:.6f}", flush=True)

    reference_score = next(r["mean_log_likelihood_per_nt"] for r in rows if r["kind"] == "reference")
    natural_scores = np.asarray([r["mean_log_likelihood_per_nt"] for r in rows if r["kind"] == "natural_variant"])
    if len(natural_scores) < 5:
        raise RuntimeError(f"Too few natural variants after filtering: {len(natural_scores)}")
    mean, std = float(natural_scores.mean()), float(natural_scores.std(ddof=1))
    for row in rows:
        value = row["mean_log_likelihood_per_nt"]
        row["delta_vs_phix174"] = value - reference_score
        row["natural_variant_zscore"] = (value - mean) / std if std else ""
        row["natural_variant_percentile"] = float(np.mean(natural_scores <= value))
    final_path = a.output_dir / "full_genome_likelihood_evidence.tsv"
    with final_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    partial.unlink(missing_ok=True)
    if rejected:
        with (a.output_dir / "rejected_natural_variants.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rejected[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(rejected)
    summary = {
        "status": "complete",
        "checkpoint": str(a.checkpoint),
        "prefix": "+~",
        "designed_count": sum(r["kind"] == "designed" for r in rows),
        "natural_variant_count": len(natural_scores),
        "rejected_natural_records": len(rejected),
        "reference_mean_log_likelihood_per_nt": reference_score,
        "natural_distribution": {
            "mean": mean, "sd": std,
            "min": float(natural_scores.min()), "max": float(natural_scores.max()),
        },
        "interpretation": "Higher mean log-likelihood indicates greater compatibility with the model distribution; it is not a viability probability.",
    }
    (a.output_dir / "full_genome_likelihood_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
