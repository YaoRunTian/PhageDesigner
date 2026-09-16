#!/usr/bin/env python3
"""Generate PhiX174 G-like proteins by controlled ESM2 masked-LM sampling.

The exact G reference used by the test_king synteny analysis is selected from
``phix174_reference_proteins.fasta``.  Four equally sized groups are generated
by masking approximately 5%, 10%, 20%, or 30% of the 175-aa reference.  The
initial methionine is preserved and masked positions are forced to change so
that the requested perturbation rate is explicit and auditable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_REFERENCE = Path(
    "/public8/lilab/student/rtyao/phage/test_king/04b_synteny/"
    "phix174_reference_proteins.fasta"
)
DEFAULT_MODEL = PROJECT / "models/esm2/esm2_t33_650M_UR50D"
DEFAULT_OUT = PROJECT / "results/16_g_like_esm2_3000"
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-fasta", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--reference-id", default="NC_001422.1_ORF.3")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gpu", default="1")
    p.add_argument("--num-sequences", type=int, default=3000)
    p.add_argument("--mask-rates", default="0.05,0.10,0.20,0.30")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260908)
    return p.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    parts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header, parts = line[1:].split()[0], []
            elif line:
                parts.append(line.upper())
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def allocate_groups(total: int, rates: list[float]) -> list[int]:
    base, remainder = divmod(total, len(rates))
    return [base + (i < remainder) for i in range(len(rates))]


def main() -> int:
    a = parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {a.output_dir}")
    rates = [float(x) for x in a.mask_rates.split(",")]
    if not rates or any(not 0 < x < 1 for x in rates):
        raise ValueError("mask rates must be between 0 and 1")
    if a.num_sequences < len(rates):
        raise ValueError("num-sequences must be at least the number of mask rates")
    a.output_dir.mkdir(parents=True)

    refs = dict(read_fasta(a.reference_fasta))
    if a.reference_id not in refs:
        raise KeyError(f"Reference ID not found: {a.reference_id}")
    reference = refs[a.reference_id].rstrip("*")
    if len(reference) != 175 or reference[0] != "M" or set(reference) - set(CANONICAL_AA):
        raise ValueError("Expected a complete, canonical 175-aa PhiX174 G protein")

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        a.model, local_files_only=True, dtype=torch.float16
    ).cuda().eval()
    model.requires_grad_(False)

    encoded = tokenizer(reference, return_tensors="pt")
    base_ids = encoded["input_ids"].cuda()
    attention = encoded["attention_mask"].cuda()
    if base_ids.shape[1] != len(reference) + 2:
        raise RuntimeError("Unexpected ESM2 tokenization of the reference")
    aa_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(x) for x in CANONICAL_AA],
        device="cuda",
        dtype=torch.long,
    )
    if len(set(aa_ids.tolist())) != len(CANONICAL_AA):
        raise RuntimeError("Canonical amino-acid token mapping is not one-to-one")

    plan: list[dict] = []
    counts = allocate_groups(a.num_sequences, rates)
    global_index = 0
    for rate, count in zip(rates, counts):
        n_mask = max(1, int(math.floor(rate * len(reference) + 0.5)))
        for within_group in range(count):
            rng = random.Random(a.seed + global_index * 1009)
            # Preserve the initiator methionine at position 1.
            positions = sorted(rng.sample(range(1, len(reference)), n_mask))
            plan.append(
                {
                    "index": global_index,
                    "within_group": within_group,
                    "mask_rate": rate,
                    "positions": positions,
                }
            )
            global_index += 1

    generated: list[dict] = []
    seen: set[str] = set()
    with torch.no_grad():
        for start in range(0, len(plan), a.batch_size):
            batch_plan = plan[start : start + a.batch_size]
            ids = base_ids.repeat(len(batch_plan), 1)
            masks = attention.repeat(len(batch_plan), 1)
            for b, item in enumerate(batch_plan):
                token_positions = torch.tensor(
                    [p + 1 for p in item["positions"]], device="cuda"
                )
                ids[b, token_positions] = tokenizer.mask_token_id
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=masks).logits.float()

            for b, item in enumerate(batch_plan):
                chars = list(reference)
                logps: list[float] = []
                generator = torch.Generator(device="cuda").manual_seed(
                    a.seed + item["index"] * 7919
                )
                for pos in item["positions"]:
                    values = logits[b, pos + 1, aa_ids] / max(a.temperature, 1e-6)
                    original_idx = CANONICAL_AA.index(reference[pos])
                    values[original_idx] = -torch.inf
                    k = min(a.top_k, len(CANONICAL_AA) - 1)
                    top_values, top_order = torch.topk(values, k=k)
                    probs = torch.softmax(top_values, dim=-1)
                    sampled_local = int(
                        torch.multinomial(probs, 1, generator=generator).item()
                    )
                    aa_order = int(top_order[sampled_local].item())
                    chars[pos] = CANONICAL_AA[aa_order]
                    logps.append(float(torch.log(probs[sampled_local]).item()))
                sequence = "".join(chars)
                if sequence in seen:
                    raise RuntimeError(
                        f"Duplicate generated sequence at index {item['index']}; rerun with a different seed"
                    )
                seen.add(sequence)
                mutation_count = sum(x != y for x, y in zip(sequence, reference))
                rate_tag = int(round(item["mask_rate"] * 100))
                generated.append(
                    {
                        "candidate_id": f"GLIKE_M{rate_tag:02d}_{item['within_group']:04d}",
                        "mask_rate": item["mask_rate"],
                        "masked_count": len(item["positions"]),
                        "masked_positions_1based": ",".join(
                            str(p + 1) for p in item["positions"]
                        ),
                        "mutation_count": mutation_count,
                        "identity_to_reference": 1.0 - mutation_count / len(reference),
                        "mean_sample_log_probability": sum(logps) / len(logps),
                        "sequence": sequence,
                    }
                )
            print(f"generated {min(start + a.batch_size, len(plan))}/{len(plan)}", flush=True)

    fasta = a.output_dir / "g_like_3000.fasta"
    metadata = a.output_dir / "generation_metadata.tsv"
    with fasta.open("w", encoding="utf-8") as handle:
        for row in generated:
            handle.write(
                f">{row['candidate_id']}|mask_rate={row['mask_rate']:.2f}|mutations={row['mutation_count']}\n"
                f"{row['sequence']}\n"
            )
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(generated[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(generated)

    summary = {
        "status": "passed",
        "reference_source": str(a.reference_fasta),
        "reference_id": a.reference_id,
        "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        "reference_length_aa": len(reference),
        "model": str(a.model),
        "sampling": {
            "total": len(generated),
            "mask_rates": rates,
            "counts": dict(Counter(f"{x['mask_rate']:.2f}" for x in generated)),
            "force_change_at_masked_sites": True,
            "preserve_initial_methionine": True,
            "temperature": a.temperature,
            "top_k": a.top_k,
            "seed": a.seed,
        },
        "unique_sequences": len(seen),
        "outputs": {"fasta": str(fasta), "metadata": str(metadata)},
    }
    (a.output_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
