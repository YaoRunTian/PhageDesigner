#!/usr/bin/env python3
"""Audit template integrity and King-style nucleotide QC for designed genomes."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_FASTA = PROJECT / "results/21_phix174_g_replacement_smoke/designed_phix174_genomes.fasta"
DEFAULT_MANIFEST = PROJECT / "results/21_phix174_g_replacement_smoke/insertion_manifest.tsv"
DEFAULT_GENBANK = PROJECT / "references/NC_001422.1.gb"
DEFAULT_OUT = PROJECT / "results/22_designed_genome_screen_smoke"
G_START0, G_AA_END0, G_CDS_END0 = 2394, 2919, 2922


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--genbank", type=Path, default=DEFAULT_GENBANK)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-length", type=int, default=4000)
    p.add_argument("--max-length", type=int, default=6000)
    p.add_argument("--min-gc", type=float, default=30.0)
    p.add_argument("--max-gc", type=float, default=65.0)
    p.add_argument("--max-homopolymer", type=int, default=10)
    return p.parse_args()


def max_homopolymer(sequence: str) -> int:
    best = run = 0; previous = None
    for base in sequence:
        run = run + 1 if base == previous else 1
        previous = base; best = max(best, run)
    return best


def main() -> int:
    a = parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    from Bio import SeqIO
    from Bio.Seq import Seq
    reference = str(SeqIO.read(a.genbank, "genbank").seq).upper()
    records = {r.id: str(r.seq).upper() for r in SeqIO.parse(a.fasta, "fasta")}
    with a.manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {r["genome_id"]: r for r in csv.DictReader(handle, delimiter="\t")}
    results = []
    for genome_id, sequence in records.items():
        expected_protein = manifest[genome_id]["protein_sequence"]
        observed = str(Seq(sequence[G_START0:G_AA_END0]).translate())
        legal = not (set(sequence) - set("ACGT"))
        gc = 100.0 * (sequence.count("G") + sequence.count("C")) / len(sequence)
        homo = max_homopolymer(sequence)
        outside = sequence[:G_START0] == reference[:G_START0] and sequence[G_CDS_END0:] == reference[G_CDS_END0:]
        length_pass = a.min_length <= len(sequence) <= a.max_length
        gc_pass = a.min_gc <= gc <= a.max_gc
        homopolymer_pass = homo <= a.max_homopolymer
        translation_pass = observed == expected_protein and "*" not in observed
        terminal_stop_pass = sequence[G_AA_END0:G_CDS_END0] in {"TAA", "TAG", "TGA"}
        all_pass = all([legal, length_pass, gc_pass, homopolymer_pass, outside, translation_pass, terminal_stop_pass])
        results.append({
            "genome_id": genome_id, "candidate_id": manifest[genome_id]["candidate_id"],
            "length": len(sequence), "gc_percent": gc, "max_homopolymer": homo,
            "legal_dna_pass": legal, "length_pass": length_pass, "gc_pass": gc_pass,
            "homopolymer_pass": homopolymer_pass, "outside_g_unchanged": outside,
            "g_translation_match": translation_pass, "g_terminal_stop_pass": terminal_stop_pass,
            "all_smoke_checks_pass": all_pass,
        })
    fields = list(results[0]) if results else []
    with (a.output_dir / "genome_screening.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(results)
    passed_ids = {r["genome_id"] for r in results if r["all_smoke_checks_pass"]}
    with (a.output_dir / "smoke_pass_genomes.fasta").open("w", encoding="utf-8") as handle:
        for genome_id in records:
            if genome_id in passed_ids:
                handle.write(f">{genome_id}\n")
                sequence = records[genome_id]
                for i in range(0, len(sequence), 60): handle.write(sequence[i:i + 60] + "\n")
    summary = {
        "status": "passed" if results and len(passed_ids) == len(results) else "failed",
        "input_genomes": len(results), "all_smoke_checks_pass": len(passed_ids),
        "thresholds": {"length": [a.min_length, a.max_length], "gc_percent": [a.min_gc, a.max_gc], "max_homopolymer": a.max_homopolymer},
        "failed_checks": dict(Counter(k for r in results for k, v in r.items() if k.endswith("_pass") and v is False)),
        "scope": "Template/translation integrity and King-style nucleotide QC; protein-hit, tropism and diversification stages are reported separately.",
    }
    (a.output_dir / "genome_screen_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
