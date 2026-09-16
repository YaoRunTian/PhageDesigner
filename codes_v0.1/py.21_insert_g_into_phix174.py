#!/usr/bin/env python3
"""Insert Evo2-ranked G-like CDSs into the NC_001422.1 PhiX174 template."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_INPUT = PROJECT / "results/20_evo2_reverse_translation_smoke/selected_cds.tsv"
DEFAULT_GENBANK = PROJECT / "references/NC_001422.1.gb"
DEFAULT_OUT = PROJECT / "results/21_phix174_g_replacement_smoke"
G_START0, G_AA_END0, G_CDS_END0 = 2394, 2919, 2922


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--genbank", type=Path, default=DEFAULT_GENBANK)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    from Bio import SeqIO
    from Bio.Seq import Seq
    record = SeqIO.read(a.genbank, "genbank")
    reference = str(record.seq).upper()
    with a.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    outputs = []
    with (a.output_dir / "designed_phix174_genomes.fasta").open("w", encoding="utf-8") as fasta:
        for row in rows:
            cds = row["cds_sequence"].upper()
            protein = row["protein_sequence"].upper()
            stop = row["stop_codon"].upper()
            designed = reference[:G_START0] + cds + stop + reference[G_CDS_END0:]
            translation = str(Seq(designed[G_START0:G_AA_END0]).translate())
            outside_unchanged = designed[:G_START0] == reference[:G_START0] and designed[G_CDS_END0:] == reference[G_CDS_END0:]
            passed = len(designed) == len(reference) and translation == protein and stop in {"TAA", "TAG", "TGA"} and outside_unchanged
            genome_id = f"PHIX174_G_{row['candidate_id']}"
            # Keep the FASTA identifier free of descriptions because the
            # reproduced King QC maps ORF IDs back to the complete header.
            # All provenance remains in insertion_manifest.tsv.
            fasta.write(f">{genome_id}\n")
            for i in range(0, len(designed), 60):
                fasta.write(designed[i:i + 60] + "\n")
            outputs.append({
                "genome_id": genome_id, "candidate_id": row["candidate_id"], "mask_rate": row["mask_rate"],
                "genome_length": len(designed), "g_cds_start_1based": G_START0 + 1,
                "g_cds_end_1based": G_CDS_END0, "g_translation_match": translation == protein,
                "outside_g_unchanged": outside_unchanged, "template_insertion_pass": passed,
                "evo2_log_likelihood": row["evo2_log_likelihood"], "genome_sha256": hashlib.sha256(designed.encode()).hexdigest(),
                "protein_sequence": protein, "cds_sequence": cds + stop,
            })
    fields = list(outputs[0]) if outputs else []
    with (a.output_dir / "insertion_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(outputs)
    summary = {
        "status": "passed" if outputs and all(r["template_insertion_pass"] for r in outputs) else "failed",
        "input_cds": len(rows), "output_genomes": len(outputs),
        "template_insertion_pass": sum(bool(r["template_insertion_pass"]) for r in outputs),
        "reference_accession": record.id, "reference_length": len(reference),
        "g_cds_1based_inclusive": "2395..2922", "outside_g_cds_preserved": True,
    }
    (a.output_dir / "insertion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
