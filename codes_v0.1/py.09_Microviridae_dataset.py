#!/usr/bin/env python3
"""Merge PhageScope v2 Microviridae records and perform reproducible QC.

The source tarballs contain one combined FASTA per database.  This script uses
the source metadata as the taxonomy filter, then verifies genome DNA, protein
AA, CDS coordinates and translated protein/CDS agreement.  Exact duplicate
genomes are retained once, with all source databases recorded in the manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
SOURCE = PROJECT / "data/phagescope_v2/source"
DEFAULT_OUT = PROJECT / "results/09_microviridae_dataset"
DATASETS = ("genbank", "refseq", "ddbj", "embl", "phagesdb", "gvd", "gpd",
            "mgv", "temphd", "chvd", "igvd", "img_vr", "gov2", "stv", "gsv",
            "uhgv", "hpgc", "elgv", "ovd", "opd", "svd", "vmgc", "urpc", "baps",
            "tymeflies_viral", "metavr")
DNA_RE = re.compile(r"^[ACGTN]+$")
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY*]+$")
CODONS = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","TCT":"S","TCC":"S","TCA":"S","TCG":"S",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","TGT":"C","TGC":"C","TGA":"*","TGG":"W",
    "CTT":"L","CTC":"L","CTA":"L","CTG":"L","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K","AGT":"S","AGC":"S","AGA":"R","AGG":"R",
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-genome-length", type=int, default=1000)
    p.add_argument("--min-protein-length", type=int, default=20)
    p.add_argument("--max-n-fraction", type=float, default=0.05)
    p.add_argument("--min-translation-identity", type=float, default=0.95)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--max-genomes", type=int, default=0)
    return p.parse_args()


def clean(value):
    return "".join(str(value or "").split()).upper()


def read_tsv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace", newline="") as h:
        return list(csv.DictReader(h, delimiter="\t"))


def fasta_records(raw):
    name, parts = None, []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                yield name.split()[0], clean("".join(parts))
            name, parts = line[1:], []
        else:
            parts.append(line)
    if name is not None:
        yield name.split()[0], clean("".join(parts))


def read_tar_fasta(path, logger, wanted=None, protein=False, protein_ids=None):
    result = {}
    if not path.exists():
        logger.warning("missing FASTA tarball: %s", path)
        return result
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith((".fa", ".faa", ".fna", ".fasta")):
                continue
            stream = tar.extractfile(member)
            if stream:
                for name, seq in fasta_records(stream.read()):
                    if wanted is not None:
                        if protein:
                            if protein_ids is not None:
                                if name not in protein_ids:
                                    continue
                            elif not any(name == gid or name.startswith(gid + "_") for gid in wanted):
                                continue
                        elif name not in wanted:
                            continue
                    result[name] = seq
    return result


def translate(dna):
    return "".join(CODONS.get(dna[i:i+3], "X") for i in range(0, len(dna) - 2, 3))


def revcomp(seq):
    return seq.translate(str.maketrans("ATCGN", "TAGCN"))[::-1]


def extract_cds(genome, start, stop, strand):
    if not (start > 0 and stop >= start and stop <= len(genome)):
        return ""
    seq = genome[start - 1:stop]
    return revcomp(seq) if strand == "-" else seq


def translation_identity(aa, cds):
    if not cds or len(cds) % 3:
        return 0.0
    observed, expected = translate(cds).rstrip("*"), aa.rstrip("*")
    if not observed or not expected:
        return 0.0
    n = min(len(observed), len(expected))
    return sum(a == b for a, b in zip(observed[:n], expected[:n])) / max(len(observed), len(expected))


def metadata(rows, source):
    out = {}
    for row in rows:
        gid = (row.get("Phage_ID") or row.get("phage_id") or "").strip()
        tax = (row.get("Taxonomy") or row.get("taxonomy") or "").strip()
        if gid and tax.casefold() == "microviridae":
            out[gid] = {"phage_id": gid, "source": source, "taxonomy": tax,
                "length_metadata": row.get("Length", ""), "gc_metadata": row.get("GC_content", ""),
                "completeness": row.get("Completeness", ""), "host": row.get("Host", ""),
                "lifestyle": row.get("Lifestyle", ""), "cluster": row.get("Cluster", ""),
                "subcluster": row.get("Subcluster", "")}
    return out


def protein_annotations(rows, selected):
    out = defaultdict(dict)
    for row in rows:
        gid = (row.get("Phage_ID") or row.get("phage_id") or "").strip()
        pid = (row.get("Protein_ID") or row.get("protein_id") or "").strip()
        if gid not in selected or not pid:
            continue
        classification = row.get("Protein_classification") or row.get("Classification") or ""
        out[gid][pid] = {"start": int(float(row.get("Start") or 0)), "stop": int(float(row.get("Stop") or 0)),
            "strand": row.get("Strand") if row.get("Strand") in ("+", "-") else "+",
            "product": row.get("Product") or "unknown", "classification": classification}
    return out


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main():
    a = args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("py09"); started = time.time(); counts = Counter()
    candidates = {}
    proteins = {}
    for source in a.datasets:
        source = source.lower();
        if source not in DATASETS:
            raise ValueError(f"unsupported dataset: {source}")
        meta = metadata(read_tsv(a.source / "metadata" / f"{source}_phage_meta_data.tsv"), source)
        counts[f"{source}_taxonomy_rows"] = len(meta)
        if not meta:
            log.info("%s: no Microviridae metadata; skipping FASTA tarballs", source)
            continue
        selected_ids = set(meta)
        ann = protein_annotations(read_tsv(a.source / "metadata" / f"{source}_phage_annotated_protein_meta_data.tsv"), meta)
        wanted_protein_ids = {pid for rows in ann.values() for pid in rows}
        genome_fa = read_tar_fasta(a.source / "genome_fasta" / f"{source}.tar.gz", log, selected_ids)
        protein_fa = read_tar_fasta(a.source / "protein_fasta" / f"{source}.tar.gz", log, selected_ids, protein=True, protein_ids=wanted_protein_ids)
        for gid, m in meta.items():
            if gid not in genome_fa:
                counts["missing_genome_fasta"] += 1; continue
            candidates.setdefault(gid, {**m, "genome": genome_fa[gid], "sources": []})["sources"].append(source)
            for pid, item in ann.get(gid, {}).items():
                if pid in protein_fa:
                    proteins.setdefault((gid, pid), {**item, "protein_id": pid, "aa": protein_fa[pid], "source": source})
        log.info("%s: %d Microviridae metadata, %d genomes, %d proteins", source, len(meta), len(genome_fa), len(protein_fa))
    genome_rows, protein_rows, retained = [], [], {}
    proteins_by_gid = defaultdict(list)
    for (pgid, _pid), protein in proteins.items():
        proteins_by_gid[pgid].append(protein)
    seen_genomes = {}
    for gid, item in candidates.items():
        genome = clean(item["genome"]); reason = ""
        if len(genome) < a.min_genome_length: reason = "short_genome"
        elif not DNA_RE.fullmatch(genome): reason = "illegal_dna"
        elif genome.count("N") / len(genome) > a.max_n_fraction: reason = "excessive_N"
        digest = hashlib.sha256(genome.encode()).hexdigest()
        if not reason and digest in seen_genomes: reason = "duplicate_genome"
        valid = []
        if not reason:
            for p in proteins_by_gid.get(gid, []):
                aa = clean(p["aa"]); cds = extract_cds(genome, p["start"], p["stop"], p["strand"])
                identity = translation_identity(aa, cds)
                if len(aa) >= a.min_protein_length and AA_RE.fullmatch(aa) and DNA_RE.fullmatch(cds) and identity >= a.min_translation_identity:
                    valid.append((p.get("protein_id", ""), p, aa, cds, identity))
            if not valid: reason = "no_valid_protein_cds_pair"
        if reason:
            counts[f"filtered_{reason}"] += 1; continue
        seen_genomes[digest] = gid; retained[gid] = (item, genome, digest, valid)
    for gid, (item, genome, digest, valid) in retained.items():
        counts["retained_genomes"] += 1; counts["retained_proteins"] += len(valid)
        genome_rows.append({"phage_id": gid, "sources": ";".join(sorted(set(item["sources"]))), "taxonomy": item["taxonomy"], "genome_length_bp": len(genome), "gc_fraction": round((genome.count("G")+genome.count("C"))/len(genome), 6), "n_fraction": round(genome.count("N")/len(genome), 6), "sequence_sha256": digest, "genome_cluster": item["cluster"], "completeness": item["completeness"], "host": item["host"], "genome_sequence": genome, "valid_protein_count": len(valid)})
        for pid, p, aa, cds, identity in valid:
            protein_rows.append({"phage_id": gid, "protein_id": pid, "sources": p["source"], "protein_sequence": aa, "cds_sequence": cds, "protein_length_aa": len(aa), "cds_length_nt": len(cds), "start": p["start"], "stop": p["stop"], "strand": p["strand"], "product": p["product"], "classification": p["classification"], "translation_identity": round(identity, 6)})
    genome_fields = list(genome_rows[0]) if genome_rows else ["phage_id"]
    protein_fields = list(protein_rows[0]) if protein_rows else ["phage_id"]
    write_tsv(a.output_dir / "microviridae_genomes.tsv", genome_rows, genome_fields)
    write_tsv(a.output_dir / "microviridae_proteins.tsv", protein_rows, protein_fields)
    summary = {"dataset": "Microviridae", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "parameters": {"datasets": a.datasets, "min_genome_length": a.min_genome_length, "min_protein_length": a.min_protein_length, "max_n_fraction": a.max_n_fraction, "min_translation_identity": a.min_translation_identity}, "counts": dict(counts), "outputs": {"genomes": str(a.output_dir/"microviridae_genomes.tsv"), "proteins": str(a.output_dir/"microviridae_proteins.tsv")}, "elapsed_seconds": round(time.time()-started, 2)}
    (a.output_dir / "microviridae_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.output_dir / "microviridae_qc_report.md").write_text("# Microviridae merged dataset\n\n" + "\n".join(f"- **{k}**: {v:,}" for k,v in counts.items()) + "\n\nGenome DNA and protein/CDS pairs were filtered before exact genome deduplication. Source database membership is retained in the manifests.\n", encoding="utf-8")
    log.info("retained %d genomes and %d protein/CDS pairs", len(genome_rows), len(protein_rows))


if __name__ == "__main__":
    main()
