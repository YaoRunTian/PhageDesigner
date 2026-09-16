# PhageDesigner v0.1

PhageDesigner is a computational prototype for protein-conditioned phage genome design. Version 0.1 explores a complete research workflow using the PhiX174 major spike protein (G protein) as the demonstration case:

```text
protein-function / phage-template input
        ↓
reference retrieval and ESM-2 masked sampling
        ↓
G-like protein sequence screening and ranking
        ↓
Evo2-constrained synonymous reverse translation
        ↓
PhiX174 G-CDS replacement
        ↓
genome, structure, RNA, and Evo2-likelihood evidence
```

The repository is intended for method development and reproducibility. It does **not** establish infectivity, assembly, host range, or experimental viability of any generated sequence.

## What is included

- PhageScope v2 data auditing and organization utilities.
- GPD quality control and rule-based protein-function normalization.
- Leakage-controlled ESM-2 single-function classification data preparation.
- ESM-2 model-size comparison and fixed-split hyperparameter-tuning code.
- Microviridae data merging, annotation auditing, and cluster-level splitting.
- Frozen ESM-2 reference embeddings and an ESM-2 → Evo2 soft-prefix interface probe.
- PhiX174 G-like protein generation by controlled ESM-2 masked sampling.
- Sequence, homology, PHROG, conservation, and monomer-structure screening.
- Candidate ranking and diversity selection.
- Evo2 likelihood ranking of synonymous CDS candidates.
- Template CDS replacement and whole-genome quality control.
- Multimer-structure input preparation and evidence extraction.
- G-start RNA-structure analysis and full-genome Evo2 likelihood scoring.
- Publication-style figures and editable workflow artwork.

## Normalized protein-function labels

The classifier experiments use nine first-level labels:

`assembly`, `replication`, `infection`, `packaging`, `integration`, `regulation`, `lysis`, `immune`, and `tRNA_related`.

These labels are obtained by deterministic text rules over database `classification` and (when needed) `product` annotations. They are development labels for supervised learning, not manually curated functional ground truth.

## Repository layout

```text
phage_designer_v0.1/
├── codes_v0.1/       # numbered analysis and plotting scripts
├── figures/          # PNG/PDF figures and editable workflow artwork
├── results/          # reports and generated analysis artifacts
├── workflow/         # workflow diagrams (PPTX/PDF)
└── README.md
```

Large raw databases, model checkpoints, temporary caches, and server-specific environments are intentionally not bundled with this repository. The scripts accept explicit input/output/model paths so that the workflow can be reproduced on another machine.

## Script map

Scripts are numbered in execution order. Later scripts may be run independently when their input manifests already exist.

| Scripts | Purpose |
|---|---|
| `py.01`–`py.02` | Audit and organize PhageScope v2 databases. |
| `py.03`–`py.04` | GPD quality control and protein-function normalization. |
| `py.05`–`py.07` | Prepare ESM-2 classification data and fixed train/validation tuning. |
| `py.08` | Probe the frozen ESM-2 → Evo2 soft-prefix interface. |
| `py.09`–`py.11` | Merge Microviridae data, normalize annotations, and split by genome cluster. |
| `py.12`–`py.15` | Build ESM-2 embeddings, test the interface, prepare paired examples, and train the adapter. |
| `py.16`–`py.19` | Generate 3,000 G-like proteins, screen them, predict monomer structures, and select 100 candidates. |
| `py.20`–`py.22` | Rank synonymous CDSs with Evo2, replace the PhiX174 G CDS, and screen designed genomes. |
| `py.23`–`py.27` | Prepare/analyze multimer structures, assess G-start RNA structure, score whole genomes, and compile evidence. |
| `py.28`–`py.32` | Create workflow diagrams and supporting scientific figures. |

The JavaScript workflow builder (`js.30_build_editable_flowchart_ppt.mjs`) creates editable PowerPoint artwork. Plotting scripts save both vector PDF and raster PNG outputs where applicable.

## Demonstration configuration

The v0.1 G-like demonstration uses:

- Reference: PhiX174 (`NC_001422.1`) major spike protein G, 175 aa.
- Protein generator: `facebook/esm2_t33_650M_UR50D` masked language model.
- Masking groups: approximately 5%, 10%, 20%, and 30% of the reference positions.
- Total generated protein candidates: 3,000, with 750 candidates per masking group.
- Protein screening: sequence integrity, PhiX174 G homology, PHROG major-spike homology, conservation, and ESMFold monomer structure.
- Candidate selection: combined quality and diversity ranking, yielding 100 protein candidates.
- CDS generation: legal synonymous codons only; every proposed CDS is translated back and checked against its requested protein.
- DNA context model: Microviridae-fine-tuned Evo2-7B, used to rank synonymous CDS alternatives in PhiX174 context.
- Genome construction: replacement of the native G CDS while preserving the remaining PhiX174 template sequence.

## Reproducibility outline

The following commands illustrate the intended order. Paths are examples and should be changed for the local installation.

```bash
PROJECT=/path/to/phage_designer_v0.1
PYTHON=/path/to/conda/envs/phage/bin/python

# Data preparation
$PYTHON $PROJECT/codes_v0.1/py.01_phagescope_audit.py --mode full
$PYTHON $PROJECT/codes_v0.1/py.02_organize_phagescope_v2.py
$PYTHON $PROJECT/codes_v0.1/py.03_filter_gpd.py
$PYTHON $PROJECT/codes_v0.1/py.04_normalize_gpd_functions.py

# ESM-2 classification data and tuning
$PYTHON $PROJECT/codes_v0.1/py.06_prepare_dataset_for_esm2_finetune.py
$PYTHON $PROJECT/codes_v0.1/py.07_finetune_esm2_fixed_split.py --prepare-cache

# PhiX174 G-like generation and screening
$PYTHON $PROJECT/codes_v0.1/py.16_generate_g_like_esm2.py \
  --reference-fasta /path/to/phix174_reference_proteins.fasta \
  --model /path/to/esm2_t33_650M_UR50D \
  --output-dir $PROJECT/results/16_g_like_esm2_3000

$PYTHON $PROJECT/codes_v0.1/py.17_screen_g_like_sequences.py
$PYTHON $PROJECT/codes_v0.1/py.18_screen_g_like_structures.py
$PYTHON $PROJECT/codes_v0.1/py.19_rank_g_like_candidates.py

# CDS, genome, and evidence stages
$PYTHON $PROJECT/codes_v0.1/py.20_evo2_constrained_reverse_translation.py
$PYTHON $PROJECT/codes_v0.1/py.21_insert_g_into_phix174.py
$PYTHON $PROJECT/codes_v0.1/py.22_screen_designed_genomes.py
$PYTHON $PROJECT/codes_v0.1/py.23_prepare_multimer_structure_inputs.py
$PYTHON $PROJECT/codes_v0.1/py.24_analyze_multimer_structures.py
$PYTHON $PROJECT/codes_v0.1/py.25_analyze_g_start_mrna.py
$PYTHON $PROJECT/codes_v0.1/py.26_score_full_genomes_evo2.py
$PYTHON $PROJECT/codes_v0.1/py.27_compile_computational_evidence.py
```

Most scripts are safe to import and do not start computation automatically. Check each script's `--help` output before a long run. Use a new output directory for each reproducible run; several scripts intentionally refuse to overwrite existing results.

## Dependencies

The exact environment is machine-specific. In broad terms, the workflow uses:

- Python 3.10+.
- PyTorch and Hugging Face Transformers for ESM-2.
- BioPython for FASTA/GenBank handling.
- NumPy, SciPy, pandas, scikit-learn, and Matplotlib.
- ESMFold for monomer structure screening.
- MMseqs2 and PHROG resources for homology/function evidence.
- Evo2 and its Microviridae checkpoint for nucleotide likelihood scoring.
- Boltz-2 or a compatible structure-prediction environment for multimer analysis.
- ViennaRNA Python bindings for local RNA folding analysis.

External model weights and databases must be downloaded separately. For Hugging Face models, set a project-local cache, for example:

```bash
export HF_HOME=/path/to/phage_designer_v0.1/models/huggingface
```

Do not commit credentials, private server paths, model weights, raw databases, or generated sensitive data to a public repository.

## Main v0.1 results

The included reports summarize the development-stage experiments:

- `results/esm2_model_comparison.md`: frozen ESM-2 8M/650M/3B classification comparison on 100k proteins.
- `results/g_like_3000_screening_report.md`: 3,000 G-like protein generation and screening report.
- `results/g_like_to_phix174_smoke_report.md`: protein → synonymous CDS → PhiX174-template genome interface smoke test.
- `results/08_soft_prompt_smoke_report.md`: short ESM-2 → Evo2 soft-prefix mechanics test.
- `results/microviridae_pipeline_summary.md`: Microviridae data and adapter-pipeline status.

The PhiX174 G-like demonstration generated 3,000 proteins, retained 473 candidates after the sequence/conservation/monomer-structure screen, selected 100 candidates by quality-and-diversity ranking, and used Evo2 to rank synonymous CDS alternatives. These counts describe this particular development run and are not universal acceptance thresholds.

The v0.1 reports describe development baselines rather than independent final-test performance. In particular, the early ESM-2 model comparison used a protein-level split that was not genome-cluster separated; later fixed-split code was added to address this leakage risk.

## Interpretation boundaries

The computational evidence table is deliberately descriptive and does not contain a `viable` label. A candidate that passes sequence, structure, RNA, or Evo2-likelihood filters may still fail because of unmodeled interactions, expression constraints, genome packaging, host range, or experimental handling. Structure predictions and likelihood scores should therefore be interpreted as prioritization evidence only.

The function labels used for the ESM-2 classifier are rule-based normalizations of database `classification` and `product` text, not manually verified functional gold standards. Multi-function annotations and unknown annotations require separate evaluation and should not be silently treated as single-function truth.

## Data and reference resources

The demonstration depends on public or separately obtained resources, including PhageScope v2, GPD annotations, PHROG, PhiX174 RefSeq `NC_001422.1`, the 36CQ structural reference, ESM-2, ESMFold, and Evo2. Please follow each resource's license and citation requirements. Keep a machine-readable record of resource versions, checksums, and download dates for a reproducible study.

## Citation and license

This repository is a research prototype. Before public release, add:

1. the preferred citation for the associated manuscript or preprint;
2. citations for ESM-2, Evo2, ESMFold, PHROG, MMseqs2, Boltz-2, and source databases;
3. a repository license compatible with all redistributed code and assets.

Until then, treat the code and figures as research-use material and verify third-party terms before redistribution.
