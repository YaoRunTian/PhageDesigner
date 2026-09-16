#!/usr/bin/env python3
"""Tune frozen ESM2-650M on leakage-controlled fixed GPD splits.

Safety defaults:
* No training starts unless --grid-id or --run-all is supplied.
* Only train_100k.tsv and validation.tsv are accepted; final_test is rejected.
* A fixed 30k validation tuning subset is cached and reused by all 18 runs.
* Model selection uses validation macro-F1 with epoch evaluation and early stopping.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import pickle
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_SPLIT_DIR = PROJECT / "results/06_prepare_dataset_for_esm2_finetune"
DEFAULT_TRAIN = DEFAULT_SPLIT_DIR / "train_100k.tsv"
DEFAULT_VALIDATION = DEFAULT_SPLIT_DIR / "validation.tsv"
DEFAULT_PICKLE = PROJECT / "results/02_phagescope_v2/gpd_phage_data.pkl"
DEFAULT_MODEL = PROJECT / "models/esm2/esm2_t33_650M_UR50D"
DEFAULT_OUT = PROJECT / "results/07_esm2_650m_fixed_split_tuning"

LABELS = (
    "assembly", "replication", "infection", "packaging", "integration",
    "regulation", "lysis", "immune", "tRNA_related",
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


@dataclass(frozen=True)
class GridConfig:
    grid_id: int
    pooling: str
    learning_rate: float
    class_weighting: str

    @property
    def slug(self) -> str:
        lr = f"{self.learning_rate:.0e}".replace("-0", "-")
        return f"grid_{self.grid_id:02d}_{self.pooling}_lr{lr}_{self.class_weighting}"


def build_grid() -> list[GridConfig]:
    grid = []
    index = 1
    for pooling in ("cls", "masked_mean"):
        for learning_rate in (1e-4, 3e-4, 1e-3):
            for class_weighting in ("none", "inverse_sqrt", "balanced"):
                grid.append(GridConfig(index, pooling, learning_rate, class_weighting))
                index += 1
    return grid


GRID = build_grid()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--pickle", dest="pickle_path", type=Path, default=DEFAULT_PICKLE)
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--grid-id", type=int, choices=range(1, 19), metavar="1-18")
    parser.add_argument("--run-all", action="store_true", help="sequentially run all 18 groups")
    parser.add_argument("--list-grid", action="store_true")
    parser.add_argument("--prepare-cache", action="store_true",
                        help="prepare the fixed sequence cache and exit")
    parser.add_argument("--validation-max-samples", type=int, default=30_000,
                        help="fixed stratified tuning subset; 0 uses full validation")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def print_grid() -> None:
    print("grid_id\tpooling\tlearning_rate\tclass_weighting\tslug")
    for config in GRID:
        print(f"{config.grid_id}\t{config.pooling}\t{config.learning_rate:g}\t"
              f"{config.class_weighting}\t{config.slug}")


def validate_args(args: argparse.Namespace) -> None:
    if args.grid_id and args.run_all:
        raise SystemExit("choose either --grid-id or --run-all, not both")
    for role, path in (("train", args.train_manifest), ("validation", args.validation_manifest)):
        if "final_test" in path.name.lower():
            raise SystemExit(f"{role} manifest may not be final_test: {path}")
    if not 64 <= args.max_seq_len <= 1024:
        raise SystemExit("--max-seq-len must be between 64 and 1024")
    if args.validation_max_samples < 0:
        raise SystemExit("--validation-max-samples must be >= 0")
    if args.early_stopping_patience < 1:
        raise SystemExit("--early-stopping-patience must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fixed_manifest(path: Path, expected_split: str | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"phage_id", "protein_id", "genome_cluster", "sequence_sha256",
                    "classification_labels", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            label = row["classification_labels"].strip()
            if label not in LABEL_TO_ID:
                raise SystemExit(f"non-single label in {path}: {label!r}")
            if expected_split and row["split"] != expected_split:
                raise SystemExit(f"unexpected split {row['split']!r} in {path}")
            rows.append(row)
    return rows


def deterministic_stratified_subset(rows: list[dict], target: int, seed: int) -> list[dict]:
    if target == 0 or target >= len(rows):
        return list(rows)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["classification_labels"]].append(row)
    counts = Counter({label: len(buckets[label]) for label in LABELS})
    exact = {label: target * counts[label] / len(rows) for label in LABELS}
    quotas = {label: int(exact[label]) for label in LABELS}
    remaining = target - sum(quotas.values())
    order = sorted(LABELS, key=lambda label: exact[label] - quotas[label], reverse=True)
    for label in order[:remaining]:
        quotas[label] += 1
    selected = []
    for label in LABELS:
        bucket = buckets[label]
        bucket.sort(key=lambda row: hashlib.sha256(
            f"{seed}|validation_tune|{row['sequence_sha256']}".encode()).digest())
        selected.extend(bucket[:quotas[label]])
    selected.sort(key=lambda row: hashlib.sha256(
        f"{seed}|validation_order|{row['sequence_sha256']}".encode()).digest())
    if len(selected) != target:
        raise RuntimeError(f"validation subset has {len(selected)} rows, expected {target}")
    return selected


def audit_disjoint(train_rows: list[dict], validation_rows: list[dict]) -> dict:
    train_hashes = {row["sequence_sha256"] for row in train_rows}
    validation_hashes = {row["sequence_sha256"] for row in validation_rows}
    train_clusters = {row["genome_cluster"] for row in train_rows}
    validation_clusters = {row["genome_cluster"] for row in validation_rows}
    audit = {
        "sequence_overlap": len(train_hashes & validation_hashes),
        "genome_cluster_overlap": len(train_clusters & validation_clusters),
    }
    if any(audit.values()):
        raise SystemExit(f"train/validation leakage detected: {audit}")
    return audit


def cache_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    val_tag = "full" if args.validation_max_samples == 0 else str(args.validation_max_samples)
    stem = f"fixed_sequences_len{args.max_seq_len}_val{val_tag}"
    return args.output_dir / f"{stem}.pkl", args.output_dir / f"{stem}.json"


def write_validation_subset(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def prepare_cache(args: argparse.Namespace) -> tuple[list[tuple], list[tuple], dict]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path, metadata_path = cache_paths(args)
    expected_fingerprints = {
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "validation_manifest_sha256": sha256_file(args.validation_manifest),
        "max_seq_len": args.max_seq_len,
        "validation_max_samples": args.validation_max_samples,
        "seed": args.seed,
    }
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected_fingerprints.items()):
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            return payload["train"], payload["validation"], metadata
        if not args.overwrite:
            raise SystemExit(f"stale cache exists: {cache_path}; use --overwrite to rebuild")

    train_rows = read_fixed_manifest(args.train_manifest, expected_split="train_100k")
    if len(train_rows) != 100_000:
        raise SystemExit(f"train_100k must contain exactly 100,000 rows, found {len(train_rows)}")
    validation_all = read_fixed_manifest(args.validation_manifest, expected_split="validation")
    audit = audit_disjoint(train_rows, validation_all)
    validation_rows = deterministic_stratified_subset(
        validation_all, args.validation_max_samples, args.seed)
    subset_path = args.output_dir / (
        "validation_tune_full.tsv" if args.validation_max_samples == 0
        else f"validation_tune_{args.validation_max_samples // 1000}k.tsv"
    )
    write_validation_subset(subset_path, validation_rows)

    needed = defaultdict(dict)
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for row in rows:
            needed[row["phage_id"]][row["protein_id"]] = (split, row)
    with args.pickle_path.open("rb") as handle:
        source = pickle.load(handle)
    examples = {"train": [], "validation": []}
    extraction = Counter()
    max_residues = args.max_seq_len - 2
    for phage_id, proteins in needed.items():
        source_proteins = (source.get(phage_id, {}).get("proteins") or {})
        for protein_id, (split, row) in proteins.items():
            sequence = (source_proteins.get(protein_id, {}).get("seq") or "").strip().upper().rstrip("*")
            digest = hashlib.sha256(sequence.encode("ascii", errors="ignore")).hexdigest()
            if not sequence or "*" in sequence or digest != row["sequence_sha256"]:
                raise SystemExit(f"sequence integrity failure: {phage_id}/{protein_id}")
            if len(sequence) > max_residues:
                sequence = sequence[:max_residues]
                extraction[f"{split}_truncated"] += 1
            examples[split].append((
                phage_id, protein_id, sequence, LABEL_TO_ID[row["classification_labels"]],
                row["sequence_sha256"], row["genome_cluster"],
            ))
    del source
    random.Random(args.seed).shuffle(examples["train"])
    random.Random(args.seed + 1).shuffle(examples["validation"])
    metadata = {
        **expected_fingerprints,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_records": len(examples["train"]),
        "validation_records": len(examples["validation"]),
        "train_class_counts": dict(Counter(LABELS[row[3]] for row in examples["train"])),
        "validation_class_counts": dict(Counter(LABELS[row[3]] for row in examples["validation"])),
        "extraction": dict(extraction),
        "leakage_audit": audit,
        "validation_subset_manifest": str(subset_path),
    }
    with cache_path.open("wb") as handle:
        pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata["cache_sha256"] = sha256_file(cache_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return examples["train"], examples["validation"], metadata


def class_weights(train_rows: list[tuple], mode: str, torch):
    counts = Counter(row[3] for row in train_rows)
    if mode == "none":
        values = [1.0] * len(LABELS)
    elif mode == "balanced":
        total = len(train_rows)
        values = [total / (len(LABELS) * counts[index]) for index in range(len(LABELS))]
    else:
        maximum = max(counts.values())
        values = [(maximum / counts[index]) ** 0.5 for index in range(len(LABELS))]
    mean = sum(values) / len(values)
    return torch.tensor([value / mean for value in values], dtype=torch.float32)


def metric_values(true_ids, predicted_ids) -> tuple[dict, list[list[int]]]:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for true_id, predicted_id in zip(true_ids, predicted_ids):
        matrix[int(true_id)][int(predicted_id)] += 1
    per_class = {}
    precisions, recalls, f1s, weighted = [], [], [], 0.0
    total = len(true_ids)
    for index, label in enumerate(LABELS):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": support}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted += f1 * support
    correct = sum(matrix[index][index] for index in range(len(LABELS)))
    return {
        "accuracy": correct / total,
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "weighted_f1": weighted / total,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "per_class": per_class,
    }, matrix


def run_one(args: argparse.Namespace, config: GridConfig,
            train_rows: list[tuple], validation_rows: list[tuple], cache_metadata: dict) -> None:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset
        from transformers import (AutoConfig, AutoTokenizer, EarlyStoppingCallback,
                                  EsmModel, EsmPreTrainedModel, Trainer,
                                  TrainingArguments, set_seed)
        from transformers.modeling_outputs import SequenceClassifierOutput
    except ImportError as error:
        raise SystemExit(f"missing training dependency: {error.name}") from error

    run_dir = args.output_dir / config.slug
    completed = run_dir / "validation_metrics.json"
    if completed.exists() and not args.overwrite:
        raise SystemExit(f"completed run exists: {run_dir}; use --overwrite to replace")
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_config = AutoConfig.from_pretrained(args.model_name)
    model_limit = int(getattr(model_config, "max_position_embeddings", 1026)) - 2
    if args.max_seq_len > model_limit:
        raise SystemExit(f"max sequence length exceeds model limit {model_limit}")
    model_config.num_labels = len(LABELS)
    model_config.id2label = {index: label for index, label in enumerate(LABELS)}
    model_config.label2id = LABEL_TO_ID
    model_config.pooling_mode = config.pooling
    model_config.special_token_ids = [
        value for value in (tokenizer.cls_token_id, tokenizer.eos_token_id,
                            tokenizer.pad_token_id) if value is not None
    ]

    class ClassificationHead(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            dropout = getattr(cfg, "hidden_dropout_prob", 0.0)
            self.dropout = nn.Dropout(dropout)
            self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
            self.out_proj = nn.Linear(cfg.hidden_size, cfg.num_labels)

        def forward(self, features):
            features = self.dropout(features)
            features = torch.tanh(self.dense(features))
            features = self.dropout(features)
            return self.out_proj(features)

    class FixedPoolEsmClassifier(EsmPreTrainedModel):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.num_labels = cfg.num_labels
            self.esm = EsmModel(cfg, add_pooling_layer=False)
            self.classifier = ClassificationHead(cfg)
            self.post_init()

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden = outputs.last_hidden_state
            if self.config.pooling_mode == "cls":
                pooled = hidden[:, 0]
            else:
                mask = attention_mask.bool()
                for token_id in self.config.special_token_ids:
                    mask = mask & input_ids.ne(token_id)
                denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
                pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator
            logits = self.classifier(pooled)
            return SequenceClassifierOutput(logits=logits, hidden_states=outputs.hidden_states,
                                            attentions=outputs.attentions)

    model = FixedPoolEsmClassifier.from_pretrained(
        args.model_name, config=model_config, ignore_mismatched_sizes=True)
    for parameter in model.esm.parameters():
        parameter.requires_grad = False

    class ProteinDataset(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            return {"sequence": row[2], "labels": row[3]}

    def collate(batch):
        encoded = tokenizer([item["sequence"] for item in batch], padding=True,
                            truncation=True, max_length=args.max_seq_len,
                            return_tensors="pt")
        encoded["labels"] = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return encoded

    weights = class_weights(train_rows, config.class_weighting, torch)

    class WeightedTrainer(Trainer):
        def __init__(self, *trainer_args, class_weight_tensor, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.class_weight_tensor = class_weight_tensor

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs.logits, labels,
                                   weight=self.class_weight_tensor.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(prediction):
        predictions = np.argmax(prediction.predictions, axis=1)
        metrics, _ = metric_values(prediction.label_ids, predictions)
        flat = {key: value for key, value in metrics.items() if key != "per_class"}
        for label, values in metrics["per_class"].items():
            for key in ("precision", "recall", "f1"):
                flat[f"{key}_{label}"] = values[key]
        return flat

    steps_per_epoch = math.ceil(len(train_rows) / (args.batch_size * args.gradient_accumulation))
    planned_steps = math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = math.ceil(planned_steps * args.warmup_ratio)
    training_kwargs = {
        "output_dir": str(run_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": config.learning_rate,
        "weight_decay": args.weight_decay,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "fp16": not bool(torch.cuda.is_bf16_supported()),
        "remove_unused_columns": False,
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": True,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        raise SystemExit("installed transformers has no evaluation strategy argument")
    if "warmup_ratio" in signature.parameters:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in signature.parameters:
        training_kwargs["warmup_steps"] = warmup_steps
    if "overwrite_output_dir" in signature.parameters:
        training_kwargs["overwrite_output_dir"] = args.overwrite
    training_args = TrainingArguments(**training_kwargs)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ProteinDataset(train_rows),
        eval_dataset=ProteinDataset(validation_rows),
        data_collator=collate,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weight_tensor=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    run_config = {
        "grid": asdict(config), "slug": config.slug,
        "model_name": args.model_name, "labels": list(LABELS),
        "train_records": len(train_rows), "validation_records": len(validation_rows),
        "max_seq_len": args.max_seq_len, "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
        "steps_per_epoch": steps_per_epoch, "planned_steps": planned_steps,
        "gpu_physical_index": args.gpu_index, "seed": args.seed,
        "selection_metric": "validation macro_f1",
        "final_test_accessed": False,
        "cache_metadata": cache_metadata,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    result = trainer.train()
    trainer.save_model(str(run_dir / "best_model"))
    tokenizer.save_pretrained(str(run_dir / "best_model"))
    (run_dir / "trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")
    epoch_rows: dict[float, dict] = {}
    for item in trainer.state.log_history:
        if "epoch" not in item:
            continue
        epoch = float(item["epoch"])
        row = epoch_rows.setdefault(epoch, {"epoch": epoch})
        if "loss" in item and not any(key.startswith("eval_") for key in item):
            row["train_loss"] = item["loss"]
        for source, target in (
            ("eval_loss", "validation_loss"),
            ("eval_accuracy", "accuracy"),
            ("eval_macro_precision", "macro_precision"),
            ("eval_macro_recall", "macro_recall"),
            ("eval_macro_f1", "macro_f1"),
            ("eval_weighted_f1", "weighted_f1"),
            ("eval_balanced_accuracy", "balanced_accuracy"),
        ):
            if source in item:
                row[target] = item[source]
    epoch_fields = ("epoch", "train_loss", "validation_loss", "accuracy",
                    "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
                    "balanced_accuracy")
    with (run_dir / "epoch_metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=epoch_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(epoch_rows[key] for key in sorted(epoch_rows))
    (run_dir / "train_result.json").write_text(
        json.dumps({key: float(value) for key, value in result.metrics.items()}, indent=2),
        encoding="utf-8")

    prediction = trainer.predict(ProteinDataset(validation_rows), metric_key_prefix="validation")
    logits = prediction.predictions
    predicted_ids = np.argmax(logits, axis=1)
    true_ids = np.asarray([row[3] for row in validation_rows], dtype=np.int64)
    metrics, matrix = metric_values(true_ids, predicted_ids)
    metrics["validation_loss"] = float(prediction.metrics.get("validation_loss", float("nan")))
    metrics["best_checkpoint"] = trainer.state.best_model_checkpoint
    metrics["best_validation_macro_f1"] = trainer.state.best_metric
    completed.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "validation_logits.npz", logits=logits,
                        true_ids=true_ids, predicted_ids=predicted_ids)
    with (run_dir / "validation_confusion_matrix.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["true\\predicted", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row])
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    with (run_dir / "validation_predictions.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["phage_id", "protein_id", "true_label", "predicted_label", "confidence"])
        for row, predicted_id, probability in zip(validation_rows, predicted_ids, probabilities):
            writer.writerow([row[0], row[1], LABELS[row[3]], LABELS[int(predicted_id)],
                             f"{probability[int(predicted_id)]:.8f}"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def update_grid_summary(output_dir: Path) -> None:
    rows = []
    for config in GRID:
        metrics_path = output_dir / config.slug / "validation_metrics.json"
        row = {
            "grid_id": config.grid_id,
            "pooling": config.pooling,
            "learning_rate": config.learning_rate,
            "class_weighting": config.class_weighting,
            "status": "completed" if metrics_path.exists() else "pending",
        }
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key in ("validation_loss", "accuracy", "macro_precision", "macro_recall",
                        "macro_f1", "weighted_f1", "balanced_accuracy",
                        "best_validation_macro_f1", "best_checkpoint"):
                row[key] = metrics.get(key)
        rows.append(row)
    fields = ("grid_id", "pooling", "learning_rate", "class_weighting", "status",
              "validation_loss", "accuracy", "macro_precision", "macro_recall",
              "macro_f1", "weighted_f1", "balanced_accuracy",
              "best_validation_macro_f1", "best_checkpoint")
    with (output_dir / "grid_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "grid_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.list_grid or (not args.grid_id and not args.run_all and not args.prepare_cache):
        print_grid()
        if not args.prepare_cache:
            print("\nNo training started. Use --prepare-cache, --grid-id N, or --run-all explicitly.")
            return 0
    for path in (args.train_manifest, args.validation_manifest, args.pickle_path):
        if not path.exists():
            raise SystemExit(f"required input not found: {path}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train_rows, validation_rows, metadata = prepare_cache(args)
    if args.prepare_cache and not args.grid_id and not args.run_all:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    selected = GRID if args.run_all else [GRID[args.grid_id - 1]]
    for config in selected:
        run_one(args, config, train_rows, validation_rows, metadata)
        update_grid_summary(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
