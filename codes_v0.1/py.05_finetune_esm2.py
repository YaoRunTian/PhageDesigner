#!/usr/bin/env python3
"""Fine-tune ESM-2 on single-function GPD protein classifications.

Only the nine valid, single-valued ``classification`` labels are used.
``hypothetical``, ``unsorted`` and semicolon-delimited multi-function labels are
excluded. Protein records are randomly split 70:30 by label; genome clusters are
deliberately not used in this first experiment.

The script is safe by default: it samples at most 100,000 proteins, truncates to
512 tokens, freezes the ESM-2 backbone, and limits training to 10,000 optimizer
steps. It never starts automatically merely by being imported.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_MANIFEST = PROJECT / "results/03_gpd_qc/gpd_protein_manifest.tsv"
DEFAULT_PICKLE = PROJECT / "results/02_phagescope_v2/gpd_phage_data.pkl"
DEFAULT_OUT = PROJECT / "results/05_esm2_single_function"

LABELS = [
    "assembly", "replication", "infection", "packaging", "integration",
    "regulation", "lysis", "immune", "tRNA_related",
]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pickle", dest="pickle_path", type=Path, default=DEFAULT_PICKLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--max-samples", type=int, default=100_000,
                        help="总样本上限；0=全部2,961,179条，需--allow-large-run")
    parser.add_argument("--sampling-strategy", choices=("proportional", "balanced"),
                        default="proportional", help="自然分布或按类别近似均衡抽样")
    parser.add_argument("--allow-large-run", action="store_true",
                        help="允许max-samples=0或超过500,000条")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="tokenizer总长度（含特殊token），范围64-1024")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--gpu-index", default="1",
                        help="物理GPU编号；默认卡1，即CUDA_VISIBLE_DEVICES=1")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-train-steps", type=int, default=10_000,
                        help="优化器步数上限；0=按epochs完整训练")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--class-weighting", choices=("none", "inverse_sqrt", "balanced"),
                        default="inverse_sqrt", help="交叉熵类别权重")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=0,
                        help="0=仅分类头；正整数=最后N层；-1=全量微调")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True, help="解冻ESM-2层时以计算换显存")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="仅筛选、提取和划分数据，不加载模型、不训练")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not args.pickle_path.exists():
        raise SystemExit(f"pickle not found: {args.pickle_path}")
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be >= 0")
    if args.max_samples == 0 and not args.allow_large_run:
        raise SystemExit("安全限制：--max-samples 0 需显式添加 --allow-large-run")
    if args.max_samples > 500_000 and not args.allow_large_run:
        raise SystemExit("安全限制：--max-samples > 500000 需显式添加 --allow-large-run")
    if not 64 <= args.max_seq_len <= 1024:
        raise SystemExit("--max-seq-len must be between 64 and 1024")
    if not 0.05 <= args.train_fraction <= 0.95:
        raise SystemExit("--train-fraction must be between 0.05 and 0.95")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise SystemExit("batch sizes must be positive")
    if args.gradient_accumulation < 1:
        raise SystemExit("--gradient-accumulation must be positive")
    if args.max_train_steps < 0:
        raise SystemExit("--max-train-steps must be >= 0")
    if args.unfreeze_last_n_layers < -1:
        raise SystemExit("--unfreeze-last-n-layers must be -1 or >= 0")


def eligible_counts(manifest: Path) -> Counter:
    counts: Counter = Counter()
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            label = (row.get("classification") or "").strip()
            if label in LABEL_TO_ID:
                counts[label] += 1
    return counts


def allocate_quotas(counts: Counter, total: int, strategy: str) -> dict[str, int]:
    if total == 0 or total >= sum(counts.values()):
        return dict(counts)
    if strategy == "balanced":
        target = total // len(LABELS)
        quotas = {label: min(counts[label], target) for label in LABELS}
    else:
        denominator = sum(counts.values())
        quotas = {label: min(counts[label], int(total * counts[label] / denominator))
                  for label in LABELS}
    remaining = total - sum(quotas.values())
    while remaining > 0:
        candidates = [label for label in LABELS if quotas[label] < counts[label]]
        if not candidates:
            break
        candidates.sort(key=lambda label: (counts[label] - quotas[label], label), reverse=True)
        for label in candidates:
            if remaining == 0:
                break
            quotas[label] += 1
            remaining -= 1
    return quotas


def stratified_reservoir(manifest: Path, quotas: dict[str, int], seed: int
                         ) -> list[tuple[str, str, str]]:
    """Sample each class independently while bounding memory by max-samples."""
    rng = random.Random(seed)
    seen: Counter = Counter()
    samples: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            label = (row.get("classification") or "").strip()
            quota = quotas.get(label, 0)
            if quota == 0:
                continue
            seen[label] += 1
            item = (row["phage_id"], row["protein_id"], label)
            bucket = samples[label]
            if len(bucket) < quota:
                bucket.append(item)
            else:
                index = rng.randrange(seen[label])
                if index < quota:
                    bucket[index] = item
    return [item for label in LABELS for item in samples[label]]


def extract_sequences(rows: list[tuple[str, str, str]], pickle_path: Path,
                      max_seq_len: int
                      ) -> tuple[list[tuple[str, str, str, int]], dict[str, int]]:
    """Load selected AA sequences; tuples use less memory than millions of dicts."""
    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)
    examples: list[tuple[str, str, str, int]] = []
    stats = Counter()
    max_residues = max_seq_len - 2
    for phage_id, protein_id, label in rows:
        protein = ((data.get(phage_id, {}).get("proteins") or {}).get(protein_id) or {})
        sequence = (protein.get("seq") or "").strip().upper().rstrip("*")
        if not sequence:
            stats["missing_sequence"] += 1
            continue
        if "*" in sequence:
            stats["internal_stop"] += 1
            continue
        if len(sequence) > max_residues:
            stats["truncated_sequence"] += 1
            sequence = sequence[:max_residues]
        examples.append((phage_id, protein_id, sequence, LABEL_TO_ID[label]))
    del data
    stats["examples_with_sequence"] = len(examples)
    return examples, dict(stats)


def stratified_split(examples: list[tuple[str, str, str, int]], train_fraction: float,
                     seed: int):
    buckets = defaultdict(list)
    for example in examples:
        buckets[example[3]].append(example)
    train, test = [], []
    for label_id in range(len(LABELS)):
        bucket = buckets[label_id]
        random.Random(seed + label_id).shuffle(bucket)
        if len(bucket) < 2:
            raise SystemExit(f"not enough examples for label {LABELS[label_id]}")
        cut = max(1, min(len(bucket) - 1, int(len(bucket) * train_fraction)))
        train.extend(bucket[:cut])
        test.extend(bucket[cut:])
    random.Random(seed).shuffle(train)
    random.Random(seed + 10_000).shuffle(test)
    return train, test


def write_split_manifest(path: Path, rows, split: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["phage_id", "protein_id", "classification", "split",
                         "sequence_length_aa"])
        for phage_id, protein_id, sequence, label_id in rows:
            writer.writerow([phage_id, protein_id, LABELS[label_id], split, len(sequence)])


def make_class_weights(train_rows, mode: str, torch):
    counts = Counter(row[3] for row in train_rows)
    if mode == "none":
        weights = [1.0] * len(LABELS)
    elif mode == "balanced":
        total = sum(counts.values())
        weights = [total / (len(LABELS) * counts[idx]) for idx in range(len(LABELS))]
    else:
        maximum = max(counts.values())
        weights = [(maximum / counts[idx]) ** 0.5 for idx in range(len(LABELS))]
    mean = sum(weights) / len(weights)
    return torch.tensor([weight / mean for weight in weights], dtype=torch.float32)


def classification_metrics(true_ids, pred_ids):
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for true_id, pred_id in zip(true_ids, pred_ids):
        matrix[int(true_id)][int(pred_id)] += 1
    per_class = {}
    f1_values, weighted_f1_sum, total = [], 0.0, len(true_ids)
    correct = sum(matrix[idx][idx] for idx in range(len(LABELS)))
    for idx, label in enumerate(LABELS):
        tp = matrix[idx][idx]
        support = sum(matrix[idx])
        predicted = sum(row[idx] for row in matrix)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": f1, "support": support}
        f1_values.append(f1)
        weighted_f1_sum += f1 * support
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "weighted_f1": weighted_f1_sum / total if total else 0.0,
        "balanced_accuracy": sum(item["recall"] for item in per_class.values()) / len(LABELS),
        "per_class": per_class,
    }
    return metrics, matrix


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    counts = eligible_counts(args.manifest)
    quotas = allocate_quotas(counts, args.max_samples, args.sampling_strategy)
    sampled_rows = stratified_reservoir(args.manifest, quotas, args.seed)
    examples, extraction_stats = extract_sequences(sampled_rows, args.pickle_path,
                                                   args.max_seq_len)
    train_rows, test_rows = stratified_split(examples, args.train_fraction, args.seed)
    write_split_manifest(args.output_dir / "train_manifest.tsv", train_rows, "train")
    write_split_manifest(args.output_dir / "test_manifest.tsv", test_rows, "test")

    run_config = {
        "dataset": "GPD",
        "split_policy": "stratified random protein-level 70:30; genome_cluster not used",
        "eligible_single_function_records": sum(counts.values()),
        "eligible_class_counts": dict(counts),
        "sampling_strategy": args.sampling_strategy,
        "sampling_quotas": quotas,
        "sampled_manifest_rows": len(sampled_rows),
        "extraction": extraction_stats,
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "train_class_counts": dict(Counter(LABELS[row[3]] for row in train_rows)),
        "test_class_counts": dict(Counter(LABELS[row[3]] for row in test_rows)),
        "labels": LABELS,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "gpu_physical_index": args.gpu_index,
        "model_visible_cuda_index": 0,
        "model_name": args.model_name,
        "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
        "class_weighting": args.class_weighting,
        "max_train_steps": args.max_train_steps,
    }
    steps_per_epoch = math.ceil(
        len(train_rows) / (args.batch_size * args.gradient_accumulation)
    )
    epoch_steps = math.ceil(steps_per_epoch * args.epochs)
    planned_steps = (
        min(epoch_steps, args.max_train_steps) if args.max_train_steps else epoch_steps
    )
    run_config["steps_per_epoch"] = steps_per_epoch
    run_config["planned_optimizer_steps"] = planned_steps
    planned_warmup_steps = math.ceil(planned_steps * args.warmup_ratio)
    run_config["warmup_ratio"] = args.warmup_ratio
    run_config["planned_warmup_steps"] = planned_warmup_steps
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(run_config, ensure_ascii=False, indent=2))
        return 0

    try:
        import numpy as np
        import torch
        import torch.nn.functional as torch_functional
        from torch.utils.data import Dataset
        from transformers import (AutoConfig, AutoTokenizer, EsmForSequenceClassification,
                                  Trainer, TrainingArguments, set_seed)
    except ImportError as error:
        raise SystemExit(
            f"缺少依赖 {error.name}；训练前需在phage环境安装transformers和accelerate"
        ) from error

    if not torch.cuda.is_available():
        raise SystemExit("CUDA不可用；脚本拒绝在CPU上意外启动ESM-2训练")
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_name)
    model_limit = int(getattr(config, "max_position_embeddings", 1026)) - 2
    if args.max_seq_len > model_limit:
        raise SystemExit(f"--max-seq-len {args.max_seq_len} exceeds model limit {model_limit}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    class ProteinDataset(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index):
            _, _, sequence, label_id = self.rows[index]
            return {"sequence": sequence, "labels": label_id}

    def collate(batch):
        tokenized = tokenizer([item["sequence"] for item in batch], padding=True,
                              truncation=True, max_length=args.max_seq_len,
                              return_tensors="pt")
        tokenized["labels"] = torch.tensor([item["labels"] for item in batch],
                                            dtype=torch.long)
        return tokenized

    model = EsmForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS),
        id2label={idx: label for idx, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID, ignore_mismatched_sizes=True)
    layers = model.esm.encoder.layer
    if args.unfreeze_last_n_layers == -1:
        for parameter in model.esm.parameters(): parameter.requires_grad = True
    else:
        if args.unfreeze_last_n_layers > len(layers):
            raise SystemExit(f"model has {len(layers)} layers; cannot unfreeze "
                             f"{args.unfreeze_last_n_layers}")
        for parameter in model.esm.parameters(): parameter.requires_grad = False
        selected_layers = layers[-args.unfreeze_last_n_layers:] \
            if args.unfreeze_last_n_layers else []
        for layer in selected_layers:
            for parameter in layer.parameters(): parameter.requires_grad = True
    if args.gradient_checkpointing and args.unfreeze_last_n_layers != 0:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    weights = make_class_weights(train_rows, args.class_weighting, torch)

    class WeightedTrainer(Trainer):
        def __init__(self, *trainer_args, class_weight_tensor, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.class_weight_tensor = class_weight_tensor

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch_functional.cross_entropy(
                outputs.logits, labels,
                weight=self.class_weight_tensor.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    use_bf16 = bool(torch.cuda.is_bf16_supported())
    training_kwargs = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "logging_steps": 100,
        "save_strategy": "steps",
        "save_steps": 1000,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "remove_unused_columns": False,
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": True,
        "seed": args.seed,
        "data_seed": args.seed,
        # Positive max_steps overrides epochs in Trainer, so use the smaller of
        # the epoch-derived plan and the user safety cap. This never extends a run.
        "max_steps": planned_steps,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    # transformers 5.x removed overwrite_output_dir from TrainingArguments.
    # Pass it only on versions that still expose the parameter; this script
    # writes its own manifests/config before Trainer construction, so omitting
    # it on newer versions does not alter the requested run semantics.
    if "overwrite_output_dir" in signature.parameters:
        training_kwargs["overwrite_output_dir"] = args.overwrite_output_dir
    if "warmup_ratio" in signature.parameters:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in signature.parameters:
        training_kwargs["warmup_steps"] = planned_warmup_steps
    else:
        raise SystemExit(
            "installed transformers supports neither warmup_ratio nor warmup_steps"
        )
    if "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = "no"
    training_args = TrainingArguments(**training_kwargs)
    trainer = WeightedTrainer(
        model=model, args=training_args,
        train_dataset=ProteinDataset(train_rows), data_collator=collate,
        processing_class=tokenizer, class_weight_tensor=weights)
    trainer.train()
    final_model = args.output_dir / "final_model"
    trainer.save_model(str(final_model))
    tokenizer.save_pretrained(str(final_model))

    prediction = trainer.predict(ProteinDataset(test_rows), metric_key_prefix="test")
    logits = prediction.predictions
    predicted_ids = np.argmax(logits, axis=1)
    true_ids = np.asarray([row[3] for row in test_rows], dtype=np.int64)
    metrics, matrix = classification_metrics(true_ids, predicted_ids)
    metrics["trainer_test_loss"] = float(prediction.metrics.get("test_loss", float("nan")))
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "confusion_matrix.tsv").open("w", encoding="utf-8",
                                                          newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["true\\predicted", *LABELS])
        for label, row in zip(LABELS, matrix): writer.writerow([label, *row])
    with (args.output_dir / "test_predictions.tsv").open("w", encoding="utf-8",
                                                         newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["phage_id", "protein_id", "true_label",
                         "predicted_label", "confidence"])
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
        for row, predicted_id, probability in zip(test_rows, predicted_ids, probabilities):
            writer.writerow([row[0], row[1], LABELS[row[3]], LABELS[int(predicted_id)],
                             f"{float(probability[int(predicted_id)]):.8f}"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
