#!/usr/bin/env python3
"""Train a small ESM2→Evo2 soft-prefix adapter on true paired examples.

ESM2 and Evo2 are frozen.  Only the projection from a 1,280-d ESM2 protein
embedding to Evo2 soft-prefix tokens is updated.  The default run is bounded
to 256 optimizer steps so it is safe to validate on one RTX 5090 before
scaling to the full paired set.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_PAIRS = PROJECT / "results/14_microviridae_paired_embeddings"
DEFAULT_EVO = Path("/public8/lilab/student/rtyao/phage/results/02_microviridae_preprocess/evo2_7b_microviridae.pt")
DEFAULT_OUT = PROJECT / "results/15_esm2_evo2_adapter_training"


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--evo-checkpoint", type=Path, default=DEFAULT_EVO)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gpu", default="1")
    p.add_argument("--soft-tokens", type=int, default=8)
    p.add_argument("--hidden-adapter", type=int, default=128)
    p.add_argument("--adapter-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--log-every", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260907)
    return p.parse_args()


def read_pairs(path):
    import csv
    with path.open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h, delimiter="\t"))


def main() -> int:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "args": {k: str(v) for k, v in vars(a).items()}, "scope": "frozen ESM2/frozen Evo2; train adapter only"}
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected one visible CUDA device")
        free, total = torch.cuda.mem_get_info()
        if free < 20 * 1024**3:
            raise RuntimeError(f"Less than 20 GiB free on physical GPU {a.gpu}")
        torch.manual_seed(a.seed); random.seed(a.seed)
        pairs = read_pairs(a.pairs_dir / "paired_examples.tsv")
        embeddings = torch.load(a.pairs_dir / "protein_embeddings.pt", map_location="cpu", weights_only=True)
        if len(pairs) != embeddings.shape[0]:
            raise ValueError(f"pair/embedding mismatch: {len(pairs)} vs {embeddings.shape[0]}")
        report["device"] = {"physical_gpu": a.gpu, "name": torch.cuda.get_device_name(), "free_bytes": free, "total_bytes": total}
        report["pairs"] = len(pairs)

        import importlib.util
        import yaml
        spec = importlib.util.find_spec("evo2")
        config_path = Path(spec.origin).parent / "configs/evo2-7b-8k.yml"
        cfg = yaml.safe_load(config_path.read_text())
        cfg.update(inference_mode=False, use_fp8_input_projections=False, use_flash_attn=False, use_flashfft=False, use_hcs_kernel=False, use_hcm_kernel=False, use_hcl_kernel=False)
        from vortex.model.model import StripedHyena
        from vortex.model.utils import dotdict, load_checkpoint
        from vortex.model.tokenizer import CharLevelTokenizer
        model = StripedHyena(dotdict(cfg))
        load_checkpoint(model, str(a.evo_checkpoint))
        converted = 0
        with torch.inference_mode(False), torch.no_grad():
            for module in model.modules():
                for name, value in list(module._parameters.items()):
                    if value is not None and torch.is_inference(value):
                        module._parameters[name] = nn.Parameter(value.clone(), requires_grad=False); converted += 1
                for name, value in list(module._buffers.items()):
                    if value is not None and torch.is_inference(value):
                        module._buffers[name] = value.clone(); converted += 1
        model.requires_grad_(False); model.eval()
        tokenizer = CharLevelTokenizer(512)
        embedding_dim = int(embeddings.shape[1])
        width = int(cfg["hidden_size"])
        adapter = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, a.hidden_adapter), nn.GELU(), nn.Linear(a.hidden_adapter, a.soft_tokens * width)).cuda()
        nn.init.normal_(adapter[-1].weight, std=0.002); nn.init.zeros_(adapter[-1].bias)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=a.adapter_lr, weight_decay=a.weight_decay)

        def forward(tokens, soft):
            def prepend(_module, _inputs, output):
                return torch.cat([soft.to(output.dtype), output], dim=1)
            hook = model.embedding_layer.register_forward_hook(prepend)
            try:
                return model(tokens)
            finally:
                hook.remove()

        def loss_for(index):
            row = pairs[index]
            z = embeddings[index].float().cuda().unsqueeze(0)
            dna = row["dna_sequence"]
            ids = torch.tensor([tokenizer.tokenize("+~" + dna)], device="cuda", dtype=torch.long)
            soft = adapter(z).reshape(1, a.soft_tokens, width)
            logits, _ = forward(ids[:, :-1], soft)
            return F.cross_entropy(logits[:, a.soft_tokens + 1:].float().reshape(-1, logits.size(-1)), ids[:, 2:].reshape(-1))

        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        report.update({"evo_hidden_size": width, "adapter_trainable_parameters": trainable, "inference_tensors_replaced": converted, "steps": []})
        order = list(range(len(pairs))); random.Random(a.seed).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for step in range(a.max_steps):
            idx = order[step % len(order)]
            loss = loss_for(idx) / a.grad_accum
            if not torch.isfinite(loss): raise RuntimeError("Non-finite loss")
            loss.backward()
            if (step + 1) % a.grad_accum == 0:
                grads = [p.grad for p in adapter.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads): raise RuntimeError("Missing/non-finite adapter gradients")
                norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            else:
                norm = torch.tensor(float("nan"))
            row = {"step": step + 1, "pair_index": idx, "label": pairs[idx]["label"], "loss": float(loss.item() * a.grad_accum), "gradient_norm": float(norm.item())}
            report["steps"].append(row)
            if (step + 1) % a.log_every == 0 or step == 0:
                print(json.dumps(row), flush=True)
        torch.save(adapter.state_dict(), a.output_dir / "adapter_final.pt")
        with torch.no_grad():
            eval_indices = order[: min(32, len(order))]
            eval_losses = [float(loss_for(i).item()) for i in eval_indices]
        report.update({"final_eval_mean_loss": sum(eval_losses) / len(eval_losses), "final_eval_n": len(eval_losses), "status": "passed"})
    except Exception:
        report["status"] = "failed"; report["traceback"] = traceback.format_exc(); print(report["traceback"], flush=True)
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated()); report["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        (a.output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"STATUS={report.get('status')} REPORT={a.output_dir/'training_report.json'}", flush=True)
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
