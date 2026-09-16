#!/usr/bin/env python3
"""Smoke-test cached ESM2 protein conditioning of a frozen Evo2 backbone.

This is an interface/mechanics test only.  It verifies that a real
Microviridae reference protein embedding can be projected to Evo2 soft-prefix
tokens, that the adapter receives gradients, and that conditioned DNA logits
are finite and different from the unconditioned logits.  It does not claim
that generated DNA has the requested biological function.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

PROJECT = Path("/public8/lilab/student/rtyao/phage/phage_designer_v0.1")
DEFAULT_REF = PROJECT / "results/12_esm2_microviridae_reference_library"
DEFAULT_GENOMES = PROJECT / "results/09_microviridae_dataset/microviridae_genomes.tsv"
DEFAULT_EVO = Path("/public8/lilab/student/rtyao/phage/results/02_microviridae_preprocess/evo2_7b_microviridae.pt")


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=Path, default=DEFAULT_REF)
    p.add_argument("--genomes", type=Path, default=DEFAULT_GENOMES)
    p.add_argument("--evo-checkpoint", type=Path, default=DEFAULT_EVO)
    p.add_argument("--gpu", default="1")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dna-length", type=int, default=128)
    p.add_argument("--soft-tokens", type=int, default=8)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--generate-tokens", type=int, default=8)
    p.add_argument("--adapter-lr", type=float, default=1e-5)
    return p.parse_args()


def read_tsv(path):
    import csv
    with path.open(encoding="utf-8", newline="") as h:
        yield from csv.DictReader(h, delimiter="\t")


def main() -> int:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "cached ESM2 -> soft-prefix Evo2 interface smoke test", "args": {k: str(v) for k, v in vars(a).items()}}
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
        report["device"] = {"physical_gpu": a.gpu, "name": torch.cuda.get_device_name(), "free_bytes": free, "total_bytes": total}

        import pandas as pd
        refs = pd.read_csv(a.reference_dir / "reference_proteins.tsv", sep="\t", dtype=str).fillna("")
        target = refs[refs["reference_role"].eq("spike_or_G_product")]
        if target.empty:
            target = refs.iloc[[0]]
        ref = target.iloc[0]
        embeddings = torch.load(a.reference_dir / "reference_embeddings.pt", map_location="cpu", weights_only=True)
        embedding = embeddings[int(ref["reference_index"])].float().cuda().unsqueeze(0)
        genomes = pd.read_csv(a.genomes, sep="\t", dtype=str, usecols=["phage_id", "genome_sequence"])
        match = genomes[genomes["phage_id"].eq(ref["phage_id"])]
        genome = match.iloc[0] if not match.empty else genomes.iloc[0]
        dna = str(genome["genome_sequence"]).upper()[: a.dna_length]
        if set(dna) - set("ACGT"):
            raise ValueError("Selected Microviridae DNA contains non-ACGT characters")
        report["inputs"] = {"reference_phage_id": ref["phage_id"], "reference_protein_id": ref["protein_id"], "reference_product": ref["product"], "reference_role": ref["reference_role"], "reference_function": ref["normalized_function"], "protein_length": len(ref["protein_sequence"]), "dna_phage_id": genome["phage_id"], "dna_length": len(dna)}
        torch.save(embedding.cpu(), a.output_dir / "conditioning_embedding.pt")

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
        # Checkpoint loading under inference_mode creates inference tensors;
        # clone them before the hook-based forward pass.
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
        report["evo_hidden_size"] = int(cfg["hidden_size"])
        report["inference_tensors_replaced"] = converted
        tokenizer = CharLevelTokenizer(512)
        ids = torch.tensor([tokenizer.tokenize("+~" + dna)], device="cuda", dtype=torch.long)
        k, width = a.soft_tokens, int(cfg["hidden_size"])
        adapter = nn.Sequential(nn.LayerNorm(embedding.shape[-1]), nn.Linear(embedding.shape[-1], 128), nn.GELU(), nn.Linear(128, k * width)).cuda()
        nn.init.normal_(adapter[-1].weight, std=0.002); nn.init.zeros_(adapter[-1].bias)
        opt = torch.optim.AdamW(adapter.parameters(), lr=a.adapter_lr, weight_decay=0.0)

        def prefix():
            return adapter(embedding).reshape(1, k, width)

        def forward(tokens, soft=None):
            hook = None
            if soft is not None:
                def prepend(_module, _inputs, output):
                    return torch.cat([soft.to(output.dtype), output], dim=1)
                hook = model.embedding_layer.register_forward_hook(prepend)
            try:
                return model(tokens)
            finally:
                if hook is not None: hook.remove()

        def objective():
            logits, _ = forward(ids[:, :-1], prefix())
            return F.cross_entropy(logits[:, k + 1:].float().reshape(-1, logits.size(-1)), ids[:, 2:].reshape(-1))

        before = [p.detach().clone() for p in adapter.parameters()]
        steps = []
        for step in range(a.steps):
            opt.zero_grad(set_to_none=True)
            loss = objective()
            if not torch.isfinite(loss): raise RuntimeError("Non-finite adapter loss")
            loss.backward()
            grads = [p.grad for p in adapter.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads): raise RuntimeError("Missing/non-finite adapter gradients")
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            if norm.item() == 0: raise RuntimeError("Zero adapter gradient")
            opt.step(); steps.append({"step": step + 1, "loss": float(loss.item()), "gradient_norm": float(norm.item())})
        delta = max((p.detach() - b).abs().max().item() for p, b in zip(adapter.parameters(), before))
        with torch.no_grad(): final_loss = float(objective().item())
        with torch.no_grad():
            p = prefix(); plain, _ = forward(ids[:, 2:]); conditioned, _ = forward(ids[:, 2:], p)
            delta_logits = float((plain[:, -1] - conditioned[:, -1]).abs().max().item())
            generated = ids[:, 2:].clone()
            for _ in range(a.generate_tokens):
                out, _ = forward(generated, p)
                nxt = out[:, -1].float().argmax(-1, keepdim=True)
                generated = torch.cat([generated, nxt], dim=1)
        torch.save(adapter.state_dict(), a.output_dir / "adapter_smoke_only.pt")
        report.update({"steps": steps, "initial_loss": steps[0]["loss"], "final_loss": final_loss, "loss_decreased": final_loss < steps[0]["loss"], "adapter_max_update": delta, "conditioning_logit_delta": delta_logits, "generated_token_ids": generated[:, -a.generate_tokens:].tolist(), "gradient_and_update_pass": bool(delta > 0), "status": "passed" if final_loss < steps[0]["loss"] and delta > 0 and delta_logits > 0 else "failed"})
    except Exception:
        report["status"] = "failed"; report["traceback"] = traceback.format_exc()
    finally:
        if "torch" in locals() and torch.cuda.is_available():
            report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            report["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        (a.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
