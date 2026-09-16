#!/usr/bin/env python3
"""Bounded single-protein interface probe; not functional or genome validation.

Uses installed Vortex directly, with in-memory config overrides only. A prefix
embedding hook preserves the model's forward path. All backbone weights remain
frozen. Full-prefix recomputation is the reference for cache validation.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def parse_args():
    root = Path('/public8/lilab/student/rtyao/phage')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', choices=['1'], default='1')
    p.add_argument('--esm-path', type=Path, default=root/'phage_designer/models/esm2/esm2_t33_650M_UR50D')
    p.add_argument('--evo-checkpoint', type=Path, default=root/'results/02_microviridae_preprocess/evo2_7b_microviridae.pt')
    p.add_argument('--protein-fasta', type=Path, default=root/'evo2_repo/phage_gen/data/NC_001422.1_Gprotein.fasta')
    p.add_argument('--dna-fasta', type=Path, default=root/'test_king/01_generate/sequences_clean.fasta')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=3, choices=range(1, 11))
    p.add_argument('--dna-length', type=int, default=128, choices=range(32, 513))
    p.add_argument('--soft-tokens', type=int, default=8, choices=range(1, 33))
    p.add_argument('--generate-tokens', type=int, default=4, choices=range(1, 17))
    p.add_argument('--adapter-lr', type=float, default=1e-5)
    p.add_argument('--cache-atol', type=float, default=0.1)
    p.add_argument('--cache-fp32', action='store_true',
                   help='Diagnose cache drift in float32 after the bf16 training probe')
    return p.parse_args()


def first_fasta(path):
    header, chunks = None, []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    break
                header = line[1:]
            elif line:
                if header is None:
                    raise ValueError(f'Invalid FASTA: {path}')
                chunks.append(line)
    if not chunks:
        raise ValueError(f'Empty FASTA: {path}')
    return header, ''.join(chunks).upper()


def run(a, report):
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from torch import nn
    from torch.nn import functional as F
    from transformers import AutoTokenizer, EsmModel
    torch.manual_seed(20260907)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA device: physical GPU 1')
    free, total = torch.cuda.mem_get_info()
    report['device'] = dict(physical_gpu=a.gpu, name=torch.cuda.get_device_name(),
                            free_bytes=free, total_bytes=total, torch=torch.__version__)
    if free < 20 * 1024**3:
        raise RuntimeError('Less than 20 GiB free; refusing to start this probe')
    torch.cuda.reset_peak_memory_stats()
    protein_id, protein = first_fasta(a.protein_fasta)
    dna_id, dna = first_fasta(a.dna_fasta)
    if not set(dna) <= set('ACGT'):
        raise ValueError('DNA input contains non-ACGT characters')
    protein = protein.rstrip('*')
    if len(protein) > 510:
        raise ValueError('Reference protein exceeds smoke-test limit; do not silently truncate')
    dna = dna[:a.dna_length]
    report['inputs'] = dict(protein_id=protein_id, protein_length=len(protein),
        protein_sha256=hashlib.sha256(protein.encode()).hexdigest(), dna_id=dna_id,
        dna_length=len(dna), dna_sha256=hashlib.sha256(dna.encode()).hexdigest(),
        note='Existing generated DNA fragment for mechanics only, not paired functional training')
    print('Loading frozen ESM2-650M', flush=True)
    tok = AutoTokenizer.from_pretrained(a.esm_path, local_files_only=True)
    esm = EsmModel.from_pretrained(a.esm_path, local_files_only=True,
                                  add_pooling_layer=False, torch_dtype=torch.float32).cuda().eval()
    esm.requires_grad_(False)
    encoded = tok(protein, return_tensors='pt', return_special_tokens_mask=True)
    special = encoded.pop('special_tokens_mask').cuda().bool()
    encoded = {k:v.cuda() for k,v in encoded.items()}
    with torch.no_grad():
        hidden = esm(**encoded).last_hidden_state
        mask = encoded['attention_mask'].bool() & ~special
        z = ((hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)).float()
    report['protein_embedding_shape'] = list(z.shape)
    torch.save(z.cpu(), a.output_dir/'protein_embedding.pt')
    del esm, hidden, encoded, tok
    gc.collect()
    torch.cuda.empty_cache()

    import importlib.util
    import yaml
    # Import paths and config are read from this installed version, not guessed.
    spec = importlib.util.find_spec('evo2')
    config_path = Path(spec.origin).parent/'configs/evo2-7b-8k.yml'
    cfg = yaml.safe_load(config_path.read_text())
    cfg.update(inference_mode=False, use_fp8_input_projections=False,
               use_flash_attn=False, use_flashfft=False,
               use_hcs_kernel=False, use_hcm_kernel=False, use_hcl_kernel=False)
    from vortex.model.model import StripedHyena
    from vortex.model.utils import dotdict, load_checkpoint
    from vortex.model.tokenizer import CharLevelTokenizer
    report['evo_config'] = cfg
    print('Loading Evo2 checkpoint, physical GPU 1 only', flush=True)
    model = StripedHyena(dotdict(cfg))
    load_checkpoint(model, str(a.evo_checkpoint))
    # Loader uses inference_mode: replace inference tensors without modifying dtype.
    converted = 0
    with torch.inference_mode(False), torch.no_grad():
        for module in model.modules():
            for name, value in list(module._parameters.items()):
                if value is not None and torch.is_inference(value):
                    module._parameters[name] = nn.Parameter(value.clone(), requires_grad=False)
                    converted += 1
            for name, value in list(module._buffers.items()):
                if value is not None and torch.is_inference(value):
                    module._buffers[name] = value.clone()
                    converted += 1
    model.requires_grad_(False)
    model.eval()
    report['inference_tensors_replaced'] = converted
    tokenizer = CharLevelTokenizer(512)
    ids = torch.tensor([tokenizer.tokenize('+~'+dna)], device='cuda', dtype=torch.long)
    k, width = a.soft_tokens, int(cfg['hidden_size'])
    adapter = nn.Sequential(nn.LayerNorm(z.shape[-1]), nn.Linear(z.shape[-1], 128),
                            nn.GELU(), nn.Linear(128, k*width)).cuda()
    nn.init.normal_(adapter[-1].weight, std=0.002)
    nn.init.zeros_(adapter[-1].bias)
    opt = torch.optim.AdamW(adapter.parameters(), lr=a.adapter_lr, weight_decay=0.)

    def prefix():
        return adapter(z).reshape(1, k, width)

    def forward(tokens, p=None, cache=None):
        hook = None
        if p is not None:
            def prepend(_module, _inputs, output):
                return torch.cat([p.to(output.dtype), output], dim=1)
            hook = model.embedding_layer.register_forward_hook(prepend)
        try:
            return model(tokens, inference_params_dict=cache)
        finally:
            if hook is not None:
                hook.remove()

    def objective():
        logits, _ = forward(ids[:, :-1], prefix())
        # Fixed '+~' is supplied context. Predict DNA only, starting after '~'.
        return F.cross_entropy(logits[:, k+1:].float().reshape(-1, logits.size(-1)),
                               ids[:, 2:].reshape(-1))

    report['adapter_trainable_parameters'] = sum(p.numel() for p in adapter.parameters())
    before = [p.detach().clone() for p in adapter.parameters()]
    report['steps'] = []
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        loss.backward()
        grads = [p.grad for p in adapter.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Missing/non-finite adapter gradients')
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        if norm.item() == 0:
            raise RuntimeError('Zero gradient: interface is disconnected')
        opt.step()
        row = dict(step=step+1, loss=loss.item(), gradient_norm=norm.item())
        report['steps'].append(row)
        print(row, flush=True)
    delta = max((p.detach()-b).abs().max().item() for p,b in zip(adapter.parameters(), before))
    if delta == 0 or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Adapter failed to update or backbone received parameter gradients')
    report['adapter_max_update'] = delta
    final = objective()
    report['final_loss_grad_enabled'] = final.item()
    del final
    with torch.no_grad():
        report['final_loss'] = objective().item()
    report['loss_decreased'] = report['final_loss'] < report['steps'][0]['loss']
    torch.save(adapter.state_dict(), a.output_dir/'adapter_smoke_only.pt')
    report['gradient_and_update_pass'] = True
    # Save partial progress if an inference-kernel/cache check fails below.
    (a.output_dir/'report.json').write_text(json.dumps(report, indent=2, default=str))

    print('Checking conditioned cached decoding against full-prefix recomputation', flush=True)
    if a.cache_fp32:
        extra = sum(v.numel()*(4-v.element_size()) for v in model.parameters()
                    if v.is_floating_point() and v.element_size() < 4)
        if torch.cuda.mem_get_info()[0] < extra + 1024**3:
            raise RuntimeError('Insufficient free memory for bounded FP32 cache diagnostic')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model.float()
        report['cache_precision'] = 'float32, TF32 disabled; training remained bf16'
    else:
        report['cache_precision'] = 'checkpoint bf16 (selected filter parameters float32)'
    with torch.no_grad():
        p = prefix()
        prompt = torch.tensor([tokenizer.tokenize('+~GAGTTTTAT')], device='cuda', dtype=torch.long)
        cache = model.initialize_inference_params(max_seqlen=128)
        cached, cache = forward(prompt, p, cache)
        for state in cache.values():
            state.seqlen_offset = k + prompt.shape[1]
        whole = prompt
        errors, agreements, probability_errors = [], [], []
        for step in range(a.generate_tokens):
            full, _ = forward(whole, p)
            u, v = cached[:, -1].float(), full[:, -1].float()
            errors.append((u-v).abs().max().item())
            probability_errors.append((u.softmax(-1)-v.softmax(-1)).abs().max().item())
            agreements.append(bool((u.argmax(-1)==v.argmax(-1)).all()))
            next_id = v.argmax(-1, keepdim=True)
            whole = torch.cat([whole, next_id], dim=1)
            if step+1 < a.generate_tokens:
                cached, cache = forward(next_id, None, cache)
                for state in cache.values():
                    state.seqlen_offset += 1
        report['cache'] = dict(max_abs_logit_errors=errors, argmax_agreement=agreements,
            max_abs_probability_errors=probability_errors,
            atol=a.cache_atol, passed=max(errors) <= a.cache_atol and all(agreements))
        report['generated_token_ids'] = whole[:, prompt.shape[1]:].tolist()
        plain, _ = forward(prompt)
        conditioned, _ = forward(prompt, p)
        report['conditioning_logit_delta'] = (plain[:, -1]-conditioned[:, -1]).abs().max().item()
        # Same cache check without soft prefix: diagnose stock numerical drift.
        baseline_cache = model.initialize_inference_params(max_seqlen=128)
        cached, baseline_cache = forward(prompt, None, baseline_cache)
        for state in baseline_cache.values():
            state.seqlen_offset = prompt.shape[1]
        baseline_errors, baseline_probs, baseline_agreements = [], [], []
        whole = prompt
        for step in range(a.generate_tokens):
            full, _ = forward(whole)
            u, v = cached[:, -1].float(), full[:, -1].float()
            baseline_errors.append((u-v).abs().max().item())
            baseline_probs.append((u.softmax(-1)-v.softmax(-1)).abs().max().item())
            baseline_agreements.append(bool((u.argmax(-1)==v.argmax(-1)).all()))
            next_id = v.argmax(-1, keepdim=True)
            whole = torch.cat([whole, next_id], dim=1)
            if step+1 < a.generate_tokens:
                cached, baseline_cache = forward(next_id, None, baseline_cache)
                for state in baseline_cache.values():
                    state.seqlen_offset += 1
        report['unconditioned_cache_control'] = dict(max_abs_logit_errors=baseline_errors,
            max_abs_probability_errors=baseline_probs, argmax_agreement=baseline_agreements,
            passed=max(baseline_errors) <= a.cache_atol and all(baseline_agreements))
    report['status'] = ('passed' if report['cache']['passed'] and report['loss_decreased']
                        and report['unconditioned_cache_control']['passed']
                        else 'partial_pass')


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', args=vars(a),
                  scope='Single-protein engineering smoke test; no functional inference')
    start = time.time()
    try:
        run(a, report)
    except Exception:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], flush=True)
    finally:
        report['elapsed_seconds'] = time.time()-start
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        (a.output_dir/'report.json').write_text(json.dumps(report, indent=2, default=str))
        print(f"STATUS={report['status']} REPORT={a.output_dir/'report.json'}", flush=True)
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
