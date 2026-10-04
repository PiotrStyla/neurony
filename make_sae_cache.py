#!/usr/bin/env python3
"""Build the circuit-tracer transcoder cache for the res-v5 TopK SAE set.

Layout (what circuit_tracer.utils.caching expects for a `transcoder_set`):
  ~/.cache/circuit_tracer/local/gollem-res-v5/
    config.yaml
    layer_{0..15}.safetensors      keys: W_enc, W_dec, b_enc, b_dec  (fp32)

Orientation change vs the published SAE files (TopKSAE: pre = (x - b_dec) @ W_enc + b_enc):
SingleLayerTranscoder computes pre = x @ W_enc^T + b_enc, so with
  W_enc' = W_enc^T            [d_sae, d_model]
  b_enc' = b_enc - b_dec @ W_enc
  W_dec' = W_dec              [d_sae, d_model]
  b_dec' = b_dec
encode/decode match TopKSAE exactly (decode already identical).

Self-test: TopKSAE.encode(x) values/indices == SingleLayerTranscoder activation on random x.
"""
import argparse
import os

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SAE_DIR = os.path.join(ROOT, "data", "sae")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "circuit_tracer", "local", "gollem-res-v5")
N_LAYER, D_MODEL, D_SAE, K = 16, 768, 6144, 50

# TopKSAE (OpenAI recipe): pre = (x - b_dec) @ W_enc + b_enc; konwersja składa b_dec w bias
# enkodera, orientacja W_enc zmienia się na [d_sae, d_model].
CONFIG = {
    "model_name": "Maggio33/GoLLeM-v5-128M-Muon-v1",
    "model_kind": "transcoder_set",
    "feature_input_hook": "hook_resid_post",
    "feature_output_hook": "hook_resid_post",
    "scan_name": "res-v5",
    "activation": "topk",
    "k": K,
}


def parse_args():
    p = argparse.ArgumentParser(description="TopK SAE set -> circuit-tracer transcoder cache")
    p.add_argument("--sae-dir", default=SAE_DIR, help="katalog z plikami safetensors SAE")
    p.add_argument("--out", default=CACHE, help="katalog wyjściowy cache")
    p.add_argument("--file-template", default="res_v5_layer{layer}_final.safetensors",
                   help="szablon nazwy pliku warstwy")
    p.add_argument("--layers", type=int, default=N_LAYER)
    p.add_argument("--d-model", type=int, default=D_MODEL)
    p.add_argument("--d-sae", type=int, default=D_SAE)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--model-name", default=CONFIG["model_name"])
    p.add_argument("--scan-name", default=CONFIG["scan_name"])
    p.add_argument("--skip-test", action="store_true")
    return p.parse_args()


def convert_layer(a, layer: int):
    src = os.path.join(a.sae_dir, a.file_template.format(layer=layer))
    with safe_open(src, framework="pt") as f:
        sd = {k: f.get_tensor(k).float() for k in f.keys()}
    w_enc, b_enc, w_dec, b_dec = sd["W_enc"], sd["b_enc"], sd["W_dec"], sd["b_dec"]
    assert w_enc.shape == (a.d_model, a.d_sae) and w_dec.shape == (a.d_sae, a.d_model), (
        f"layer {layer}: {w_enc.shape}/{w_dec.shape} != {(a.d_model, a.d_sae)}/{(a.d_sae, a.d_model)}"
    )
    out = {
        "W_enc": (w_enc.T).contiguous(),
        "b_enc": (b_enc - b_dec @ w_enc).contiguous(),
        "W_dec": w_dec.contiguous(),
        "b_dec": b_dec.contiguous(),
    }
    save_file(out, os.path.join(a.out, f"layer_{layer}.safetensors"))


def self_test(a):
    """Folded SingleLayerTranscoder must reproduce the source TopKSAE math exactly."""
    import sys

    try:
        import circuit_tracer  # noqa: F401  (installed in apps/graph's venv)
    except ImportError:
        sys.path.insert(0, r"C:/Users/Hipek/circuit_tracer_fork")  # local dev checkout
    from circuit_tracer.transcoder.activation_functions import TopK
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder
    from safetensors.torch import load_file

    layer = a.layers // 2
    src = os.path.join(a.sae_dir, a.file_template.format(layer=layer))
    sd = load_file(src)
    w_enc, b_enc, w_dec, b_dec = (sd[k].float() for k in ("W_enc", "b_enc", "W_dec", "b_dec"))
    tr = load_transcoder(
        os.path.join(a.out, f"layer_{layer}.safetensors"),
        layer,
        activation_fn=TopK(a.k),
        lazy_encoder=False,
        lazy_decoder=False,
    )

    torch.manual_seed(0)
    x = torch.randn(256, a.d_model) * 5
    with torch.no_grad():
        # reference: TopKSAE math straight from the source file (no third-party code)
        pre_sae = torch.relu((x - b_dec) @ w_enc + b_enc)
        vals, idx = pre_sae.topk(a.k, dim=-1)
        f_sparse = torch.zeros(256, a.d_sae).scatter(1, idx, vals)
        xhat_sae = f_sparse @ w_dec + b_dec
        # converted SingleLayerTranscoder (bias folded, W_enc transposed)
        pre_tr = tr.encode(x, apply_activation_function=False)
        d_pre = (torch.relu(pre_tr) - pre_sae).abs().max().item()
        acts_tr = tr.encode(x)
        d_act = (acts_tr.gather(1, idx) - vals).abs().max().item()
        xhat_tr = tr.decode(acts_tr, x)
        d_dec = (xhat_sae - xhat_tr).abs().max().item()
    print(f"encode pre diff : {d_pre:.3e}")
    print(f"top-k val diff  : {d_act:.3e}")
    print(f"decode diff     : {d_dec:.3e}")
    # absolute floor + scale term (aktywacje potrafią mieć rząd 1e4)
    tol = 1e-3 + 1e-7 * float(vals.abs().max())
    assert max(d_pre, d_act, d_dec) < tol, "SAE CONVERSION MISMATCH"
    print("SAE CONVERSION OK")


def main():
    a = parse_args()
    config = {**CONFIG, "model_name": a.model_name, "scan_name": a.scan_name, "k": a.k}
    os.makedirs(a.out, exist_ok=True)
    for l in range(a.layers):
        convert_layer(a, l)
    with open(os.path.join(a.out, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    print(f"wrote {a.layers} layers + config.yaml -> {a.out}")
    if not a.skip_test:
        self_test(a)


if __name__ == "__main__":
    main()
