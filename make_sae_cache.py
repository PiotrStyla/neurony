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
import os

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SAE_DIR = os.path.join(ROOT, "data", "sae")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "circuit_tracer", "local", "gollem-res-v5")
N_LAYER, D_MODEL, D_SAE, K = 16, 768, 6144, 50

CONFIG = {
    "model_name": "Maggio33/GoLLeM-v5-128M-Muon-v1",
    "model_kind": "transcoder_set",
    "feature_input_hook": "hook_resid_post",
    "feature_output_hook": "hook_resid_post",
    "scan_name": "res-v5",
    "activation": "topk",
    "k": K,
}


def convert_layer(layer: int):
    with safe_open(os.path.join(SAE_DIR, f"res_v5_layer{layer}_final.safetensors"), framework="pt") as f:
        sd = {k: f.get_tensor(k).float() for k in f.keys()}
    w_enc, b_enc, w_dec, b_dec = sd["W_enc"], sd["b_enc"], sd["W_dec"], sd["b_dec"]
    assert w_enc.shape == (D_MODEL, D_SAE) and w_dec.shape == (D_SAE, D_MODEL)
    out = {
        "W_enc": (w_enc.T).contiguous(),
        "b_enc": (b_enc - b_dec @ w_enc).contiguous(),
        "W_dec": w_dec.contiguous(),
        "b_dec": b_dec.contiguous(),
    }
    save_file(out, os.path.join(CACHE, f"layer_{layer}.safetensors"))


def self_test():
    """Folded SingleLayerTranscoder must reproduce TopKSAE.encode/decode exactly."""
    import sys

    try:
        import circuit_tracer  # noqa: F401  (installed in apps/graph's venv)
    except ImportError:
        sys.path.insert(0, r"C:/Users/Hipek/circuit_tracer_fork")  # local dev checkout
    from circuit_tracer.transcoder.activation_functions import TopK
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder

    sys.path.insert(0, os.path.join(ROOT, "data", "sae"))
    from topk_sae import TopKSAE  # noqa: E402  (shipped in the SAE HF repo)

    layer = 8
    sae = TopKSAE(D_MODEL, D_SAE, K)
    from safetensors.torch import load_file

    sae.load_state_dict(load_file(os.path.join(SAE_DIR, f"res_v5_layer{layer}_final.safetensors")))
    tr = load_transcoder(
        os.path.join(CACHE, f"layer_{layer}.safetensors"),
        layer,
        activation_fn=TopK(K),
        lazy_encoder=False,
        lazy_decoder=False,
    )

    torch.manual_seed(0)
    x = torch.randn(256, D_MODEL) * 5
    with torch.no_grad():
        vals, idx = sae.encode(x)
        pre_tr = tr.encode(x, apply_activation_function=False)
        # TopKSAE vals are relu(pre).topk; compare like for like (post-ReLU)
        pre_sae = torch.relu((x - sae.b_dec) @ sae.W_enc + sae.b_enc)
        d_pre = (torch.relu(pre_tr) - pre_sae).abs().max().item()
        acts_tr = tr.encode(x)
        d_act = (acts_tr.gather(1, idx) - vals).abs().max().item()
        xhat_sae = sae.decode(vals, idx)
        xhat_tr = tr.decode(acts_tr, x)
        d_dec = (xhat_sae - xhat_tr).abs().max().item()
    print(f"encode pre diff : {d_pre:.3e}")
    print(f"top-k val diff  : {d_act:.3e}")
    print(f"decode diff     : {d_dec:.3e}")
    # absolute floor; the values here are O(1e4) so 2e-4 is fp32 noise (rel ~1e-8)
    assert max(d_pre, d_act, d_dec) < 1e-3, "SAE CONVERSION MISMATCH"
    print("SAE CONVERSION OK")


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    for l in range(N_LAYER):
        convert_layer(l)
    with open(os.path.join(CACHE, "config.yaml"), "w") as f:
        yaml.dump(CONFIG, f)
    print(f"wrote {N_LAYER} layers + config.yaml -> {CACHE}")
    self_test()
