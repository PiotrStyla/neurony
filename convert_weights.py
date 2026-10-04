#!/usr/bin/env python3
"""Convert GoLLeM-v5 weights to the HF-wrapper key layout + verify forward parity.

Original keys (train_gpt_ref.GPT / published safetensors) -> wrapper keys
(GollemV5ForCausalLM, Llama/Qwen3-conventional module tree so that
interp-engine's structural discovery finds decoder_layers/self_attn/mlp/lm_head):

  tok.weight                 -> model.embed_tokens.weight
  head.weight                -> lm_head.weight            (tied; kept identical)
  lnf.weight                 -> model.norm.weight
  blocks.{i}.ln1.weight      -> model.layers.{i}.input_layernorm.weight
  blocks.{i}.ln2.weight      -> model.layers.{i}.post_attention_layernorm.weight
  blocks.{i}.qkv.{w,b}       -> model.layers.{i}.self_attn.qkv.{w,b}
  blocks.{i}.q_norm.weight   -> model.layers.{i}.self_attn.q_norm.weight
  blocks.{i}.k_norm.weight   -> model.layers.{i}.self_attn.k_norm.weight
  blocks.{i}.proj.{w,b}      -> model.layers.{i}.self_attn.o_proj.{w,b}
  blocks.{i}.mlp.gate.weight -> model.layers.{i}.mlp.gate_proj.weight
  blocks.{i}.mlp.up.weight   -> model.layers.{i}.mlp.up_proj.weight
  blocks.{i}.mlp.down.weight -> model.layers.{i}.mlp.down_proj.weight
  blocks.{i}.vr_lambda       -> model.layers.{i}.vr_lambda

Parity check: wrapper logits vs train_gpt_ref.GPT logits on random tokens (tol 2e-4).
"""
import argparse
import importlib.util
import os
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "model", "model.safetensors")
DST_DIR = os.path.join(ROOT, "hf_wrap")
VOCAB, N_LAYER, N_HEAD, D_EMBD = 12288, 16, 12, 768

# reference implementation (SlayerLab/gollem-v5-ckpts); fetched on demand, not vendored
TRAIN_GPT_REF_URL = (
    "https://huggingface.co/SlayerLab/gollem-v5-ckpts/raw/"
    "963440ce6ab4ada7e95da4c1faa28ebb00c082d4/train_gpt_ref.py"
)


def ensure_train_gpt_ref() -> str:
    import urllib.request

    path = os.path.join(ROOT, "train_gpt_ref.py")
    if not os.path.exists(path):
        urllib.request.urlretrieve(TRAIN_GPT_REF_URL, path)
        print(f"downloaded {path}")
    return path


def remap_key(key: str) -> str:
    if key == "tok.weight":
        return "model.embed_tokens.weight"
    if key == "head.weight":
        return "lm_head.weight"
    if key == "lnf.weight":
        return "model.norm.weight"
    assert key.startswith("blocks."), key
    rest = key[len("blocks.") :]
    layer, tail = rest.split(".", 1)
    tail_map = {
        "ln1.weight": "input_layernorm.weight",
        "ln2.weight": "post_attention_layernorm.weight",
        "qkv.weight": "self_attn.qkv.weight",
        "qkv.bias": "self_attn.qkv.bias",
        "q_norm.weight": "self_attn.q_norm.weight",
        "k_norm.weight": "self_attn.k_norm.weight",
        "proj.weight": "self_attn.o_proj.weight",
        "proj.bias": "self_attn.o_proj.bias",
        "mlp.gate.weight": "mlp.gate_proj.weight",
        "mlp.up.weight": "mlp.up_proj.weight",
        "mlp.down.weight": "mlp.down_proj.weight",
        "vr_lambda": "self_attn.vr_lambda",
    }
    return f"model.layers.{layer}.{tail_map[tail]}"


def convert():
    sd = load_file(SRC)
    out = {remap_key(k): v.contiguous() for k, v in sd.items()}
    os.makedirs(DST_DIR, exist_ok=True)
    save_file(out, os.path.join(DST_DIR, "model.safetensors"))
    print(f"wrote {len(out)} tensors -> {DST_DIR}/model.safetensors")
    return sd, out


def check_parity(orig_sd):
    import sys
    sys.path.insert(0, ROOT)
    from hf_wrap.modeling_gollem_v5 import GollemV5Config, GollemV5ForCausalLM

    spec = importlib.util.spec_from_file_location("tgr", ensure_train_gpt_ref())
    tgr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgr)

    cfg = SimpleNamespace(pos="rope", norm="rmsnorm", norm_eps=1e-6, ffn="swiglu",
                          ffn_mult=2.667, value_residual=True, qk_norm=True, rope_theta=100000.0)
    ref = tgr.GPT(VOCAB, N_LAYER, D_EMBD, N_HEAD, 1024, cfg)
    ref.load_state_dict(orig_sd)
    ref.eval()

    hfcfg = GollemV5Config(vocab_size=VOCAB, hidden_size=D_EMBD, num_hidden_layers=N_LAYER,
                           num_attention_heads=N_HEAD, rope_theta=100000.0,
                           rms_norm_eps=1e-6, ffn_mult=2.667, max_position_embeddings=1024)
    hf = GollemV5ForCausalLM(hfcfg)
    hf.load_state_dict(load_file(os.path.join(DST_DIR, "model.safetensors")))
    hf.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (2, 128))
    with torch.no_grad():
        ref_logits, _ = ref(ids)
        hf_logits = hf(ids).logits
    diff = (ref_logits.float() - hf_logits.float()).abs().max().item()
    print(f"max |logit diff| = {diff:.2e}")
    assert diff < 2e-4, f"PARITY FAILED: {diff}"
    print("PARITY OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()
    orig_sd, _ = convert()
    if not args.skip_parity:
        check_parity(orig_sd)
