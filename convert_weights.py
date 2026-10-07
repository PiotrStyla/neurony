#!/usr/bin/env python3
"""Convert GoLLeM weights to the HF-wrapper key layout + verify forward parity.

Key remap (identical for v5/v6 — both checkpoints use the training-script names) ->
Llama/Qwen3-conventional tree (interp-engine's structural discovery):

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
  blocks.{i}.vr_lambda       -> model.layers.{i}.self_attn.vr_lambda

Parity (2e-4 na logitach) z referencją architektury:
  gollem_v5 -> train_gpt_ref.py (SlayerLab/gollem-v5-ckpts, Apache-2.0; pobierany na żądanie)
  gollem_v6 -> modeling_gollem_v6.py z repo modelu (oficjalny kod)

Usage: python convert_weights.py [--arch gollem_v5|gollem_v6] [--skip-parity]
"""
import argparse
import importlib.util
import json
import os
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.abspath(__file__))

ARCH = {
    "gollem_v5": {
        "src": os.path.join(ROOT, "data", "model", "model.safetensors"),
        "dst": os.path.join(ROOT, "hf_wrap"),
        "wrapper": ("hf_wrap.modeling_gollem_v5", "GollemV5Config", "GollemV5ForCausalLM"),
        "dims": (12288, 16, 12, 768),  # vocab, layers, heads, d_model
    },
    "gollem_v6": {
        "src": os.path.join(ROOT, "data", "model_v6", "model.safetensors"),
        "dst": os.path.join(ROOT, "hf_wrap_gollem_v6"),
        "wrapper": ("hf_wrap_gollem_v6.modeling_gollem_v6", "GollemV6Config", "GollemV6ForCausalLM"),
        "dims": (32768, 20, 15, 960),
    },
}

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
    rest = key[len("blocks."):]
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


def convert(a):
    sd = load_file(a["src"])
    out = {remap_key(k): v.contiguous() for k, v in sd.items()}
    os.makedirs(a["dst"], exist_ok=True)
    save_file(out, os.path.join(a["dst"], "model.safetensors"))
    print(f"wrote {len(out)} tensors -> {a['dst']}/model.safetensors")
    return sd


def load_reference(arch: str, orig_sd):
    """Oficjalna referencja architektury: (model, czy zwraca krotke)."""
    if arch == "gollem_v6":
        sys_path = os.path.join(ROOT, "data", "model_v6")
        spec = importlib.util.spec_from_file_location("mgv6", os.path.join(sys_path, "modeling_gollem_v6.py"))
        mgv6 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mgv6)
        cfg = SimpleNamespace(**json.load(open(os.path.join(sys_path, "config.json"))))
        ref = mgv6.GPT(cfg.vocab, cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.block, cfg)
        ref.load_state_dict(orig_sd)
        ref.eval()
        return ref, False
    spec = importlib.util.spec_from_file_location("tgr", ensure_train_gpt_ref())
    tgr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgr)
    cfg = SimpleNamespace(pos="rope", norm="rmsnorm", norm_eps=1e-6, ffn="swiglu",
                          ffn_mult=2.667, value_residual=True, qk_norm=True, rope_theta=100000.0)
    vocab, layers, heads, d_model = ARCH["gollem_v5"]["dims"]
    ref = tgr.GPT(vocab, layers, d_model, heads, 1024, cfg)
    ref.load_state_dict(orig_sd)
    ref.eval()
    return ref, True


def check_parity(arch: str, orig_sd):
    import sys

    sys.path.insert(0, ROOT)
    a = ARCH[arch]
    vocab, layers, heads, d_model = a["dims"]
    mod_name, cfg_cls, lm_cls = a["wrapper"]
    mod = importlib.import_module(mod_name)
    WrapperCfg, WrapperLM = getattr(mod, cfg_cls), getattr(mod, lm_cls)

    ref, ref_returns_tuple = load_reference(arch, orig_sd)
    hf = WrapperLM(WrapperCfg(vocab_size=vocab, hidden_size=d_model, num_hidden_layers=layers,
                              num_attention_heads=heads))
    hf.load_state_dict(load_file(os.path.join(a["dst"], "model.safetensors")))
    hf.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (2, 128))
    with torch.no_grad():
        ref_out = ref(ids)
        ref_logits = ref_out[0] if ref_returns_tuple else ref_out
        hf_logits = hf(ids).logits
    diff = (ref_logits.float() - hf_logits.float()).abs().max().item()
    print(f"max |logit diff| = {diff:.2e}")
    assert diff < 2e-4, f"PARITY FAILED: {diff}"
    print("PARITY OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCH), default="gollem_v5")
    ap.add_argument("--src", help="nadhodzacy plik model.safetensors (domyslnie z ARCH)")
    ap.add_argument("--dst", help="katalog wrapperu docelowego (domyslnie z ARCH)")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()
    arch = dict(ARCH[args.arch])
    if args.src:
        arch["src"] = args.src
    if args.dst:
        arch["dst"] = args.dst
    orig_sd = convert(arch)
    if not args.skip_parity:
        check_parity(args.arch, orig_sd)
