#!/usr/bin/env python3
"""Mine top-activating examples for a GoLLeM TopK SAE set (res_post_block).

Forward follows the official model semantics (train_gpt_ref.py for v5,
modeling_gollem_v6.py for v6): post-block residuals -> TopK SAE -> per-feature
top-N max-activating windows + density. Values are post-TopK (0 outside a
token's top-k), which is the SAE's actual output.

Examples:
  # GoLLeM-v5 + res-v5 (defaults)
  python mine_activations.py
  # GoLLeM-v6 + res_v6 (20 warstw, PL/EN)
  python mine_activations.py --arch gollem_v6 --model-dir data/model_v6 \
      --sae-dir data/sae_v6 --file-template "res_v6_layer{layer}_final.safetensors" \
      --layers 20 --d-model 960 --n-head 15 --vocab 32768 --d-sae 7680 \
      --mix-pl 0.5 --out out_v6

Modes: --mode smoke (4 batche, bez zapisu) | mine (pelny przebieg -> out/)
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="top-activating examples dla zestawu TopK SAE")
    p.add_argument("--arch", choices=["gollem_v5", "gollem_v6"], default="gollem_v5")
    p.add_argument("--model-dir", default=os.path.join(ROOT, "data", "model"))
    p.add_argument("--sae-dir", default=os.path.join(ROOT, "data", "sae"))
    p.add_argument("--file-template", default="res_v5_layer{layer}_final.safetensors")
    p.add_argument("--out", default=os.path.join(ROOT, "out"))
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--vocab", type=int, default=12288)
    p.add_argument("--d-sae", type=int, default=6144)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--n-top", type=int, default=20)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--ctx", type=int, default=8)
    p.add_argument("--windows", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--mix-pl", type=float, default=0.0,
                   help="udzial Wikipedii PL w korpusie (v6 jest dwujezyczny: 0.5)")
    p.add_argument("--mode", choices=["mine", "smoke"], default="mine")
    return p.parse_args()


A = parse_args()


def load_model():
    """Zwraca (model, tokenizer) — model ma .tok, .blocks i forward(idx) -> logits."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(A.model_dir, "tokenizer.json"))
    if A.arch == "gollem_v6":
        sys.path.insert(0, A.model_dir)
        from modeling_gollem_v6 import load_gollem_v6

        model, _ = load_gollem_v6(A.model_dir, device="cpu")
    else:
        from safetensors.torch import load_file

        spec = importlib.util.spec_from_file_location("tgr", os.path.join(ROOT, "train_gpt_ref.py"))
        tgr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tgr)
        cfg = SimpleNamespace(pos="rope", norm="rmsnorm", norm_eps=1e-6, ffn="swiglu",
                              ffn_mult=2.667, value_residual=True, qk_norm=True,
                              rope_theta=100000.0)
        model = tgr.GPT(A.vocab, A.layers, A.d_model, A.n_head, 1024, cfg)
        model.load_state_dict(load_file(os.path.join(A.model_dir, "model.safetensors")))
        model.eval()
    return model, tok


def load_saes():
    from safetensors.torch import load_file

    saes = []
    for l in range(A.layers):
        path = os.path.join(A.sae_dir, A.file_template.format(layer=l))
        saes.append({k: v.float() for k, v in load_file(path).items()})
    return saes


def forward_resid(model, idx):
    """Lista residuow po blokach [B, T, d_model] (res_post_block)."""
    x = model.tok(idx)
    v0, out = None, []
    for b in model.blocks:
        x, v0 = b(x, v0)
        out.append(x)
    return out


def merge_top(vals, locs, f, v, p, n_top):
    """Merge new hits (f, v, p) into per-feature top-N tables (numpy)."""
    F = vals.shape[0]
    cf = np.concatenate([np.repeat(np.arange(F, dtype=np.int64), n_top), f])
    cv = np.concatenate([vals.ravel(), v])
    cp = np.concatenate([locs.ravel(), p])
    order = np.lexsort((-cv, cf))
    cf, cv, cp = cf[order], cv[order], cp[order]
    uniq, start, counts = np.unique(cf, return_index=True, return_counts=True)
    rank = np.arange(len(cf)) - np.repeat(start, counts)
    sel = rank < n_top
    vals[:] = cv[sel].reshape(F, n_top)
    locs[:] = cp[sel].reshape(F, n_top)


def corpus_windows(n_windows, tok, seq):
    from datasets import load_dataset

    def collect(docs, budget):
        ids = []
        for add in docs():
            ids.extend(add)
            if len(ids) >= budget:
                break
        return ids[:budget]

    def en_docs():
        for name in ("wikitext-103-raw-v1", "wikitext-2-raw-v1"):
            for row in load_dataset("Salesforce/wikitext", name, split="train", streaming=True):
                t = row["text"]
                if t.strip():
                    yield tok.encode(t).ids

    def pl_docs():
        for row in load_dataset("wikimedia/wikipedia", "20231101.pl", split="train", streaming=True):
            t = row.get("text", "")
            if len(t) > 200:
                yield tok.encode(t[:20000]).ids

    target = n_windows * seq
    n_pl = int(target * A.mix_pl)
    parts = []
    ids = collect(en_docs, target - n_pl)
    if ids:
        parts.append(np.asarray(ids[: (len(ids) // seq) * seq], dtype=np.int64).reshape(-1, seq))
    if n_pl:
        ids = collect(pl_docs, n_pl)
        if ids:
            parts.append(np.asarray(ids[: (len(ids) // seq) * seq], dtype=np.int64).reshape(-1, seq))
    windows = np.concatenate(parts, axis=0)
    rng = np.random.default_rng(1337)
    rng.shuffle(windows)
    assert len(windows) >= n_windows, f"za malo okien: {len(windows)} < {n_windows}"
    return torch.from_numpy(windows[:n_windows])


def validate(model, tok):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "".join(r["text"] for r in ds)
    ids = tok.encode(text).ids
    ctx, burn = 1024, 32
    nll, ntok, nbytes = 0.0, 0, 0
    with torch.no_grad():
        for s in range(0, len(ids) - ctx - 1, ctx):
            w = ids[s: s + ctx]
            out = model(torch.tensor([w[:-1]]))
            logits = out[0] if isinstance(out, tuple) else out
            lp = torch.log_softmax(logits.float(), dim=-1)
            tgt = torch.tensor(w[1:])
            nll += -lp[0, burn:].gather(1, tgt[burn:, None]).sum().item()
            ntok += ctx - 1 - burn
            nbytes += len(tok.decode(w[burn + 1:]).encode("utf-8"))
    print(f"token ppl: {np.exp(nll / ntok):.4f}")
    print(f"byte  ppl: {np.exp(nll / nbytes):.4f}")


def main():
    torch.set_num_threads(os.cpu_count())
    t0 = time.time()
    model, tok = load_model()
    print(f"model {A.arch} w {time.time() - t0:.1f}s")
    saes = load_saes()
    print(f"SAE: {A.layers} warstw, {A.d_model} -> {A.d_sae}, k={A.k}")

    n_batches = 4 if A.mode == "smoke" else A.windows // A.batch
    n_windows = n_batches * A.batch
    windows = corpus_windows(n_windows, tok, A.seq)
    print(f"korpus: {n_windows} okien ({n_windows * A.seq:,} tokenow)")

    top_vals = np.full((A.layers, A.d_sae, A.n_top), -np.inf, dtype=np.float32)
    top_locs = np.full((A.layers, A.d_sae, A.n_top), -1, dtype=np.int64)
    hit_counts = np.zeros((A.layers, A.d_sae), dtype=np.int64)

    t_start = time.time()
    with torch.no_grad():
        for bi in range(n_batches):
            bw = windows[bi * A.batch: (bi + 1) * A.batch]
            idx = bw.to(torch.int64)
            res = forward_resid(model, idx)
            row_locs = (np.arange(bi * A.batch, (bi + 1) * A.batch)[:, None] * A.seq
                        + np.arange(A.seq)[None, :]).reshape(-1)
            for l in range(A.layers):
                x = res[l].reshape(-1, A.d_model)
                pre = torch.relu((x - saes[l]["b_dec"]) @ saes[l]["W_enc"] + saes[l]["b_enc"])
                vals, fidx = pre.topk(A.k, dim=-1)
                vals, fidx = vals.numpy(), fidx.numpy()
                rows, cols = np.nonzero(vals > 0)
                f = fidx[rows, cols].astype(np.int64)
                v = vals[rows, cols]
                p = row_locs[rows]
                merge_top(top_vals[l], top_locs[l], f, v.astype(np.float32), p, A.n_top)
                hit_counts[l] += np.bincount(f, minlength=A.d_sae)
            if bi == 0:
                logits = model(idx[:, :64])
                logits = logits[0] if isinstance(logits, tuple) else logits
                loss = torch.nn.functional.cross_entropy(
                    logits[0, :-1].float(), idx[0, 1:64])
                print(f"sanity LM loss: {loss.item():.4f}")
            if (bi + 1) % 4 == 0 or A.mode == "smoke":
                dt = time.time() - t_start
                eta = (n_batches - bi - 1) * dt / (bi + 1) / 60
                print(f"batch {bi + 1}/{n_batches}  {dt:.0f}s  ETA {eta:.0f} min", flush=True)

    total = n_windows * A.seq
    density = hit_counts.astype(np.float32) / total
    covered = (top_locs >= 0).sum(axis=2)
    print(f"featury z >=1 przykladem: {(covered > 0).sum()} / {A.layers * A.d_sae}")
    print(f"featury z pelnym {A.n_top}: {(covered == A.n_top).sum()}")

    if A.mode == "smoke":
        l = 0
        feat = int(np.argmax(top_vals[l, :, 0]))
        print(f"smoke: L{l} F{feat} max={top_vals[l, feat, 0]:.3f}")
        return

    os.makedirs(A.out, exist_ok=True)
    for l in range(A.layers):
        np.savez(os.path.join(A.out, f"layer{l}.npz"),
                 top_vals=top_vals[l], top_locs=top_locs[l],
                 density=density[l], hit_counts=hit_counts[l])
    np.save(os.path.join(A.out, "windows.npy"), windows)
    with open(os.path.join(A.out, "meta.json"), "w") as f:
        json.dump({"arch": A.arch, "tokens": total, "windows": n_windows, "seq": A.seq,
                   "ctx": A.ctx, "topk": A.k, "n_top": A.n_top,
                   "d_sae": A.d_sae, "layers": A.layers, "mix_pl": A.mix_pl,
                   "seconds": time.time() - t_start}, f, indent=2)
    print(f"napisano {A.out}")


if __name__ == "__main__":
    if not os.path.isdir(A.sae_dir):
        sys.exit(f"brak katalogu SAE: {A.sae_dir}")
    main()
