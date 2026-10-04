#!/usr/bin/env python3
"""Mine top-activating examples for PiotrSty/gollem-v5-128m-sae-res-v5.

Forward pass follows train_gpt_ref.py (SlayerLab/gollem-v5-ckpts) exactly:
res_post_block after each of 16 blocks -> TopK SAE (k=50) -> per-feature top-20
max-activating windows + density. Values are post-TopK (0 outside a token's
top-50), which is the SAE's actual output.

Modes:
  --validate   byte-normalized perplexity on WikiText-2 test (model card: 2.2717)
  --smoke      4 batches, timing + sanity, no output files
  (default)    full mining run -> out/layer{L}.npz + out/windows.npy
"""
import argparse
import importlib.util
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "data", "model")
SAE_DIR = os.path.join(ROOT, "data", "sae")
OUT_DIR = os.path.join(ROOT, "out")

SEQ = 512          # SAE training seq len; mining windows match
CTX = 8            # context tokens each side of the max-activating token
TOPK = 50          # SAE k
N_TOP = 20         # examples kept per feature
VOCAB, N_LAYER, N_HEAD, D_EMBD, D_SAE = 12288, 16, 12, 768, 6144


def load_gpt():
    spec = importlib.util.spec_from_file_location("tgr", os.path.join(ROOT, "train_gpt_ref.py"))
    tgr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgr)
    cfg = SimpleNamespace(pos="rope", norm="rmsnorm", norm_eps=1e-6, ffn="swiglu",
                          ffn_mult=2.667, value_residual=True, qk_norm=True,
                          rope_theta=100000.0)
    model = tgr.GPT(VOCAB, N_LAYER, D_EMBD, N_HEAD, 1024, cfg)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(os.path.join(MODEL_DIR, "model.safetensors")))
    model.eval()
    return tgr, model


def load_saes():
    from safetensors.torch import load_file
    saes = []
    for l in range(N_LAYER):
        sd = load_file(os.path.join(SAE_DIR, f"res_v5_layer{l}_final.safetensors"))
        saes.append({k: v.float() for k, v in sd.items()})
    return saes


def forward_resid(model, idx):
    """Returns list of post-block residuals [B, T, 768], one per block."""
    x = model.tok(idx)
    v0 = None
    res = []
    for b in model.blocks:
        x, v0 = b(x, v0)
        res.append(x)
    return res


def sae_topk(sae, x_flat):
    pre = torch.relu((x_flat - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"])
    return pre.topk(TOPK, dim=-1)


def merge_top(vals, locs, f, v, p):
    """Merge new hits (f, v, p) into per-feature top-N_TOP tables (numpy)."""
    F = vals.shape[0]
    cf = np.concatenate([np.repeat(np.arange(F, dtype=np.int64), N_TOP), f])
    cv = np.concatenate([vals.ravel(), v])
    cp = np.concatenate([locs.ravel(), p])
    order = np.lexsort((-cv, cf))
    cf, cv, cp = cf[order], cv[order], cp[order]
    uniq, start, counts = np.unique(cf, return_index=True, return_counts=True)
    rank = np.arange(len(cf)) - np.repeat(start, counts)
    sel = rank < N_TOP
    # every feature has >= N_TOP entries (old table pads with -inf) -> exactly F*N_TOP
    vals[:] = cv[sel].reshape(F, N_TOP)
    locs[:] = cp[sel].reshape(F, N_TOP)


def corpus_windows(n_windows, tok, seq=SEQ):
    from datasets import load_dataset
    ids = []
    total = 0
    for split_name in ("wikitext-103-raw-v1", "wikitext-2-raw-v1"):
        ds = load_dataset("Salesforce/wikitext", split_name, split="train")
        for row in ds:
            text = row["text"]
            if not text.strip():
                continue
            ids.extend(tok.encode(text).ids)
            total += 1
            if len(ids) >= (n_windows + 1) * seq:
                break
        if len(ids) >= (n_windows + 1) * seq:
            break
    ids = np.asarray(ids[: n_windows * seq], dtype=np.int32)
    return ids.reshape(n_windows, seq)


def validate(model, tok):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "".join(r["text"] for r in ds)
    ids = tok.encode(text).ids
    ctx = 1024
    burn = 32
    nll, ntok, nbytes = 0.0, 0, 0
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(ids) - ctx - 1, ctx):
            window = ids[start : start + ctx]
            x = torch.tensor([window[:-1]])
            y = torch.tensor([window[1:]])
            logits, _ = model(x)
            lp = torch.log_softmax(logits.float(), dim=-1)
            tgt = y[0]
            nll_w = -lp[0, burn:, :].gather(1, tgt[burn:, None]).sum()
            n = ctx - 1 - burn
            nll += nll_w.item()
            ntok += n
            covered = tok.decode(window[burn + 1 :])
            nbytes += len(covered.encode("utf-8"))
    tok_ppl = float(np.exp(nll / ntok))
    byte_ppl = float(np.exp(nll / nbytes))
    print(f"windows done in {time.time()-t0:.1f}s")
    print(f"token ppl     : {tok_ppl:.4f}")
    print(f"byte ppl      : {byte_ppl:.4f}   (model card WikiText-2: 2.2717)")
    return byte_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "smoke", "mine"], default="mine")
    ap.add_argument("--windows", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count())
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    t0 = time.time()
    _, model = load_gpt()
    print(f"model loaded in {time.time()-t0:.1f}s")

    if args.mode == "validate":
        validate(model, tok)
        return

    t0 = time.time()
    saes = load_saes()
    print(f"SAEs loaded in {time.time()-t0:.1f}s")

    n_windows = 4 * args.batch if args.mode == "smoke" else args.windows
    t0 = time.time()
    windows = corpus_windows(n_windows, tok)
    print(f"corpus: {n_windows} windows ({n_windows*SEQ:,} tokens) tokenized in {time.time()-t0:.1f}s")

    top_vals = np.full((N_LAYER, D_SAE, N_TOP), -np.inf, dtype=np.float32)
    top_locs = np.full((N_LAYER, D_SAE, N_TOP), -1, dtype=np.int64)
    hit_counts = np.zeros((N_LAYER, D_SAE), dtype=np.int64)
    total_tokens = 0

    n_batches = n_windows // args.batch
    t_start = time.time()
    with torch.no_grad():
        for bi in range(n_batches):
            t_b = time.time()
            bw = windows[bi * args.batch : (bi + 1) * args.batch]
            idx = torch.from_numpy(bw.astype(np.int64))
            res = forward_resid(model, idx)
            win_ids = np.arange(bi * args.batch, (bi + 1) * args.batch, dtype=np.int64)
            row_locs = (win_ids[:, None] * SEQ + np.arange(SEQ)[None, :]).reshape(-1)
            for l in range(N_LAYER):
                vals, fidx = sae_topk(saes[l], res[l].reshape(-1, D_EMBD))
                vals = vals.numpy()
                fidx = fidx.numpy()
                valid = vals > 0
                rows, cols = np.nonzero(valid)
                f = fidx[rows, cols].astype(np.int64)
                v = vals[rows, cols]
                p = np.repeat(row_locs[rows], 1)
                merge_top(top_vals[l], top_locs[l], f, v.astype(np.float32), p)
                hit_counts[l] += np.bincount(f, minlength=D_SAE)
            total_tokens += args.batch * SEQ
            if bi == 0:
                logits, _ = model(idx[:, :64])
                loss = torch.nn.functional.cross_entropy(
                    logits[0, :-1].reshape(-1, VOCAB).float(), idx[0, 1:64])
                print(f"sanity LM loss (63 predicted tokens, batch 0): {loss.item():.4f}")
            if (bi + 1) % 4 == 0 or args.mode == "smoke":
                dt = time.time() - t_start
                print(f"batch {bi+1}/{n_batches}  {dt:.1f}s elapsed  "
                      f"({dt/(bi+1):.1f}s/batch, ETA {(n_batches-bi-1)*dt/(bi+1)/60:.0f} min)",
                      flush=True)
            if args.mode == "smoke" and bi == 3:
                break

    elapsed = time.time() - t_start
    n_tok_used = total_tokens
    density = hit_counts.astype(np.float32) / max(n_tok_used, 1)
    print(f"mining done in {elapsed/60:.1f} min for {n_tok_used:,} tokens")

    covered = (top_locs >= 0).sum(axis=2)
    print(f"features with >=1 example: {(covered > 0).sum()} / {N_LAYER * D_SAE}")
    print(f"features with full {N_TOP} examples: {(covered == N_TOP).sum()}")

    if args.mode == "smoke":
        l = 0
        feat = int(np.argmax(top_vals[l, :, 0]))
        v = top_vals[l, feat]
        loc = top_locs[l, feat]
        print(f"smoke top feature L{l} F{feat}: max={v[0]:.3f} locs={loc[:5].tolist()}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    for l in range(N_LAYER):
        np.savez(os.path.join(OUT_DIR, f"layer{l}.npz"),
                 top_vals=top_vals[l], top_locs=top_locs[l],
                 density=density[l], hit_counts=hit_counts[l])
    np.save(os.path.join(OUT_DIR, "windows.npy"), windows)
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({"tokens": int(n_tok_used), "windows": n_windows, "seq": SEQ,
                   "ctx": CTX, "topk": TOPK, "n_top": N_TOP,
                   "seconds": elapsed}, f, indent=2)
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
