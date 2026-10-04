#!/usr/bin/env python3
"""Automated model-fault investigation on the Neuronpedia stack.

The loop this replaces by hand: probe a model for wrong answers, generate an
attribution graph per error, extract the top driving features, characterize them
from their top-activating examples, and check whether the same features recur
across a whole error class.

Modes (all against a running webapp + graph server):
  probe  --cases cases.json            greedy-complete each prompt, flag mismatches
  explain --prompt "..."               graph for the prompt + top drivers of the
                                       greedy token + the drivers' top activations
  class  --cases cases.json            explain every case, report features shared
                                       across the "error" group but absent from
                                       the "control" group

cases.json format:
  [{"prompt": "The capital of Japan is", "expect": "Tokyo", "group": "error"}, ...]

Graph JSON is read from the S3 shim directory (server-local) or over HTTP
(S3_PUBLIC_URL). Everything else goes through the webapp API.
"""
import argparse
import glob
import json
import os
import time
import urllib.request

WEBAPP = os.environ.get("WEBAPP_URL", "http://127.0.0.1:3000")
MODEL_ID = "gollem-v5-128m-muon-v1"
API_KEY = os.environ.get("NP_API_KEY") or open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")).read().strip().split("=")[-1]
S3_DATA = os.environ.get("S3_DATA", "/opt/gollem-np/s3_data/neuronpedia-attrib/user-graphs")
HF_WRAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_wrap")


def api(path, body=None, method="POST"):
    req = urllib.request.Request(
        WEBAPP + path,
        data=None if body is None else json.dumps(body).encode(),
        method="GET" if body is None else method,
        headers={"Content-Type": "application/json", "x-api-key": API_KEY},
    )
    return json.load(urllib.request.urlopen(req, timeout=600))


def greedy(prompt: str, n_tokens: int = 3):
    """Greedy completion with the HF wrapper; returns (token_ids, token_strings)."""
    import torch
    import sys

    sys.path.insert(0, os.path.dirname(HF_WRAP))
    from hf_wrap.configuration_gollem_v5 import GollemV5Config
    from hf_wrap.modeling_gollem_v5 import GollemV5ForCausalLM
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(HF_WRAP, "tokenizer.json"))
    model = GollemV5ForCausalLM(GollemV5Config())
    model.load_state_dict(load_file(os.path.join(HF_WRAP, "model.safetensors")))
    model.eval()
    ids = tok.encode(prompt).ids
    out_ids = []
    with torch.no_grad():
        x = torch.tensor([ids])
        for _ in range(n_tokens):
            nxt = int(model(x).logits[0, -1].argmax())
            out_ids.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    return out_ids, [tok.id_to_token(i) for i in out_ids]


def graph_json_for(slug: str) -> dict:
    paths = sorted(glob.glob(os.path.join(S3_DATA, f"*{slug}*.json")))
    if not paths:
        raise FileNotFoundError(f"no graph JSON for slug {slug} under {S3_DATA}")
    return json.load(open(paths[-1]))


def explain(prompt: str, top_n: int = 5):
    t0 = time.time()
    gen = api("/api/graph/generate", {"prompt": prompt, "modelId": MODEL_ID})
    slug = gen["url"].split("slug=")[-1]
    g = graph_json_for(slug)
    nodes = {n["node_id"]: n for n in g["nodes"]}
    targets = [n for n in g["nodes"] if n["is_target_logit"]]
    if not targets:
        print(f"== {prompt!r}: no target logit in graph ==")
        return {}
    target = max(targets, key=lambda n: n["token_prob"])
    target_tok = target["feature"]
    drivers = sorted(
        ((l["weight"], l["source"]) for l in g["links"] if l["target"] == target["node_id"]),
        reverse=True,
    )
    print(f"\n== {prompt!r} -> greedy token id {target_tok} (p={target['token_prob']:.3f}), "
          f"{gen['numNodes']} nodes, {time.time() - t0:.0f}s ==")
    picked = {}
    for w, src in drivers[:top_n]:
        n = nodes[src]
        if n["feature_type"] == "logit":
            continue
        layer, feat = src.split("_")[0], src.split("_")[1]
        picked[f"{layer}/{feat}"] = w
        detail = api(f"/api/feature/{MODEL_ID}/{layer}-res-v5/{feat}")
        acts = sorted(detail.get("activations", []), key=lambda a: -a.get("maxValue", 0))[:3]
        print(f"  {w:+7.3f}  {layer}-res-v5/{feat:<5} density={detail.get('frac_nonzero', 0):.3f}")
        for a in acts:
            toks, vals = a["tokens"], a["values"]
            i = max(range(len(vals)), key=lambda j: vals[j])
            print(f"            ...{''.join(toks[max(0, i - 3):i + 4])!r}...")
    return picked


def probe(cases):
    for c in cases:
        ids, toks = greedy(c["prompt"])
        got = "".join(toks)
        ok = c["expect"].lower() in got.lower()
        print(f"{'ok  ' if ok else 'WRONG'} {c['prompt']!r:50} expect={c['expect']!r:12} greedy={got!r}")


def error_class(cases):
    by_group = {"error": {}, "control": {}}
    for c in cases:
        feats = explain(c["prompt"], top_n=5) or {}
        for f, w in feats.items():
            by_group.get(c.get("group", "error"), {})[f] = max(by_group.get(c.get("group", "error"), {}).get(f, 0), w)
    err, ctl = by_group["error"], by_group["control"]
    shared = sorted((set(err) & set(ctl)), key=lambda f: -err[f])
    err_only = sorted((set(err) - set(ctl)), key=lambda f: -err[f])
    print("\n== shared between error and control prompts ==")
    for f in shared:
        print(f"  {f}  (err weight {err[f]:+.3f}, ctl weight {ctl[f]:+.3f})")
    print("== features driving ERROR prompts only ==")
    for f in err_only:
        print(f"  {f}  (err weight {err[f]:+.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "explain", "class"])
    ap.add_argument("--cases")
    ap.add_argument("--prompt")
    args = ap.parse_args()
    if args.mode == "probe":
        probe(json.load(open(args.cases)))
    elif args.mode == "explain":
        explain(args.prompt)
    else:
        error_class(json.load(open(args.cases)))
