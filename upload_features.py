#!/usr/bin/env python3
"""Build upload-batch payloads from mine_activations.py output and POST them
to the local Neuronpedia webapp (/api/feature/upload-batch).

Activation values are post-TopK: the mined max value at the peak token, 0.0 at
other context tokens (that is the SAE output there). Token strings are the
model's own BPE pieces (tokenizer.json id_to_token).
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "out")

MODEL_ID = "gollem-v5-128m-muon-v1"
N_LAYER, D_SAE, N_TOP, CTX = 16, 6144, 20, 8
BATCH_FEATURES = 128


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(ROOT, "data", "model", "tokenizer.json"))


def build_layer_features(layer, windows, id_to_token):
    z = np.load(os.path.join(OUT_DIR, f"layer{layer}.npz"))
    top_vals, top_locs, density = z["top_vals"], z["top_locs"], z["density"]
    seq = windows.shape[1]
    features = []
    for feat in range(D_SAE):
        activations = []
        for v, loc in zip(top_vals[feat], top_locs[feat]):
            if loc < 0 or not np.isfinite(v):
                continue
            win = int(loc) // seq
            pos = int(loc) % seq
            lo, hi = max(0, pos - CTX), min(seq, pos + CTX + 1)
            ids = windows[win, lo:hi]
            tokens = [id_to_token(int(i)) for i in ids]
            values = [0.0] * (hi - lo)
            values[pos - lo] = float(v)
            activations.append({"tokens": tokens, "values": values})
        activations.sort(key=lambda a: -max(a["values"]))
        if not activations:
            # dead feature in the 524k-token sample: register it with an honest marker
            activations = [{"tokens": ["<no activation in 524k-token sample>"], "values": [0.0]}]
        features.append({
            "index": feat,
            "density": float(density[feat]),
            "activations": activations,
        })
    return features


def post_batches(url, api_key, layer, features, retries=4):
    src = f"{layer}-res-v5"
    ok = 0
    for start in range(0, len(features), BATCH_FEATURES):
        chunk = features[start : start + BATCH_FEATURES]
        body = json.dumps({"modelId": MODEL_ID, "source": src, "features": chunk}).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json", "x-api-key": api_key})
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    resp = json.loads(r.read())
                ok += 1
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise
                print(f"  retry {attempt+1} after: {e}", flush=True)
                time.sleep(2 * (attempt + 1))
        if ok % 8 == 0:
            print(f"  layer {layer}: {ok}/48 batches", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:3000/api/feature/upload-batch")
    ap.add_argument("--api-key-file", default=os.path.join(ROOT, "api_key.txt"))
    ap.add_argument("--layers", default="all")
    args = ap.parse_args()

    api_key = open(args.api_key_file).read().strip().split("=")[-1]
    layers = range(N_LAYER) if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    tok = load_tokenizer()
    id_to_token = lambda i: tok.id_to_token(i)
    windows = np.load(os.path.join(OUT_DIR, "windows.npy"))

    t0 = time.time()
    for layer in layers:
        features = build_layer_features(layer, windows, id_to_token)
        n = post_batches(args.url, api_key, layer, features)
        print(f"layer {layer}: {n} batches posted ({time.time()-t0:.0f}s total)", flush=True)
    print("UPLOAD DONE")


if __name__ == "__main__":
    main()
