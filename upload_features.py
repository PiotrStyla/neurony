#!/usr/bin/env python3
"""Upload mined features/activations to the Neuronpedia webapp (/api/feature/upload-batch).

Works for any model/source set, e.g.:
  python upload_features.py --model-id gollem-v5-128m-muon-v1 --set-name res-v5 \
      --layers 16 --d-sae 6144 --out out
  python upload_features.py --model-id gollem-v6-250m --set-name res-v6 \
      --layers 20 --d-sae 7680 --out out_v6
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
BATCH_FEATURES = 128


def parse_args():
    p = argparse.ArgumentParser(description="wgrowanie featurow/aktywacji do Neuronpedia")
    p.add_argument("--model-id", default="gollem-v5-128m-muon-v1")
    p.add_argument("--set-name", default="res-v5")
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--d-sae", type=int, default=6144)
    p.add_argument("--n-top", type=int, default=20)
    p.add_argument("--ctx", type=int, default=8)
    p.add_argument("--out", default=os.path.join(ROOT, "out"))
    p.add_argument("--url", default="http://127.0.0.1:3000/api/feature/upload-batch")
    p.add_argument("--api-key-file", default=os.path.join(ROOT, "api_key.txt"))
    p.add_argument("--layer-list", default="all", help='"all" albo np. "0,1,2"')
    return p.parse_args()


def build_layer_features(a, layer, windows, id_to_token):
    z = np.load(os.path.join(a.out, f"layer{layer}.npz"))
    top_vals, top_locs, density = z["top_vals"], z["top_locs"], z["density"]
    seq = windows.shape[1]
    features = []
    for feat in range(a.d_sae):
        activations = []
        for v, loc in zip(top_vals[feat], top_locs[feat]):
            if loc < 0 or not np.isfinite(v):
                continue
            win, pos = int(loc) // seq, int(loc) % seq
            lo, hi = max(0, pos - a.ctx), min(seq, pos + a.ctx + 1)
            tokens = [id_to_token(int(i)) for i in windows[win, lo:hi]]
            values = [0.0] * (hi - lo)
            values[pos - lo] = float(v)
            activations.append({"tokens": tokens, "values": values})
        activations.sort(key=lambda ac: -max(ac["values"]))
        if not activations:
            # martwy featr w probce: uczciwy znacznik zamiast pustych tokenow
            activations = [{"tokens": ["<no activation in sample>"], "values": [0.0]}]
        features.append({"index": feat, "density": float(density[feat]), "activations": activations})
    return features


def post_batches(a, api_key, layer, features):
    src = f"{layer}-{a.set_name}"
    ok = 0
    for start in range(0, len(features), BATCH_FEATURES):
        chunk = features[start: start + BATCH_FEATURES]
        body = json.dumps({"modelId": a.model_id, "source": src, "features": chunk}).encode()
        req = urllib.request.Request(a.url, data=body, method="POST", headers={
            "Content-Type": "application/json", "x-api-key": api_key})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    r.read()
                ok += 1
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"  retry {attempt + 1} po: {e}", flush=True)
                time.sleep(2 * (attempt + 1))
    return ok


def main():
    a = parse_args()
    api_key = open(a.api_key_file).read().strip().split("=")[-1]
    layers = (range(a.layers) if a.layer_list == "all"
              else [int(x) for x in a.layer_list.split(",")])

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "model", "tokenizer.json")
                              if a.set_name == "res-v5"
                              else os.path.join(ROOT, "data", "model_v6", "tokenizer.json"))
    windows = np.load(os.path.join(a.out, "windows.npy"))

    t0 = time.time()
    for layer in layers:
        features = build_layer_features(a, layer, windows, tok.id_to_token)
        n = post_batches(a, api_key, layer, features)
        print(f"layer {layer}: {n} batchy ({time.time() - t0:.0f}s total)", flush=True)
    print("UPLOAD DONE")


if __name__ == "__main__":
    main()
