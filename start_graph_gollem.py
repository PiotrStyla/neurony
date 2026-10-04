#!/usr/bin/env python3
"""Launch the Neuronpedia graph server (apps/graph) for a local GoLLeM model + TopK SAE set.

Everything model-specific comes from the environment (defaults = v5 + res-v5), so one
code path serves several instances (e.g. np-graph on :5004 for v5, np-graph-v6 on :5005):

  MODEL_PATH            katalog HF wrappera / modelu (tez etykieta w np_model_to_hf.json)
  TRANSCODER_SET        nazwa zestawu SAE (np. local/gollem-res-v5)
  TRANSCODER_CACHE_DIR  katalog cache circuit-tracera dla zestawu
  N_LAYERS              liczba warstw / plikow layer_*.safetensors
  SOURCE_URLS           URL-e zrodel do metadanych grafu (przecinek)
  SECRET, DEVICE, MODEL_DTYPE, TOKEN_LIMIT, SERVER_HOST, SERVER_PORT — jak w apps/graph/start.py

Runs uvicorn in this process (unlike apps/graph/start.py, which spawns a child) so the
transcoder-loader shim below is installed in the server's own interpreter.

Shim: circuit-tracer's ``load_transcoders_from_cache`` does not forward the ``activation``/
``k`` fields from config.yaml to ``load_transcoder_set``, so a TopK set loaded through the
cache silently degrades to ReLU. Here the set is built through ``load_transcoders(config)``,
which does pass them. Falls through to the stock hub loader for every other ref.
"""
import os
import sys

GOLLEM_NP = os.path.dirname(os.path.abspath(__file__))
APPS_GRAPH = os.environ.get("APPS_GRAPH", os.path.join(os.path.dirname(GOLLEM_NP), "neuronpedia", "apps", "graph"))
sys.path.insert(0, APPS_GRAPH)  # neuronpedia_graph package lives in apps/graph (start.py runs from there)

MODEL_PATH = os.environ.get("GOLLEM_HF_MODEL_PATH", os.path.join(GOLLEM_NP, "hf_wrap")).replace("\\", "/")
LOCAL_REF = os.environ.get("TRANSCODER_SET", "local/gollem-res-v5")
CACHE_DIR = os.environ.get("TRANSCODER_CACHE_DIR",
                           os.path.join(os.path.expanduser("~"), ".cache", "circuit_tracer", "local", "gollem-res-v5"))
N_LAYERS = int(os.environ.get("N_LAYERS", "16"))
SOURCE_URLS = os.environ.get("SOURCE_URLS", "").split(",") if os.environ.get("SOURCE_URLS") else [
    "http://localhost:3000/gollem-v5-128m-muon-v1/res-v5",
    "https://huggingface.co/PiotrSty/gollem-v5-128m-sae-res-v5",
]

os.environ.setdefault("ATTRIBUTION_ENGINE", "circuit-tracer")
os.environ.setdefault("MODEL_ENGINE", "interp_engine")
os.environ.setdefault("MODEL_ID", MODEL_PATH)
os.environ.setdefault("TRANSCODER_SET", LOCAL_REF)
os.environ.setdefault("SECRET", "localhost-secret")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("MODEL_DTYPE", "float32")
os.environ.setdefault("TOKEN_LIMIT", "64")
os.environ.setdefault("MAX_FEATURE_NODES", "10000")
os.environ.setdefault("UPDATE_INTERVAL", "1000")
os.environ.setdefault("SERVER_HOST", "127.0.0.1")
os.environ.setdefault("SERVER_PORT", "5004")


def install_shim() -> None:
    import yaml

    import circuit_tracer.utils.hf_utils as hf_utils

    original = hf_utils.load_transcoder_from_hub

    def load_transcoder_from_hub(hf_ref, device=None, dtype=None, **kwargs):
        if hf_ref != LOCAL_REF:
            return original(hf_ref, device=device, dtype=dtype, **kwargs)
        with open(os.path.join(CACHE_DIR, "config.yaml")) as f:
            config = yaml.safe_load(f)
        config["transcoders"] = [os.path.join(CACHE_DIR, f"layer_{i}.safetensors") for i in range(N_LAYERS)]
        transcoders = hf_utils.load_transcoders(
            config,
            device=device,
            dtype=dtype,
            lazy_encoder=kwargs.get("lazy_encoder", False),
            lazy_decoder=kwargs.get("lazy_decoder", False),
        )
        return transcoders, config

    hf_utils.load_transcoder_from_hub = load_transcoder_from_hub
    import circuit_tracer.replacement_model.replacement_model_interp_engine as rmi

    rmi.load_transcoder_from_hub = load_transcoder_from_hub

    def ensure_tokenized(prompt, tokenizer, device, model_name=""):
        """Upstream z odpowiednikiem `filter(None, ...)` — specjalny token o id 0
        (np. |endoftext| w GoLLeM-v6) jest faliwy i upstream traci go przez
        next(filter(None, ...)) -> StopIteration. Tu: `is not None`."""
        import torch

        tokens = tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        tokens = torch.as_tensor(tokens).to(device).reshape(-1).long()
        specials = [i for i in ([tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id]
                                + list(tokenizer.all_special_ids)) if i is not None]
        if int(tokens[0]) not in {int(i) for i in tokenizer.all_special_ids}:
            tokens = torch.cat([torch.tensor([specials[0]], device=device), tokens])
        return tokens

    rmi.ensure_tokenized = ensure_tokenized


def main() -> None:
    install_shim()
    import neuronpedia_graph.server as graph_server

    # graph metadata "source_urls": the stock table only knows hub transcoder sets
    graph_server.TRANSCODER_SET_TO_SOURCE_URL_ARRAYS[LOCAL_REF] = SOURCE_URLS
    import uvicorn

    uvicorn.run(
        "neuronpedia_graph.server:app",
        host=os.environ["SERVER_HOST"],
        port=int(os.environ["SERVER_PORT"]),
    )


if __name__ == "__main__":
    sys.exit(main())
