#!/usr/bin/env python3
"""Launch the Neuronpedia graph server (apps/graph) for GoLLeM-v5 + res-v5 TopK SAEs.

Runs uvicorn in this process (unlike apps/graph/start.py, which spawns a child) so the
transcoder-loader shim below is installed in the server's own interpreter.

Shim: circuit-tracer's ``load_transcoders_from_cache`` does not forward the ``activation``/
``k`` fields from config.yaml to ``load_transcoder_set``, so a TopK set loaded through the
cache silently degrades to ReLU. Here the local set is built through ``load_transcoders(config)``,
which does pass them (and resolves the layer files from explicit local paths). Falls through to
the stock hub loader for every other ref.

Model: the HF wrapper in ../gollem-np/hf_wrap (GollemV5ForCausalLM, auto_map/trust_remote_code),
parity-checked against train_gpt_ref.GPT to 3e-5 logits.
"""
import os
import sys

GOLLEM_NP = os.path.dirname(os.path.abspath(__file__))
APPS_GRAPH = os.environ.get("APPS_GRAPH", os.path.join(os.path.dirname(GOLLEM_NP), "neuronpedia", "apps", "graph"))
sys.path.insert(0, APPS_GRAPH)  # neuronpedia_graph package lives in apps/graph (start.py runs from there)
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "circuit_tracer", "local", "gollem-res-v5")
LOCAL_REF = "local/gollem-res-v5"
MODEL_PATH = os.path.join(GOLLEM_NP, "hf_wrap").replace("\\", "/")  # must match np_model_to_hf.json exactly

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

N_LAYERS = 16


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


def main() -> None:
    install_shim()
    import neuronpedia_graph.server as graph_server

    # graph metadata "source_urls": the stock table only knows hub transcoder sets
    graph_server.TRANSCODER_SET_TO_SOURCE_URL_ARRAYS[LOCAL_REF] = [
        "http://localhost:3000/gollem-v5-128m-muon-v1/res-v5",
        "https://huggingface.co/PiotrSty/gollem-v5-128m-sae-res-v5",
    ]
    import uvicorn

    uvicorn.run(
        "neuronpedia_graph.server:app",
        host=os.environ["SERVER_HOST"],
        port=int(os.environ["SERVER_PORT"]),
    )


if __name__ == "__main__":
    sys.exit(main())
