#!/usr/bin/env python3
"""Generates sae_training_gollem_v6.ipynb (nbformat 4) with compile-checked cells."""
import json
import os

MD_INTRO = """# GoLLeM-v6-250M → zestaw SAE (TopK, strumień resztkowy po bloku)

Trenuje **20 TopK-SAE** (po jednym na warstwę, `res_post_block`) dla [SlayerLab/GoLLeM-v6-250M](https://huggingface.co/SlayerLab/GoLLeM-v6-250M) — ten sam przepis co `res-v5` dla v5-128M:

* `f = TopK(ReLU((x - b_dec) @ W_enc + b_enc))`, `x_hat = f_sparse @ W_dec + b_dec` (k=50, niewiązane macierze, wiersze dekodera renormowane do 1 po każdym kroku, `b_dec` inicjalizowane średnią resztą)
* dane: **WikiText (EN) + Wikipedia PL** po połowie — v6 jest dwujęzyczny (2/3 PL), trening wyłącznie na angielskim zniekształciłby featury
* rozmiar SAE: 960 → **7680** (8×), 655 kroków × batch 8 × 512 tokenów ≈ 2.68M tokenów — identycznie jak v5

**Wyjście** (zgodne z `res-v5`, wchodzi prosto do stosu Neuronpedia): `res_v6_layer{L}_final.safetensors` (`W_enc` [960, 7680], `b_enc`, `W_dec` [7680, 960], `b_dec`, fp32), `sae_config.json`, `README.md` z SHA-256.

**Wymagania:** Colab GPU (T4/L4), ~2–3 h na pełny przebieg. Model ładowany jest oficjalnym `modeling_gollem_v6.py` z repo modelu (inferencja-only, `strict=True`).

**Licencje:** model CC-BY-SA-4.0 → wytrenowane SAE są bytem pochodnym — przy publikacji zachowaj atrybucję (SlayerLab/Fabryka AI) i tę samą licencję. Kod w komórkach możesz używać swobodnie (Apache-2.0, repo `neurony`)."""

MD_1 = "## 1. Instalacja i pobranie modelu"

C_PIP = "!pip install -q huggingface_hub datasets tokenizers safetensors"

C_LOAD = """import json, os, time, types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

assert torch.cuda.is_available(), 'Włącz runtime GPU: Runtime > Change runtime type > T4'
DEVICE = 'cuda'
MODEL_ID = 'SlayerLab/GoLLeM-v6-250M'
MODEL_DIR = '/content/gollem-v6'
OUT_DIR = '/content/gollem-v6-sae'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

for f in ('config.json', 'model.safetensors', 'tokenizer.json', 'modeling_gollem_v6.py'):
    hf_hub_download(MODEL_ID, f, local_dir=MODEL_DIR)

import sys
sys.path.insert(0, MODEL_DIR)
from modeling_gollem_v6 import load_gollem_v6  # oficjalny kod z repo modelu

cfg = types.SimpleNamespace(**json.load(open(f'{MODEL_DIR}/config.json')))
model, tok = load_gollem_v6(MODEL_DIR, device=DEVICE)
print(f'model: {cfg.n_layer} warstw, d_model={cfg.n_embd}, vocab={cfg.vocab} | device: {next(model.parameters()).device}')"""

MD_2 = """## 2. Parametry SAE i dane

Parametry są zgodne z przepisem `res-v5`; `DATA_MIX` ustawia proporcję EN/PL (v6 jest 2/3 polski — domyślnie 50/50 dla zrównoważonych featurów)."""

C_PARAMS = """D_MODEL = cfg.n_embd        # 960
D_SAE = 8 * D_MODEL        # 7680 (8x — jak w res-v5)
K = 50                     # TopK
SEQ = 512                  # długość okna (jak w res-v5)
BATCH = 8                  # okien na krok (jak w res-v5)
STEPS = 655                # 655 x 8 x 512 = 2.68M tokenów (jak w res-v5)
LR = 1e-3                  # Adam (jak w res-v5)
DATA_MIX = {'en': 0.5, 'pl': 0.5}
SEED = 1337
WARMUP = 24               # okna na estymację b_dec (poza treningiem)
torch.manual_seed(SEED)
n_tokens_target = (STEPS * BATCH + WARMUP) * SEQ   # trening + zapas na b_dec
print(f'cel: {n_tokens_target:,} tokenów (w tym {WARMUP} okien na b_dec) | SAE: {D_MODEL} -> {D_SAE}, k={K}')"""

C_DATA = """from datasets import load_dataset

def collect(ids_fn, budget):
    ids = []
    for add in ids_fn():
        ids.extend(add)
        if len(ids) >= budget:
            break
    return ids[:budget]

def en_docs():
    # WikiText-103 + WikiText-2 raw — ten sam korpus co dla res-v5
    for name in ('wikitext-103-raw-v1', 'wikitext-2-raw-v1'):
        for row in load_dataset('Salesforce/wikitext', name, split='train', streaming=True):
            t = row['text']
            if t.strip():
                yield tok.encode(t).ids

def pl_docs():
    # Wikipedia PL — dla dwujęzycznego v6
    for row in load_dataset('wikimedia/wikipedia', '20231101.pl', split='train', streaming=True):
        t = row.get('text', '')
        if len(t) > 200:
            yield tok.encode(t[:20000]).ids

t0 = time.time()
n_en = int(n_tokens_target * DATA_MIX['en'])
n_pl = n_tokens_target - n_en
ids_en = collect(en_docs, n_en) if n_en else []
ids_pl = collect(pl_docs, n_pl) if n_pl else []
print(f'EN: {len(ids_en):,} tok | PL: {len(ids_pl):,} tok | {time.time() - t0:.0f}s')

# mieszanie okien EN/PL (kolejność okien losowa, wewnątrz okna tekst spójny)
windows = []
for ids in (ids_en, ids_pl):
    ids = ids[:(len(ids) // SEQ) * SEQ]
    windows.append(np.asarray(ids, dtype=np.int64).reshape(-1, SEQ))
windows = np.concatenate(windows, axis=0)
rng = np.random.default_rng(SEED)
rng.shuffle(windows)
windows = torch.from_numpy(windows)
assert len(windows) >= STEPS * BATCH + 24, f'za mało danych: {len(windows)} okien < {STEPS * BATCH + 24}'
windows = windows[:STEPS * BATCH + 24]  # WARMUP okien na estymację b_dec
print('okna:', tuple(windows.shape))"""

MD_3 = """## 3. TopKSAE i zbieranie `res_post_block`

Klasa SAE jest zgodna z `topk_sae.py` z repo [`PiotrSty/gollem-v5-128m-sae-res-v5`](https://huggingface.co/PiotrSty/gollem-v5-128m-sae-res-v5). Strumień resztkowy pobieramy dokładnie jak w treningu v5: **po wyjściu każdego bloku** (po obu rezydualach)."""

C_SAE = '''class TopKSAE(nn.Module):
    # zgodne z topk_sae.py z repo res-v5 (PiotrSty)
    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.d_in, self.d_sae, self.k = d_in, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        w = torch.randn(d_sae, d_in)
        with torch.no_grad():
            w /= w.norm(dim=1, keepdim=True)
            self.W_dec.copy_(w)
            self.W_enc.copy_(w.t() * 0.2)

    @torch.no_grad()
    def renorm_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def encode(self, x):
        pre = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        vals, idx = pre.topk(self.k, dim=-1)
        return vals, idx

    def decode(self, vals, idx):
        f = torch.zeros(vals.size(0), self.d_sae, device=vals.device, dtype=vals.dtype)
        f.scatter_(-1, idx, vals)
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        vals, idx = self.encode(x)
        return self.decode(vals, idx), vals, idx


@torch.no_grad()
def resid_post(idx):
    """[n_layer, B*T, d_model] — strumień resztkowy po każdym bloku (res_post_block)."""
    x = model.tok(idx.to(DEVICE))
    v0, out = None, []
    for b in model.blocks:
        x, v0 = b(x, v0)
        out.append(x)
    return torch.stack([o.reshape(-1, D_MODEL) for o in out])'''

MD_4 = """## 4. Inicjalizacja `b_dec` średnią resztą i trening

Jeden przebieg danych: forward modelu daje residua wszystkich 20 warstw, a każda warstwa robi własny krok Adama. (Dzięki temu model liczymy raz, a nie 20 razy.)"""

C_BDEC = """saes = [TopKSAE(D_MODEL, D_SAE, K).to(DEVICE) for _ in range(cfg.n_layer)]
opts = [torch.optim.Adam(s.parameters(), lr=LR) for s in saes]

# b_dec = średni strumień resztkowy (jak w res-v5)
t0 = time.time()
mean = torch.zeros(cfg.n_layer, D_MODEL, device=DEVICE)
n_mean = 0
for i in range(0, 24, BATCH):
    rs = resid_post(windows[i:i + BATCH])
    mean += rs.mean(dim=1)
    n_mean += 1
mean /= n_mean
for l in range(cfg.n_layer):
    saes[l].b_dec.data.copy_(mean[l])
print(f'b_dec zainicjalizowane w {time.time() - t0:.0f}s')"""

C_TRAIN = """t0 = time.time()
for step in range(STEPS):
    batch = windows[24 + step * BATCH: 24 + (step + 1) * BATCH]
    with torch.no_grad():
        rs = resid_post(batch)                     # [20, 4096, 960]
    for l in range(cfg.n_layer):
        x = rs[l]
        with torch.autocast('cuda', dtype=torch.float16):
            x_hat, vals, idx = saes[l](x)
            loss = F.mse_loss(x_hat, x)
        opts[l].zero_grad(set_to_none=True)
        loss.backward()
        opts[l].step()
        saes[l].renorm_decoder()
    if (step + 1) % 50 == 0:
        dt = time.time() - t0
        print(f'step {step + 1}/{STEPS}  loss(last layer) {loss.item():.4f}  {dt:.0f}s  ETA {(STEPS - step - 1) * dt / (step + 1) / 60:.0f} min', flush=True)
print(f'trening: {(time.time() - t0) / 60:.1f} min')"""

MD_5 = "## 5. Metryki: FVU i odsetek martwych featurów\n\nOdnośnik `res-v5`: FVU 0.07–0.24, dead frac ≈ 0%."

C_EVAL = '''@torch.no_grad()
def evaluate():
    fvu_all, dead_all, hits = [], [], torch.zeros(cfg.n_layer, D_SAE, device=DEVICE)
    for i in range(0, 24, BATCH):           # okna trzymane z boku (poza treningiem)
        rs = resid_post(windows[STEPS * BATCH + i: STEPS * BATCH + i + BATCH])
        for l in range(cfg.n_layer):
            x = rs[l]
            x_hat, vals, idx = saes[l](x)
            err = (x - x_hat).pow(2).sum()
            tot = (x - x.mean(0, keepdim=True)).pow(2).sum()
            fvu_all.append((err / tot.clamp_min(1e-12)).item())
            hits.scatter_add_(1, idx, torch.ones_like(vals))
    n = len(fvu_all) // cfg.n_layer
    fvu_per_layer = [sum(fvu_all[l::cfg.n_layer]) / n for l in range(cfg.n_layer)]
    dead_per_layer = [(hits[l] == 0).float().mean().item() for l in range(cfg.n_layer)]
    return fvu_per_layer, dead_per_layer

fvus, deads = evaluate()
print(f'{"layer":>5} {"FVU":>7} {"dead_frac":>9}')
for l in range(cfg.n_layer):
    print(f'{l:>5} {fvus[l]:>7.4f} {deads[l]:>9.4f}')'''

MD_6 = "## 6. Zapis artefaktów (jak `res-v5`) + SHA-256"

C_SAVE = '''import hashlib, zipfile

sha_rows = []
for l in range(cfg.n_layer):
    s = saes[l]
    path = f'{OUT_DIR}/res_v6_layer{l}_final.safetensors'
    save_file({
        'W_enc': s.W_enc.detach().float().cpu().contiguous(),   # [d_in, d_sae] — orientacja jak w res-v5
        'b_enc': s.b_enc.detach().float().cpu().contiguous(),
        'W_dec': s.W_dec.detach().float().cpu().contiguous(),  # [d_sae, d_in]
        'b_dec': s.b_dec.detach().float().cpu().contiguous(),
    }, path)
    sha_rows.append((os.path.basename(path), hashlib.sha256(open(path, 'rb').read()).hexdigest()))

sae_config = {
    'model_id': 'gollem-v6-250m', 'hook': 'res_post_block', 'layers': cfg.n_layer,
    'd_in': D_MODEL, 'd_sae': D_SAE, 'k': K, 'lr': LR,
    'batch': BATCH, 'seq': SEQ, 'step': STEPS, 'tokens': STEPS * BATCH * SEQ,
}
json.dump(sae_config, open(f'{OUT_DIR}/sae_config.json', 'w'), indent=1)

lines = [
    '# GoLLeM-v6-250M SAE set - residual stream (res_v6)\\n',
    f'TopK SAEs on the post-block residual stream of {MODEL_ID}, one per layer, {cfg.n_layer} layers.\\n',
    f'd_in {D_MODEL} -> d_sae {D_SAE} (8x), k = {K}. Data: WikiText + Polish Wikipedia '
    f'({int(DATA_MIX["en"] * 100)}/{int(DATA_MIX["pl"] * 100)}).\\n',
    f'Training: {STEPS} steps x batch {BATCH} x seq {SEQ} = {STEPS * BATCH * SEQ:,} tokens, '
    f'Adam lr {LR}, decoder renorm per step, b_dec = mean residual.\\n',
    '\\n## Metrics\\n', '| layer | FVU | dead frac |\\n|---|---:|---:|\\n',
]
lines += [f'| {l} | {fvus[l]:.4f} | {deads[l]:.4f} |\\n' for l in range(cfg.n_layer)]
lines += ['\\n## Files\\n| file | sha256 |\\n|---|---|\\n']
lines += [f'| {n} | `{h}` |\\n' for n, h in sha_rows]
open(f'{OUT_DIR}/README.md', 'w').writelines(lines)

with zipfile.ZipFile('/content/gollem-v6-sae.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT_DIR)):
        z.write(f'{OUT_DIR}/{f}', f)
print('gotowe:', sorted(os.listdir(OUT_DIR)))
print('paczka: /content/gollem-v6-sae.zip (File > Download)')'''

MD_7 = """## 7. (Opcjonalnie) publikacja na Hugging Face

Odkomentuj, jeśli chcesz opublikować zestaw jak `PiotrSty/gollem-v5-128m-sae-res-v5`. Pamiętaj o licencji modelu (CC-BY-SA-4.0) — opis w README paczki jest już przygotowany."""

C_PUSH = """# from huggingface_hub import login, HfApi
# login()  # token z write
# HfApi().upload_folder(folder_path=OUT_DIR, repo_id='TwojLogin/gollem-v6-250m-sae-res-v6', repo_type='model')"""

MD_8 = """## 8. Dalsze kroki w stosie Neuronpedia (repo `neurony`)

1. Zmień stałe w `make_sae_cache.py` na `D_MODEL=960, D_SAE=7680, K=50, N_LAYER=20` i wygeneruj cache circuit-tracera (self-test sprawdzi dokładność konwersji).
2. Zarejestruj model i source set `res-v6` w webappie (przewodnik PDF, sekcje 3–5) i wgraj featury przez `upload_features.py` (najpierw `mine_activations.py` — analogicznie, z `modeling_gollem_v6.py` zamiast `hf_wrap`).
3. Grafy atrybucji: `start_graph_gollem.py` z modelem v6 — ten sam schemat, nowy `TRANSCODER_SET`.

*Szczegóły: `GoLLeM-Neuronpedia-przewodnik.pdf` oraz `github.com/PiotrStyla/neurony` (NOTICE — licencje i atrybucje).*"""

CELLS = [
    ("markdown", MD_INTRO),
    ("markdown", MD_1),
    ("code", C_PIP),
    ("code", C_LOAD),
    ("markdown", MD_2),
    ("code", C_PARAMS),
    ("code", C_DATA),
    ("markdown", MD_3),
    ("code", C_SAE),
    ("markdown", MD_4),
    ("code", C_BDEC),
    ("code", C_TRAIN),
    ("markdown", MD_5),
    ("code", C_EVAL),
    ("markdown", MD_6),
    ("code", C_SAVE),
    ("markdown", MD_7),
    ("code", C_PUSH),
    ("markdown", MD_8),
]

cells = []
for kind, src in CELLS:
    if kind == "code" and not src.lstrip().startswith(("!", "%")):
        compile(src, "<cell>", "exec")  # kazda komorka musi sie kompilowac (poza magiami IPythona)
    cells.append({
        "cell_type": kind,
        "metadata": {},
        "source": src,
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    })

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sae_training_gollem_v6.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {out}: {len(cells)} cells, wszystkie komorki kodu skompilowane")
