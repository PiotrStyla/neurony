#!/usr/bin/env python3
"""Generates sft_fix_gollem_v6.ipynb (nbformat 4) with compile-checked cells."""
import json
import os

MD_INTRO = """# GoLLeM-v6 — naprawa odpowiedzi: format + wiedza (SFT)

Diagnoza z 32-promptowej baterii (PL/EN) i rankingu logitów: **wiedza w dużej części JEST w wagach, ale model wybiera tryb „encyklopedycznej kontynuacji" zamiast odpowiedzi** (Wisla rank 2, Warsaw 2, Paris 2, Jupiter 3, Mieszko 7 — a greedy generuje „rzeka,", „the city of"…). Prawdziwe luki wiedzy: Warszawa-PL, Mickiewicz, Orwell.

Ten notebook trenuje dwie fazy (mały LR, ~30–45 min na T4):

1. **Format odpowiedzi** — pary pytanie→nazwa (z AUGMENTACJĄ formatów PL/EN); CE tylko na odpowiedzi, prompt maskowany. Uczy ODPOWIADAĆ z istniejącej wiedzy (rank 2 → 1).
2. **Wiedza** — pary dla realnych luk (fakty PL/EN) + mix LM-loss na WikiText (1:1) przeciw zapominaniu.

**Pomiar**: ta sama bateria 32 promptów co baseline — greedy verdict + rank pierwszego tokenu odpowiedzi, tabela PRZED/PO. Część par jest trzymana z treningu (HOLD_OUT) jako uczciwy test generalizacji.

**Licencja**: model bazowy CC-BY-SA-4.0 → wariant pochodny: ta sama licencja + atrybucja (SlayerLab/Fabryka AI)."""

MD_1 = "## 1. Instalacja i model bazowy"

C_PIP = "!pip install -q huggingface_hub tokenizers safetensors"

C_LOAD = """import json, os, sys, time
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download

assert torch.cuda.is_available(), 'Wlacz runtime GPU: Runtime > Change runtime type > T4'
DEVICE = 'cuda'
MODEL_ID = 'SlayerLab/GoLLeM-v6-250M'
MODEL_DIR = '/content/gollem-v6'
OUT_DIR = '/content/gollem-v6-sft'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

for f in ('config.json', 'model.safetensors', 'tokenizer.json', 'modeling_gollem_v6.py'):
    hf_hub_download(MODEL_ID, f, local_dir=MODEL_DIR)
sys.path.insert(0, MODEL_DIR)
from modeling_gollem_v6 import load_gollem_v6  # oficjalny kod modelu

model, tok = load_gollem_v6(MODEL_DIR, device=DEVICE)
print('model v6 gotowy:', next(model.parameters()).shape, '| device:', DEVICE)"""

MD_2 = """## 2. Pomiar PRZED (baseline) — bateria 32 + ranking odpowiedzi

Ta sama bateria co baseline. `rank` = pozycja pierwszego tokenu poprawnej odpowiedzi w rozkładzie (1 = pewniak; <=10 = wiedza obecna, przegrywa z trybem kontynuacji; >100 = luka)."""

C_EVAL = '''BATTERY = [
    ('Stolica Polski to', 'Warszawa', 'pl-geo'),
    ('Najwyzszym szczytem Polski jest', 'Rysy', 'pl-geo'),
    ('Najdluzsza rzeka w Polsce to', 'Wisla', 'pl-geo'),
    ('Stolica Francji to', 'Paryz', 'pl-geo'),
    ('Stolica Niemiec to', 'Berlin', 'pl-geo'),
    ('Najwieksze jezioro w Polsce to', 'Sniardwy', 'pl-geo'),
    ('Autor Pana Tadeusza to', 'Mickiewicz', 'pl-lit'),
    ('Tworca Lalki to', 'Prus', 'pl-lit'),
    ('Nobla z literatury otrzymala Olga', 'Tokarczuk', 'pl-lit'),
    ('Pierwszym krolem Polski byl', 'Mieszko', 'pl-hist'),
    ('Woda zamarza w temperaturze', '0', 'pl-sci'),
    ('Predkosc swiatla w proznie wynosi', '300', 'pl-sci'),
    ('Ziemia obiega Slonce w', '365', 'pl-sci'),
    ('Symbol chemiczny zlota to', 'Au', 'pl-sci'),
    ('Liczba kontynentow na Ziemi to', '7', 'pl-sci'),
    ('The capital of Poland is', 'Warsaw', 'en-geo'),
    ('The capital of France is', 'Paris', 'en-geo'),
    ('The capital of Germany is', 'Berlin', 'en-geo'),
    ('The capital of Japan is', 'Tokyo', 'en-geo'),
    ('The capital of Italy is', 'Rome', 'en-geo'),
    ('The longest river in the world is', 'Nile', 'en-geo'),
    ('The author of Hamlet was', 'Shakespeare', 'en-lit'),
    ('The author of 1984 was', 'Orwell', 'en-lit'),
    ('The author of Harry Potter is', 'Rowling', 'en-lit'),
    ('Romeo and Juliet was written by', 'Shakespeare', 'en-lit'),
    ('Water freezes at', '0', 'en-sci'),
    ('The chemical symbol for gold is', 'Au', 'en-sci'),
    ('The largest planet in the solar system is', 'Jupiter', 'en-sci'),
    ('The number of continents on Earth is', '7', 'en-sci'),
    ('The sun rises in the', 'east', 'en-sci'),
    ('Mount Everest is the highest', 'mountain', 'en-sci'),
]

# pary trzymane z treningu (uczciwy test generalizacji na format)
HOLD_OUT = {
    'Stolica Polski to', 'Najdluzsza rzeka w Polsce to', 'Autor Pana Tadeusza to',
    'The capital of France is', 'The author of Hamlet was', 'The largest planet in the solar system is',
    'Symbol chemiczny zlota to', 'The number of continents on Earth is',
}


@torch.no_grad()
def evaluate(tag):
    rows = []
    for prompt, answer, group in BATTERY:
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], device=DEVICE)
        out = []
        for _ in range(3):
            logits = model(x)[0, -1].float()
            nxt = int(logits.argmax())
            out.append(tok.id_to_token(nxt))
            x = torch.cat([x, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        first = tok.encode(' ' + answer).ids[0]
        last_logits = model(torch.tensor([ids], device=DEVICE))[0, -1].float()
        rank = int((last_logits > last_logits[first]).sum()) + 1
        hit = answer.lower() in ''.join(out).lower().replace('Ġ', ' ')
        rows.append({'prompt': prompt, 'answer': answer, 'group': group,
                     'greedy': ''.join(out), 'hit': hit, 'rank': rank,
                     'holdout': prompt in HOLD_OUT})
    n = len(rows)
    hits = sum(r['hit'] for r in rows)
    top10 = sum(r['rank'] <= 10 for r in rows)
    print(f'== {tag}: greedy trafien {hits}/{n} | odpowiedz w top-10: {top10}/{n}')
    for r in rows:
        mark = 'ok  ' if r['hit'] else 'WRONG'
        h = 'H' if r['holdout'] else ' '
        print(f"  {mark} {h} {r['prompt'][:38]:38} expect={r['answer'][:10]:10} rank={r['rank']:<5} {r['greedy'][:28]!r}")
    return rows


before = evaluate('PRZED')'''

MD_3 = """## 3. Zbiór treningowy: pytanie → nazwa, z augmentacją formatów

Pary z baterii (poza `HOLD_OUT`) + ~40 dodatkowych faktów PL/EN + warianty formatów (pytań, uzupełnień, „X to stolica…"). Faza 1 = format (wszystkie pary), faza 2 = wiedza (rozszerzona lista faktów, z LM-mix)."""

C_DATA = '''# dodatkowe fakty (wiedza) — PL i EN
EXTRA_FACTS = [
    ('Stolica Hiszpanii to', 'Madryt'), ('Stolica Wloch to', 'Rzym'),
    ('Stolica Anglii to', 'Londyn'), ('Stolica Portugalii to', 'Lizbona'),
    ('Stolica Szwecji to', 'Sztokholm'), ('Stolica Ukrainy to', 'Kijow'),
    ('Stolica Czech to', 'Praga'), ('Stolica Grecji to', 'Ateny'),
    ('Najwyzszy szczyt swiata to', 'Everest'), ('Najdluzsza rzeka swiata to', 'Nil'),
    ('Najwiekszy ocean to', 'Spokojny'), ('Najglebsze jezioro swiata to', 'Bajkal'),
    ('Autorem Quo Vadis jest', 'Sienkiewicz'), ('Autorem Wesela jest', 'Wyspianski'),
    ('Autorem Sonetow krymskich jest', 'Mickiewicz'), ('Tworca Ferdydurke to', 'Gombrowicz'),
    ('Pierwsza stolica Polski byla', 'Gniezno'), ('Bitwa pod Grunwaldem odbyla sie w', '1410'),
    ('Rok odzyskania niepodleglosci przez Polske to', '1918'),
    ('Polski noblista z fizyki to', 'Curie'),
    ('The capital of Spain is', 'Madrid'), ('The capital of England is', 'London'),
    ('The capital of Portugal is', 'Lisbon'), ('The capital of Sweden is', 'Stockholm'),
    ('The capital of China is', 'Beijing'), ('The capital of India is', 'New Delhi'),
    ('The capital of Australia is', 'Canberra'), ('The capital of Egypt is', 'Cairo'),
    ('The tallest mountain in the world is', 'Everest'),
    ('The longest river in Africa is', 'Nile'), ('The largest ocean is', 'Pacific'),
    ('The deepest lake in the world is', 'Baikal'),
    ('The author of The Great Gatsby was', 'Fitzgerald'),
    ('The author of Pride and Prejudice was', 'Austen'),
    ('The author of The Hobbit was', 'Tolkien'), ('The author of Crime and Punishment was', 'Dostoevsky'),
    ('World War II ended in', '1945'), ('World War I started in', '1914'),
    ('The moon landing happened in', '1969'), ('The speed of sound is about', '343'),
    ('Water boils at', '100'), ('The chemical symbol for silver is', 'Ag'),
    ('The chemical symbol for iron is', 'Fe'), ('The closest planet to the sun is', 'Mercury'),
    ('The smallest planet in the solar system is', 'Mercury'),
    ('The human heart has', '4'), ('The square root of 144 is', '12'),
]

# warianty formatow: (szablon promptu, szablon odpowiedzi)
PL_TEMPLATES = [
    ('{q} {a}', None),                 # uzupelnienie: pytanie + odpowiedz
    ('Pytanie: {q} Odpowiedz: {a}', None),
    ('{q} Poprawna odpowiedz to {a}', None),
]
EN_TEMPLATES = [
    ('{q} {a}', None),
    ('Question: {q} Answer: {a}', None),
    ('{q} The correct answer is {a}', None),
]


def build_pairs(facts, holdout):
    pairs = []
    for q, a in facts:
        if q in holdout:
            continue
        templates = EN_TEMPLATES if q.startswith('The ') or q[0].isupper() and ' to' not in q[-4:] and not any(w in q for w in ('Stolica', 'rzeka', 'szczytem', 'Autorem', 'Tworca', 'krolem', 'zamarza', 'Predkosc', 'obiega', 'Symbol', 'Liczba', 'Rok', 'Bitwa', 'stolica', 'szczyt', 'ocean', 'jezioro', 'noblista')) else PL_TEMPLATES
        for prompt_t, _ in templates:
            pairs.append({'text': prompt_t.format(q=q, a=a)})
    return pairs


phase1 = build_pairs([(q, a) for q, a, _ in BATTERY] + EXTRA_FACTS, HOLD_OUT)
phase2 = build_pairs(EXTRA_FACTS, set())
print('faza 1 (format):', len(phase1), 'par | faza 2 (wiedza):', len(phase2), 'par')

# LM-mix przeciw zapominaniu: losowe akapity WikiText
from datasets import load_dataset
lm_texts = []
for row in load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train', streaming=True):
    t = row['text'].strip()
    if 200 < len(t) < 1200:
        lm_texts.append({'text': t})
    if len(lm_texts) >= 200:
        break
print('teksty LM-mix:', len(lm_texts))'''

MD_4 = "## 4. Trening: CE na odpowiedzi (prompt maskowany) + LM-mix w fazie 2"

C_TRAIN = '''LR = 1e-5            # konserwatywnie: naprawic format, nie zepsuc modelu
BATCH = 8
EPOCHS_1 = 2
EPOCHS_2 = 1
LM_MIX = 1.0         # co drugi krok fazy 2 = LM-loss na WikiText

def build_batch(items, with_answer_mask=True):
    xs, ys = [], []
    for it in items:
        if with_answer_mask:
            text = it['text']
            ans = text.rsplit(' ', 1)[-1]
            full = tok.encode(text).ids
            prompt_ids = tok.encode(text[: text.rfind(' ' + ans)]).ids
            y = [-100] * len(prompt_ids) + full[len(prompt_ids):]
            xs.append(full)
            ys.append(y[: len(full)])
        else:
            ids = tok.encode(it['text']).ids
            xs.append(ids)
            ys.append(ids[1:] + [0])
    T = max(len(a) for a in xs)
    X = torch.zeros(len(xs), T, dtype=torch.long)
    Y = torch.full((len(xs), T), -100, dtype=torch.long)
    for i, (a, b) in enumerate(zip(xs, ys)):
        X[i, : len(a)] = torch.tensor(a)
        Y[i, : len(b)] = torch.tensor(b[: len(a)])
    return X.to(DEVICE), Y.to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR)

def run_epoch(items, tag, with_mask=True):
    t0 = time.time()
    losses = []
    for i in range(0, len(items), BATCH):
        chunk = items[i: i + BATCH]
        X, Y = build_batch(chunk, with_answer_mask=with_mask)
        logits = model(X)  # oficjalny modeling_gollem_v6 zwraca tensor logitow
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                               Y[:, 1:].reshape(-1), ignore_index=-100)
        assert torch.isfinite(loss), 'nie-skonczona strata'
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    print(f'{tag}: srednia strata {sum(losses)/len(losses):.4f} ({time.time()-t0:.0f}s)')

t0 = time.time()
for ep in range(EPOCHS_1):
    run_epoch(phase1, f'faza1 epoka {ep+1}', with_mask=True)
for ep in range(EPOCHS_2):
    inter = [x for pair in zip(phase2, lm_texts * (1 + len(phase2) // max(len(lm_texts), 1))) for x in pair]
    run_epoch(inter, f'faza2 epoka {ep+1} (z LM-mix)', with_mask=True)
print(f'trening: {(time.time()-t0)/60:.1f} min')'''

MD_5 = "## 5. Pomiar PO — ta sama bateria (porównanie z baseline)"

C_EVAL2 = '''after = evaluate('PO')
print()
print('== ZMIANA (przed -> po) ==')
for b, a in zip(before, after):
    if b['hit'] != a['hit'] or abs(b['rank'] - a['rank']) > 5:
        h = 'H' if b['holdout'] else ' '
        print(f"  {h} {b['prompt'][:38]:38} rank {b['rank']:>5} -> {a['rank']:<5} greedy {b['greedy'][:18]!r} -> {a['greedy'][:18]!r}")
hb = [r for r in before if r['holdout']]; ha = [r for r in after if r['holdout']]
print(f'HOLD-OUT (generalizacja): greedy {sum(r["hit"] for r in hb)}/{len(hb)} -> {sum(r["hit"] for r in ha)}/{len(ha)} | top-10 rank {sum(r["rank"]<=10 for r in hb)}/{len(hb)} -> {sum(r["rank"]<=10 for r in ha)}/{len(ha)}')'''

MD_6 = "## 6. Zapis wariantu + publikacja (opcjonalnie)"

C_SAVE = '''import hashlib
save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
          f'{OUT_DIR}/model.safetensors')
for f in ('config.json', 'tokenizer.json', 'modeling_gollem_v6.py'):
    import shutil
    shutil.copy(f'{MODEL_DIR}/{f}', f'{OUT_DIR}/{f}')
sha = hashlib.sha256(open(f'{OUT_DIR}/model.safetensors', 'rb').read()).hexdigest()
readme = ''' + '"""' + '''
---
license: cc-by-sa-4.0
language:
- pl
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- gollem
- sft
---

# GoLLeM-v6 250M - answer-format + knowledge SFT

Dwie fazy SFT na bazie SlayerLab/GoLLeM-v6-250M: (1) format odpowiedzi
(pytanie -> nazwa, augmentacja formatow PL/EN, CE tylko na odpowiedzi),
(2) wiedza (fakty PL/EN + LM-mix na WikiText przeciw zapominaniu).
Pomiar: 32-promptowa bateria PL/EN (greedy + rank pierwszego tokenu).

model.safetensors sha256: `''' + ''' + sha + ''' + '''`
Pochodna CC-BY-SA-4.0 (model bazowy SlayerLab/Fabryka AI).
''' + '"""' + '''
open(f'{OUT_DIR}/README.md', 'w').write(readme)
print('zapisano:', sorted(os.listdir(OUT_DIR)), '| sha256:', sha[:16], '...')

# publikacja (odkomentuj):
# from huggingface_hub import login, HfApi
# login()  # token z write
# HfApi().upload_folder(folder_path=OUT_DIR, repo_id='PiotrSty/gollem-v6-250m-sft-answers', repo_type='model')'''

MD_7 = """## 7. Dalsze kroki

1. Wynik traktujemy jako **poprawę formatu** jeśli HOLD-OUT rank →≤10 rośnie bez wzrostu greedy na starych trafieniach (kontrola: Tokarczuk/east/mountain nadal ok).
2. Weryfikacja obwodów: `fault_scan class` na nowym wariancie (uwaga: SAE `res-v6` liczone na bazie — dla grafów nowego wariantu przelicz `mine_activations.py --model-dir <wariant>`).
3. Rejestracja w Neuronpedia jako `gollem-v6-250m-sft-answers` (przewodnik PDF, sekcje 3–5) — wtedy działa fault_scan i grafy na nowym wariancie."""

CELLS = [
    ("markdown", MD_INTRO),
    ("markdown", MD_1),
    ("code", C_PIP),
    ("code", C_LOAD),
    ("markdown", MD_2),
    ("code", C_EVAL),
    ("markdown", MD_3),
    ("code", C_DATA),
    ("markdown", MD_4),
    ("code", C_TRAIN),
    ("markdown", MD_5),
    ("code", C_EVAL2),
    ("markdown", MD_6),
    ("code", C_SAVE),
    ("markdown", MD_7),
]

cells = []
for kind, src in CELLS:
    if kind == "code" and not src.lstrip().startswith(("!", "%")):
        compile(src, "<cell>", "exec")
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
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_fix_gollem_v6.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {out}: {len(cells)} cells, compile OK")
