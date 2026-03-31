# HISTORY.md — What Has Happened So Far

This document records every experiment run, every bug found, and the analysis of
why results dropped on Luxury_Beauty. It is meant to be a complete narrative of
the project from the original A-LLMRec baseline to the current state.

---

## 1. The Original System (A-LLMRec, Baseline)

### What it does

A-LLMRec is a two-stage LLM-based recommender system (Kim et al., KDD 2024).

**Stage 1 — CF-Text Alignment**

A pre-trained, frozen SASRec model provides item embeddings `E_i` (d_CF dim) and
user representations `x_u`. A Sentence-BERT (SBERT) model encodes item
title+description text into `Q_i` (768-dim). Two small autoencoder MLPs compress
both into a shared **128-dimensional** latent space:

```
SASRec item emb  → mlp  → 128-d   (e_i, joint CF-text embedding)
SBERT text emb   → mlp2 → 128-d   (q_i, text-only embedding)

Loss = L_matching + L_item_recon + L_text_recon + L_rec
```

After Stage 1, `e_i = mlp(E_i)` captures both collaborative filtering signal (who
buys this) and textual knowledge (what it is). This is frozen for Stage 2.

**Stage 2 — LLM Alignment and Text Generation**

Two 2-layer MLPs (`log_emb_proj`, `item_emb_proj`) project the user and item
embeddings into the LLM's token space (4096-dim for OPT-6.7B). The frozen OPT-6.7B
receives a text prompt where special tokens `[HistoryEmb]` and `[CandidateEmb]`
are replaced with the projected CF item embeddings, and the user representation is
prepended as a soft prompt. The model is trained to **generate the correct item
title** as text output.

At inference: `llm.generate()` → decode text → string-match against known titles.

### Baseline Results

| Dataset        | Branch | Hit@1 |
|----------------|--------|-------|
| All_Beauty     | master | 44%   |
| Luxury_Beauty  | master | 58%   |

The Luxury_Beauty baseline is significantly higher because OPT-6.7B has strong
pre-trained knowledge of luxury brand names (Chanel, La Mer, Estée Lauder, etc.),
which it leverages directly in text generation.

---

## 2. The ID-Prediction Extension (Branch: id-pred-1)

### Motivation

The text-generation approach has a key weakness: the LLM must generate a title
string, which is then matched via string comparison. This is fragile — any
hallucination, partial match, or casing mismatch counts as a miss. More
fundamentally, the model is trained to produce English tokens that spell a title,
not to directly rank items.

Inspired by AlmostRec (Wu et al., ICMR 2025), the idea was to replace text
generation with **ID prediction**: run the same enriched prompt through the LLM,
extract the last token's hidden state, and use that as a query to score the 20
candidates directly via their CF embeddings.

### Architecture

```
Stage 1: (unchanged)
  SASRec → mlp → 128-d   (e_i)

Stage 2 (ID-prediction):
  log_emb_proj  : d_CF → 4096    (user rep → LLM token space)
  item_emb_proj : 128  → 4096    (item CF-text emb → LLM token space)

  Frozen OPT-6.7B processes the enriched prompt
  → last_hidden = hidden_states[-1][:, -1, :]   (batch, 4096)

  Scoring:
    cand_embs = item_emb_proj(e_i)              (batch, 20, 4096)
    logits    = cosine(last_hidden, cand_embs)  (batch, 20)

  Loss = CrossEntropyLoss(logits, target_indices)
```

Only `log_emb_proj` and `item_emb_proj` receive gradient updates. The LLM is fully
frozen. No string matching at inference — just `argmax(logits)` → item ID.

---

## 3. Bugs Found and Fixed (documented in Changes-1.md)

### Bug 1 — Position-Blind Linear Head (Hit@1 = 4.81% = random)

**Original implementation:**
```python
self.id_pred_head = nn.Linear(d_model, 20)
logits = self.id_pred_head(last_hidden)   # (batch, 20)
```

**Why it failed:**
`make_candidate_text` randomly shuffles the 20 candidates for every user on every
call. The ground-truth label is the *slot index* of the positive item (e.g., 3 or
17). But `nn.Linear(d_model, 20)` produces the same 20 projections regardless of
which items are in which slots. Because the correct slot shifts randomly, the
gradient signal is pure noise. The model cannot converge and stays at random
performance (1/20 = 5%).

**Fix applied:**
Replaced with dot-product scoring against the candidates' own embeddings:
```python
cand_stack = torch.stack(candidate_embs)  # (batch, 20, 4096)
logits = torch.bmm(cand_stack, last_hidden.unsqueeze(-1)).squeeze(-1)
```
Now each logit is tied to a specific item's embedding, not a fixed slot number.
Shuffling the candidates no longer scrambles the learning signal.

### Bug 2 — Exploding Dot Products (loss oscillating 0.0 ↔ 30+)

**Why it happened:**
Raw dot products between `last_hidden` and `cand_embs` are unbounded. When the
L2 norms grow large during training, a few logits become very large, causing
CrossEntropy to produce near-zero loss (overconfident correct) or massive loss
(overconfident wrong) alternately. This gradient explosion cycle prevents
convergence.

**Fix applied:**
L2-normalize both vectors before the dot product (cosine similarity):
```python
h_norm = F.normalize(last_hidden, dim=-1)    # (batch, 4096)
c_norm = F.normalize(cand_stack,  dim=-1)    # (batch, 20, 4096)
logits = torch.bmm(c_norm, h_norm.unsqueeze(-1)).squeeze(-1)
```
Logits are now bounded in [-1, 1]. Initial loss is near log(20) ≈ 3.0 and
decreases smoothly.

---

## 4. Experiments and Results

### Experiment 1 — ID-Prediction on All_Beauty (id-pred-1, frozen LLM)

- Model: OPT-6.7B
- Dataset: All_Beauty
- Architecture: ID-prediction with fixed dot-product + cosine scoring, frozen LLM
- **Result: 51%**
- Comparison: Baseline on All_Beauty = 44%
- **Verdict: +7pp improvement over baseline. ID-prediction works here.**

### Experiment 2 — Baseline on All_Beauty (master)

- Model: OPT-6.7B
- Dataset: All_Beauty
- Architecture: Original text-generation
- **Result: 44%**

### Experiment 3 — ID-Prediction on Luxury_Beauty (id-pred-1, frozen LLM)

- Model: OPT-6.7B
- Dataset: Luxury_Beauty
- Architecture: ID-prediction with fixed dot-product + cosine scoring, frozen LLM
- **Result: 31%**
- Comparison: Baseline on Luxury_Beauty = 58%
- **Verdict: -27pp drop from baseline. ID-prediction fails here.**

### Experiment 4 — ID-Prediction + LoRA on Luxury_Beauty (id-pred-1)

- Model: OPT-6.7B + LoRA (r=8, target: q_proj/v_proj)
- Dataset: Luxury_Beauty
- Architecture: ID-prediction with LoRA fine-tuning added
- **Result: 25%**
- **Verdict: Even worse. LoRA made things worse, not better.**

### Experiment 5 — Baseline on Luxury_Beauty (master)

- Model: OPT-6.7B
- Dataset: Luxury_Beauty
- Architecture: Original text-generation, fully frozen LLM
- **Result: 58%**
- **Verdict: Very strong. This is the bar to beat.**

### Summary Table

| Exp | Dataset        | Architecture          | Hit@1 | vs Baseline |
|-----|----------------|-----------------------|-------|-------------|
| 2   | All_Beauty     | Baseline (text-gen)   | 44%   | —           |
| 1   | All_Beauty     | ID-pred (frozen)      | 51%   | +7pp        |
| 5   | Luxury_Beauty  | Baseline (text-gen)   | 58%   | —           |
| 3   | Luxury_Beauty  | ID-pred (frozen)      | 31%   | -27pp       |
| 4   | Luxury_Beauty  | ID-pred + LoRA        | 25%   | -33pp       |

---