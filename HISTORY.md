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

## 5. Why Results Dropped on Luxury_Beauty

### 5.1 The Scoring Mechanism and Its Dependency

The ID-prediction score is:
```
logits[k] = cosine(last_hidden, item_emb_proj(e_k))
```

For this to work, the LLM's `last_hidden` (output of OPT-6.7B's top transformer
layer after processing the full enriched prompt) must "point toward" the correct
candidate's CF embedding in 4096-dimensional space.

This requires `last_hidden` and the projected CF embeddings to live in compatible
regions of the same vector space — i.e., the LLM's internal representation of
"what comes next" must be alignable with the CF embedding of the correct item.

### 5.2 Why It Works on All_Beauty

All_Beauty is a smaller dataset with more generic beauty products. OPT-6.7B has
weaker pre-trained associations for generic products (e.g., "Daily Moisturizer 200ml"
is not culturally distinctive). As a result:

- `last_hidden` is more neutral and more influenced by the CF signals injected into
  the prompt via `log_emb_proj` (user rep) and `item_emb_proj` (history/candidate
  embeddings).
- The training can push `item_emb_proj` to produce embeddings in a region of
  4096-d space that `last_hidden` naturally points toward.
- Cosine scoring works: `last_hidden` aligns with the correct candidate's embedding.

### 5.3 Why It Fails on Luxury_Beauty

Luxury_Beauty has a larger dataset, meaning SASRec produces stronger, richer CF
embeddings. However, this is not what causes the failure.

The failure is caused by **OPT-6.7B's strong pre-trained knowledge of luxury brands**.
When the prompt contains luxury brand names (Chanel, La Mer, Dior, etc.), the LLM's
internal representations are dominated by its pre-training on these culturally
distinctive names. This means:

- `last_hidden` is a **text-semantic vector** — it encodes "luxury perfume/skincare
  brand associations" from OPT's pre-training, not a CF-aligned recommendation signal.
- The projected CF embeddings from `item_emb_proj(e_i)` encode **behavioral patterns**
  — who buys what together, collaborative filtering signals.
- These two are in fundamentally different regions of 4096-d space.
- Cosine similarity between them is essentially random → 31%.

This is the same phenomenon documented in the original A-LLMRec paper: TALLRec
(LoRA fine-tuned on recommendation text) underperforms the pure CF baseline
(SASRec) in warm scenarios, because the LLM's text knowledge can dominate over the
CF signal without explicit alignment.

The reason the baseline (58%) works so well on Luxury_Beauty is precisely because
it exploits OPT's luxury brand knowledge directly — the LLM generates the correct
title because it "knows" luxury brands from pre-training. ID-prediction cannot
exploit this same advantage.

### 5.4 Why LoRA Made It Even Worse

LoRA was added to Experiment 4 motivated by AlmostRec (Wu et al., ICMR 2025), which
uses LoRA successfully in a multimodal recommendation framework.

However, LoRA made things worse for three reasons:

**Reason 1 — It destroys the advantage that made the baseline work.**
The 58% baseline relied on OPT's intact pre-trained luxury brand knowledge. LoRA
modifies OPT's attention layers (`q_proj`, `v_proj`) during training. On a dataset
where the LLM's pre-trained text knowledge is the strongest signal, this is
destructive. You lose the pre-trained advantage without gaining CF alignment.

**Reason 2 — The training signal is too indirect for LoRA to converge.**
LoRA is trying to reshape how OPT-6.7B's 32 transformer layers attend to tokens,
guided only by a cosine cross-entropy loss over 20 items. This is a weak and indirect
signal to retrain attention patterns in a 6.7B parameter model.

**Reason 3 — AlmostRec's LoRA is in a fundamentally different setup.**
AlmostRec uses LoRA on LLaVA-1.5 with:
- Datasets of 100K+ users and 719K+ interactions (MicroLens-100K)
- A dedicated content-behavior adapter explicitly designed for multimodal alignment
- Rich visual + text + ID information from the start

Luxury_Beauty has ~64K interactions. The LoRA training does not have enough signal
to overcome OPT's pre-trained representations.

### 5.5 The Circular Dependency in `item_emb_proj`

There is an additional architectural issue: `item_emb_proj` serves two roles
simultaneously.

**Role 1 (injection):** Projects CF embeddings for injection into the LLM prompt at
`[HistoryEmb]` and `[CandidateEmb]` token positions. This changes what the LLM sees,
which changes `last_hidden`.

**Role 2 (scoring target):** Projects CF embeddings to form the scoring targets
against which `last_hidden` is compared via cosine similarity.

When `item_emb_proj` is updated during training:
- The CF signals injected into the LLM change → `last_hidden` changes
- The scoring targets change

The model is simultaneously moving what it shows to the LLM AND what it's trying
to align with. This circular dependency makes optimisation harder, particularly on
more complex datasets.

---

## 6. Summary of Root Causes

| Root Cause | Impact |
|---|---|
| OPT's strong luxury brand knowledge dominates `last_hidden` | Cosine scoring against CF embeddings fails (31%) |
| Text-semantic LLM space ≠ CF-behavioral space in 4096-d | Fundamental misalignment in scoring |
| LoRA degrades OPT's pre-trained text knowledge | Drops from 31% to 25%, loses baseline advantage |
| `item_emb_proj` serves as both injection input and scoring target | Circular optimization, unstable training |
| 128-d Stage 1 bottleneck compresses Luxury_Beauty's richer CF space | Scoring targets are less informative |

The core insight: **ID-prediction as currently implemented asks `item_emb_proj` to
lift 128-d CF embeddings up into OPT's 4096-d text-semantic space so that cosine
similarity works. This is the wrong direction.** The LLM's `last_hidden` should
instead be brought DOWN into the well-trained 128-d CF-text space where Stage 1
alignment already lives.
