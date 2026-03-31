# PLAN.md — What We Are Doing Now (Branch: id-pred-2)

This document describes the changes being made in the `id-pred-2` branch, why each
change is being made, and what we expect from each experiment.

---

## 1. The Problem We Are Solving

From the analysis in HISTORY.md, the current ID-prediction approach fails on
Luxury_Beauty because:

**Current scoring direction (wrong):**
```
last_hidden (4096-d, text-semantic)
      ↑ OPT's luxury brand knowledge dominates this

cosine similarity
      ↓
item_emb_proj(e_i)   (128 → 4096-d, CF embedding lifted UP to LLM space)
      ↑ trying to match OPT's text representations with CF behavioral patterns
```

`item_emb_proj` is trying to map 128-d CF embeddings into OPT's 4096-d text space.
On luxury products, OPT's text representations are so strongly shaped by pre-training
that there is no region of 4096-d space where CF embeddings can naturally meet them.

**What we want (correct direction):**
```
score_head(last_hidden)   (4096 → 128-d, LLM output pulled DOWN to CF-text space)
      ↑ learn what part of last_hidden is recommendation-relevant

cosine similarity
      ↓
e_i   (128-d, Stage 1 CF-text joint embedding, already well-trained)
      ↑ this space explicitly encodes both CF and text signals
```

The Stage 1 128-d space is the natural alignment point. It was explicitly trained
(via matching loss, reconstruction loss, and recommendation loss) to encode both
collaborative filtering patterns and textual item descriptions. Scoring in this
space is more principled than trying to score in OPT's raw 4096-d output space.

---

## 2. Changes Being Made

### Change 1 — Remove LoRA (Revert to Frozen LLM)

**File:** `models/llm4rec.py`

LoRA (Experiment 4) made results worse on Luxury_Beauty (31% → 25%) and contradicts
the original design intent documented in CHANGES.md. The A-LLMRec paper itself shows
that fine-tuning the LLM on recommendation data hurts performance in warm scenarios
(this is exactly what happened).

The LLM is reverted to fully frozen. Only `log_emb_proj`, `item_emb_proj`, and the
new `score_head` will receive gradient updates. This restores A-LLMRec's core
advantage: efficiency (no LLM fine-tuning) and preserved pre-trained representations.

**Before (current, with LoRA):**
```python
self.llm_model = prepare_model_for_kbit_training(self.llm_model)
lora_config = LoraConfig(r=8, lora_alpha=16, ...)
self.llm_model = get_peft_model(self.llm_model, lora_config)
```

**After (reverted to frozen):**
```python
for _, param in self.llm_model.named_parameters():
    param.requires_grad = False
```

---

### Change 2 — Add `score_head` and Separate Injection from Scoring

**File:** `models/a_llmrec_model.py`

**The key architectural change.**

Currently `item_emb_proj` (128 → 4096) is used for two purposes:
1. Injecting CF embeddings into the LLM prompt at `[HistoryEmb]`/`[CandidateEmb]`
   positions — this is correct and stays as-is.
2. Producing the scoring targets against which `last_hidden` is compared via cosine
   similarity — this is the problem.

We separate these two roles:

**Keep for injection (unchanged):**
```python
# item_emb_proj still maps 128 → 4096 for prompt injection
interact_embs  = item_emb_proj(get_item_emb(interact_ids))   # → LLM prompt
candidate_embs = item_emb_proj(get_item_emb(candidate_ids))  # → LLM prompt
```

**New for scoring:**
```python
# score_head maps last_hidden (4096) → 128-d (Stage 1 CF-text space)
self.score_head = nn.Linear(d_model, 128)

# Scoring targets are now the RAW 128-d Stage 1 embeddings, NOT item_emb_proj(...)
scoring_embs = get_item_emb(candidate_ids)   # (20, 128) — Stage 1 embeddings

# Project last_hidden down into Stage 1 space
h_proj = score_head(last_hidden)             # (batch, 128)

# Cosine similarity in 128-d Stage 1 space
h_norm = F.normalize(h_proj, dim=-1)
c_norm = F.normalize(scoring_embs, dim=-1)
logits = bmm(c_norm, h_norm.unsqueeze(-1)).squeeze(-1)   # (batch, 20)
```

**Why 128-d:**
- Stage 1 was trained specifically to produce a 128-d space where CF and text signals
  are aligned. This is the most semantically meaningful compact space in the pipeline.
- Projecting `last_hidden` (4096→128) is far easier to train than lifting CF
  embeddings (128→4096) — we are extracting a small recommendation-relevant subspace
  from a large representation.
- The circular dependency is broken: `item_emb_proj` only affects what the LLM sees
  (injection), not the scoring targets. Scoring targets are fixed (frozen Stage 1
  embeddings via frozen `mlp`).

**What gets gradients:**
- `log_emb_proj` — user representation projection (unchanged)
- `item_emb_proj` — item embedding injection only (unchanged, but no longer scoring)
- `score_head` — new projection, learns to extract recommendation signal from LLM output

**What stays frozen:**
- OPT-6.7B (all 6.7B weights)
- SASRec (CF backbone)
- `mlp` from Stage 1 (the 128-d encoder)

---

### Change 3 — Ablation: Score at `[UserRep]` Token Position

**File:** `models/llm4rec.py` (new method parameter or separate method)

As an experiment alongside the primary fix, try using the **first token's hidden
state** (index 0 — the prepended user representation `log_emb`) instead of the
last token.

**Rationale:**
- The `[UserRep]` token starts as the projected CF user representation.
- After full attention over the entire sequence, it has attended over all history
  items, all candidate items, and all text.
- For a recommendation task ("given this user's history, predict next item"), the
  user representation token may be a more natural query vector than the last text
  token after "The recommendation is".
- The last token is more likely to be dominated by the text fragment at the end of
  the prompt; the user rep token is seeded from CF information.

**Implementation:**
```python
# In forward_id, controlled by a parameter:
if use_user_token:
    last_hidden = outputs.hidden_states[-1][:, 0, :]   # [UserRep] position
else:
    last_hidden = outputs.hidden_states[-1][:, -1, :]  # last token (current)
```

This is a low-cost ablation — train both and compare.

---

## 3. What We Are NOT Changing

- **Stage 1** (`pre_train_phase1`, SBERT alignment, all losses) — untouched. The
  128-d joint CF-text embedding space from Stage 1 is the foundation of the new
  scoring approach.
- **The text-generation path** (`pre_train_phase2`, `generate`) — untouched. This
  is kept as the baseline comparison.
- **SASRec backbone** — untouched.
- **The prompt structure** — the same enriched prompt (user rep + history + candidates)
  is used. We are only changing what we extract and how we score.
- **Candidate set size (20)** — unchanged.
- **Evaluation protocol** — Hit@1, same as before.

---

## 4. Experiments Planned

### Experiment 6 — Remove LoRA Only (Sanity Check)

- Branch: id-pred-2
- Change: Revert to frozen LLM, keep cosine scoring in 4096-d (current approach)
- Dataset: Luxury_Beauty
- Expected: Should recover ~31% (what we had before LoRA)
- Purpose: Confirm LoRA removal doesn't break anything, establish the new baseline

### Experiment 7 — score_head in 128-d Stage 1 Space (Primary Fix)

- Branch: id-pred-2
- Change: Remove LoRA + add `score_head` (4096→128), score against Stage 1 embeddings
- Dataset: Luxury_Beauty
- Expected: Significantly higher than 31%. Should approach or exceed 58% if the
  theory is correct — the CF-text alignment in Stage 1 space is the right scoring
  space.
- Purpose: Test the primary hypothesis

### Experiment 8 — score_head on All_Beauty (Regression Check)

- Branch: id-pred-2
- Change: Same as Exp 7
- Dataset: All_Beauty
- Expected: At least maintain 51% (ideally improve)
- Purpose: Confirm the fix doesn't hurt the dataset where ID-pred was already working

### Experiment 9 — Score at [UserRep] Token (Ablation)

- Branch: id-pred-2
- Change: Use hidden state at index 0 ([UserRep] token) instead of index -1
- Dataset: Luxury_Beauty
- Expected: May improve further if user token is a better recommendation query
- Purpose: Understand which token position is most informative

---

## 5. Why We Expect This to Work

The key insight from all experiments so far:

**All_Beauty (51%):** The CF signal from SASRec is moderate (smaller dataset). OPT
doesn't have strong pre-trained associations for generic products. `item_emb_proj`
successfully lifts CF embeddings into a region of 4096-d space that `last_hidden`
can match. Cosine scoring in 4096-d works.

**Luxury_Beauty (31%):** OPT's luxury brand knowledge dominates `last_hidden` in
4096-d space. CF embeddings lifted to 4096-d cannot reach OPT's text representations.

**The fix:** By projecting `last_hidden` DOWN to 128-d (instead of lifting CF
embeddings UP to 4096-d), we ask the model to extract the recommendation-relevant
portion of `last_hidden` and map it into the Stage 1 CF-text space. The Stage 1
space was built to be the intersection of CF and text knowledge — it is the correct
alignment target regardless of whether the LLM's raw output is text-dominated or
CF-dominated.

Additionally, since `score_head` is a new trainable parameter (128 weights +
bias from a 4096-d input), the model now has a dedicated component for this
alignment rather than relying on `item_emb_proj` to do the impossible task of
matching OPT's text representations from the CF side.

---

## 6. How to Run the New Experiments

### Train Stage 1 (if not already done for Luxury_Beauty)
```bash
python main.py --pretrain_stage1 \
               --rec_pre_trained_data Luxury_Beauty \
               --recsys sasrec \
               --num_epochs 10
```

### Train Stage 2 — ID-Prediction (new score_head approach)
```bash
python main.py --pretrain_stage2 --id_prediction \
               --rec_pre_trained_data Luxury_Beauty \
               --llm opt \
               --num_epochs 5
```

### Inference — ID-Prediction
```bash
python main.py --inference --id_prediction \
               --rec_pre_trained_data Luxury_Beauty \
               --llm opt

python eval.py --id
```

### Inference — Baseline (for comparison)
```bash
python main.py --inference \
               --rec_pre_trained_data Luxury_Beauty \
               --llm opt

python eval.py
```
