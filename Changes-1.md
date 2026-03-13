# Changes-1.md — Bug Report & Fix: ID Prediction Scoring

This document focuses specifically on the bug discovered after the first implementation
of the ID-prediction head, why it caused random performance, and exactly what was
changed to fix it.

---

## The Error

After training Stage 2 with `--id_prediction` and running inference, the model
achieved:

```
ID-prediction Hit@1: 0.0481  (101/2099)
```

With 20 candidates (1 positive + 19 negatives), **random chance is exactly 1/20 = 0.05
(5.0%)**. A result of 4.81% is indistinguishable from random — the model had learned
absolutely nothing.

---

## Root Cause: Position-Blind Linear Head

### What was implemented

The first implementation added a small linear head on top of the LLM's last hidden
state:

```python
# In __init__
self.id_pred_head = nn.Linear(d_model, 20)

# In pre_train_phase2_id and generate_id
logits = self.id_pred_head(last_hidden)   # (batch, 20)
loss   = self.ce_criterion(logits, target_indices)
```

### Why this is fundamentally broken

`nn.Linear(d_model, 20)` maps a hidden vector to exactly 20 fixed output dimensions.
These 20 outputs have no relationship to the actual items in the candidate list —
they are just 20 fixed linear projections of the hidden state, the same 20 projections
regardless of which items happen to be candidates for this user.

The critical detail is that `make_candidate_text` **randomly shuffles** the 20
candidates for every single user on every call:

```python
random_ = np.random.permutation(len(candidate_text))
candidate_text = np.array(candidate_text)[random_]
candidate_ids  = np.array(candidate_ids)[random_]
```

This means:
- For user A, the target item might land in slot 3.
- For user B (same target item), the target item might land in slot 17.

The label fed to `CrossEntropyLoss` is the **slot index** (e.g., 3 or 17). But the
model has no way to know which item is in which slot — the linear head produces the
same 20 numbers regardless. The correct slot index changes randomly from sample to
sample, so the gradient signal is **pure noise**. The model cannot converge and stays
at random performance.

In short: the head was trained to predict "which slot number is correct", but slot
numbers are meaningless because they are assigned randomly.

---

## The Fix: Dot-Product Scoring Against Candidate Embeddings

### Core idea

The candidate item embeddings (`candidate_embs`) are already projected into `d_model`
space by `item_emb_proj`. Every item has a **unique, consistent embedding** that
represents its collaborative and textual characteristics.

The correct way to score a candidate is:

```
score_k = last_hidden · candidate_embs[k]
```

This is a dot product between the LLM's output representation of the user context and
the embedding of candidate item `k`. Items that better match the user's context will
have a higher dot product, regardless of which slot they are in.

This is the same principle used in AlmostRec (their `W_T` transformation layer) and in
standard retrieval/recommendation models: score = query · key.

### Why this works where the linear head does not

| Property | Linear head `(d_model → 20)` | Dot-product scoring |
|---|---|---|
| Aware of which item is in each slot? | No | Yes — uses the item's own embedding |
| Affected by random candidate shuffling? | Yes — label changes randomly | No — score is tied to the item, not its position |
| Consistent gradient signal? | No | Yes |
| Learns anything useful? | No | Yes |

### Code change

Replaced in both `pre_train_phase2_id` and `generate_id`:

```python
# BEFORE (broken)
logits = self.id_pred_head(last_hidden)   # (batch, 20)
```

```python
# AFTER (fixed)
# Stack candidate embeddings into a batch tensor
cand_stack = torch.stack([c for c in candidate_embs])  # (batch, 20, d_model)
# Dot product: for each item in the candidate list, how well does the
# user hidden state align with that item's embedding?
logits = torch.bmm(cand_stack, last_hidden.unsqueeze(-1)).squeeze(-1)  # (batch, 20)
```

`torch.bmm` performs batched matrix multiplication:
- `cand_stack` shape: `(batch, 20, d_model)`
- `last_hidden.unsqueeze(-1)` shape: `(batch, d_model, 1)`
- Result: `(batch, 20, 1)` → squeezed to `(batch, 20)`

Each of the 20 output values is now a dot product between the user's hidden state
and a specific item's embedding — a meaningful relevance score.

`CrossEntropyLoss(logits, target_indices)` then works correctly because the correct
item always has the same embedding, so the model receives a consistent signal to push
that item's score higher.

---

## Files Changed

Only `models/a_llmrec_model.py` was modified:

1. **`__init__`**: Removed `self.id_pred_head = nn.Linear(...)` and its Xavier
   initialisation. `candidate_num` and `ce_criterion` were kept.

2. **`pre_train_phase2_id`**: Replaced `self.id_pred_head(last_hidden)` with
   `torch.bmm` dot-product scoring.

3. **`generate_id`**: Same replacement.

4. **`save_model`**: Removed saving of `id_pred_head.pt` (no longer exists).

5. **`load_model`**: Removed loading of `id_pred_head.pt`.

No changes to `llm4rec.py`, `train_model.py`, `main.py`, or `eval.py`.

---

## How to Retrain After Fix 1

Stage 1 is unchanged — you do **not** need to retrain it. Only retrain Stage 2:

```bash
# Retrain Stage 2 with the fixed scoring
python main.py --pretrain_stage2 --id_prediction \
               --rec_pre_trained_data Movies_and_TV --llm opt

# Run inference
python main.py --inference --id_prediction \
               --rec_pre_trained_data Movies_and_TV --llm opt

# Evaluate
python eval.py --id
```

---

## Bug 2 — Exploding Dot Products / Unstable Loss

### Symptom

After Fix 1 was applied, training loss alternated wildly between `0.0000` and
very large spikes (7.0, 11.5, 16.2, 20.1, 31.4, …). The pattern continued for
hundreds of iterations with no sign of settling, indicating a training
instability rather than normal early-epoch noise.

### Root Cause

The dot-product score between `last_hidden` and the candidate embeddings is:

```
score_k = last_hidden · candidate_embs[k]
```

Both `last_hidden` (the last token of the LLM) and `candidate_embs` (the
projected item embeddings) are **unbounded in magnitude**. When their L2 norms
grow large, the dot products can become enormous positive or negative values.
CrossEntropyLoss exponentiation then pushes the probability mass to near 0 or
near 1 in a hard, brittle way:

- If the correct candidate happens to have the largest logit, the loss is
  ~0.0 (model appears "perfect" even though it is unstable).
- If the correct candidate has a small logit, the loss explodes to 20–30+.

The resulting gradients also explode, making the next step worse — a classic
gradient explosion cycle. The loss does not decrease because every large-loss
step corrupts the projectors and restarts the cycle.

### Fix Applied

**L2-normalise both vectors before the dot product** (i.e. use cosine
similarity instead of raw dot product). This bounds every score to `[-1, 1]`,
which keeps CrossEntropyLoss well-conditioned regardless of embedding magnitude.

**Files changed:**

**`models/a_llmrec_model.py`**

1. Added `import torch.nn.functional as F` at the top.
2. In `pre_train_phase2_id` — scoring block changed from:

```python
cand_stack = torch.stack([c for c in candidate_embs])
logits = torch.bmm(cand_stack, last_hidden.unsqueeze(-1)).squeeze(-1)
```

to:

```python
cand_stack = torch.stack([c for c in candidate_embs])
h_norm = F.normalize(last_hidden, dim=-1)       # (batch, d_model)
c_norm = F.normalize(cand_stack, dim=-1)         # (batch, candidate_num, d_model)
logits = torch.bmm(c_norm, h_norm.unsqueeze(-1)).squeeze(-1)
```

3. Identical change applied inside `generate_id`.

### Why This Works

- Cosine similarity is scale-invariant. The model cannot boost a score simply
  by increasing the norm of its projectors — it must align the direction of
  `last_hidden` with the target candidate's direction.
- All logits are in `[-1, 1]`, so CrossEntropyLoss gradients are always
  reasonable in magnitude.
- Training loss should now decrease smoothly from roughly `log(20) ≈ 3.0`
  towards lower values as the model learns.

---

## How to Retrain After Fix 2

Stop any running training run, then retrain Stage 2 from scratch (Stage 1
checkpoint is unaffected):

```bash
# Retrain Stage 2 with the normalised scoring
python main.py --pretrain_stage2 --id_prediction \
               --rec_pre_trained_data Movies_and_TV --llm opt

# Run inference
python main.py --inference --id_prediction \
               --rec_pre_trained_data Movies_and_TV --llm opt

# Evaluate
python eval.py --id
```

Expected: initial loss near 3.0 (random, 20 candidates), steadily decreasing
each epoch.
