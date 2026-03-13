# A-LLMRec — ID Prediction Extension: Design Document

This document records the motivation, design decisions, and exact code changes made
to extend the original A-LLMRec codebase with an **ID-prediction head** as an
alternative to text-generation for the recommendation task.

---

## 1. Background

### Original A-LLMRec (the baseline)

The original system (KDD 2024, Kim et al.) follows a two-stage pipeline:

**Stage 1 — Alignment between CF and text**
- A pre-trained, frozen CF recommender (e.g., SASRec) provides item embeddings `E_i`
  and user representations `x_u`.
- A Sentence-BERT (SBERT) model encodes item title+description text into `Q_i`.
- Two small autoencoder MLPs (`mlp` for CF, `mlp2` for text) map both into a shared
  128-dimensional latent space.
- Four losses are combined: matching loss (MSE between latent CF and text), two
  reconstruction losses, and a BPR-style recommendation loss.
- After training, `e_i = mlp_encoder(E_i)` is the **joint collaborative-text embedding**.

**Stage 2 — Alignment with LLM**
- Two 2-layer MLPs (`log_emb_proj`, `item_emb_proj`) project `x_u` and `e_i` into
  the LLM's token-embedding space (`d_token`).
- A frozen OPT-6.7B model receives a text prompt where special marker tokens
  `[HistoryEmb]` and `[CandidateEmb]` are replaced by the projected item embeddings,
  and the projected user embedding `O_u` is prepended as a soft prompt.
- The model is trained to **generate the correct next item's title** as text.
- **Loss**: causal language-model cross-entropy on the output tokens.
- **Inference**: `llm.generate()` → decode output text → match to item titles.

### Problem with text generation

The user observed a key limitation of the text-generation objective:
- The LM head predicts over a **vocabulary of ~50K word-piece tokens**, not over items.
- At inference, the model must generate a title string, which is then matched against
  known item titles — this string-matching step is fragile (hallucinations, partial
  matches, case sensitivity).
- The training signal is indirect: the model learns to produce English tokens that
  happen to spell the item title, rather than directly learning to rank items.

---

## 2. Inspiration: AlmostRec (ICMR 2025)

The second paper read (Wu et al., ICMR 2025) proposes **AlmostRec**, which uses a
Large Multimodal Model (LMM, specifically LLaVA-1.5) with:
- A **content-behavior adapter** that maps pre-trained sequential recommender embeddings
  into the LMM token space.
- A **transformation layer** `W_T ∈ R^{d_m × |I|}` that maps the LMM's output hidden
  state to one logit per item in the candidate/full item set.
- An **ID prediction objective**: cross-entropy over item IDs (not text tokens).

```
q(i_{N+1} | i_1,...,i_N) = Softmax(F_{d_m} W_T)
Loss = -sum_i p_i * log(q_i)
```

This is more directly aligned with the recommendation task and eliminates hallucination.

---

## 3. What We Built

We added an **AlmostRec-style ID prediction head** on top of the existing A-LLMRec
pipeline as an **optional, backward-compatible alternative** to text generation.

### Architecture overview

```
                     ┌─────────────────────────────────────────┐
                     │               STAGE 1 (unchanged)        │
  CF-RecSys ──E_i──► │  mlp (autoencoder)  ──► e_i (128-d)     │
  SBERT     ──Q_i──► │  mlp2 (autoencoder) ──► q_i (128-d)     │
                     │  Losses: L_match + L_recon + L_rec       │
                     └─────────────────────────────────────────┘
                                         │
                              e_i (128-d) │  x_u (d_CF)
                                         ▼
                     ┌─────────────────────────────────────────┐
                     │               STAGE 2 (extended)         │
                     │                                          │
                     │  item_emb_proj: 128 → d_token            │
                     │  log_emb_proj:  d_CF → d_token           │
                     │                                          │
                     │  ┌─────────────────────────────────┐    │
                     │  │  Frozen OPT-6.7B LLM             │    │
                     │  │  Input: [O_u | prompt+emb tokens]│    │
                     │  │  Output:                         │    │
                     │  │   (A) token logits ──► text gen  │    │
                     │  │   (B) last hidden ──► id_head    │    │
                     │  └─────────────────────────────────┘    │
                     │                          │               │
                     │               id_pred_head (d_token→20)  │
                     │                          │               │
                     │             CE loss over candidate IDs   │
                     └─────────────────────────────────────────┘
```

### Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| LLM frozen or LoRA? | **Fully frozen** | Preserves A-LLMRec's efficiency advantage; only the small head trains |
| Score over full item set or candidate set? | **Candidate set (20 items)** | Consistent with existing evaluation protocol; avoids a `d_model × 60K` head |
| Which hidden state to use? | **Last token of last layer** | Standard for causal/decoder LMs; attends to the full enriched prompt |
| Backward compatible? | **Yes** | `--id_prediction` flag selects new path; original text-gen path untouched |
| New checkpoint? | **`id_pred_head.pt`** | Small file; saved alongside existing `log_proj.pt` and `item_proj.pt` |

---

## 4. File-by-File Changes

### `models/llm4rec.py`

**Added method: `forward_id(log_emb, samples)`**

Does the same input construction as the existing `forward()`:
- Tokenise `text_input`, get initial embeddings.
- Call `replace_hist_candi_token()` to inject history/candidate item embeddings.
- Prepend the projected user representation `O_u`.

But instead of calling the LLM with `labels` to get LM loss, it calls with
`output_hidden_states=True` and returns:
```python
last_hidden = outputs.hidden_states[-1][:, -1, :].float()  # (batch, d_model)
```

The existing `forward()` is untouched.

---

### `models/a_llmrec_model.py`

**`__init__`** — added under the `pretrain_stage2 or inference` block:
```python
self.candidate_num = 20
self.id_pred_head = nn.Linear(llm_hidden_size, self.candidate_num)
self.ce_criterion = nn.CrossEntropyLoss()
```

**`save_model`** — saves `id_pred_head.pt` alongside existing stage 2 checkpoints.

**`load_model`** — loads `id_pred_head.pt` if it exists (graceful fallback if not present).

**`forward()`** — two new dispatch modes added:
- `mode='phase2_id'` → `pre_train_phase2_id()`
- `mode='generate_id'` → `generate_id()`

**New method: `_build_id_samples(u, seq, pos)`**

Shared helper extracted from the prompt-building logic. Does everything
`pre_train_phase2` does per-user, but also:
- Records `target_idx`: the **integer index** of the positive item in the shuffled
  candidate list (the ground-truth label for cross-entropy).
- Returns: `(text_input, interact_embs, candidate_embs, candidate_ids_list, target_indices)`.

**New method: `pre_train_phase2_id(data, optimizer, batch_iter)`**

Calls `_build_id_samples()`, runs `self.llm.forward_id()`, applies `id_pred_head`,
computes `CrossEntropyLoss(logits, target_indices)`, backpropagates.

Only `log_emb_proj`, `item_emb_proj`, and `id_pred_head` receive gradients.

**New method: `generate_id(data)`**

Same setup as `pre_train_phase2_id` but under `torch.no_grad()`. Returns a list of
dicts: `{predicted_id, target_id, candidate_ids, hit}`. Also appends to
`recommendation_output_id.txt`.

---

### `train_model.py`

Added public launchers:
- `train_model_phase2_id(args)` — multi-GPU aware, mirrors `train_model_phase2`.
- `inference_id(args)` — mirrors `inference`.

Added private worker functions:
- `train_model_phase2_id_(rank, world_size, args)` — same as `train_model_phase2_`
  but calls `mode='phase2_id'`.
- `inference_id_(rank, world_size, args)` — same as `inference_` but calls
  `mode='generate_id'` and prints Hit@1 at the end.

---

### `main.py`

Added CLI flag:
```bash
--id_prediction   # switch Stage 2 to ID-prediction mode
```

New routing logic (evaluated in order):
```
--pretrain_stage2 --id_prediction  →  train_model_phase2_id(args)
--pretrain_stage2                  →  train_model_phase2(args)       [unchanged]
--inference --id_prediction        →  inference_id(args)
--inference                        →  inference(args)                 [unchanged]
```

---

### `eval.py`

Added function `evaluate_id(results_file)`:
- Reads `recommendation_output_id.txt` (written by `generate_id()`).
- Parses `Target | Predicted | Hit` lines.
- Returns `(hit_at_1_rate, total_count)`.
- No string matching needed — IDs are integers.

Updated `__main__` block:
```bash
python eval.py --id     # evaluates ID-prediction results
python eval.py          # evaluates text-generation results (original behaviour)
```

---

## 5. How to Run

### Train Stage 1 (unchanged)
```bash
python main.py --pretrain_stage1 --rec_pre_trained_data Movies_and_TV --recsys sasrec
```

### Train Stage 2 — text generation (original, unchanged)
```bash
python main.py --pretrain_stage2 --rec_pre_trained_data Movies_and_TV --llm opt
```

### Train Stage 2 — ID prediction (new)
```bash
python main.py --pretrain_stage2 --id_prediction --rec_pre_trained_data Movies_and_TV --llm opt
```

### Inference — text generation (original)
```bash
python main.py --inference --rec_pre_trained_data Movies_and_TV --llm opt
python eval.py
```

### Inference — ID prediction (new)
```bash
python main.py --inference --id_prediction --rec_pre_trained_data Movies_and_TV --llm opt
python eval.py --id
```

---

## 6. What Is NOT Changed

- Stage 1 training (`pre_train_phase1`, SBERT alignment, all losses) — **untouched**.
- The original `pre_train_phase2` and `generate` (text-generation path) — **untouched**.
- The CF backbone (`models/recsys_model.py`, `pre_train/sasrec/`) — **untouched**.
- The LLM itself (`OPTForCausalLM`) — **still fully frozen in both paths**.
- Existing checkpoint files — the new `id_pred_head.pt` is additive; old checkpoints
  still load correctly for text-generation inference.

---

## 7. Bug Fix: ID Scoring (v2)

### Problem discovered

After the first implementation was trained and evaluated, Hit@1 was 0.0481 —
essentially random chance (1/20 = 0.05 for 20 candidates).

**Root cause:** `id_pred_head = Linear(d_model, 20)` is **position-blind**.
It always outputs 20 fixed projections of the hidden state with no knowledge of
which items occupy the 20 slots. Because `make_candidate_text` randomly shuffles
candidates for every user, the ground-truth label (an integer 0–19) shifts
randomly too. The gradient signal is pure noise, so the model cannot learn.

### Fix applied

Removed `id_pred_head` entirely. Scoring is now a **dot-product** between the
LLM's last hidden state `h` and each candidate's embedding `c_k` (both already
in `d_model` space via `item_emb_proj`):

```
score_k = h · c_k
logits  = bmm(cand_stack, h.unsqueeze(-1)).squeeze(-1)   # (batch, 20)
```

This is item-aware: every candidate has a unique embedding regardless of its
shuffled position, so the model receives a consistent learning signal.

**Files changed:**

- `models/a_llmrec_model.py`:
  - `__init__`: removed `id_pred_head` Linear + `xavier_normal_` init; kept
    `candidate_num` and `ce_criterion`.
  - `pre_train_phase2_id`: replaced `self.id_pred_head(last_hidden)` with
    `torch.bmm(cand_stack, last_hidden.unsqueeze(-1)).squeeze(-1)`.
  - `generate_id`: same replacement.
  - `save_model` / `load_model`: removed `id_pred_head.pt` save/load blocks.

---

## 8. Differences vs AlmostRec

| Aspect | AlmostRec | This implementation |
|---|---|---|
| LLM backbone | LLaVA-1.5 (vision-language) | OPT-6.7B (text only) |
| LLM tuning | LoRA on LMM | Fully frozen |
| Scoring scope | Full item set `|I|` | Candidate set (20 items) |
| Modalities | Image + text + ID | Text + ID (no images) |
| Adapter name | Content-behavior adapter | `item_emb_proj` (already existed) |
| Transformation layer | `W_T ∈ R^{d_m × |I|}` | `id_pred_head` (Linear, `d_m → 20`) |
