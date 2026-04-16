# Project Context: A-LLMRec with SmolVLM Multimodal Extension

## 1. Research Background

### Paper 1 — A-LLMRec (KDD 2024)
**"Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System"**
Kim et al., KAIST + NAVER. [`3637528.3671931`]

**The core problem this paper solves:**
Traditional collaborative filtering (CF) models like SASRec are excellent in "warm" scenarios (items with many interactions) but fail on cold-start (new/sparse items). Conversely, LLM-based recommenders (e.g., TALLRec) do well in cold scenarios due to their language understanding but fail in warm scenarios because they lack collaborative knowledge from user-item interaction patterns.

**Key idea:**
Instead of fine-tuning the LLM or the CF model, A-LLMRec trains only a small "alignment network" that bridges the CF model's embedding space with the LLM's token space. This gives:
- The LLM access to rich collaborative knowledge (from the CF model)
- Model-agnostic design (swap out any CF backbone)
- ~2.5× faster training and ~1.7× faster inference than TALLRec

**Two-stage architecture:**

*Stage 1 — Align CF embeddings with text embeddings:*
- A frozen SASRec CF model provides item embeddings
- A fine-tuned Sentence-BERT (`nq-distilbert-base-v1`) provides text embeddings from item titles/descriptions
- Two **2-layer** MLP autoencoders (one for CF embeddings, one for SBERT embeddings) are trained to match their latent spaces
- Loss = latent matching (MSE) + item reconstruction + text reconstruction + BPR recommendation loss
- Output: "joint collaborative-text embeddings" (128-dim) per item

*Stage 2 — Project into LLM token space and train prompts:*
- Two 2-layer MLPs project (a) CF user representations and (b) joint item embeddings into the LLM's token hidden dimension
- Special markers `[HistoryEmb]` and `[CandidateEmb]` are inserted in the text prompt
- At forward pass, these marker token positions in the embedding matrix are replaced with the projected CF/item vectors
- The user representation is prepended as a continuous embedding (not a text token) at position 0 of the LLM input sequence
- Only the two projection heads (and LoRA adapters for SmolVLM) are trained; SASRec and the LLM base weights stay frozen

**Results (Hit@1):**
| Dataset | SASRec | TALLRec | A-LLMRec |
|---------|--------|---------|----------|
| Movies & TV | 0.6154 | 0.2345 | 0.6237 |
| Video Games | 0.5402 | 0.4403 | 0.5282 |
| Beauty | 0.5298 | 0.5542 | 0.5809 |

---

### Paper 2 — AlmostRec (ICMR 2025)
**"Aligning Large Multimodal Model with Sequential Recommendation via Content-Behavior Guidance"**
Wu et al., Tsinghua University. [`3731715.3733273`]

**The problem this paper addresses:**
Existing LLM-based recommenders are text-only and generate free-form text output, leading to:
- Ignoring users' visual preferences
- Hallucinations (recommending items not in the candidate set)
- Inability to interpret ID-based collaborative signals

**Key idea:**
Use a **Large Multimodal Model (LMM)** — specifically LLaVA-1.5-7B — as the backbone instead of a textual LLM. Feed it three modalities simultaneously: item images (V), text prompts (T), and sequential item IDs (S).

**Architecture:**
- Pre-trained sequential recommender (e.g., SASRec) provides item embeddings
- A "Content-Behavior Adapter" (`W_B`, a learnable linear projection) maps item embeddings into the LMM's token space
- LoRA fine-tunes the LMM for the recommendation task
- A "Transformation Layer" (`W_T`) projects the LMM's output to a probability distribution over all candidate items (cross-entropy loss)
- Crucially: outputs an **item ID prediction** (not free-form text), preventing hallucinations

---

## 2. The Codebase: `A-LLMRec-Forked`

This is a fork of the official A-LLMRec repository, extended to use **SmolVLM2-2.2B-Instruct** as the LLM backbone with product image inputs. Active development branch: `smol-img`.

### Directory Structure
```
A-LLMRec-Forked/
├── main.py                     # CLI entrypoint — args parsed here, dispatches to train_model.py
├── train_model.py              # Training/inference loops: phase1, phase2, inference
├── eval.py                     # Hit@1 / NDCG@1 evaluation from output text files
├── utils.py                    # Utility: create_dir, find_filepath
├── requirements.txt
├── models/
│   ├── a_llmrec_model.py       # Core model class: A_llmrec_model (all 3 stages + generate)
│   ├── llm4rec.py              # LLM wrapper: llm4rec class (OPT-6.7B or SmolVLM2-2.2B)
│   └── recsys_model.py         # CF wrapper: RecSys class (loads frozen pre-trained SASRec)
│   └── saved_models/           # Checkpoint directory (created at runtime)
├── pre_train/
│   └── sasrec/
│       ├── model.py            # SASRec transformer implementation
│       ├── main.py             # SASRec standalone training entrypoint
│       ├── utils.py            # SeqDataset, SeqDataset_Inference, data_partition, evaluate
│       └── data_preprocess.py  # Amazon review JSON → .txt interactions + text dicts + id_to_asin
├── data/
│   ├── amazon/                 # Amazon review datasets (.txt, metadata JSON, text dicts)
│   └── images/
│       └── download_images.py  # Downloads product JPEG images keyed by ASIN
└── results/smol/               # Inference output text files (default output location)
```

---

## 3. Data Files

### Generated by `data_preprocess.py` (run via `pre_train/sasrec/main.py`)
For a dataset named `All_Beauty`:
- `data/amazon/All_Beauty.txt` — user-item interaction sequences, one `user_id item_id` pair per line (both are 1-indexed integers, sorted by timestamp per user)
- `data/amazon/All_Beauty_text_name_dict.json.gz` — pickle of `{'title': {int_id: str}, 'description': {int_id: str}}`
- `data/amazon/All_Beauty_id_to_asin.json.gz` — pickle of `{int_id: asin_string}` — maps internal integer item IDs back to Amazon ASINs; used to look up image filenames at runtime
- `data/images/All_Beauty/{asin}.jpg` — product images downloaded separately by `download_images.py`

### SASRec checkpoint format
Pre-trained SASRec is saved as a `.pth` file:
```
pre_train/sasrec/{dataset}/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth
```
The file is a list `[kwargs_dict, state_dict]` where `kwargs_dict = {'user_num': int, 'item_num': int, 'args': Namespace}`.
`recsys_model.py:load_checkpoint()` finds this file via glob, loads it, and reconstructs the model.

---

## 4. Model Classes

### `two_layer_mlp` (`models/a_llmrec_model.py`)
Used in Stage 1. A symmetric autoencoder with:
- `fc1`: `nn.Linear(dims, 128)` — encoder, compresses to 128-dim latent
- `sigmoid` activation
- `fc2`: `nn.Linear(128, dims)` — decoder, reconstructs back to original dimension

`forward(x)` returns `(latent, reconstruction)`. The 128-dim latent is the "joint collaborative-text embedding".

### `RecSys` (`models/recsys_model.py`)
Thin wrapper around a frozen SASRec checkpoint.
- Loads the `.pth` checkpoint from `pre_train/{recsys}/{pre_trained_data}/`
- Freezes all parameters (`requires_grad = False`)
- Exposes `self.model` (the SASRec), `self.item_num`, `self.hidden_units`
- `self.hidden_units` = 50 (SASRec default, from checkpoint's `args.hidden_units`)

### `SASRec` (`pre_train/sasrec/model.py`)
Standard SASRec transformer:
- Item embedding: `nn.Embedding(item_num+1, hidden_units, padding_idx=0)`
- Positional embedding: `nn.Embedding(maxlen, hidden_units)`
- Stacked multi-head attention + pointwise FFN blocks
- `forward(mode='log_only')` → returns final hidden state at last position `[:, -1, :]` (shape: `[batch, hidden_units]`) — this is the CF user representation
- `forward(mode='item')` → returns `(log_feats, pos_embs, neg_embs)` all reshaped flat — used in Stage 1 to get item embeddings

### `llm4rec` (`models/llm4rec.py`)
Wrapper around the language model. For SmolVLM:
- Loads `HuggingFaceTB/SmolVLM2-2.2B-Instruct` via `AutoModelForImageTextToText`
- Uses `flash_attention_2` attention implementation
- Loaded in `bfloat16` (or 4-bit if `--load_in_4bit`)
- Stores `self.processor` (full `Idefics3Processor` with `do_image_splitting=False`)
- `self.llm_tokenizer = self.processor.tokenizer`
- Registers special tokens: `[UserRep]`, `[HistoryEmb]`, `[CandidateEmb]` — all models
- Resizes token embeddings after adding special tokens: `llm_model.resize_token_embeddings(len(tokenizer))`
- Freezes all base model weights first, then applies LoRA with `get_peft_model()`
- LoRA config: `r=16`, `lora_alpha=32`, `target_modules=["q_proj","k_proj","v_proj","o_proj"]`, `lora_dropout=0.05`, `bias="none"`
- `llm_hidden_size` property: reads `cfg.text_config.hidden_size` for SmolVLM (nested config), `cfg.hidden_size` for OPT

### `A_llmrec_model` (`models/a_llmrec_model.py`)
The main model. Composed of:
- `self.recsys` — frozen RecSys (SASRec)
- `self.mlp` — `two_layer_mlp(rec_sys_dim)` — CF item autoencoder (Stage 1)
- `self.mlp2` — `two_layer_mlp(768)` — SBERT autoencoder (Stage 1 only, not created otherwise)
- `self.sbert` — `SentenceTransformer('nq-distilbert-base-v1')` (Stage 1 only)
- `self.llm` — `llm4rec` (Stage 2 / inference only)
- `self.log_emb_proj` — `nn.Sequential(Linear(rec_sys_dim→llm_hidden), LayerNorm, LeakyReLU, Linear)` — projects CF user rep into LLM token space
- `self.item_emb_proj` — `nn.Sequential(Linear(128→llm_hidden), LayerNorm, GELU, Linear)` — projects joint item embedding into LLM token space
- `self._image_cache` — `{int_id: PIL.Image}` dict loaded at `__init__` time by `_preload_images()`

---

## 5. Training Stages

### Stage 1: CF ↔ Text Alignment (`pre_train_phase1`)

**What's trained:** `self.sbert`, `self.mlp`, `self.mlp2`

**Data flow per batch:**
1. SASRec forward (`mode='item'`) with no grad → `log_emb`, `pos_emb`, `neg_emb` (all `[batch*maxlen, 50]`)
2. Take only last-position indices: `indices = [maxlen*(i+1)-1 for i in range(batch_size)]`
3. Process in sub-batches of 60 (to keep SBERT memory manageable):
   - Look up item text via `find_item_text(pos_ids)` and `find_item_text(neg_ids)`
   - SBERT encodes text → `pos_text_embedding`, `neg_text_embedding` (768-dim)
   - CF autoencoder: `mlp(pos_emb)` → `(pos_latent_cf, pos_recon_cf)`
   - Text autoencoder: `mlp2(pos_text_embedding)` → `(pos_latent_text, pos_recon_text)`
4. Losses:
   - BPR rec loss: `BCEWithLogitsLoss` on `(log_emb * recon_cf).mean(axis=1)` for pos/neg
   - Matching loss: `MSE(pos_latent_cf, pos_latent_text)` — aligns CF and text latent spaces
   - CF reconstruction: `MSE(pos_recon_cf, pos_emb)` (weight 0.5)
   - Text reconstruction: `MSE(pos_recon_text, pos_text_embedding.data)` (weight 0.2)
   - **Total: `bpr + matching + 0.5*cf_recon + 0.2*text_recon`**

**Checkpoint saved after each epoch and every `max(10, num_batch//100)` steps:**
```
models/saved_models/{dataset}_{recsys}_{epoch1}_{sbert,mlp,mlp2}.pt
```

### Stage 2: LLM Alignment (`pre_train_phase2`)

**What's trained:** `self.log_emb_proj`, `self.item_emb_proj`, LoRA adapters in `self.llm.llm_model`

**Loads Stage 1 checkpoint:** `phase1_epoch = 10` (hardcoded in `train_model.py`)

**Data flow per batch:**
1. SASRec forward (`mode='log_only'`) with no grad → `log_emb` (`[batch, 50]`)
2. For each user `i`:
   - Target: `pos[i][-1]` (last positive item in sequence)
   - History: `seq[i][seq[i]>0]` non-padded item IDs; take last 10 via `make_interact_text`
   - Candidates: 1 positive + 19 random negatives, shuffled via `make_candidate_text`
   - Build prompt text (see Section 6 below)
   - For SmolVLM: collect PIL images for last `min(5, len(history[-10:]))` history items
3. Project user rep: `log_emb_proj(log_emb)` → `[batch, llm_hidden]`
4. Project item embeddings: `item_emb_proj(get_item_emb(item_ids))` for history + candidates
5. Pass to `self.llm.forward(log_emb, samples)` which computes LM loss

**Checkpoint saved:**
```
models/saved_models/{dataset}_{recsys}_{epoch1}_{llm}_{epoch2}_{log_proj,item_proj}.pt
models/saved_models/{dataset}_{recsys}_{epoch1}_{llm}_{epoch2}_lora.pt  # SmolVLM only
```

### Inference (`generate`)

**Loads both Stage 1 and Stage 2 checkpoints.** `phase2_epoch = args.num_epochs`.

**Same prompt construction as Stage 2** but uses `model.eval()` and `torch.no_grad()`.

**For SmolVLM, uses a manual KV-cache greedy decode loop** (not `model.generate()`):
1. First forward pass with full `input_ids`, `inputs_embeds`, `pixel_values` → gets `past_key_values`
2. Then loops up to 49 more steps, feeding only the last generated token each time
3. Stops early if all sequences hit EOS
4. This avoids a bug in `prepare_inputs_for_generation` that corrupts image-token positions when `inputs_embeds` is provided

**For OPT**, uses standard `model.generate()` with greedy search (`do_sample=False`, `num_beams=1`, `max_new_tokens=50`).

**Output format** (appended to `args.inference_output_file`):
```
Answer: "correct item title"

LLM: predicted text

--------------------------------
```

---

## 6. Prompt Construction (SmolVLM Path)

The prompt is built per-user in `pre_train_phase2` and `generate`:

```python
# Raw text (before wrap_prompt):
" is a user representation."
"This user has bought "
"\"title_1\"[HistoryEmb], ..., \"title_5\"[HistoryEmb], "
"\"title_6\"[HistoryEmb]<image>, ..., \"title_10\"[HistoryEmb]<image>"
" in the previous. Recommend one next item for this user to buy next "
"from the following item title set, "
"\"cand_1\"[CandidateEmb], ..., \"cand_20\"[CandidateEmb]"
". The recommendation is "

# After wrap_prompt (SmolVLM only):
"<|im_start|>User: {raw_text}<end_of_utterance>\nAssistant: "
```

**Key point:** the prompt text starts with a **space followed by "is a user representation."**. There is NO `[UserRep]` in the text string. The user representation is prepended as a continuous embedding vector via `torch.cat([log_emb, inputs_embeds], dim=1)` — it occupies position 0 in the embedding sequence but has no corresponding text token.

**`[HistoryEmb]` and `[CandidateEmb]`** are text tokens in the prompt. Their positions in `inputs_embeds` are located by token ID and replaced in-place by projected item embeddings (`replace_hist_candi_token`).

**`<image>`** tokens are added by `make_interact_text` for the last `min(5, len(history[-10:]))` history items (items 6–10 when history length ≥ 10). The `Idefics3Processor` expands each `<image>` into a sequence of image patch tokens when tokenizing.

---

## 7. Image Handling

### `_preload_images()` — called in `__init__`
Iterates `self.id_to_asin` and loads every `data/images/{dataset}/{asin}.jpg` into `self._image_cache = {int_id: PIL.Image}`. Failed loads are silently skipped. This warms the entire image dataset into RAM at startup to avoid per-sample disk I/O during training.

### `load_history_images(item_ids, n=5)`
- Takes the last `n` items from `item_ids`
- Looks each up in `self._image_cache`
- Falls back to a 100×100 black RGB image (`Image.fromarray(np.zeros((100,100,3), dtype=np.uint8))`) for cache misses
- Pads with black images at the **front** if history length < `n`
- Always returns exactly `n` PIL Images

**Critical invariant:** the `n` passed here must exactly match the number of `<image>` tokens emitted by `make_interact_text`. Both use `n_images = min(5, len(interact_ids[-10:]))`. If they diverge, `Idefics3Processor` crashes.

---

## 8. Non-Obvious Implementation Details

### dtype alignment for `inputs_embeds`
The projection heads (`log_emb_proj`, `item_emb_proj`) run in float32. SmolVLM's vision encoder runs in bfloat16. When `Idefics3`'s `inputs_merger` scatters vision features into `inputs_embeds`, both tensors must share dtype. Fix: before the model forward, cast `inputs_embeds` to `next(self.llm_model.parameters()).dtype` (bfloat16) when `pixel_values is not None`. See `a_llmrec_model.py` in `generate()` and `llm4rec.py` in `forward()`.

### Dummy pad token prepended to `input_ids`
The user representation embedding is prepended to `inputs_embeds` via `torch.cat`, making it one token longer than `llm_tokens.input_ids`. But `Idefics3` uses `input_ids` to locate `<image>` token positions in `inputs_embeds`. If lengths mismatch, the index lookup is off-by-one. Fix: prepend a single `pad_token_id` to `input_ids`. The pad ID is not the image-token ID so `Idefics3` ignores position 0 (the user-rep slot).

### LoRA applied AFTER the freeze loop
All base LLM parameters are frozen first (`param.requires_grad = False`). LoRA is applied with `get_peft_model()` afterwards. Order matters: if LoRA is applied before the freeze loop, the loop will freeze the LoRA adapter weights (A, B matrices), which are the only parameters that should train.

LoRA weights are saved/loaded separately from the projection heads as `lora.pt`. At load time:
```python
lora_state = {k: v for k, v in self.llm_model.state_dict().items() if 'lora_' in k}
torch.save(lora_state, out_dir + 'lora.pt')
# ...
lora_state = torch.load(out_dir + 'lora.pt', ...)
self.llm.llm_model.load_state_dict(lora_state, strict=False)
```

### SmolVLM manual greedy decode (inference only)
`model.generate()` internally sets `input_ids=None` whenever `inputs_embeds` is provided (inside `prepare_inputs_for_generation`). This causes `inputs_merger` to fall back to an embedding-comparison heuristic that gives wrong image-token counts and crashes with "not divisible by patch_size". The fix: run the first forward pass manually (passing both `input_ids` and `inputs_embeds` so `inputs_merger` uses the correct branch), take the KV cache, then decode token-by-token without `inputs_embeds`.

### OPT token ID 0 remapping at decode
OPT uses token ID 0 as padding. `batch_decode(..., skip_special_tokens=True)` doesn't strip it since it's not registered as a special token. Fix (OPT only): `outputs[outputs == 0] = 2` before decoding. This line is intentionally absent for SmolVLM where token 0 is `<|endoftext|>`, a real content token.

### Chat template for SmolVLM-Instruct
SmolVLM2-2.2B-Instruct was trained with the Idefics3 chat format. Without the `<|im_start|>User: ...<end_of_utterance>\nAssistant:` wrapper, the instruct-tuned model generates verbose off-topic text. OPT (not instruction-tuned) receives the raw prompt unchanged. See `llm4rec.wrap_prompt()`.

### `[UserRep]` special token is registered but never appears in prompt text
`[UserRep]` is added as a special token (so it gets a unique token ID that won't collide with vocabulary). It is referenced conceptually — the user representation *occupies* the role of a `[UserRep]` token — but the actual text string passed to the tokenizer never contains `[UserRep]`. The user rep embedding is injected by prepending `log_emb` directly in embedding space.

### Stage 1 `phase1_epoch` hardcoded to 10
In `train_model.py`, `train_model_phase2_` and `inference_` both hardcode `phase1_epoch = 10`. This means Stage 2 always loads the Stage 1 checkpoint from epoch 10 regardless of `--num_epochs`. If you train Stage 1 for a different number of epochs, update this value manually.

### Wandb is completely disabled by default
`train_model.py` sets `wandb = None` at module level. The `--use_wandb` flag and related args exist but `wandb` is never imported, so all wandb calls are guarded by `_should_use_wandb()` which returns `False` unless `--use_wandb` is passed.

---

## 9. Checkpoint Save/Load Summary

### Stage 1 saves (path prefix: `models/saved_models/{dataset}_{recsys}_{epoch1}_`):
- `sbert.pt` — fine-tuned SBERT state dict
- `mlp.pt` — CF item autoencoder weights
- `mlp2.pt` — text (SBERT) autoencoder weights

### Stage 2 saves (path prefix: `models/saved_models/{dataset}_{recsys}_{epoch1}_{llm}_{epoch2}_`):
- `log_proj.pt` — user rep projection head weights
- `item_proj.pt` — item embedding projection head weights
- `lora.pt` — LoRA adapter weights (SmolVLM only, extracted by filtering keys containing `'lora_'`)

### Load sequence at inference:
1. Load `mlp.pt` into `self.mlp`, then **freeze** it
2. Load `log_proj.pt` → `self.log_emb_proj`
3. Load `item_proj.pt` → `self.item_emb_proj`
4. Load `lora.pt` → `self.llm.llm_model` with `strict=False` (SmolVLM only)

---



---

## 11. Running the Code

```bash
# Pre-train SASRec backbone (also generates text_name_dict and id_to_asin)
cd pre_train/sasrec
python main.py --dataset All_Beauty

# Download product images (one JPEG per ASIN)
cd ../../
python data/images/download_images.py \
  --dataset All_Beauty \
  --reviews data/amazon/All_Beauty.json.gz \
  --metadata data/amazon/meta_All_Beauty.json \
  --output_dir data/images/All_Beauty

# Stage 1: align CF embeddings with SBERT
python main.py --pretrain_stage1 --rec_pre_trained_data All_Beauty --llm smolvlm

# Stage 2: train projection heads + LoRA (with images for SmolVLM)
python main.py --pretrain_stage2 --rec_pre_trained_data All_Beauty --llm smolvlm

# Optional: 4-bit quantization to reduce GPU memory
python main.py --pretrain_stage2 --rec_pre_trained_data All_Beauty --llm smolvlm --load_in_4bit

# Inference
python main.py --inference --rec_pre_trained_data All_Beauty --llm smolvlm \
  --inference_output_file ./results/smol/recommendation_output_smol_v1_2B.txt

# Evaluate Hit@1
python eval.py --file results/smol/recommendation_output_smol_v1_2B.txt
```

### Key CLI args (main.py):
| Arg | Default | Notes |
|-----|---------|-------|
| `--llm` | `opt` | `opt` or `smolvlm` |
| `--recsys` | `sasrec` | only SASRec supported |
| `--rec_pre_trained_data` | `Movies_and_TV` | dataset name, must match checkpoint dir name |
| `--load_in_4bit` | off | 4-bit NF4 quantization for SmolVLM |
| `--batch_size1` | 32 | Stage 1 batch size |
| `--batch_size2` | 4 | Stage 2 batch size |
| `--batch_size_infer` | 32 | Inference batch size |
| `--num_epochs` | 10 | epochs for whichever stage is active |
| `--stage1_lr` / `--stage2_lr` | 1e-4 | learning rates |
| `--multi_gpu` | off | enables DDP via `mp.spawn` |
| `--inference_output_file` | `./results/smol/recommendation_output_smol_v1_2B.txt` | output path |
