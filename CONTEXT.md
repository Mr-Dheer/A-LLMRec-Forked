# Project Context: A-LLMRec with Multimodal Extension

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
- A frozen+fine-tuned Sentence-BERT (SBERT) provides text embeddings from item titles/descriptions
- Two 1-layer MLP autoencoders (one for CF embeddings, one for SBERT embeddings) are trained to match their latent spaces
- Loss = latent matching (MSE) + item reconstruction + text reconstruction + BPR recommendation loss
- Output: "joint collaborative-text embeddings" (128-dim) per item

*Stage 2 — Project into LLM token space and train prompts:*
- Two 2-layer MLPs project (a) CF user representations and (b) joint item embeddings into the LLM's token hidden dimension
- Special markers `[UserRep]`, `[HistoryEmb]`, `[CandidateEmb]` are inserted in the text prompt
- At forward pass, these marker token positions in the embedding matrix are replaced with the projected CF/item vectors
- The LLM (frozen OPT-6.7B) is given the combined prompt and trained to generate the correct next item title
- Only the two projection heads are trained; both SASRec and the LLM stay frozen

**Results (Hit@1):**
| Dataset | SASRec | TALLRec | A-LLMRec |
|---------|--------|---------|----------|
| Movies & TV | 0.6154 | 0.2345 | 0.6237 |
| Video Games | 0.5402 | 0.4403 | 0.5282 |
| Beauty | 0.5298 | 0.5542 | 0.5809 |

Outperforms all baselines in cold, warm, few-shot, cold-user, and cross-domain scenarios.

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

**Results (HR@10):**
- Netflix: 0.1128 (+18% over best baseline E4SRec at 0.0956)
- MicroLens-100K: 0.1249 (+22% over E4SRec at 0.1018)

---

## 2. The Codebase: `A-LLMRec-Forked`

This is a fork of the official A-LLMRec GitHub repository, extended to incorporate ideas from AlmostRec. The active development branch is `smol-img`.

### Directory Structure
```
A-LLMRec-Forked/
├── main.py                     # CLI: --pretrain_stage1, --pretrain_stage2, --inference
├── train_model.py              # Training loops: phase1, phase2, inference
├── eval.py                     # Hit@1 evaluation from output text files
├── utils.py                    # Utility: create_dir, find_filepath
├── requirements.txt
├── models/
│   ├── a_llmrec_model.py       # Core model: two stages + generate
│   ├── llm4rec.py              # LLM wrapper (OPT-6.7B or SmolVLM-Instruct)
│   └── recsys_model.py         # CF wrapper (loads frozen pre-trained SASRec)
├── pre_train/
│   └── sasrec/
│       ├── model.py            # SASRec transformer implementation
│       ├── main.py             # SASRec training entrypoint
│       ├── utils.py            # SeqDataset, data_partition, etc.
│       └── data_preprocess.py  # Amazon review data → interaction sequences + text dicts
├── data/
│   ├── amazon/                 # Amazon review datasets (.txt interaction files, metadata JSON)
│   └── images/
│       └── download_images.py  # Downloads product JPEG images by ASIN
└── results/smol/               # Inference output text files
```

### Data Files (generated by preprocessing)
For a dataset named `All_Beauty`, preprocessing generates:
- `data/amazon/All_Beauty.txt` — user-item interaction sequences (int user_id, int item_id)
- `data/amazon/All_Beauty_text_name_dict.json.gz` — `{title: {int_id: str}, description: {int_id: str}}`
- `data/amazon/All_Beauty_id_to_asin.json.gz` — `{int_id: asin_string}` (added in current work)
- `data/images/All_Beauty/{asin}.jpg` — product images (downloaded separately)

---

## 3. Experiments Run Prior to Current Work

All experiments used the `All_Beauty` Amazon dataset.

| # | Model | Architecture | Hit@1 | Notes |
|---|-------|-------------|-------|-------|
| 1 | SmolVLM2-2.2B-Instruct | Original A-LLMRec, LLM swapped from OPT to SmolVLM2-2.2B, with chat template wrapping | **45%** | First successful SmolVLM integration |
| 2 | SmolVLM-500M-Instruct | Same swap, no chat template | 14% | Too small; also missing chat template |
| 3 | OPT-6.7B | Original A-LLMRec, no changes | 44% | Baseline to beat |
| 4 | OPT-6.7B + ID prediction | Modified to predict item ID instead of text title (branch `id-pred-1`) | **51%** | Best result so far |

**Key finding:** SmolVLM-2B matches OPT-6.7B at 45% hit rate, and ID prediction (instead of text generation) gives a significant boost to 51%. The goal is to now combine SmolVLM's vision capability with the ID prediction approach, but we are implementing them incrementally — images first (text generation still), then integrate ID prediction later.

---

## 4. What Was Just Implemented: Image Integration into Stage 2

### Motivation
AlmostRec demonstrated that feeding product images as an additional modality meaningfully improves recommendation quality. SmolVLM2-2.2B-Instruct (`HuggingFaceTB/SmolVLM2-2.2B-Instruct`) is an LMM (Large Multimodal Model, architecture: `Idefics3ForConditionalGeneration`) that natively supports image inputs via a SigLIP vision encoder. The hypothesis is that showing product images from the user's purchase/watch history gives the model visual context about the user's preferences that text alone cannot capture.

### Design Decision
We feed the **5 most recent** history items as images. Specifically:
- History is shown as 10 items in the text prompt
- Items 6–10 (the 5 most recent) each have both a `[HistoryEmb]` CF embedding injection AND an `<image>` visual token
- Items 1–5 (older history) have only `[HistoryEmb]`
- If an image file is missing or corrupt, a 100×100 black RGB image is used as a silent fallback

This design keeps the existing CF embedding injection mechanism intact while adding visual information for the most contextually relevant (recent) items.

### How SmolVLM Processes Images (Technical Detail)
SmolVLM2 uses the `Idefics3Processor` which expands each `<image>` text placeholder into a variable number of image patch tokens in `input_ids` (with surrounding `<fake_token_around_image>` delimiters). Image splitting is disabled (`do_image_splitting=False`) to keep sequence lengths manageable. With 5 images the total extra tokens are well within the 16k context window.

The model is loaded with `flash_attention_2` for faster inference. The model's own `forward()` method handles the vision injection:
1. Text+images are tokenized by `Idefics3Processor` → expanded `input_ids` + `pixel_values`
2. `inputs_embeds` is computed from `input_ids` via the embedding layer
3. **Our code** replaces `[HistoryEmb]`/`[CandidateEmb]` positions in `inputs_embeds` with projected CF embeddings (unchanged from before)
4. **The model internally** uses `input_ids` to locate `<image>` token positions in `inputs_embeds` and replaces them with vision encoder output
5. The two injection mechanisms operate at different token positions — no conflict

### Files Changed

#### `pre_train/sasrec/data_preprocess.py`
Added saving of an `id_to_asin` mapping (`{int_id: asin_string}`) alongside the existing `text_name_dict`. This is needed at inference/training time to look up the JPEG filename for a given integer item ID. Previously, the `itemmap` (ASIN → int_id) was built but never persisted.

#### `models/llm4rec.py`
- **Processor**: Now stores `self.processor` (full `Idefics3Processor`) in addition to `self.llm_tokenizer`. Previously the processor was discarded after extracting the tokenizer.
- **`forward()`**: When `samples['images']` is present (SmolVLM path), uses `self.processor(text=..., images=...)` to tokenize, getting back expanded `input_ids` + `pixel_values`. Falls back to plain `self.llm_tokenizer` for OPT.
- **Model call**: Now passes `input_ids`, `pixel_values`, and `image_attention_mask` to `self.llm_model()`. `input_ids` is required alongside `inputs_embeds` for Idefics3 to locate image token positions; OPT ignores it when `inputs_embeds` is provided.

#### `models/a_llmrec_model.py`
- **Imports**: Added `os` and `PIL.Image` at module level.
- **`__init__`**: Loads `{dataset}_id_to_asin.json.gz` into `self.id_to_asin`; sets `self.image_dir = ./data/images/{dataset}`; calls `_preload_images()` to warm an in-memory cache.
- **`_preload_images()`**: New method. At init time, iterates all ASINs in `id_to_asin` and loads every JPEG into `self._image_cache = {int_id: PIL.Image}`. Skips failures silently. This avoids repeated disk I/O during training/inference.
- **`load_history_images(item_ids, n=5)`**: Looks up the last `n` item IDs in `self._image_cache`. Falls back to a 100×100 black image for cache misses, and pads with black images at the front if history has fewer than `n` items. Always returns exactly `n` PIL Images.
- **`make_interact_text(interact_ids, interact_max_num, use_images=False)`**: Added `use_images` parameter. When `True`, the last 5 items in the history slice get `[HistoryEmb]<image>` instead of just `[HistoryEmb]`. The number of `<image>` tokens inserted here must exactly match the number of images passed to the processor — a mismatch causes a runtime crash.
- **`pre_train_phase2()`**: Sets `use_images = (self.args.llm == 'smolvlm')`, collects `images_batch` (one list of PIL Images per sample), calculates `n_images = min(5, len(interact_ids[-10:]))` to handle short histories. Includes images in the `samples` dict passed to `self.llm.forward()`.
- **`generate()`**: Same `use_images` flag and `n_images` calculation. Selects between full processor and plain tokenizer for prompt tokenization. Passes `pixel_values`, `image_attention_mask`, and `input_ids` to `llm_model.generate()`.

---

## 5. Current Model Architecture (Stage 2, SmolVLM path)

```
Input per sample:
  - seq[i]         : user's interaction history (integer item IDs, padded to maxlen=50)
  - pos[i]         : target item ID (next item to predict)

Step 1 — CF user representation:
  SASRec(seq[i]) → log_emb (50-dim, frozen)
  log_emb_proj(log_emb) → O_u (2048-dim, trainable 2-layer MLP)

Step 2 — Joint item embeddings (from Stage 1):
  For each history/candidate item_id:
    SASRec.item_emb(item_id) → CF_emb (50-dim, frozen)
    mlp(CF_emb) → latent_emb (128-dim, frozen after Stage 1)
    item_emb_proj(latent_emb) → O_i (2048-dim, trainable 2-layer MLP)

Step 3 — Prompt construction:
  Text: "[UserRep is user representation. This user has bought
          title_1[HistoryEmb], ..., title_5[HistoryEmb],
          title_6[HistoryEmb]<image>, ..., title_10[HistoryEmb]<image>
          in the previous. Recommend one next item ... from
          cand_1[CandidateEmb], ..., cand_20[CandidateEmb].
          The recommendation is"
  Images: [PIL_img_6, PIL_img_7, PIL_img_8, PIL_img_9, PIL_img_10]

Step 4 — Idefics3Processor tokenization:
  processor(text, images) →
    input_ids (expanded: each <image> → ~1k tokens)
    pixel_values (preprocessed image tensors for vision encoder)

Step 5 — Embedding injection:
  inputs_embeds = embedding_layer(input_ids)
  replace [HistoryEmb] positions  → O_i for each history item
  replace [CandidateEmb] positions → O_i for each candidate item
  prepend O_u (user rep) at position 0

Step 6 — LMM forward pass:
  llm_model(input_ids, inputs_embeds, pixel_values, ...)
    ↳ vision encoder processes pixel_values
    ↳ replaces <image> positions in inputs_embeds with visual features
    ↳ standard autoregressive transformer forward
  → generates next item title as text
```

---

## 6. Non-Obvious Implementation Details (Why Things Are Done This Way)

### dtype alignment for inputs_embeds
The projection heads (`log_emb_proj`, `item_emb_proj`) run in float32. SmolVLM's vision encoder runs in bfloat16. When Idefics3's `inputs_merger` scatters vision features into `inputs_embeds`, the two tensors must share the same dtype or PyTorch raises a type mismatch error. Fix: before passing `inputs_embeds` to the model, cast it to `next(self.llm_model.parameters()).dtype` (bfloat16) when `pixel_values` is not None.

### Dummy pad token for input_ids alignment
When we prepend the user representation vector to `inputs_embeds` (via `torch.cat`), `inputs_embeds` becomes one token longer than `llm_tokens.input_ids`. But Idefics3 uses `input_ids` to locate `<image>` token positions. If lengths don't match, the index lookup is off-by-one. Fix: prepend a single pad token ID to `input_ids` before passing to the model. The pad ID is not the image-token ID, so Idefics3 ignores position 0 (the user-rep slot).

### LoRA applied AFTER freezing
All base LLM parameters are frozen in a loop first (`param.requires_grad = False`). LoRA is then applied with `get_peft_model()`. This order matters: if you call `get_peft_model` first, the freeze loop will also freeze the LoRA adapter weights (A, B matrices) which are the only things that should be trainable. LoRA targets `q_proj`, `k_proj`, `v_proj`, `o_proj` with `r=16`, `lora_alpha=32`, `dropout=0.05`. LoRA weights are saved/loaded separately from the projection heads.

### OPT token ID 0 remapping at decode time
OPT uses token ID 0 as the padding token. `batch_decode(..., skip_special_tokens=True)` doesn't strip it by default since it's not registered as a special token. Fix (OPT only): `outputs[outputs == 0] = 2` remaps all 0s to the EOS token (2) before decoding. This line is intentionally skipped for SmolVLM where token ID 0 is `<|endoftext|>`, a real content token.

### Chat template for SmolVLM-Instruct
SmolVLM2-2.2B-Instruct was trained with the Idefics3 chat format. Without the `<|im_start|>User: ... <end_of_utterance>\nAssistant:` wrapper, the model generates verbose off-topic text rather than a product title. OPT (not instruction-tuned) receives the raw prompt unchanged.

### Image count must equal `<image>` token count
`make_interact_text` emits exactly `n_images = min(5, len(interact_ids[-10:]))` `<image>` tokens. The same `n_images` is passed to `load_history_images`. If these two counts differ, the Idefics3Processor will fail to match images to tokens. This coupling is fragile — any future change to how many `<image>` tokens are inserted must be reflected in the image loading call.

### Wandb integration (optional)
`train_model.py` supports optional Weights & Biases logging via `--use_wandb`, `--wandb_project`, `--wandb_run_name`, and `--wandb_log_interval` args. Only rank 0 logs in multi-GPU mode. Phase 1 logs all four loss components; Phase 2 logs the combined LM loss.

---

## 7. What Remains To Do (Planned Next Steps)

1. **Integrate ID prediction** (from branch `id-pred-1`) into the SmolVLM+images pipeline. This replaces text generation with a softmax over all candidate item IDs using a linear transformation layer, preventing hallucinations and giving a 51% baseline. This is the main technique from AlmostRec.

2. **Evaluate images + text generation** first (current state) to measure the impact of images in isolation before adding ID prediction.

3. **Potentially add candidate images** as well (currently only history images are shown).

4. **Explore different numbers of history images** (currently 5; could try 3 or all 10).

---

## 8. Running the Code

```bash
# Pre-train SASRec backbone (generates id_to_asin mapping too)
cd pre_train/sasrec
python main.py --dataset All_Beauty

# Download product images
cd ../../
python data/images/download_images.py \
  --dataset All_Beauty \
  --reviews data/amazon/All_Beauty.json.gz \
  --metadata data/amazon/meta_All_Beauty.json \
  --output_dir data/images/All_Beauty

# Stage 1: align CF embeddings with SBERT
python main.py --pretrain_stage1 --rec_pre_trained_data All_Beauty --llm smolvlm

# Stage 2: train projection heads (now with images for SmolVLM)
python main.py --pretrain_stage2 --rec_pre_trained_data All_Beauty --llm smolvlm

# Inference
python main.py --inference --rec_pre_trained_data All_Beauty --llm smolvlm

# Evaluate
python eval.py --file results/smol/recommendation_output_smol_v1_2B.txt
```
