# Design: Diagnosing and Addressing Visual Neglect in LLM-Based Sequential Recommendation

**Date:** 2026-04-09
**Branch:** `visual-dropout-ablation` (to be forked from `smol-img`)
**Dataset:** Luxury_Beauty (9912 valid test users)
**Model:** SmolVLM2-2.2B-Instruct + SASRec backbone

---

## 1. Motivation

The A-LLMRec system with SmolVLM achieved 59% Hit@1 on Luxury_Beauty when images (5 history items) and LoRA were added simultaneously, compared to 56% without either. However, this +3% gain is **confounded** — LoRA (which makes the LLM partially trainable) and images were introduced together. We do not know whether images contribute anything, or whether LoRA alone explains the improvement.

Three papers from the literature review ground this concern:

- **"Hidden in plain sight" (Paper 3):** VLMs consistently fail to use visual tokens, defaulting to language priors. Vision representations are preserved throughout the model — the LLM just doesn't attend to them. Fine-tuning the LLM (via LoRA) provides the largest gains, specifically by overcoming language biases.

- **"Are Multimodal Embeddings Truly Beneficial" (Paper 4):** Text dominates in multimodal recommenders. Image alone doesn't improve performance. Simple fusion shows limited gains — sophisticated fusion mechanisms are needed. Critically: "merely demonstrating superior performance over a baseline is insufficient proof of efficacy." Their modality knockout methodology (replacing one modality with noise/constants) is a diagnostic tool.

- **"Lost in Embeddings" (Paper 6):** VLM connectors distort visual representations (40-60% k-NN divergence post-projection), but the information is recoverable. Reconstruction loss at answer-relevant image patches correlates with task failure. Exception: Qwen2.5-VL's unfrozen vision encoder produces more semantically meaningful post-projection embeddings.

**Core problem:** The training objective is "predict the next item's title." The LLM can minimize this loss entirely through text (history titles + candidate titles + CF embeddings). Images are redundant given text, so the model ignores them.

---

## 2. Research Questions

**RQ1:** In A-LLMRec with SmolVLM, does the performance gain from adding images come from images themselves, or from LoRA fine-tuning introduced alongside them?

**RQ2:** Does the LLM attend to visual tokens, or does it rely entirely on text and CF embeddings?

**RQ3:** Can training-time modality dropout — repurposing the diagnostic knockout from Paper 4 as an intervention — force the LLM to utilize visual tokens?

---

## 3. Experimental Design

### 3.1 Ablation Grid (answers RQ1)

A 2x2 factorial design isolating LoRA and image contributions:

| Run | LoRA | Images | Dropout | --experiment name        | Status    |
|-----|------|--------|---------|--------------------------|-----------|
| A   | No   | No     | 0.0     | ablation-no-lora-no-img  | Exists (~56%) |
| B   | Yes  | No     | 0.0     | ablation-lora-no-img     | **New**   |
| C   | No   | Yes    | 0.0     | ablation-no-lora-img     | **New**   |
| D   | Yes  | Yes    | 0.0     | ablation-lora-img        | **New** (retrain) |

- A vs B: isolates LoRA contribution (if B >> A, LoRA is the driver)
- A vs C: isolates image contribution without LoRA (if C >> A, images help even with frozen LLM)
- B vs D: isolates image contribution on top of LoRA (the key comparison)
- Interaction effect: (D - B) vs (C - A) — do LoRA and images interact?

Run A already exists as the `smol` branch result (~56% on Luxury_Beauty).
Run D corresponds to the current `smol-img` configuration (~59%), but must be retrained because:
1. The existing checkpoints were overwritten by `smol-img-3-rmv-blk-img` (confirmed via LoRA key fingerprinting — 54 `out_proj` keys present that `smol-img` doesn't use)
2. The existing inference output (`smol_images_in_prompt_LuxBeauty_v4.txt`) was only a partial run (2099/9912 users = 21.2%)

### 3.2 Attention Analysis (answers RQ2)

Using Run D's trained model (LoRA + images, no dropout):

1. Run inference on a sample of test users
2. Extract per-layer attention weights from the LLM's transformer layers
3. Compute mean attention at `<image>` token positions vs text token positions per layer
4. Compare against a "uniform attention" baseline

If the model ignores images: attention at `<image>` positions will be near-uniform or below average — confirming visual neglect in the RecSys setting, extending Paper 3's VQA findings.

Also compare Run B (LoRA, no images) vs Run D (LoRA, with images) — does the presence of images change how the LLM attends to text tokens?

### 3.3 Text Modality Dropout (answers RQ3)

One additional run beyond the ablation grid:

| Run | LoRA | Images | Dropout | --experiment name        |
|-----|------|--------|---------|--------------------------|
| E   | Yes  | Yes    | 0.3     | dropout-lora-img-p03     |

During Stage 2 training, with probability p=0.3, replace a history item's title text with `"[MASKED]"` but keep its `[HistoryEmb]` CF embedding and `<image>` token intact.

**Mechanism:** When a title is masked, the LLM still has:
- `[HistoryEmb]` — the CF embedding (what users who bought this also bought)
- `<image>` — the visual appearance (what this item looks like)

The LLM cannot rely on text matching for masked items. To minimize the LM loss, it must learn to extract useful signal from the image tokens. Over thousands of training steps, this builds visual attention patterns that persist at inference (where no masking occurs).

**Novel contribution:** Paper 4 uses modality knockout as a **diagnostic** tool (replace modality, measure performance drop). We repurpose it as a **training strategy** — a novel application not previously explored in the literature.

Compare:
- D vs E (Hit@1): does dropout improve recommendation accuracy?
- D vs E (attention weights at `<image>` positions): does dropout increase visual attention?

**If dropout works:** Higher attention to `<image>` tokens + improved Hit@1 = evidence that forced visual attention helps recommendations.

**If dropout doesn't improve Hit@1 but increases attention:** The LLM can be taught to attend to images, but visual information doesn't improve recommendation for this dataset/domain — a significant negative finding for the multimodal RecSys community.

### 3.4 Total Runs

**5 runs total (4 new):**

| Run | LoRA | Images | Dropout | Status |
|-----|------|--------|---------|--------|
| A   | No   | No     | 0.0     | Exists |
| B   | Yes  | No     | 0.0     | New    |
| C   | No   | Yes    | 0.0     | New    |
| D   | Yes  | Yes    | 0.0     | New    |
| E   | Yes  | Yes    | 0.3     | New    |

---

## 4. Implementation

### 4.1 Branch and Experiment Directory

- Fork `visual-dropout-ablation` from `smol-img`
- All checkpoints saved to `models/saved_models/{experiment}/` (using the `--experiment` flag already added to `main.py` and `a_llmrec_model.py`)
- Inference outputs to `results/{experiment}/` with descriptive filenames

### 4.2 Code Changes

**Change 1 — `main.py`: New CLI arguments**

```python
parser.add_argument("--use_lora", action="store_true",
                    help="Enable LoRA on SmolVLM (for ablation runs without LoRA)")
parser.add_argument("--use_images", action="store_true",
                    help="Enable image injection in prompt (for ablation runs without images)")
parser.add_argument("--visual_dropout", type=float, default=0.0,
                    help="Probability of masking history item titles during Stage 2 training (0.0 = disabled)")
```

Currently LoRA is always applied when `llm=smolvlm`, and images are always used when `llm=smolvlm`. These flags decouple them so the ablation grid is possible from a single branch.

**Change 2 — `models/llm4rec.py`: Conditional LoRA**

The LoRA block in `__init__` becomes gated on `args.use_lora`:

```python
if llm_model == "smolvlm" and load_lora:  # new parameter
    lora_config = LoraConfig(...)
    self.llm_model = get_peft_model(self.llm_model, lora_config)
```

When `use_lora=False`, the LLM is fully frozen — only `log_emb_proj` and `item_emb_proj` are trainable (identical to the `smol` branch behavior, but with the SmolVLM model).

**Change 3 — `models/a_llmrec_model.py`: Conditional images**

The `use_images` flag reads from `args.use_images` instead of being hardcoded to `args.llm == 'smolvlm'`:

```python
# In pre_train_phase2 and generate:
use_images = getattr(self.args, 'use_images', False)  # was: (self.args.llm == 'smolvlm')
```

When `use_images=False`: no `<image>` tokens, no image collection, no Idefics3Processor — pure text path with SmolVLM. The llm4rec.forward() takes the plain tokenizer path (no pixel_values).

**Change 4 — `models/a_llmrec_model.py`: Text dropout in `make_interact_text`**

```python
def make_interact_text(self, interact_ids, interact_max_num, use_images=False):
    interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
    interact_text = []
    if interact_max_num == 'all':
        for title in interact_item_titles_:
            interact_text.append(title + '[HistoryEmb]')
    else:
        titles_slice = interact_item_titles_[-interact_max_num:]
        for j, title in enumerate(titles_slice):
            suffix = '[HistoryEmb]'
            if use_images and j >= len(titles_slice) - 5:
                suffix += '<image>'
                # NEW: Training-time modality dropout
                # Mask the title text but keep CF embedding + image token.
                # Forces the LLM to extract signal from visual tokens.
                if self.training and random.random() < self.args.visual_dropout:
                    title = '"[MASKED]"'
            interact_text.append(title + suffix)
        interact_ids = interact_ids[-interact_max_num:]

    interact_text = ','.join(interact_text)
    return interact_text, interact_ids
```

Note: dropout only applies to items that have `<image>` tokens (the last 5 of 10 history items). Items 1-5 (text + CF only, no image) are never masked — masking them would remove all signal since they have no image to fall back on.

**Change 5 — `extract_attention.py`: New analysis script**

A standalone script (not part of training) that:
1. Loads a trained checkpoint
2. Runs a forward pass on a batch of test users
3. Extracts attention weights from each LLM transformer layer
4. For each layer, computes:
   - Mean attention weight at `<image>` token positions (across all heads)
   - Mean attention weight at text token positions
   - Mean attention weight at `[HistoryEmb]` token positions
   - Mean attention weight at `[CandidateEmb]` token positions
5. Saves results to a JSON file for plotting

This produces the data for Paper 3-style attention analysis figures.

### 4.3 Files Modified

| File | Change | Lines |
|------|--------|-------|
| `main.py` | Add `--use_lora`, `--use_images`, `--visual_dropout` args | ~6 |
| `models/llm4rec.py` | Gate LoRA on `use_lora` flag | ~5 |
| `models/a_llmrec_model.py` | Gate images on `use_images` flag; add dropout in `make_interact_text` | ~15 |
| `extract_attention.py` | New file — attention extraction utility | ~80-100 |

Total code change to existing files: ~25 lines. One new analysis script.

### 4.4 What is NOT changed

- Stage 1 training: completely untouched
- `train_model.py`: untouched (training loop stays the same)
- `eval.py`: untouched (already fixed for whitespace + bidirectional matching)
- SASRec: untouched
- `pre_train/sasrec/`: untouched
- LoRA configuration (r=16, lora_alpha=32, target_modules): same when LoRA is enabled
- Image handling (`_preload_images`, `load_history_images`): same when images are enabled
- Prompt format: same (except `[MASKED]` replacing titles during dropout)
- Number of images: 5 (last 5 history items)
- Number of candidates: 20 (1 positive + 19 negative)

---

## 5. Command Lines (all on GPU 2)

### Stage 1 (shared — only needs to run once)

```bash
python main.py \
  --pretrain_stage1 \
  --rec_pre_trained_data Luxury_Beauty \
  --llm smolvlm \
  --experiment shared-stage1 \
  --num_epochs 10 \
  --gpu_num 2
```

### Stage 2 Training (4 new runs)

**Run B — LoRA, no images:**
```bash
python main.py \
  --pretrain_stage2 \
  --rec_pre_trained_data Luxury_Beauty \
  --llm smolvlm \
  --use_lora \
  --experiment ablation-lora-no-img \
  --stage1_experiment shared-stage1 \
  --num_epochs 10 \
  --gpu_num 2
```

**Run C — No LoRA, images:**
```bash
python main.py \
  --pretrain_stage2 \
  --rec_pre_trained_data Luxury_Beauty \
  --llm smolvlm \
  --use_images \
  --experiment ablation-no-lora-img \
  --stage1_experiment shared-stage1 \
  --num_epochs 10 \
  --gpu_num 2
```

**Run D — LoRA + images (baseline with both):**
```bash
python main.py \
  --pretrain_stage2 \
  --rec_pre_trained_data Luxury_Beauty \
  --llm smolvlm \
  --use_lora \
  --use_images \
  --experiment ablation-lora-img \
  --stage1_experiment shared-stage1 \
  --num_epochs 10 \
  --gpu_num 2
```

**Run E — LoRA + images + dropout 0.3:**
```bash
python main.py \
  --pretrain_stage2 \
  --rec_pre_trained_data Luxury_Beauty \
  --llm smolvlm \
  --use_lora \
  --use_images \
  --visual_dropout 0.3 \
  --experiment dropout-lora-img-p03 \
  --stage1_experiment shared-stage1 \
  --num_epochs 10 \
  --gpu_num 2
```

### Inference (for each run)

```bash
# Run B
python main.py --inference --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_lora --experiment ablation-lora-no-img --num_epochs 10 --gpu_num 2 \
  --inference_output_file ./results/ablation-lora-no-img/output.txt

# Run C
python main.py --inference --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_images --experiment ablation-no-lora-img --num_epochs 10 --gpu_num 2 \
  --inference_output_file ./results/ablation-no-lora-img/output.txt

# Run D
python main.py --inference --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_lora --use_images --experiment ablation-lora-img --num_epochs 10 --gpu_num 2 \
  --inference_output_file ./results/ablation-lora-img/output.txt

# Run E
python main.py --inference --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_lora --use_images --experiment dropout-lora-img-p03 --num_epochs 10 --gpu_num 2 \
  --inference_output_file ./results/dropout-lora-img-p03/output.txt
```

### Evaluation

```bash
python eval.py --file results/ablation-lora-no-img/output.txt
python eval.py --file results/ablation-no-lora-img/output.txt
python eval.py --file results/ablation-lora-img/output.txt
python eval.py --file results/dropout-lora-img-p03/output.txt
```

### Attention Analysis

```bash
# Compare D (no dropout) vs E (dropout)
python extract_attention.py \
  --experiment ablation-lora-img \
  --output results/attention/no-dropout.json \
  --gpu_num 2

python extract_attention.py \
  --experiment dropout-lora-img-p03 \
  --output results/attention/dropout-p03.json \
  --gpu_num 2
```

---

## 6. Expected Outcomes

### Scenario 1 — LoRA explains everything (B ~ 59%, C ~ 56%)
Images don't help. The 56->59% gain was entirely LoRA.
- Paper story: "First empirical evidence that images are ignored in LLM-based sequential RecSys, consistent with Paper 3's findings in VQA"
- Dropout result (E): If E > D, dropout forced the model to use images for the first time. If E ~ D, even forcing doesn't help — visual info isn't useful for this domain.

### Scenario 2 — Images help modestly (B ~ 57-58%, gap between B and D is 1-2%)
Both LoRA and images contribute, images give a small boost.
- Paper story: "Images provide marginal gains under naive injection, consistent with Paper 4's finding that simple fusion underperforms"
- Dropout result becomes the main contribution: if E > D, dropout amplifies the visual signal beyond what naive injection achieves.

### Scenario 3 — Images help substantially (B ~ 56%, C ~ 58%)
Images are genuinely valuable even without LoRA. This would be surprising.
- Paper story: "Contrary to Paper 3, visual tokens ARE used in the recommendation setting"
- Dropout result: if E > D, further improvement through forced attention.

**All three scenarios produce a publishable paper.** The ablation grid is the diagnostic contribution. The dropout is the intervention. The attention analysis is the evidence. The numbers determine which story you tell, not whether you have a story.

---

## 7. Risks and Mitigations

**Risk 1: Run C crashes (no LoRA + images)**
Without LoRA, the LLM is fully frozen. The `llm.train()` call in `pre_train_phase2` would have no effect on the LLM. The processor/image path should still work since it's independent of LoRA.
Mitigation: Test Run C first with 1 epoch to verify it runs.

**Risk 2: Text dropout hurts performance (E < D)**
If masking titles damages the training signal more than it helps visual learning. Titles are a critical signal — they tell the LLM exactly what product the user interacted with. However, the disruption is limited:
- Only 5 of 10 history items are eligible for masking (only those with `<image>` tokens)
- At p=0.3, on average only **1.5 items** per step lose their title — the other 8.5 retain full text
- All 20 candidate items always keep full text — the decision-making signal is never degraded
- Training loss will be higher (noisier optimization) but the model has enough text context to still learn

The deeper risk is whether visual attention patterns learned during training **persist at inference**, where no masking occurs and all titles are available. The model might default back to text-only when text returns, making the training-time disruption pointless.

Mitigation: If E < D, this is still a publishable result — "modality dropout does not improve recommendation despite increasing visual attention, suggesting visual information is not discriminative for beauty products, or that visual attention patterns do not transfer from masked training to unmasked inference."

**Risk 3: Stage 1 checkpoint compatibility**
All runs share the same Stage 1 checkpoint. The current code uses `args.experiment` in both `save_model` and `load_model`, so Stage 2 runs would look for Stage 1 checkpoints in their own experiment directory (where none exist).
Mitigation: Add a `--stage1_experiment` CLI argument that tells `load_model` where to find Stage 1 checkpoints. Default it to `args.experiment` for backward compatibility. All Stage 2 runs pass `--stage1_experiment shared-stage1` to load from the shared Stage 1 directory. This requires ~3 extra lines: one in `main.py` (add argument), one in `load_model` (use `args.stage1_experiment` for the Stage 1 path prefix), and the `train_model.py` change is zero since it already passes `args` through.

**Risk 4: Checkpoints already overwritten**
We confirmed the current `models/saved_models/` checkpoints belong to `smol-img-3-rmv-blk-img`. All runs in this plan use the `--experiment` flag to save to separate subdirectories, preventing future overwrites.

---

## 8. Paper Outline (Draft)

**Title:** "Does Your Multimodal Recommender Actually See? Diagnosing Visual Neglect in LLM-Based Sequential Recommendation"

1. **Introduction:** LLM-based recommenders with VLM backbones can process images, but do they actually use visual information? We present evidence of visual neglect and propose modality dropout as an intervention.

2. **Related Work:** A-LLMRec, AlmostRec, visual neglect in VLMs (Paper 3), multimodal embeddings in RecSys (Paper 4), information loss in VLM connectors (Paper 6).

3. **Method:**
   - A-LLMRec architecture with SmolVLM2-2.2B-Instruct
   - Ablation design: 2x2 factorial (LoRA x images)
   - Training-time modality dropout: knockout as intervention, not just diagnostic
   - Attention analysis methodology

4. **Experiments:**
   - Table 1: Ablation grid results (RQ1)
   - Figure 1: Attention at visual token positions in baseline model (RQ2)
   - Table 2: Dropout vs no-dropout results (RQ3)
   - Figure 2: Attention comparison before/after dropout training (RQ3)

5. **Discussion:** What the results mean for the multimodal RecSys community. Whether visual features are inherently useful for recommendation depends on domain and fusion strategy.

6. **Conclusion**
