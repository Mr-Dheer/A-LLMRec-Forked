# Visual Neglect Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CLI flags to independently toggle LoRA and image injection, add text modality dropout during training, and create an attention extraction script for analysis — enabling a 5-run ablation study of visual neglect in LLM-based sequential recommendation.

**Architecture:** Three CLI flags (`--use_lora`, `--use_images`, `--visual_dropout`) decouple LoRA and image injection that are currently hard-coupled to `--llm smolvlm`. A `--stage1_experiment` flag separates Stage 1 checkpoint loading from Stage 2 saving. Text dropout masks history item titles during Stage 2 training with probability `p`, keeping CF embeddings and `<image>` tokens intact. A standalone `extract_attention.py` script hooks into the LLM's attention layers for post-training analysis.

**Tech Stack:** PyTorch, Transformers (SmolVLM2-2.2B-Instruct / Idefics3), PEFT (LoRA), PIL

**Working directory:** `/users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation/`

---

### File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `main.py` | Modify | Add `--use_lora`, `--use_images`, `--visual_dropout`, `--stage1_experiment` CLI args |
| `models/llm4rec.py` | Modify | Gate LoRA application on `use_lora` flag |
| `models/a_llmrec_model.py` | Modify | Gate image injection on `use_images` flag; add text dropout in `make_interact_text`; use `stage1_experiment` in `load_model` |
| `extract_attention.py` | Create | Standalone script to extract and save per-layer attention weights at visual/text token positions |

---

### Task 1: Add CLI Arguments to `main.py`

**Files:**
- Modify: `main.py:14-51`

- [ ] **Step 1: Add the four new arguments**

In `main.py`, after line 15 (`parser.add_argument('--gpu_num', ...)`), add:

```python
    # ablation flags — decouple LoRA and images from --llm choice
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA adapters on SmolVLM (off = fully frozen LLM).")
    parser.add_argument("--use_images", action="store_true",
                        help="Enable image injection in prompt (off = text-only prompt).")
    parser.add_argument("--visual_dropout", type=float, default=0.0,
                        help="Probability of masking a history item title during Stage 2 training.")
    parser.add_argument("--stage1_experiment", type=str, default=None,
                        help="Experiment name to load Stage 1 checkpoints from. Defaults to --experiment.")
```

After `args = parser.parse_args()`, before `args.device = ...`, add:

```python
    if args.stage1_experiment is None:
        args.stage1_experiment = args.experiment
```

- [ ] **Step 2: Verify the arguments parse correctly**

Run:
```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
python main.py --help
```
Expected: the four new arguments (`--use_lora`, `--use_images`, `--visual_dropout`, `--stage1_experiment`) appear in the help output alongside existing args.

- [ ] **Step 3: Commit**

```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
git add main.py
git commit -m "feat: add --use_lora, --use_images, --visual_dropout, --stage1_experiment CLI args"
```

---

### Task 2: Gate LoRA in `llm4rec.py`

**Files:**
- Modify: `models/llm4rec.py:28-34` (constructor signature)
- Modify: `models/llm4rec.py:113-121` (LoRA application block)

- [ ] **Step 1: Add `use_lora` parameter to constructor**

Change the `__init__` signature from:

```python
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
        load_in_4bit=False,
    ):
```

to:

```python
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
        load_in_4bit=False,
        use_lora=False,
    ):
```

- [ ] **Step 2: Gate the LoRA block on `use_lora`**

Change the LoRA application block (currently at lines 113-121) from:

```python
        if llm_model == "smolvlm":
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)
```

to:

```python
        if llm_model == "smolvlm" and use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)
```

- [ ] **Step 3: Update the call site in `a_llmrec_model.py`**

In `models/a_llmrec_model.py`, change the `llm4rec` instantiation (currently at lines 92-96) from:

```python
            self.llm = llm4rec(
                device=self.device,
                llm_model=args.llm,
                load_in_4bit=args.load_in_4bit,
            )
```

to:

```python
            self.llm = llm4rec(
                device=self.device,
                llm_model=args.llm,
                load_in_4bit=args.load_in_4bit,
                use_lora=getattr(args, 'use_lora', False),
            )
```

- [ ] **Step 4: Verify it parses without crashing**

Run:
```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
python -c "from models.llm4rec import llm4rec; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 5: Commit**

```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
git add models/llm4rec.py models/a_llmrec_model.py
git commit -m "feat: gate LoRA application on --use_lora flag"
```

---

### Task 3: Gate Image Injection in `a_llmrec_model.py`

**Files:**
- Modify: `models/a_llmrec_model.py` — `pre_train_phase2` method (line ~433), `generate` method (line ~516)

- [ ] **Step 1: Change `use_images` from hardcoded to flag-driven in `pre_train_phase2`**

In `pre_train_phase2`, change line 433 from:

```python
        use_images = (self.args.llm == 'smolvlm')
```

to:

```python
        use_images = getattr(self.args, 'use_images', False)
```

- [ ] **Step 2: Same change in `generate`**

In `generate`, change line 516 from:

```python
        use_images = (self.args.llm == 'smolvlm')
```

to:

```python
        use_images = getattr(self.args, 'use_images', False)
```

- [ ] **Step 3: Verify the module still imports**

Run:
```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
python -c "from models.a_llmrec_model import A_llmrec_model; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 4: Commit**

```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
git add models/a_llmrec_model.py
git commit -m "feat: gate image injection on --use_images flag"
```

---

### Task 4: Add Text Modality Dropout in `make_interact_text`

**Files:**
- Modify: `models/a_llmrec_model.py` — `make_interact_text` method (lines 352-379)

- [ ] **Step 1: Add dropout logic to `make_interact_text`**

Replace the current `make_interact_text` method (lines 352-379) with:

```python
    def make_interact_text(self, interact_ids, interact_max_num, use_images=False):
        """
        Build the textual part of the user history for the LLM prompt.

        Appends a special marker [HistoryEmb] to each title so we can
        later replace it with the aligned item embedding in the LLM input.

        When use_images=True (SmolVLM path), the last 5 items in the slice
        also get an <image> token appended directly after [HistoryEmb].  The
        Idefics3Processor will expand each <image> into the correct sequence
        of visual-patch tokens when the prompt is tokenized.

        When visual_dropout > 0 and the model is in training mode, each
        image-eligible item's title is replaced with "[MASKED]" with
        probability p.  The [HistoryEmb] CF embedding and <image> token
        remain, forcing the LLM to extract signal from visual tokens.
        """
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        interact_text = []
        if interact_max_num == 'all':
            for title in interact_item_titles_:
                interact_text.append(title + '[HistoryEmb]')
        else:
            titles_slice = interact_item_titles_[-interact_max_num:]
            dropout_p = getattr(self.args, 'visual_dropout', 0.0)
            for j, title in enumerate(titles_slice):
                suffix = '[HistoryEmb]'
                if use_images and j >= len(titles_slice) - 5:
                    suffix += '<image>'
                    # Training-time modality dropout: mask title, keep CF embedding + image
                    if self.training and dropout_p > 0 and random.random() < dropout_p:
                        title = '"[MASKED]"'
                interact_text.append(title + suffix)
            interact_ids = interact_ids[-interact_max_num:]

        interact_text = ','.join(interact_text)
        return interact_text, interact_ids
```

- [ ] **Step 2: Verify the module still imports**

Run:
```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
python -c "from models.a_llmrec_model import A_llmrec_model; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
git add models/a_llmrec_model.py
git commit -m "feat: add text modality dropout in make_interact_text (--visual_dropout)"
```

---

### Task 5: Use `stage1_experiment` in `load_model`

**Files:**
- Modify: `models/a_llmrec_model.py` — `load_model` method (line 137)

- [ ] **Step 1: Change `load_model` to use `stage1_experiment` for Stage 1 path**

Change line 137 from:

```python
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f'./models/saved_models/{args.experiment}/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_'
```

to:

```python
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        stage1_exp = getattr(args, 'stage1_experiment', args.experiment)
        out_dir = f'./models/saved_models/{stage1_exp}/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_'
```

The rest of `load_model` that builds the Stage 2 path (line 147: `out_dir += f'{args.llm}_{phase2_epoch}_'`) continues to use the **original** `out_dir` which is based on `stage1_exp`. This needs fixing — Stage 2 checkpoints should load from `args.experiment`, not `stage1_exp`.

- [ ] **Step 2: Split Stage 1 and Stage 2 paths in `load_model`**

Replace the full `load_model` method with:

```python
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        # Stage 1 checkpoints may live in a different experiment directory
        # (shared across ablation runs).
        stage1_exp = getattr(args, 'stage1_experiment', args.experiment)
        stage1_dir = f'./models/saved_models/{stage1_exp}/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_'

        # Load Stage 1 alignment MLP and freeze it for later stages.
        mlp = torch.load(stage1_dir + 'mlp.pt', map_location=args.device)
        self.mlp.load_state_dict(mlp)
        del mlp
        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        if args.inference:
            # Stage 2 checkpoints use the current experiment directory.
            stage2_dir = f'./models/saved_models/{args.experiment}/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_{args.llm}_{phase2_epoch}_'

            log_emb_proj_dict = torch.load(stage2_dir + 'log_proj.pt', map_location=args.device)
            self.log_emb_proj.load_state_dict(log_emb_proj_dict)
            del log_emb_proj_dict

            item_emb_proj_dict = torch.load(stage2_dir + 'item_proj.pt', map_location=args.device)
            self.item_emb_proj.load_state_dict(item_emb_proj_dict)
            del item_emb_proj_dict

            # Load LoRA adapter weights (SmolVLM only).
            if args.llm == 'smolvlm' and getattr(args, 'use_lora', False):
                lora_state = torch.load(stage2_dir + 'lora.pt', map_location=args.device)
                self.llm.llm_model.load_state_dict(lora_state, strict=False)
```

Note: the LoRA loading is now also gated on `use_lora` — if LoRA is disabled, no `lora.pt` was saved and we don't try to load it.

- [ ] **Step 3: Update `save_model` to also gate LoRA saving**

In `save_model` (lines 131-134), change:

```python
            if args.llm == 'smolvlm':
                lora_state = {k: v for k, v in self.llm.llm_model.state_dict().items() if 'lora_' in k}
                torch.save(lora_state, out_dir + 'lora.pt')
```

to:

```python
            if args.llm == 'smolvlm' and getattr(args, 'use_lora', False):
                lora_state = {k: v for k, v in self.llm.llm_model.state_dict().items() if 'lora_' in k}
                torch.save(lora_state, out_dir + 'lora.pt')
```

- [ ] **Step 4: Verify the module still imports**

Run:
```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
python -c "from models.a_llmrec_model import A_llmrec_model; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 5: Commit**

```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
git add models/a_llmrec_model.py
git commit -m "feat: separate stage1/stage2 experiment paths in load_model, gate LoRA save/load on --use_lora"
```

---

### Task 6: Create `extract_attention.py`

**Files:**
- Create: `extract_attention.py`

- [ ] **Step 1: Create the attention extraction script**

Create `extract_attention.py` in the project root:

```python
"""
Extract per-layer attention weights at <image> vs text token positions.

Usage:
    python extract_attention.py \
        --experiment ablation-lora-img \
        --rec_pre_trained_data Luxury_Beauty \
        --num_epochs 10 \
        --gpu_num 2 \
        --num_samples 100 \
        --output results/attention/ablation-lora-img.json
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from models.a_llmrec_model import A_llmrec_model
from pre_train.sasrec.utils import data_partition, SeqDataset_Inference
from torch.utils.data import DataLoader


def get_token_positions(input_ids, tokenizer):
    """Classify each token position as 'image', 'history', 'candidate', or 'text'."""
    history_id = tokenizer("[HistoryEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
    candidate_id = tokenizer("[CandidateEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()

    # <image> tokens get expanded by Idefics3 into fake_token_around_image + patch tokens.
    # We identify image tokens as those whose ID matches the image token.
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    fake_token_id = tokenizer.convert_tokens_to_ids("<fake_token_around_image>")

    positions = {"image": [], "history": [], "candidate": [], "text": []}
    for idx in range(input_ids.size(0)):
        tok = input_ids[idx].item()
        if tok == image_token_id or tok == fake_token_id:
            positions["image"].append(idx)
        elif tok == history_id:
            positions["history"].append(idx)
        elif tok == candidate_id:
            positions["candidate"].append(idx)
        else:
            positions["text"].append(idx)
    return positions


def extract_attention(model, data_loader, tokenizer, device, num_samples):
    """Run forward passes and collect attention statistics."""
    model.eval()

    # Storage: per-layer mean attention at each token type
    layer_stats = {}
    samples_processed = 0

    for _, data in enumerate(data_loader):
        if samples_processed >= num_samples:
            break
        u, seq, pos, neg = data
        u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()

        use_images = getattr(model.args, 'use_images', False)

        with torch.no_grad():
            log_emb = model.recsys.model(u, seq, pos, neg, mode='log_only')

            # Build prompt for first sample only (attention analysis is per-sample)
            i = 0
            target_item_id = pos[i]
            target_item_title = model.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
            interact_text, interact_ids = model.make_interact_text(seq[i][seq[i] > 0], 10, use_images=use_images)
            candidate_num = 20
            candidate_text, candidate_ids = model.make_candidate_text(seq[i][seq[i] > 0], candidate_num, target_item_id, target_item_title)

            input_text = ' is a user representation.'
            input_text += 'This user has bought '
            input_text += interact_text
            input_text += ' in the previous. Recommend one next item for this user to buy next from the following item title set, '
            input_text += candidate_text
            input_text += '. The recommendation is '

            text_input = model.llm.wrap_prompt(input_text)

            # Tokenize
            if use_images:
                images = model.load_history_images(interact_ids, n=min(5, len(interact_ids[-10:])))
                model.llm.processor.tokenizer.padding_side = "left"
                processed = model.llm.processor(
                    text=[text_input],
                    images=[images],
                    padding="longest",
                    return_tensors="pt",
                ).to(device)
                input_ids = processed.input_ids[0]
            else:
                model.llm.llm_tokenizer.padding_side = "left"
                tokens = model.llm.llm_tokenizer(
                    [text_input],
                    padding="longest",
                    return_tensors="pt",
                ).to(device)
                input_ids = tokens.input_ids[0]

            positions = get_token_positions(input_ids, model.llm.llm_tokenizer)

            # Forward pass with output_attentions=True
            if use_images:
                outputs = model.llm.llm_model(
                    input_ids=processed.input_ids,
                    pixel_values=processed.pixel_values,
                    attention_mask=processed.attention_mask,
                    output_attentions=True,
                    return_dict=True,
                )
            else:
                outputs = model.llm.llm_model(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    output_attentions=True,
                    return_dict=True,
                )

            attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)

            for layer_idx, attn in enumerate(attentions):
                attn_matrix = attn[0].float().cpu().numpy()  # [heads, seq, seq]
                mean_attn = attn_matrix.mean(axis=0)  # [seq, seq] — averaged over heads

                # For each token type, compute mean attention RECEIVED from all other tokens
                stats = {}
                for token_type, idxs in positions.items():
                    if len(idxs) > 0:
                        stats[token_type] = float(mean_attn[:, idxs].mean())
                    else:
                        stats[token_type] = 0.0

                if layer_idx not in layer_stats:
                    layer_stats[layer_idx] = {t: [] for t in ["image", "history", "candidate", "text"]}
                for t in stats:
                    layer_stats[layer_idx][t].append(stats[t])

        samples_processed += 1

    # Average across samples
    result = {}
    for layer_idx in sorted(layer_stats.keys()):
        result[f"layer_{layer_idx}"] = {}
        for t in ["image", "history", "candidate", "text"]:
            vals = layer_stats[layer_idx][t]
            result[f"layer_{layer_idx}"][t] = float(np.mean(vals)) if vals else 0.0

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--stage1_experiment", type=str, default=None)
    parser.add_argument("--rec_pre_trained_data", type=str, default="Luxury_Beauty")
    parser.add_argument("--llm", type=str, default="smolvlm")
    parser.add_argument("--recsys", type=str, default="sasrec")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_images", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--batch_size_infer", type=int, default=1,
                        help="Must be 1 for per-sample attention extraction")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.stage1_experiment is None:
        args.stage1_experiment = args.experiment
    args.device = f'cuda:{args.gpu_num}'
    args.inference = True
    args.pretrain_stage1 = False
    args.pretrain_stage2 = False
    args.visual_dropout = 0.0

    model = A_llmrec_model(args).to(args.device)
    phase1_epoch = 10
    phase2_epoch = args.num_epochs
    model.load_model(args, phase1_epoch=phase1_epoch, phase2_epoch=phase2_epoch)

    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt')
    user_train, user_valid, user_test, usernum, itemnum = dataset

    users = range(1, usernum + 1)
    user_list = [u for u in users if len(user_train[u]) >= 1 and len(user_test[u]) >= 1]
    random.seed(42)
    user_list = random.sample(user_list, min(args.num_samples, len(user_list)))

    inference_data = SeqDataset_Inference(user_train, user_valid, user_test, user_list, itemnum, args.maxlen)
    data_loader = DataLoader(inference_data, batch_size=1, pin_memory=True)

    result = extract_attention(model, data_loader, model.llm.llm_tokenizer, args.device, args.num_samples)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved attention stats for {args.num_samples} samples to {args.output}")
    print("\nSummary (mean attention received by token type):")
    for layer_name in sorted(result.keys(), key=lambda x: int(x.split("_")[1])):
        stats = result[layer_name]
        parts = [f"{t}: {v:.6f}" for t, v in stats.items() if v > 0]
        print(f"  {layer_name}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses arguments**

Run:
```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
python extract_attention.py --help
```
Expected: help output listing `--experiment`, `--output`, `--use_lora`, `--use_images`, etc.

- [ ] **Step 3: Commit**

```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
git add extract_attention.py
git commit -m "feat: add extract_attention.py for per-layer visual attention analysis"
```

---

### Task 7: Smoke Test — Verify All Configurations Parse

**Files:** None (testing only)

- [ ] **Step 1: Test Run B config (LoRA, no images)**

```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
python -c "
import argparse
import sys
sys.argv = ['main.py', '--pretrain_stage2', '--rec_pre_trained_data', 'Luxury_Beauty',
            '--llm', 'smolvlm', '--use_lora', '--experiment', 'test-b',
            '--stage1_experiment', 'shared-stage1', '--num_epochs', '1', '--gpu_num', '2']
exec(open('main.py').read().split('if __name__')[0])
print('Args parsed OK:', args)
print('use_lora:', args.use_lora, 'use_images:', args.use_images, 'visual_dropout:', args.visual_dropout)
print('experiment:', args.experiment, 'stage1_experiment:', args.stage1_experiment)
"
```
Expected: `use_lora: True use_images: False visual_dropout: 0.0`

- [ ] **Step 2: Test Run E config (LoRA + images + dropout)**

```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
python -c "
import argparse
import sys
sys.argv = ['main.py', '--pretrain_stage2', '--rec_pre_trained_data', 'Luxury_Beauty',
            '--llm', 'smolvlm', '--use_lora', '--use_images', '--visual_dropout', '0.3',
            '--experiment', 'test-e', '--stage1_experiment', 'shared-stage1',
            '--num_epochs', '1', '--gpu_num', '2']
exec(open('main.py').read().split('if __name__')[0])
print('Args parsed OK:', args)
print('use_lora:', args.use_lora, 'use_images:', args.use_images, 'visual_dropout:', args.visual_dropout)
"
```
Expected: `use_lora: True use_images: True visual_dropout: 0.3`

- [ ] **Step 3: Test Run C config (no LoRA, images)**

```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
python -c "
import argparse
import sys
sys.argv = ['main.py', '--pretrain_stage2', '--rec_pre_trained_data', 'Luxury_Beauty',
            '--llm', 'smolvlm', '--use_images', '--experiment', 'test-c',
            '--stage1_experiment', 'shared-stage1', '--num_epochs', '1', '--gpu_num', '2']
exec(open('main.py').read().split('if __name__')[0])
print('Args parsed OK:', args)
print('use_lora:', args.use_lora, 'use_images:', args.use_images, 'visual_dropout:', args.visual_dropout)
"
```
Expected: `use_lora: False use_images: False visual_dropout: 0.0`

---

### Task 8: Copy Spec and Plan Docs

**Files:**
- Copy: `docs/superpowers/specs/2026-04-09-visual-neglect-ablation-design.md` from main repo

- [ ] **Step 1: Copy the spec from the main repo to the worktree**

```bash
cp /users/kavach_d/projects/idea-3/A-LLMRec-Forked/docs/superpowers/specs/2026-04-09-visual-neglect-ablation-design.md \
   /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation/docs/superpowers/specs/
```

- [ ] **Step 2: Commit spec and plan**

```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
git add docs/
git commit -m "docs: add visual neglect ablation spec and implementation plan"
```

---

## Run Commands Reference

After all tasks are implemented, these are the exact commands for each experimental run:

### Stage 1 (shared, run once)

```bash
cd /users/kavach_d/projects/idea-3/A-LLMRec-visual-ablation
python main.py --pretrain_stage1 --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --experiment shared-stage1 --num_epochs 10 --gpu_num 2
```

### Run B: LoRA, no images

```bash
# Train
python main.py --pretrain_stage2 --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_lora --experiment ablation-lora-no-img --stage1_experiment shared-stage1 \
  --num_epochs 10 --gpu_num 2

# Inference
python main.py --inference --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_lora --experiment ablation-lora-no-img --stage1_experiment shared-stage1 \
  --num_epochs 10 --gpu_num 2 \
  --inference_output_file ./results/ablation-lora-no-img/output.txt

# Eval
python eval.py --file results/ablation-lora-no-img/output.txt
```

### Run C: No LoRA, images

```bash
# Train
python main.py --pretrain_stage2 --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_images --experiment ablation-no-lora-img --stage1_experiment shared-stage1 \
  --num_epochs 10 --gpu_num 2

# Inference
python main.py --inference --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_images --experiment ablation-no-lora-img --stage1_experiment shared-stage1 \
  --num_epochs 10 --gpu_num 2 \
  --inference_output_file ./results/ablation-no-lora-img/output.txt

# Eval
python eval.py --file results/ablation-no-lora-img/output.txt
```

### Run D: LoRA + images (no dropout)

```bash
# Train
python main.py --pretrain_stage2 --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_lora --use_images --experiment ablation-lora-img --stage1_experiment shared-stage1 \
  --num_epochs 10 --gpu_num 2

# Inference
python main.py --inference --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_lora --use_images --experiment ablation-lora-img --stage1_experiment shared-stage1 \
  --num_epochs 10 --gpu_num 2 \
  --inference_output_file ./results/ablation-lora-img/output.txt

# Eval
python eval.py --file results/ablation-lora-img/output.txt
```

### Run E: LoRA + images + dropout 0.3

```bash
# Train
python main.py --pretrain_stage2 --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_lora --use_images --visual_dropout 0.3 \
  --experiment dropout-lora-img-p03 --stage1_experiment shared-stage1 \
  --num_epochs 10 --gpu_num 2

# Inference (no dropout at inference — visual_dropout defaults to 0.0)
python main.py --inference --rec_pre_trained_data Luxury_Beauty --llm smolvlm \
  --use_lora --use_images \
  --experiment dropout-lora-img-p03 --stage1_experiment shared-stage1 \
  --num_epochs 10 --gpu_num 2 \
  --inference_output_file ./results/dropout-lora-img-p03/output.txt

# Eval
python eval.py --file results/dropout-lora-img-p03/output.txt
```

### Attention Analysis

```bash
# Run D (baseline — no dropout)
python extract_attention.py --experiment ablation-lora-img --stage1_experiment shared-stage1 \
  --use_lora --use_images --num_samples 100 --gpu_num 2 \
  --output results/attention/no-dropout.json

# Run E (with dropout)
python extract_attention.py --experiment dropout-lora-img-p03 --stage1_experiment shared-stage1 \
  --use_lora --use_images --num_samples 100 --gpu_num 2 \
  --output results/attention/dropout-p03.json
```
