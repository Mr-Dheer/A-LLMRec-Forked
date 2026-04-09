"""
Extract per-layer attention weights by token type from a trained A-LLMRec model.

For each transformer layer, computes the mean attention *received* by four
token categories: image, history ([HistoryEmb]), candidate ([CandidateEmb]),
and text (everything else).  Results are saved as JSON and printed as a table.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.a_llmrec_model import A_llmrec_model
from pre_train.sasrec.utils import data_partition, SeqDataset_Inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-layer attention weights by token type from a trained A-LLMRec model."
    )
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name (checkpoint directory).")
    parser.add_argument("--stage1_experiment", type=str, default=None,
                        help="Experiment name for Stage 1 checkpoints (defaults to --experiment).")
    parser.add_argument("--rec_pre_trained_data", type=str, default="Luxury_Beauty",
                        help="Dataset name (default: Luxury_Beauty).")
    parser.add_argument("--llm", type=str, default="smolvlm",
                        help="LLM backend (default: smolvlm).")
    parser.add_argument("--recsys", type=str, default="sasrec",
                        help="RecSys backbone (default: sasrec).")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Stage 2 epoch to load (default: 10).")
    parser.add_argument("--gpu_num", type=int, default=0,
                        help="GPU device index (default: 0).")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of test users to sample (default: 100).")
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA adapters.")
    parser.add_argument("--use_images", action="store_true",
                        help="Enable image injection in prompts.")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load LLM with 4-bit quantization.")
    parser.add_argument("--maxlen", type=int, default=50,
                        help="Max sequence length for SASRec (default: 50).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save JSON results.")
    return parser.parse_args()


def build_token_masks(input_ids, tokenizer, use_images):
    """
    Build boolean masks for each token type in a single input_ids tensor.

    Returns a dict with keys: image, history, candidate, text.
    Each value is a 1-D boolean tensor of shape (seq_len,).
    """
    history_id = tokenizer("[HistoryEmb]", add_special_tokens=False).input_ids[0]
    candidate_id = tokenizer("[CandidateEmb]", add_special_tokens=False).input_ids[0]

    image_token_ids = set()
    if use_images:
        for tok_name in ["<image>", "<fake_token_around_image>"]:
            tid = tokenizer.convert_tokens_to_ids(tok_name)
            # convert_tokens_to_ids returns unk_token_id for unknown tokens
            if tid != tokenizer.unk_token_id:
                image_token_ids.add(tid)

    seq_len = input_ids.shape[0]
    image_mask = torch.zeros(seq_len, dtype=torch.bool)
    history_mask = (input_ids == history_id)
    candidate_mask = (input_ids == candidate_id)

    if image_token_ids:
        for tid in image_token_ids:
            image_mask |= (input_ids == tid)

    text_mask = ~(image_mask | history_mask | candidate_mask)

    return {
        "image": image_mask,
        "history": history_mask,
        "candidate": candidate_mask,
        "text": text_mask,
    }


def compute_attention_by_type(attentions, masks):
    """
    For each layer, compute mean attention *received* by each token type.

    attentions: tuple of (1, num_heads, seq, seq) tensors, one per layer.
    masks: dict of boolean tensors (seq_len,) for each token type.

    Returns dict: {layer_idx: {type_name: float, ...}, ...}
    """
    results = {}
    for layer_idx, attn in enumerate(attentions):
        # attn shape: (1, num_heads, seq_q, seq_k)
        # Mean over batch and heads -> (seq_q, seq_k)
        attn_mean = attn[0].float().mean(dim=0)  # (seq_q, seq_k)

        # Mean attention received = mean over all query positions of
        # attention weight going to each key position, then averaged
        # within the token-type group.
        # attn_mean[q, k] = how much query q attends to key k
        # received_by_k = mean over q of attn_mean[q, k]
        received = attn_mean.mean(dim=0)  # (seq_k,)

        layer_result = {}
        for name, mask in masks.items():
            if mask.any():
                layer_result[name] = received[mask].mean().item()
            else:
                layer_result[name] = 0.0
        results[f"layer_{layer_idx}"] = layer_result

    return results


def main():
    args = parse_args()

    if args.stage1_experiment is None:
        args.stage1_experiment = args.experiment

    # Required model-loading attributes
    args.device = f"cuda:{args.gpu_num}"
    args.inference = True
    args.pretrain_stage1 = False
    args.pretrain_stage2 = False
    args.visual_dropout = 0.0

    print(f"Loading model from experiment '{args.experiment}' ...")
    model = A_llmrec_model(args).to(args.device)
    model.load_model(args, phase1_epoch=10, phase2_epoch=args.num_epochs)
    model.eval()

    # Prepare data
    dataset = data_partition(
        args.rec_pre_trained_data,
        path=f"./data/amazon/{args.rec_pre_trained_data}.txt",
    )
    user_train, user_valid, user_test, usernum, itemnum = dataset
    print(f"user num: {usernum}, item num: {itemnum}")

    # Build user list (same filter as inference)
    users = range(1, usernum + 1)
    user_list = [u for u in users if len(user_train[u]) >= 1 and len(user_test[u]) >= 1]

    # Sub-sample
    if len(user_list) > args.num_samples:
        random.seed(42)
        user_list = random.sample(user_list, args.num_samples)

    inference_data_set = SeqDataset_Inference(
        user_train, user_valid, user_test, user_list, itemnum, args.maxlen,
    )
    data_loader = DataLoader(inference_data_set, batch_size=1, pin_memory=True)

    use_images = args.use_images
    tokenizer = model.llm.llm_tokenizer

    # Accumulators: per-layer, per-type running sums and counts
    layer_sums = {}  # layer_name -> {type -> running_sum}
    n_samples = 0

    print(f"Processing {len(user_list)} users ...")
    for _, batch_data in enumerate(tqdm(data_loader, desc="Extracting attention")):
        u, seq, pos, neg = batch_data
        u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()

        with torch.no_grad():
            log_emb = model.recsys.model(u, seq, pos, neg, mode="log_only")

            i = 0  # batch_size=1
            target_item_id = pos[i][-1] if pos[i].ndim > 0 else pos[i]
            target_item_title = model.find_item_text_single(
                target_item_id, title_flag=True, description_flag=False,
            )

            interact_text, interact_ids = model.make_interact_text(
                seq[i][seq[i] > 0], 10, use_images=use_images,
            )
            candidate_text, candidate_ids = model.make_candidate_text(
                seq[i][seq[i] > 0], 20, target_item_id, target_item_title,
            )

            input_text = " is a user representation."
            if args.rec_pre_trained_data == "Movies_and_TV":
                input_text += "This user has watched "
            elif args.rec_pre_trained_data == "Video_Games":
                input_text += "This user has played "
            else:
                input_text += "This user has bought "
            input_text += interact_text
            if args.rec_pre_trained_data == "Movies_and_TV":
                input_text += " in the previous. Recommend one next movie for this user to watch next from the following movie title set, "
            elif args.rec_pre_trained_data == "Video_Games":
                input_text += " in the previous. Recommend one next game for this user to play next from the following game title set, "
            else:
                input_text += " in the previous. Recommend one next item for this user to buy next from the following item title set, "
            input_text += candidate_text
            input_text += ". The recommendation is "

            text_input = model.llm.wrap_prompt(input_text)

            # Tokenize
            if use_images:
                n_images = min(5, len(interact_ids[-10:]))
                sample_images = model.load_history_images(interact_ids, n=n_images)
                model.llm.processor.tokenizer.padding_side = "left"
                processed = model.llm.processor(
                    text=[text_input],
                    images=[sample_images],
                    padding="longest",
                    return_tensors="pt",
                ).to(args.device)
                llm_tokens = processed
                pixel_values = processed.pixel_values
            else:
                model.llm.llm_tokenizer.padding_side = "left"
                llm_tokens = model.llm.llm_tokenizer(
                    [text_input], padding="longest", return_tensors="pt",
                ).to(args.device)
                pixel_values = None

            # Prepare embeddings (same as generate)
            interact_embs = [model.item_emb_proj(model.get_item_emb(interact_ids))]
            candidate_embs = [model.item_emb_proj(model.get_item_emb(candidate_ids))]

            inputs_embeds = model.llm.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            llm_tokens, inputs_embeds = model.llm.replace_hist_candi_token(
                llm_tokens, inputs_embeds, interact_embs, candidate_embs,
            )

            # Prepend user representation
            log_emb_proj = model.log_emb_proj(log_emb).unsqueeze(1)
            atts_llm = torch.ones(log_emb_proj.size()[:-1], dtype=torch.long, device=args.device)
            atts_llm = atts_llm.unsqueeze(1)

            inputs_embeds = torch.cat([log_emb_proj, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            # Prepend dummy pad token for input_ids alignment (same as generate)
            if pixel_values is not None:
                dummy = torch.full(
                    (llm_tokens.input_ids.size(0), 1),
                    tokenizer.pad_token_id, dtype=torch.long, device=args.device,
                )
                input_ids_for_model = torch.cat([dummy, llm_tokens.input_ids], dim=1)
            else:
                input_ids_for_model = torch.cat([
                    torch.full(
                        (llm_tokens.input_ids.size(0), 1),
                        tokenizer.pad_token_id, dtype=torch.long, device=args.device,
                    ),
                    llm_tokens.input_ids,
                ], dim=1)

            # Cast to model dtype
            model_dtype = next(model.llm.llm_model.parameters()).dtype
            inputs_embeds = inputs_embeds.to(model_dtype)

            # Build token-type masks on the full input_ids (including dummy prefix)
            masks = build_token_masks(input_ids_for_model[0].cpu(), tokenizer, use_images)

            # Forward pass with attention output
            # Note: flash_attention_2 does not support output_attentions.
            # We temporarily switch to eager attention for this analysis.
            orig_attn_impl = getattr(model.llm.llm_model.config, "_attn_implementation", None)
            model.llm.llm_model.config._attn_implementation = "eager"
            # Also update per-layer config for models that read it there
            if hasattr(model.llm.llm_model, "model"):
                inner = model.llm.llm_model.model
                if hasattr(inner, "text_model"):
                    inner = inner.text_model
                if hasattr(inner, "layers"):
                    for layer in inner.layers:
                        if hasattr(layer, "self_attn"):
                            if hasattr(layer.self_attn, "config"):
                                layer.self_attn.config._attn_implementation = "eager"

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.llm.llm_model(
                    input_ids=input_ids_for_model,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    output_attentions=True,
                    return_dict=True,
                )

            # Restore original attention implementation
            if orig_attn_impl is not None:
                model.llm.llm_model.config._attn_implementation = orig_attn_impl
                if hasattr(model.llm.llm_model, "model"):
                    inner = model.llm.llm_model.model
                    if hasattr(inner, "text_model"):
                        inner = inner.text_model
                    if hasattr(inner, "layers"):
                        for layer in inner.layers:
                            if hasattr(layer, "self_attn"):
                                if hasattr(layer.self_attn, "config"):
                                    layer.self_attn.config._attn_implementation = orig_attn_impl

            attentions = outputs.attentions
            sample_result = compute_attention_by_type(attentions, masks)

            # Accumulate
            for layer_name, type_dict in sample_result.items():
                if layer_name not in layer_sums:
                    layer_sums[layer_name] = {}
                for ttype, val in type_dict.items():
                    layer_sums[layer_name][ttype] = layer_sums[layer_name].get(ttype, 0.0) + val
            n_samples += 1

    # Average across samples
    results = {}
    for layer_name in sorted(layer_sums.keys(), key=lambda x: int(x.split("_")[1])):
        results[layer_name] = {
            ttype: layer_sums[layer_name][ttype] / n_samples
            for ttype in ["image", "history", "candidate", "text"]
            if ttype in layer_sums[layer_name]
        }

    # Save JSON
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'Layer':<12} {'Image':>10} {'History':>10} {'Candidate':>10} {'Text':>10}")
    print("-" * 55)
    for layer_name, vals in results.items():
        print(
            f"{layer_name:<12} "
            f"{vals.get('image', 0.0):>10.6f} "
            f"{vals.get('history', 0.0):>10.6f} "
            f"{vals.get('candidate', 0.0):>10.6f} "
            f"{vals.get('text', 0.0):>10.6f}"
        )


if __name__ == "__main__":
    main()
