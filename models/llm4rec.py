import torch
import torch.nn as nn

from transformers import AutoTokenizer, BitsAndBytesConfig, OPTForCausalLM


class llm4rec(nn.Module):
    """
    Wrapper around OPT-6.7B for A-LLMRec.

    The LLM is fully frozen — no LoRA, no fine-tuning of any kind.
    Only the surrounding projection heads (log_emb_proj, item_emb_proj,
    score_head) in A_llmrec_model receive gradient updates.

    Why frozen (not LoRA):
      - A-LLMRec's core advantage is that only the small alignment network
        trains, not the LLM itself.
      - Experiment 4 showed LoRA hurt performance on Luxury_Beauty (31%→25%)
        by degrading OPT's pre-trained luxury brand knowledge without
        successfully learning CF alignment.
      - The original CHANGES.md design explicitly specified frozen LLM.
    """
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
    ):
        super().__init__()
        self.device = device

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        if llm_model == "opt":
            self.llm_model = OPTForCausalLM.from_pretrained(
                "facebook/opt-6.7b",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                use_safetensors=True,
                device_map="cuda:1",
            )
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                "facebook/opt-6.7b",
                use_fast=False,
            )
        else:
            raise Exception(f"{llm_model} is not supported")

        # Register A-LLMRec special tokens.
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[CandidateEmb]']}
        )

        # Resize embedding matrix for the new special tokens.
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # Freeze all LLM weights. Only the projection heads in A_llmrec_model train.
        for _, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.max_output_txt_len = max_output_txt_len

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        """
        Construct a single sequence per example:

          [input (non-pad)] + [output (without first token)] + [input (pad tail)]

        Allows feeding both prompt and answer to the LLM while masking the
        prompt part in the loss.
        """
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def replace_hist_candi_token(self, llm_tokens, inputs_embeds, interact_embs, candidate_embs):
        """
        Replace embeddings at [HistoryEmb] / [CandidateEmb] token positions
        with projected item embeddings from A_llmrec_model.

        This is how collaborative knowledge is injected into the LLM's
        token embedding space.
        """
        if len(interact_embs) == 0:
            return llm_tokens, inputs_embeds
        inputs_embeds = inputs_embeds.clone()
        history_token_id = self.llm_tokenizer(
            "[HistoryEmb]", return_tensors="pt", add_special_tokens=False
        ).input_ids.item()
        candidate_token_id = self.llm_tokenizer(
            "[CandidateEmb]", return_tensors="pt", add_special_tokens=False
        ).input_ids.item()

        for inx in range(len(llm_tokens["input_ids"])):
            idx_tensor = (llm_tokens["input_ids"][inx] == history_token_id).nonzero().view(-1)
            for idx, item_emb in zip(idx_tensor, interact_embs[inx]):
                inputs_embeds[inx][idx] = item_emb

            idx_tensor = (llm_tokens["input_ids"][inx] == candidate_token_id).nonzero().view(-1)
            for idx, item_emb in zip(idx_tensor, candidate_embs[inx]):
                inputs_embeds[inx][idx] = item_emb

        return llm_tokens, inputs_embeds

    def forward(self, log_emb, samples):
        """
        Stage-2 language modelling loss (text-generation path, unchanged).

        Args:
          log_emb: projected user representations (batch, d_token).
          samples: dict with 'text_input', 'text_output', 'interact', 'candidate'.
        """
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)

        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(self.device)

        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(self.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        llm_tokens, inputs_embeds = self.replace_hist_candi_token(
            llm_tokens, inputs_embeds, samples['interact'], samples['candidate']
        )
        attention_mask = llm_tokens['attention_mask']

        log_emb = log_emb.unsqueeze(1)
        inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with torch.cuda.amp.autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        return outputs.loss

    def forward_id(self, log_emb, samples, use_user_token=False):
        """
        ID-prediction forward pass.

        Runs the enriched prompt through the frozen LLM and returns a hidden
        state to be used as the recommendation query vector.

        Args:
          log_emb        : projected user representations (batch, d_token).
          samples        : dict with 'text_input', 'interact', 'candidate'.
          use_user_token : if True, return the hidden state at position 0
                           (the prepended [UserRep] token) instead of the
                           last token. The [UserRep] token is seeded from
                           the CF user representation and attends over the
                           entire prompt — it may be a better recommendation
                           query than the final text token.

        Returns:
          hidden : (batch, d_model) — the chosen token's hidden state from
                   the top transformer layer, cast to float32.
        """
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)

        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(self.device)

        inputs_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)
        text_input_tokens, inputs_embeds = self.replace_hist_candi_token(
            text_input_tokens, inputs_embeds, samples['interact'], samples['candidate']
        )
        attention_mask = text_input_tokens['attention_mask']

        # Prepend the user representation as the first token (soft prompt).
        log_emb = log_emb.unsqueeze(1)
        inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, attention_mask], dim=1)

        with torch.cuda.amp.autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )

        # Top transformer layer hidden states: (batch, seq_len, d_model)
        top_hidden = outputs.hidden_states[-1]

        if use_user_token:
            # Index 0 is the prepended [UserRep] soft-prompt token.
            # It starts as the CF user representation and attends over the
            # full prompt — potentially a better recommendation query.
            hidden = top_hidden[:, 0, :].float()
        else:
            # Default: last token (after "The recommendation is").
            hidden = top_hidden[:, -1, :].float()

        return hidden
