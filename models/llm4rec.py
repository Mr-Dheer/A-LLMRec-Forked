import torch
import torch.nn as nn

from transformers import AutoTokenizer, BitsAndBytesConfig, OPTForCausalLM


class llm4rec(nn.Module):
    """
    Wrapper around an OPT language model used in A-LLMRec.

    Responsibilities:
      - Load and freeze a pretrained OPT-6.7B model and tokenizer.
      - Register special tokens used by A-LLMRec (user rep, history, candidate).
      - Provide utilities to:
          * splice together prompt (input) and target (output) text,
          * replace special marker tokens with projected item embeddings,
          * compute the LM loss for Stage-2 training.
    """
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
    ):
        super().__init__()
        self.device = device
        
        # Currently only OPT is supported as the backbone LLM.
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        if llm_model == "opt":
            self.llm_model = OPTForCausalLM.from_pretrained(
                "facebook/opt-6.7b",
                quantization_config=bnb_config,
                dtype=torch.float16,
                use_safetensors=True,
                device_map="auto",
            )
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                "facebook/opt-6.7b",
                use_fast=False,
            )
        else:
            raise Exception(f"{llm_model} is not supported")
            
        # Define pad/BOS/EOS/UNK plus special marker tokens for A-LLMRec.
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[UserRep]','[HistoryEmb]','[CandidateEmb]']})

        # Resize embeddings so the model can consume the new tokens.
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        
        # Freeze all LLM weights; only surrounding projection heads are trainable.
        for _, param in self.llm_model.named_parameters():
            param.requires_grad = False
            
        self.max_output_txt_len = max_output_txt_len

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        """
        Construct a single sequence per example:

          [input (non-pad)] + [output (without first token)] + [input (pad tail)]

        This allows us to feed both prompt and answer to the LLM while
        masking out the prompt part in the loss.
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
        with projected item embeddings passed in from A_llmrec_model.

        This is how collaborative knowledge is injected into the LLM's
        token embedding space.
        """
        if len(interact_embs) == 0:
            return llm_tokens, inputs_embeds
        history_token_id = self.llm_tokenizer("[HistoryEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        candidate_token_id = self.llm_tokenizer("[CandidateEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        
        for inx in range(len(llm_tokens["input_ids"])):
            idx_tensor=(llm_tokens["input_ids"][inx]==history_token_id).nonzero().view(-1)
            for idx, item_emb in zip(idx_tensor, interact_embs[inx]):
                inputs_embeds[inx][idx]=item_emb
        
            idx_tensor=(llm_tokens["input_ids"][inx]==candidate_token_id).nonzero().view(-1)
            for idx, item_emb in zip(idx_tensor, candidate_embs[inx]):
                inputs_embeds[inx][idx]=item_emb
        return llm_tokens, inputs_embeds
    
    def forward(self, log_emb, samples):
        """
        Compute the Stage-2 language modeling loss.

        Args:
          log_emb: projected user representations (batch, d_token),
                   already mapped from CF space into LLM token space.
          samples: dict with:
              - 'text_input': list of prompt strings,
              - 'text_output': list of target titles (answers),
              - 'interact': list of tensors of projected history item embeddings,
              - 'candidate': list of tensors of projected candidate item embeddings.
        """
        # Attention mask for the prepended user representation token.
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
            
        # Tokenize the target text (e.g., correct item title) and append EOS.
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
        
        # Merge input and output tokens into a single sequence for each example.
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # Initialize targets from token IDs and ignore pad positions.
        targets = llm_tokens['input_ids'].masked_fill(llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100)

        # Ignore the prompt (input) part in the loss; only train on the answer.
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100
        
        # Also ignore the prepended user representation token.
        empty_targets = (torch.ones(atts_llm.size(), dtype=torch.long).to(self.device).fill_(-100))

        targets = torch.cat([empty_targets, targets], dim=1)
        
        # Convert tokens to embeddings, then inject history/candidate item embeddings.
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        llm_tokens, inputs_embeds = self.replace_hist_candi_token(llm_tokens, inputs_embeds, samples['interact'], samples['candidate'])
        attention_mask = llm_tokens['attention_mask']
        
        # Prepend the user representation embedding as the first token (soft prompt).
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
        loss = outputs.loss

        return loss