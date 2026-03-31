import torch
import torch.nn as nn
import json
import time
import uuid

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    OPTForCausalLM,
    AutoModelForImageTextToText
)
from peft import LoraConfig, get_peft_model


class llm4rec(nn.Module):
    """
    Wrapper around a language model used in A-LLMRec.

    Responsibilities:
      - Load and freeze a pretrained LLM and tokenizer.
      - Register special tokens used by A-LLMRec (user rep, history, candidate).
      - Provide utilities to:
          * splice together prompt (input) and target (output) text,
          * replace special marker tokens with projected item embeddings,
          * compute the LM loss for Stage-2 training.

    Supported llm_model values: "opt", "smolvlm"
    """
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
        load_in_4bit=False,
    ):
        super().__init__()
        self.device = device

        if llm_model == "opt":
            if load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
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
            # OPT has no pad/bos/eos/unk by default — set them explicitly.
            self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
            self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
            self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        elif llm_model == "smolvlm":
            model_kwargs = {
                "_attn_implementation": "flash_attention_2",
                "device_map": {"": 0},
            }
            if load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16

            self.llm_model = AutoModelForImageTextToText.from_pretrained(
                "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                **model_kwargs,
            )
            # SmolVLM2 uses Idefics3Processor. We keep the full processor so that
            # image inputs can be processed alongside text in Stage 2.
            # bos=<|im_start|>, eos/pad=<|im_end|>, unk=<|endoftext|> are
            # already correctly defined — do NOT override them.
            self.processor = AutoProcessor.from_pretrained(
                "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                do_image_splitting=False,
            )
            self.llm_tokenizer = self.processor.tokenizer

        else:
            raise Exception(f"{llm_model} is not supported")

        # Add A-LLMRec marker tokens for all models.
        self.llm_tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[CandidateEmb]']}
        )

        # Resize embeddings so the model can consume the new tokens.
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # Freeze all LLM weights; only surrounding projection heads are trainable.
        for _, param in self.llm_model.named_parameters():
            param.requires_grad = False

        # Apply LoRA to SmolVLM after freezing base weights.
        # LoRA adds small trainable adapter matrices (A, B) alongside frozen
        # attention projections. Only these new parameters have requires_grad=True.
        # Must be applied AFTER the freeze loop above so the loop doesn't touch them.
        if llm_model == "smolvlm":
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)

        self.max_output_txt_len = max_output_txt_len
        self._llm_model_name = llm_model

    @property
    def llm_hidden_size(self) -> int:
        """
        Return the text-decoder hidden size regardless of model architecture.

        OPT exposes hidden_size at the top-level config.
        SmolVLM (Idefics3) nests it under config.text_config.hidden_size.
        """
        cfg = self.llm_model.config
        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size
        if hasattr(cfg, "text_config"):
            return cfg.text_config.hidden_size
        raise ValueError(f"Cannot determine hidden_size from config: {type(cfg)}")

    def wrap_prompt(self, text: str) -> str:
        """
        Wrap a raw recommendation prompt with the model's expected input format.

        SmolVLM-Instruct is trained with the Idefics3 chat template:
            <|im_start|>User: {text}<end_of_utterance>\nAssistant:
        Without this wrapper the instruct-tuned 2B model sees an out-of-distribution
        prompt and generates verbose, off-topic text instead of a product title.

        OPT has no chat template and receives the raw prompt unchanged.
        """
        if self._llm_model_name == "smolvlm":
            return f"<|im_start|>User: {text}<end_of_utterance>\nAssistant: "
        return text

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

    def _debug_log(self, hypothesis_id, location, message, data, run_id="pre-fix"):
        payload = {
            "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
            "timestamp": int(time.time() * 1000),
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
        }
        try:
            with open("/users/kavach_d/projects/idea-3/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            pass
    
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
        # Output is always text-only, so we always use the plain tokenizer here.
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(self.device)

        # Tokenize the input prompt. When images are present (SmolVLM path) we
        # use the full Idefics3Processor so that <image> placeholders in the
        # text are expanded to the correct number of image-token IDs and
        # pixel_values are returned for the vision encoder.
        if samples.get('images') is not None:
            image_groups = samples['images']
            image_group_lengths = [len(g) if isinstance(g, list) else -1 for g in image_groups]
            image_token_counts = [t.count("<image>") for t in samples['text_input']]
            has_any_nonempty_image_group = any(v > 0 for v in image_group_lengths)
            # region agent log
            self._debug_log(
                "H1_H2_H5",
                "models/llm4rec.py:247",
                "processor_inputs_summary",
                {
                    "batch_size_text": int(len(samples['text_input'])),
                    "batch_size_images": int(len(image_groups)),
                    "image_group_lengths_first10": image_group_lengths[:10],
                    "empty_image_group_count": int(sum(1 for v in image_group_lengths if v == 0)),
                    "image_token_counts_first10": image_token_counts[:10],
                    "has_any_nonempty_image_group": bool(has_any_nonempty_image_group),
                },
                run_id="post-fix",
            )
            # endregion
            if has_any_nonempty_image_group:
                try:
                    processed = self.processor(
                        text=samples['text_input'],
                        images=image_groups,
                        return_tensors="pt",
                        padding="longest",
                    ).to(self.device)
                except Exception as e:
                    # region agent log
                    self._debug_log(
                        "H3",
                        "models/llm4rec.py:267",
                        "processor_call_exception",
                        {
                            "error_type": type(e).__name__,
                            "error": str(e),
                            "batch_size_text": int(len(samples['text_input'])),
                            "batch_size_images": int(len(image_groups)),
                        },
                        run_id="post-fix",
                    )
                    # endregion
                    raise
                text_input_tokens = processed
                pixel_values = processed.pixel_values
                image_attention_mask = processed.get('image_attention_mask')
            else:
                # region agent log
                self._debug_log(
                    "H1",
                    "models/llm4rec.py:281",
                    "fallback_to_text_only_tokenizer_due_to_empty_image_groups",
                    {
                        "batch_size_text": int(len(samples['text_input'])),
                        "image_group_lengths_first10": image_group_lengths[:10],
                        "image_token_counts_first10": image_token_counts[:10],
                    },
                    run_id="post-fix",
                )
                # endregion
                text_input_tokens = self.llm_tokenizer(
                    samples['text_input'],
                    return_tensors="pt",
                    padding="longest",
                    truncation=False,
                ).to(self.device)
                pixel_values = None
                image_attention_mask = None
        else:
            text_input_tokens = self.llm_tokenizer(
                samples['text_input'],
                return_tensors="pt",
                padding="longest",
                truncation=False,
            ).to(self.device)
            pixel_values = None
            image_attention_mask = None

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

        # When pixel_values are present (SmolVLM path), Idefics3's inputs_merger
        # uses input_ids to locate <image> token positions in inputs_embeds.
        # inputs_embeds has had the user-rep token prepended (line above), making
        # it one token longer than llm_tokens['input_ids'].  We prepend a single
        # pad token so both tensors have identical sequence length; the pad id is
        # not the image-token id, so Idefics3 leaves position-0 (user-rep) alone.
        if pixel_values is not None:
            dummy = torch.full(
                (llm_tokens['input_ids'].size(0), 1),
                self.llm_tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            input_ids_for_model = torch.cat([dummy, llm_tokens['input_ids']], dim=1)
        else:
            input_ids_for_model = llm_tokens['input_ids']

        with torch.cuda.amp.autocast():
            outputs = self.llm_model(
                input_ids=input_ids_for_model,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_attention_mask=image_attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return loss