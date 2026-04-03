import os
import random
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np
from PIL import Image

from models.recsys_model import *
from models.llm4rec import *
from sentence_transformers import SentenceTransformer


class two_layer_mlp(nn.Module):
    """
    Lightweight 2-layer MLP used as an encoder/decoder block.

    Forward returns:
      - latent: 128-dim representation used for matching (shared space)
      - recon: reconstructed vector in the original input dimension
    """
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1


class A_llmrec_model(nn.Module):
    """
    A-LLMRec model:

    - Stage 1: Aligns CF item embeddings with SBERT text embeddings using
      two autoencoders + several losses (matching, reconstruction, rec loss).
    - Stage 2: Projects the CF user representation and aligned item embeddings
      into the LLM token space and injects them into prompts for recommendation.
    """
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        
        with open(f'./data/amazon/{args.rec_pre_trained_data}_text_name_dict.json.gz','rb') as ft:
            self.text_name_dict = pickle.load(ft)

        id_to_asin_path = f'./data/amazon/{args.rec_pre_trained_data}_id_to_asin.json.gz'
        if os.path.exists(id_to_asin_path):
            with open(id_to_asin_path, 'rb') as f:
                self.id_to_asin = pickle.load(f)
        else:
            self.id_to_asin = {}
        self.image_dir = f'./data/images/{args.rec_pre_trained_data}'
        self._image_cache = self._preload_images()

        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768
        
        self.mlp = two_layer_mlp(self.rec_sys_dim)
        # Components below are only created when running Stage 1 (alignment).
        if args.pretrain_stage1:
            self.sbert = SentenceTransformer('nq-distilbert-base-v1')
            self.mlp2 = two_layer_mlp(self.sbert_dim)
        
        self.mse = nn.MSELoss()
        
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG=0
        self.lan_HIT=0
        self.num_user = 0
        self.yes = 0
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()

        # Stage 2 components (LLM and projection heads) are only needed
        # when training Stage 2 or doing inference.
        if args.pretrain_stage2 or args.inference:
            self.llm = llm4rec(
                device=self.device,
                llm_model=args.llm,
                load_in_4bit=args.load_in_4bit,
            )
            
            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, self.llm.llm_hidden_size),
                nn.LayerNorm(self.llm.llm_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.llm.llm_hidden_size, self.llm.llm_hidden_size)
            )
            nn.init.xavier_normal_(self.log_emb_proj[0].weight)
            nn.init.xavier_normal_(self.log_emb_proj[3].weight)

            self.item_emb_proj = nn.Sequential(
                nn.Linear(128, self.llm.llm_hidden_size),
                nn.LayerNorm(self.llm.llm_hidden_size),
                nn.GELU(),
                nn.Linear(self.llm.llm_hidden_size, self.llm.llm_hidden_size)
            )
            nn.init.xavier_normal_(self.item_emb_proj[0].weight)
            nn.init.xavier_normal_(self.item_emb_proj[3].weight)
            
    def save_model(self, args, epoch1=None, epoch2=None):
        out_dir = f'./models/saved_models/'
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_{args.recsys}_{epoch1}_'
        # Save Stage 1 modules (alignment between CF and SBERT).
        if args.pretrain_stage1:
            torch.save(self.sbert.state_dict(), out_dir + 'sbert.pt')
            torch.save(self.mlp.state_dict(), out_dir + 'mlp.pt')
            torch.save(self.mlp2.state_dict(), out_dir + 'mlp2.pt') 
        
        out_dir += f'{args.llm}_{epoch2}_'
        # Save Stage 2 projection heads (CF -> LLM token space).
        if args.pretrain_stage2:
            torch.save(self.log_emb_proj.state_dict(), out_dir + 'log_proj.pt')
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            # Save LoRA adapter weights (SmolVLM only).
            if args.llm == 'smolvlm':
                lora_state = {k: v for k, v in self.llm.llm_model.state_dict().items() if 'lora_' in k}
                torch.save(lora_state, out_dir + 'lora.pt')
            
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f'./models/saved_models/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_'
        
        # Load Stage 1 alignment MLP and freeze it for later stages.
        mlp = torch.load(out_dir + 'mlp.pt', map_location = args.device)
        self.mlp.load_state_dict(mlp)
        del mlp
        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        if args.inference:
            out_dir += f'{args.llm}_{phase2_epoch}_'

            log_emb_proj_dict = torch.load(out_dir + 'log_proj.pt', map_location = args.device)
            self.log_emb_proj.load_state_dict(log_emb_proj_dict)
            del log_emb_proj_dict

            item_emb_proj_dict = torch.load(out_dir + 'item_proj.pt', map_location = args.device)
            self.item_emb_proj.load_state_dict(item_emb_proj_dict)
            del item_emb_proj_dict

            # Load LoRA adapter weights (SmolVLM only).
            if args.llm == 'smolvlm':
                lora_state = torch.load(out_dir + 'lora.pt', map_location=args.device)
                self.llm.llm_model.load_state_dict(lora_state, strict=False)

    def find_item_text(self, item, title_flag=True, description_flag=True):
        """
        Lookup titles/descriptions for a list of item IDs and format as strings.
        """
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]
    
    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        """
        Single-item version of find_item_text.
        """
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'
        
    def _preload_images(self):
        """Pre-load all product images into memory to avoid repeated disk I/O."""
        cache = {}
        if not self.id_to_asin or not os.path.isdir(self.image_dir):
            return cache
        for int_id, asin in self.id_to_asin.items():
            path = os.path.join(self.image_dir, f'{asin}.jpg')
            try:
                cache[int(int_id)] = Image.open(path).convert('RGB')
            except Exception:
                pass
        print(f'Pre-loaded {len(cache)} / {len(self.id_to_asin)} product images into memory')
        return cache

    def load_history_images(self, item_ids, n=5):
        """
        Load the last n product images for the given item_ids sequence.

        Looks up images from the in-memory cache (populated at init).
        Falls back to a 100x100 black image for missing items.
        The returned list always contains exactly n PIL Images (padded at the
        front with black images if the history is shorter than n).
        """
        black = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        images = []
        for item_id in list(item_ids)[-n:]:
            img = self._image_cache.get(int(item_id), black)
            images.append(img)
        while len(images) < n:
            images.insert(0, black)
        return images

    def get_item_emb(self, item_ids):
        """
        Get CF item embeddings and map them into the shared latent space (128-d).

        Returns the latent representation (used as joint collaborative-text
        embedding after Stage 1 training).
        """
        with torch.no_grad():
            item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
            item_embs, _ = self.mlp(item_embs)
        
        return item_embs
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        """
        Dispatch to the corresponding pipeline:
          - mode='phase1': Stage 1 alignment training
          - mode='phase2': Stage 2 alignment with LLM
          - mode='generate': inference / text generation
        """
        if mode == 'phase1':
            return self.pre_train_phase1(data, optimizer, batch_iter)
        if mode == 'phase2':
            return self.pre_train_phase2(data, optimizer, batch_iter)
        if mode =='generate':
            return self.generate(data)
        return None

    def pre_train_phase1(self,data,optimizer, batch_iter):
        """
        Stage 1 training:
          - Align CF item embeddings with SBERT text embeddings
          - Add reconstruction and recommendation losses
        """
        epoch, total_epoch, step, total_step = batch_iter
        
        self.sbert.train()
        optimizer.zero_grad()

        u, seq, pos, neg = data
        # Only use the last position in each sequence (for efficiency),
        # consistent with the paper's training strategy.
        indices = [self.maxlen*(i+1)-1 for i in range(u.shape[0])]
        
        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode='item')
            
        log_emb_ = log_emb[indices]
        pos_emb_ = pos_emb[indices]
        neg_emb_ = neg_emb[indices]
        pos_ = pos.reshape(pos.size)[indices]
        neg_ = neg.reshape(neg.size)[indices]
        
        start_inx = 0
        end_inx = 60
        iterss = 0
        mean_loss = 0
        bpr_loss = 0
        gt_loss = 0
        rc_loss = 0
        text_rc_loss = 0
        original_loss = 0
        # Process samples in sub-batches of size 60 to keep SBERT calls manageable.
        while start_inx < len(log_emb_):
            log_emb = log_emb_[start_inx:end_inx]
            pos_emb = pos_emb_[start_inx:end_inx]
            neg_emb = neg_emb_[start_inx:end_inx]
            
            pos__ = pos_[start_inx:end_inx]
            neg__ = neg_[start_inx:end_inx]
            
            start_inx = end_inx
            end_inx += 60
            iterss +=1
            
            # Convert item IDs to textual descriptions (title, description).
            pos_text = self.find_item_text(pos__)
            neg_text = self.find_item_text(neg__)
            
            # Encode positive/negative item texts using SBERT.
            pos_token = self.sbert.tokenize(pos_text)
            pos_text_embedding= self.sbert({'input_ids':pos_token['input_ids'].to(self.device),'attention_mask':pos_token['attention_mask'].to(self.device)})['sentence_embedding']
            neg_token = self.sbert.tokenize(neg_text)
            neg_text_embedding= self.sbert({'input_ids':neg_token['input_ids'].to(self.device),'attention_mask':neg_token['attention_mask'].to(self.device)})['sentence_embedding']
            
            # CF item autoencoder: latent matching space + reconstruction.
            pos_text_matching, pos_proj = self.mlp(pos_emb)
            neg_text_matching, neg_proj = self.mlp(neg_emb)
            
            # Text autoencoder: latent matching space + reconstruction.
            pos_text_matching_text, pos_text_proj = self.mlp2(pos_text_embedding)
            neg_text_matching_text, neg_text_proj = self.mlp2(neg_text_embedding)
            
            # Recommendation loss: classify positive vs negative items
            # based on interaction between user representation and item embeddings.
            pos_logits, neg_logits = (log_emb*pos_proj).mean(axis=1), (log_emb*neg_proj).mean(axis=1)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=pos_logits.device), torch.zeros(neg_logits.shape, device=pos_logits.device)

            loss = self.bce_criterion(pos_logits, pos_labels)
            loss += self.bce_criterion(neg_logits, neg_labels)
            
            # Latent-space alignment loss (CF item vs text latent representations).
            matching_loss = self.mse(pos_text_matching,pos_text_matching_text) + self.mse(neg_text_matching,neg_text_matching_text)
            # Reconstruction losses for CF item embeddings and SBERT embeddings.
            reconstruction_loss = self.mse(pos_proj,pos_emb) + self.mse(neg_proj,neg_emb)
            text_reconstruction_loss = self.mse(pos_text_proj,pos_text_embedding.data) + self.mse(neg_text_proj,neg_text_embedding.data)
            
            # Full Stage 1 loss = rec loss + matching + weighted recon terms
            # (matches Eq. (6) in the paper).
            total_loss = loss + matching_loss + 0.5*reconstruction_loss + 0.2*text_reconstruction_loss
            total_loss.backward()
            optimizer.step()
            
            mean_loss += total_loss.item()
            bpr_loss += loss.item()
            gt_loss += matching_loss.item()
            rc_loss += reconstruction_loss.item()
            text_rc_loss += text_reconstruction_loss.item()
            
        print("loss in epoch {}/{} iteration {}/{}: {} / BPR loss: {} / Matching loss: {} / Item reconstruction: {} / Text reconstruction: {}".format(epoch, total_epoch, step, total_step, mean_loss/iterss, bpr_loss/iterss, gt_loss/iterss, rc_loss/iterss, text_rc_loss/iterss))
        return {
            "loss": mean_loss / iterss,
            "bpr_loss": bpr_loss / iterss,
            "matching_loss": gt_loss / iterss,
            "item_reconstruction_loss": rc_loss / iterss,
            "text_reconstruction_loss": text_rc_loss / iterss,
        }
    
    def make_interact_text(self, interact_ids, interact_max_num, use_images=False):
        """
        Build the textual part of the user history for the LLM prompt.

        Appends a special marker [HistoryEmb] to each title so we can
        later replace it with the aligned item embedding in the LLM input.

        When use_images=True (SmolVLM path), the last 5 items in the slice
        also get an <image> token appended directly after [HistoryEmb].  The
        Idefics3Processor will expand each <image> into the correct sequence
        of visual-patch tokens when the prompt is tokenized.
        """
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
                interact_text.append(title + suffix)
            interact_ids = interact_ids[-interact_max_num:]

        interact_text = ','.join(interact_text)
        return interact_text, interact_ids
    
    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title):
        """
        Build candidate set text (1 positive + sampled negatives) for the LLM prompt.

        Uses [CandidateEmb] markers which will be replaced by projected
        item embeddings in the token embedding sequence.
        """
        neg_item_id = []
        while len(neg_item_id)<50:
            t = np.random.randint(1, self.item_num+1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)
        random.shuffle(neg_item_id)
        
        candidate_ids = [target_item_id]
        candidate_text = [target_item_title + '[CandidateEmb]']

        for neg_candidate in neg_item_id[:candidate_num - 1]:
            candidate_text.append(self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + '[CandidateEmb]')
            candidate_ids.append(neg_candidate)
                
        random_ = np.random.permutation(len(candidate_text))
        candidate_text = np.array(candidate_text)[random_]
        candidate_ids = np.array(candidate_ids)[random_]
            
        return ','.join(candidate_text), candidate_ids
    
    def pre_train_phase2(self, data, optimizer, batch_iter):
        """
        Stage 2 training:
          - Construct prompts that include history and candidate items
          - Project CF user representations and joint item embeddings into
            the LLM token space
          - Optimize to predict the correct next item title

        When using SmolVLM, the last 5 history items also have <image> tokens
        in the prompt and their PIL images are collected into images_batch so
        that Idefics3Processor can process them together with the text.
        """
        epoch, total_epoch, step, total_step = batch_iter

        optimizer.zero_grad()
        u, seq, pos, neg = data
        mean_loss = 0

        text_input = []
        text_output = []
        interact_embs = []
        candidate_embs = []
        images_batch = []
        self.llm.train()

        use_images = (self.args.llm == 'smolvlm')

        # Get CF user representations for the batch (frozen CF-RecSys).
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')

        for i in range(len(u)):
            # Use the last positive item as the target (next item).
            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)

            # User interaction history (titles + [HistoryEmb] markers).
            # For SmolVLM, the last 5 also get <image> appended.
            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, use_images=use_images)
            candidate_num = 20
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title)

            input_text = ''
            input_text += ' is a user representation.'

            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text += 'This user has watched '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text += 'This user has played '
            else:
                input_text += 'This user has bought '

            input_text += interact_text

            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text +=' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text +=' in the previous. Recommend one next game for this user to play next from the following game title set, '
            else:
                input_text +=' in the previous. Recommend one next item for this user to buy next from the following item title set, '

            input_text += candidate_text
            input_text += '. The recommendation is '

            text_input.append(self.llm.wrap_prompt(input_text))
            text_output.append(target_item_title)

            # Project latent joint embeddings into LLM token space for all
            # history and candidate items.
            interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
            candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))

            # Collect images for the last min(5, history_len) history items (SmolVLM only).
            # Must match the number of <image> tokens emitted by make_interact_text.
            if use_images:
                n_images = min(5, len(interact_ids[-10:]))
                sample_images = self.load_history_images(interact_ids, n=n_images)
                images_batch.append(sample_images)

        samples = {
            'text_input': text_input,
            'text_output': text_output,
            'interact': interact_embs,
            'candidate': candidate_embs,
            'images': images_batch if use_images else None,
        }
        # Project CF user representations into LLM token space.
        log_emb = self.log_emb_proj(log_emb)
        loss_rm = self.llm(log_emb, samples)
        loss_rm.backward()
        optimizer.step()
        mean_loss += loss_rm.item()
        print("A-LLMRec model loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, mean_loss))
        return {"loss": mean_loss}
        
    def generate(self, data):
        """
        Inference routine:
          - Build prompts and aligned embeddings as in Stage 2 training
          - Call the frozen LLM generate() API to get text outputs

        When using SmolVLM, the last 5 history items have <image> tokens in the
        prompt.  The full Idefics3Processor tokenizes text + images together,
        returning expanded input_ids and pixel_values which are passed to
        llm_model.generate() so the vision encoder can inject visual features.
        """
        u, seq, pos, neg, rank = data

        use_images = (self.args.llm == 'smolvlm')

        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []
        images_batch = []
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            for i in range(len(u)):
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)

                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, use_images=use_images)
                candidate_num = 20
                candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title)

                input_text = ''
                input_text += ' is a user representation.'
                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text += 'This user has watched '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += 'This user has played '
                else:
                    input_text += 'This user has bought '

                input_text += interact_text

                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text +=' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text +=' in the previous. Recommend one next game for this user to play next from the following game title set, '
                else:
                    input_text +=' in the previous. Recommend one next item for this user to buy next from the following item title set, '

                input_text += candidate_text
                input_text += '. The recommendation is '

                answer.append(target_item_title)
                text_input.append(self.llm.wrap_prompt(input_text))

                # Pre-compute projected embeddings for history and candidates.
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
                candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))

                # Collect images for the last min(5, history_len) history items (SmolVLM only).
                # Must match the number of <image> tokens emitted by make_interact_text.
                if use_images:
                    n_images = min(5, len(interact_ids[-10:]))
                    images_batch.append(self.load_history_images(interact_ids, n=n_images))

        # Add user representation token at the beginning of the LLM input.
        log_emb = self.log_emb_proj(log_emb)
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)

        with torch.no_grad():
            # Tokenize: use full processor for SmolVLM (expands <image> tokens
            # and prepares pixel_values), plain tokenizer for OPT.
            if use_images:
                self.llm.processor.tokenizer.padding_side = "left"
                processed = self.llm.processor(
                    text=text_input,
                    images=images_batch,
                    padding="longest",
                    return_tensors="pt",
                ).to(self.device)
                llm_tokens = processed
                pixel_values = processed.pixel_values
                image_attention_mask = processed.get('image_attention_mask')
            else:
                self.llm.llm_tokenizer.padding_side = "left"
                llm_tokens = self.llm.llm_tokenizer(
                    text_input,
                    padding="longest",
                    return_tensors="pt"
                ).to(self.device)
                pixel_values = None
                image_attention_mask = None

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens.input_ids)

                # Replace [HistoryEmb] / [CandidateEmb] token positions with
                # the projected joint item embeddings in the embedding matrix.
                llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(llm_tokens, inputs_embeds, interact_embs, candidate_embs)

                attention_mask = llm_tokens.attention_mask
                # Prepend the user representation embedding at the very front.
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                # Prepend a dummy pad token so input_ids length matches
                # inputs_embeds (which has the extra user-rep token at pos 0).
                if pixel_values is not None:
                    dummy = torch.full(
                        (llm_tokens.input_ids.size(0), 1),
                        self.llm.llm_tokenizer.pad_token_id,
                        dtype=torch.long,
                        device=self.device,
                    )
                    input_ids_for_model = torch.cat([dummy, llm_tokens.input_ids], dim=1)
                else:
                    input_ids_for_model = llm_tokens.input_ids

                # Match inputs_embeds dtype to model dtype (see llm4rec.forward).
                if pixel_values is not None:
                    model_dtype = next(self.llm.llm_model.parameters()).dtype
                    inputs_embeds = inputs_embeds.to(model_dtype)

                if use_images:
                    # SmolVLM + images path: model.generate() internally calls
                    # prepare_inputs_for_generation which sets input_ids=None
                    # whenever inputs_embeds is provided.  inputs_merger then
                    # falls back to an embedding-comparison heuristic that gives
                    # wrong image-token counts, raising "not divisible by
                    # patch_size".
                    #
                    # Fix: run the first forward pass manually (input_ids stays
                    # intact → inputs_merger uses the correct branch), then
                    # decode greedily from the KV cache without inputs_embeds.
                    first_out = self.llm.llm_model(
                        input_ids=input_ids_for_model,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_attention_mask=image_attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_kv = first_out.past_key_values
                    next_tok = first_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    all_new = [next_tok]
                    cur_attn = torch.cat(
                        [attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.long, device=self.device)],
                        dim=1,
                    )
                    for _ in range(49):  # max_new_tokens - 1
                        if (next_tok == self.llm.llm_tokenizer.eos_token_id).all():
                            break
                        step_out = self.llm.llm_model(
                            input_ids=next_tok,
                            attention_mask=cur_attn,
                            past_key_values=past_kv,
                            use_cache=True,
                            return_dict=True,
                        )
                        past_kv = step_out.past_key_values
                        next_tok = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        all_new.append(next_tok)
                        cur_attn = torch.cat(
                            [cur_attn, torch.ones((cur_attn.size(0), 1), dtype=torch.long, device=self.device)],
                            dim=1,
                        )
                    outputs = torch.cat(all_new, dim=1)  # [batch, num_new_tokens]
                    output_text_new_only = self.llm.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    output_text_new_only = [text.strip() for text in output_text_new_only]
                else:
                    outputs = self.llm.llm_model.generate(
                        input_ids=input_ids_for_model,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_attention_mask=image_attention_mask,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=50,
                        min_length=1,
                        eos_token_id=self.llm.llm_tokenizer.eos_token_id,
                        pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                        repetition_penalty=1.5,
                        length_penalty=1,
                        num_return_sequences=1,
                    )
                    # OPT uses token 0 as padding — remap to EOS (2) so
                    # batch_decode works.
                    if self.args.llm == 'opt':
                        outputs[outputs == 0] = 2
                    only_new_tokens = outputs[:, input_ids_for_model.shape[1]:]
                    output_text_new_only = self.llm.llm_tokenizer.batch_decode(only_new_tokens, skip_special_tokens=True)
                    output_text_new_only = [text.strip() for text in output_text_new_only]

        output_path = self.args.inference_output_file
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'a') as f:
            for i in range(len(text_input)):
                f.write('Answer: ' + str(answer[i]) + '\n\n')
                f.write('LLM: ' + str(output_text_new_only[i]) + '\n\n')
                f.write('--------------------------------\n\n')

        return output_text_new_only