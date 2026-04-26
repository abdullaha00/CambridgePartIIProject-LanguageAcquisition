from typing import List

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.modular_qg.qg.data import resize_prompt
from models.modular_qg.common.tokens import TOK_BOS, TOK_EOS, TOK_G, TOK_N, TOK_PAD, TOK_Q, TOK_A, TOK_Y

BODY_TOKS = [TOK_Q, TOK_A, TOK_Y, TOK_N, TOK_G]

class LMKTQG(nn.Module):
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        added = self.tokenizer.add_special_tokens(
            {
                "bos_token": TOK_BOS,
                "eos_token": TOK_EOS,
                "pad_token": TOK_PAD, # GPT2 has no pad token so we use our own
                "additional_special_tokens": BODY_TOKS}
        )
        
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        emb_dim = self.model.config.n_embd
        
        # scalar difficulty -> embedding is done via a linear layer
        self.diff_embd = nn.Linear(1, emb_dim)
        nn.init.normal_(self.diff_embd.weight, mean=0.0, std=self.model.config.initializer_range)
        nn.init.zeros_(self.diff_embd.bias)

        self.g_id = self.tokenizer.convert_tokens_to_ids(TOK_G)
        assert self.g_id != self.tokenizer.unk_token_id, \
            f"Special token {TOK_G} not found in tokenizer."

        self.to(self.device)
    
    def forward(self, input_ids, attention_mask, difficulty, labels=None):
        """
        input_ids: (B, T) token IDs containing <G> marker
        attention_mask: (B, T) 
        difficulty: (B, 1) scalar difficulty values (scaled up)
        labels: (B, T) with -100 for ignore
        """

        B, T = input_ids.shape

        #=== move to device
        dev = next(self.parameters()).device
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)
        difficulty = difficulty.to(dev)
        if labels is not None:
            labels = labels.to(dev)
        
        #=== token embds
        wte = self.model.get_input_embeddings()
        input_embs = wte(input_ids) # (B, T, E)

        # == FIND <G> POSITION
        g_mask = (input_ids == self.g_id).int()  # (B, T)
        g_pos = g_mask.argmax(dim=1) 

        # Encode difficulty (B, 1) -> (B, E)
        diff_embs = self.diff_embd(difficulty).unsqueeze(1)  # (B, 1, E)

        # Insert difficulty embedding before <G> pos

        new_embs = []
        new_masks = []
        new_labels = []

        # TODO: vectorise
        for i in range(B):
            emb = input_embs[i, :, :] # (T, E)
            
            # We want [tokens_preceding_<G>, diff_emb, <G>, tokens_following_<G>]
            # Where tokens_following_<G> is the question_text followed by <EOS>
            new_emb = torch.cat([emb[:g_pos[i], :], 
                                 diff_embs[i, :],  
                                 emb[g_pos[i]:, :]], dim=0) # (T+1, E)
            
            new_embs.append(new_emb)
            
            # Extend the attention mask by 1 for the inserted diff embedding
            one_mask = torch.tensor([1], device=attention_mask.device)
            new_mask = torch.cat([attention_mask[i, :g_pos[i]], 
                                  one_mask, 
                                  attention_mask[i, g_pos[i]:]], dim=0) # (T+1)

            new_masks.append(new_mask)

            # We need to also adjust labels if provided
            if labels is not None:
                # For labels, we want to insert a -100 at the same position to ignore the diff_emb in loss
                new_label = torch.cat([labels[i, :g_pos[i]], 
                                       torch.tensor([-100], device=labels.device), 
                                       labels[i, g_pos[i]:]], dim=0) # (T+1)
                new_labels.append(new_label)


        input_embs = torch.stack(new_embs, dim=0)  # (B, T+1, E)
        attention_mask = torch.stack(new_masks, dim=0)  # (B, T+1)
        if labels is not None:
            labels = torch.stack(new_labels, dim=0)  # (B, T+1)

        outputs = self.model(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            labels=labels,
        ) # (loss, logits, ...)

        return outputs

    def generate(
        self,
        history_prefix: str,
        target_diff: float,
        num_gen_seqs: int = 30,
        max_new_toks: int = 20,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.99
    ):
        """
        We encode history + <G> as the prompt
        We inject difficulty before <G> as a scalar embedding
        We sample tokens after <G> until <EOS> or max length
        """

        assert (
            0.0 <= target_diff <= 1.0
            and temperature > 0
            and top_k >= 0
            and 0.0 < top_p <= 1.0
        ), "Invalid parameters."

        dev = next(self.parameters()).device  # Ensure generation happens on the same device as the model

        self.eval()
        tok = self.tokenizer

        prompt = f"{history_prefix} {TOK_G}".strip()
        prompt = resize_prompt(prompt, length=800)
        prompt_enc = tok.encode(prompt, add_special_tokens=False, truncation=False) # (T,)

        max_prompt_toks = self.model.config.max_position_embeddings - 1 - max_new_toks # leave room for generated tokens and diff emb
        if len(prompt_enc) > max_prompt_toks:
            prompt_enc = prompt_enc[-max_prompt_toks:]
        
        assert self.g_id in prompt_enc, f"Prompt encoding does not contain {TOK_G}: {prompt_enc}"

        prompt_ids = torch.tensor([prompt_enc], device=dev)  # (1, T)

        all_results = []
        
        minibatch_size = min(num_gen_seqs, 10) # generate in batches to avoid OOM

        for start in range(0, num_gen_seqs, minibatch_size):
            end = min(start + minibatch_size, num_gen_seqs) # non inclusive end index
            batch_size = end - start # (0, minibatch_size, 2*minibatch_size, ... minibatch_size-1)

            batch_res = self._generate_batch(
                prompt_ids=prompt_ids,
                target_diff=target_diff,
                num_seqs=batch_size,
                max_new_toks=max_new_toks,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p
            )

            all_results.extend(batch_res)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_results

        
    @torch.no_grad()
    def _generate_batch(
        self, 
        prompt_ids: torch.Tensor, # (1, T)
        target_diff: float, 
        num_seqs: int,
        max_new_toks: int,
        temperature: float,
        repetition_penalty: float,
        top_k: int,
        top_p: float = 0.99
    ) -> List[str]:
        
        tok = self.tokenizer
        wte = self.model.get_input_embeddings()
        dev = next(self.parameters()).device
        
        prompt_embs = wte(prompt_ids)  # (1, T, E)

        target_diff_scaled = torch.tensor([[target_diff * 100]], device=dev)  # (1, 1)
        diff_emb = self.diff_embd(target_diff_scaled)  # (1, E)
        
        # Find <G> position, insert diff embedding, and prepare attention mask
        g_mask = (prompt_ids == self.g_id)  # (S, T)
        g_pos = g_mask.int().argmax(dim=1).item()  # scalar position of <G>

        input_embs = torch.cat([
            prompt_embs[0, :g_pos], # before <G>, (..., E)
            diff_emb, # diff embedding, (1, E)
            prompt_embs[0, g_pos:] # after <G>, (..., E)
        ], dim=0)  # (T+1, E)

        attention_mask = torch.ones(input_embs.shape[0], device=dev).unsqueeze(0)  # (1, T+1)

        input_embs = input_embs.unsqueeze(0).repeat(num_seqs, 1, 1)  # (1, T+1, E) -> (S, T+1, E)
        attention_mask = attention_mask.repeat(num_seqs, 1)  # (1, T+1) -> (S, T+1)
        prompt_tok_hist = prompt_ids.repeat(num_seqs, 1)  # (1, T) -> (S, T)

        # Auto-regressive generation loop
        eos_id = tok.eos_token_id
        pad_id = tok.pad_token_id

        assert tok.eos_token_id == self.tokenizer.convert_tokens_to_ids(TOK_EOS), "EOS token ID mismatch"
        assert tok.pad_token_id == self.tokenizer.convert_tokens_to_ids(TOK_PAD), "PAD token ID mismatch"

        generated_ids = []  # collects elements of (S, max_new_toks)
        finished = torch.zeros(num_seqs, dtype=torch.bool, device=dev)  # (S,)

        past_key_values = None
        current_embs = input_embs  # (S, T+1, E)

        for _ in range(max_new_toks):
            outputs = self.model(
                inputs_embeds=current_embs,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True  
            )

            past_key_values = outputs.past_key_values  # faster generation
            next_tok_logits = outputs.logits[:, -1, :]  # (S, V)
            
            # Apply temp
            if temperature != 1.0:
                next_tok_logits = next_tok_logits / temperature
            
            # Apply repetition penalty to discourage exact repetition of prompt tokens
            if repetition_penalty != 1.0:
                if generated_ids:
                    gen_ids = torch.stack(generated_ids, dim=1) # (S, gen_len)
                    prev_tokens = torch.cat([prompt_tok_hist, gen_ids], dim=1) # (S, T + gen_len)
                else:
                    prev_tokens = prompt_tok_hist

                # get the current logits of previous token
                prev_tok_scores = torch.gather(next_tok_logits, dim=1, index=prev_tokens)
                prev_tok_scores = torch.where(
                    prev_tok_scores < 0,
                    prev_tok_scores * repetition_penalty,
                    prev_tok_scores / repetition_penalty,
                )

                #next_tok_logits[i, prev_tokens[i, j]] = prev_tok_logits[i, j]
                next_tok_logits.scatter_(dim=1, index=prev_tokens, src=prev_tok_scores)

            # Top-k filtering

            S,V = next_tok_logits.shape

            if 0 < top_k < V:
                topk_vals, _ = torch.topk(next_tok_logits, top_k, dim=-1) # (S, top_k)
                threshold = topk_vals[:, -1].unsqueeze(-1) # (S, 1) the lowest logit in the top-k (cutoff val)
                next_tok_logits = next_tok_logits.masked_fill(next_tok_logits < threshold, -float('inf'))
            
            # Nucelus filtering
            # We select the smallest set of tokens whose cumulative probability mass exceeds top_p
            # This samples from the most probable tokens only

            if top_p < 1.0:
                # Sort and calculate cumulative probs
                s_logits, s_indices = torch.sort(next_tok_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(s_logits, dim=-1), dim=-1)

                # Create a mask for tokens to remove
                s_rem_mask = cumulative_probs > top_p
                # We want to keep the next token exceeding the treshold so that the total prob mass is above threshold
                # So we shift to the right 
                s_rem_mask[:, 1:] = s_rem_mask[:, :-1].clone()
                s_rem_mask[:, 0] = False

                # Convert back to original indexing and set logits of removed tokens to -inf
                ind_rem_mask = torch.zeros_like(s_rem_mask)
                ind_rem_mask.scatter_(dim=1, index=s_indices, src=s_rem_mask)

                next_tok_logits[ind_rem_mask] = -float('inf')

            #sample
            probs = torch.softmax(next_tok_logits, dim=-1)  # (S, V)
            next_tok_ids = torch.multinomial(probs, num_samples=1).squeeze(1)  # (S, 1) -> (S,)

            next_tok_ids[finished] = pad_id  # Replace early-finished sequences with pad if finished

            generated_ids.append(next_tok_ids)

            # Update finished sequences
            finished = finished | (next_tok_ids == eos_id)
            if finished.all():
                break

            # Prepare next step
            current_embs = wte(next_tok_ids.unsqueeze(1))  # (S, 1, E)
            attention_mask = torch.cat([attention_mask, torch.ones((num_seqs, 1), device=dev)], dim=1)  # (S, step+1)

        assert generated_ids, "No tokens were generated. Check if the model can generate any tokens given prompt."
        
        gen_tensor = torch.stack(generated_ids, dim=1)  # (S, gen_len)

        results = []
        for i in range(num_seqs):
            seq_ids = gen_tensor[i, :] # (gen_len,)

            end_pos = seq_ids.shape[0] # non inclusive

            # Truncate end_pos to EOS if present
            eos_indices = (seq_ids == eos_id).nonzero(as_tuple=False)  # (num_eos, 1)

            if eos_indices.numel() > 0:
                end_pos = eos_indices[0].item()  # first EOS position

            trimmed_ids = seq_ids[:end_pos]
            text = tok.decode(trimmed_ids, skip_special_tokens=True).strip()

            results.append(text)

        return results
        
