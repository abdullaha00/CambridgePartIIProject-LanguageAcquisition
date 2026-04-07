from dataclasses import dataclass
import logging

from torch import nn
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from models.adaptive_qg.aqg_dkt.adaptive_data import ExerciseRecord
from models.adaptive_qg.aqg_dkt.aqg_kt import DKT

logger = logging.getLogger(__name__)

def tokenize_qg_input(keywords: list[str], tokenizer: AutoTokenizer, max_length: int = 15, prefix_reserve: int = 0) -> tuple[str]:
    
    prefix = [tokenizer.pad_token] * prefix_reserve # reserve with pad toks

    text = " ".join(prefix + [f"{kw}" for kw in keywords]).strip()
    enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

    return enc["input_ids"], enc["attention_mask"]

def tokenize_qg_output(toks: list[str], tokenizer: AutoTokenizer, max_length: int = 30) -> list[str]:
    text = " ".join(toks)

    enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

    label_ids = enc["input_ids"].clone()
    label_ids[label_ids == tokenizer.pad_token_id] = -100
        
    return label_ids

def build_projection_layer(projection_type: str, input_dim: int, output_dim: int, mlp_hidden_dim: int) -> nn.Module:
    if projection_type == "linear":
        return nn.Linear(input_dim, output_dim)
    elif projection_type == "mlp":
        if not mlp_hidden_dim:
            logger.warning("MLP hidden dimension not specified, defaulting to 512")
            mlp_hidden_dim = 512
        return nn.Sequential(

            nn.Linear(input_dim, mlp_hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, output_dim, bias=True)
        )
    else:
        raise ValueError(f"Unsupported projection type: {projection_type}")

class ExerciseGenerator(nn.Module):
    def __init__(self, model_name: str, vocab_size: int = 5, projection_type: str = "linear", mlp_hidden_dim: int = 512, extra_feat_dim: int = 0):
        
        super().__init__()
                
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tok_embeddings = self.model.get_input_embeddings()  
        self.positional_embeddings = self.model.model.encoder.embed_positions.weight

        assert self.tok_embeddings is not None, "Model does not have token embeddings"
        assert self.positional_embeddings is not None, "Model does not have positional embeddings"
        
        emb_dim = self.tok_embeddings.embedding_dim
        assert emb_dim == self.positional_embeddings.shape[1], "Token and positional embedding dimensions do not match"
    
        self.ffn_diff = build_projection_layer(projection_type, 1, emb_dim, mlp_hidden_dim)
        self.ffn_student_state = build_projection_layer(projection_type, vocab_size, emb_dim, mlp_hidden_dim)
        self.ffn_extra_feats = build_projection_layer(projection_type, extra_feat_dim, emb_dim, mlp_hidden_dim) if extra_feat_dim > 0 else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_input_embeddings(
        self, 
        input_ids: torch.Tensor, # (B, T)
        difficulty: torch.Tensor, # (B, 1)
        student_state: torch.Tensor, # (B, V)
        extra_features: torch.Tensor = None
    ) -> torch.Tensor:

        B, T = input_ids.shape
        input_emb = self.tok_embeddings(input_ids)  # (B, T, E)

        control_embeddings: list[torch.Tensor] = []
        
        if student_state is not None:
            student_emb = self.ffn_student_state(student_state)  # (B, E)
            control_embeddings.append(student_emb)  
        if difficulty is not None:
            diff_emb = self.ffn_diff(difficulty)  # (B, E)
            control_embeddings.append(diff_emb)  

        assert (extra_features is not None) == (self.ffn_extra_feats is not None), "Extra features provided but FFN for extra features not initialized (/vice versa)"
        
        if extra_features is not None:
            extra_emb = self.ffn_extra_feats(extra_features)  # (B, E)
            control_embeddings.append(extra_emb)  

        for i, control_emb in enumerate(control_embeddings):
            if i >= T-1:
                logger.warning(f"More control features ({len(control_embeddings)}) than input sequence length ({T}); truncating control features")
                break
            input_emb[:, i+1, :] = control_emb  # Add control embedding to reserved
            # NOTE: ensure correct reserved slots
            # NOTE: check indices (i+1)

        # POSITION EMBEDDINGS
        # pos_emb = self.positional_embeddings[:T, :].unsqueeze(0)  # (1, T, E)
        # input_emb = input_emb + pos_emb  # (B, T, E)

        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)  # (B, T)
        pos_emb = self.positional_embeddings[pos_ids]  # (B, T, E)
        input_emb = input_emb + pos_emb  # (B, T, E)

        return input_emb
    
    def forward(self, 
        input_ids: torch.Tensor,  # (B, T)
        attention_mask: torch.Tensor , # (B, T) 
        student_state: torch.Tensor,  # (B, V)
        difficulty: torch.Tensor,      # (B, 1)
        decoder_input_ids: torch.Tensor,  # (B, D)
        labels: torch.Tensor = None,  # (B, D)
        extra_features: torch.Tensor = None,  # (B, F)
        past_key_values = None,
        use_cache: bool = False,
    ) -> str:
        
        input_embds_controlled = self.get_input_embeddings(input_ids, difficulty, student_state, extra_features)  # (B, T, E)
        
        outputs = self.model(
            inputs_embeds=input_embds_controlled,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        return outputs
    

def qg_input_difficulties(
        student_states: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: AutoTokenizer,
        sub_word_ids: torch.Tensor,
        oov_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    sub_word_difficulties = student_states[:, sub_word_ids] * oov_mask  # (B, S)

    true_labels = labels.clone() # (B, T_dec)
    true_labels[true_labels == -100] = tokenizer.pad_token_id  # Ignore padding

    label_difficulties = torch.gather(sub_word_difficulties, 1, true_labels)  # (B, T_dec) with [b,t] -> diff assigned to each target decoder tok
    sum_difficulties = label_difficulties.sum(dim=1, keepdim=True)  # (B, 1)

    return sum_difficulties, sub_word_difficulties

def build_qg_batch_user(
    exercises: list[ExerciseRecord],
    knowledge_states: torch.Tensor, # (T, V)
    difficulties: torch.Tensor,
    model: ExerciseGenerator,
    tokenizer: AutoTokenizer,
    sub_word_ids: torch.Tensor,
    oov_mask: torch.Tensor,
    word_vocab: dict[str, int],
    word_error_rates: dict[str, float],
    use_extra_feats: bool = False,
    max_in_length: int = 15,
    max_out_length: int = 30
):
    
    enc_ids_list = []
    attn_mask_list = []
    
    labels_list = []
    dec_ids_list = []
    state_positions_list = []

    for ex in exercises:
        # Tokenize exercise text
        prefix_reserve = 3 if use_extra_feats else 2

        kw_ids, kw_attn_mask = tokenize_qg_input(ex.keywords, tokenizer, max_in_length, prefix_reserve)  # (input_ids, attention_mask)
        labels = tokenize_qg_output(ex.tokens, tokenizer, max_out_length)
        
        enc_ids_list.append(kw_ids)
        attn_mask_list.append(kw_attn_mask)

        labels_list.append(labels)
        dec_ids_list.append(model.model.prepare_decoder_input_ids_from_labels(labels)) 

        assert ex.state_position < knowledge_states.size(0), \
              f"State position {ex.state_position} out of bounds for knowledge states with size {knowledge_states.size(0)}"
        
        state_positions_list.append(ex.state_position)

    batch_input_ids = torch.cat(enc_ids_list, dim=0).to(knowledge_states.device)  # (B, T)
    batch_attn_mask = torch.cat(attn_mask_list, dim=0).to(knowledge_states.device)  # (B, T)
    batch_labels = torch.cat(labels_list, dim=0).to(knowledge_states.device)  # (B, D)
    batch_dec_ids = torch.cat(dec_ids_list, dim=0).to(knowledge_states.device)  # (B, D)

    batch_state_positions = torch.tensor(state_positions_list, dtype=torch.long, device=knowledge_states.device)  # (B,)
    student_states = knowledge_states[batch_state_positions]  # (B, V)

    # (B, T_dec), (B, S)
    input_difficulties, sub_word_difficulties = qg_input_difficulties(student_states, batch_labels, tokenizer, sub_word_ids, oov_mask) 
    
    return batch_input_ids, batch_attn_mask, student_states, input_difficulties, batch_dec_ids, batch_labels, sub_word_difficulties

# === GENERATION

@dataclass
class CandidateExercise:
    decoded_ids: torch.Tensor
    length: int = 0
    likelihood: float = 0.0
    difficulty: float = 0.0
    future_diff: float = 0.0
    final_score: float = 0.0

class DCDecoder:
    def __init__(
        self, 
        model: ExerciseGenerator, 
        tokenizer: AutoTokenizer,
        sub_word_ids: torch.Tensor,
        oov_mask: torch.Tensor,
        beam_size: int,
        lookahead_steps: int,
        factor_likelihood: float = 0.5,
        factor_diff: float = 0.5,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.sub_word_ids = sub_word_ids
        self.oov_mask = oov_mask
        self.beam_size = beam_size
        self.lookahead_steps = lookahead_steps

        self.factor_likelihood = factor_likelihood
        self.factor_diff = factor_diff

        self.special_token_ids = {
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.unk_token_id
        }

    def extend_candidate(
        self,
        base_cand: CandidateExercise,
        token_id: int,
        token_log_prob: float,
        sub_word_difficulties: torch.Tensor,
        device: torch.device
    ) -> CandidateExercise:
        
        tok_id_tens = torch.tensor([[token_id]], device=device)  # (1, 1)
        is_special = token_id in self.tokenizer.all_special_ids

        return CandidateExercise(
            decoded_ids = torch.cat([base_cand.decoded_ids, tok_id_tens.squeeze(0)]),  # (L+1)
            length = base_cand.length + 1 if not is_special else base_cand.length,  # Only count non-special tokens towards length
            likelihood = base_cand.likelihood + token_log_prob if not is_special else base_cand.likelihood,
            difficulty = base_cand.difficulty + sub_word_difficulties[token_id].item()
        )

    def top_k(
        self,
        extensions: tuple[CandidateExercise, torch.Tensor],  # (CandidateExercise, log_probs)
        sub_word_difficulties: torch.Tensor,
    ) -> list[CandidateExercise]: 

        cands: list[CandidateExercise] = []
        added: set[tuple[int, ...]] = set()

        # PRUNING
        for base_cand, log_probs in extensions:
            
            # === K=1 MAX PRUNING
            vals, idxs = torch.topk(log_probs[0], k=1)  # (k,), (k,)
            tok_id = int(idxs[0].item())
            tok_log_prob = vals[0].item()

            new_cand = self.extend_candidate(base_cand, tok_id, tok_log_prob, sub_word_difficulties, log_probs.device)
            key_cand = tuple(tok_id for tok_id in new_cand.decoded_ids.tolist() if tok_id not in self.special_token_ids) 

            if key_cand not in added:
                cands.append(new_cand)
                added.add(key_cand)

        # === GLOBAL TOP-K PRUNING

        flat_full_log_probs = torch.cat(
            [log_probs + cand.likelihood for cand, log_probs in extensions], # E entries of(1, V) 
            dim=-1
        ) # (1, E*V) with each cand's likelihood added to its extensions' log probs


        vals, idxs = torch.topk(
            flat_full_log_probs[0],
            k=min(self.beam_size, flat_full_log_probs.shape[-1])
        )  # (k,), (k,)

        V = extensions[0][1].shape[-1]  # Number of extensions, vocab size

        for _, flat_idx in zip(vals, idxs):
            ext_idx = flat_idx.item() // V  # Which candidate's extensions
            tok_id = flat_idx.item() % V  # Which token in that candidate's extensions

            base_cand, _ = extensions[ext_idx]

            new_cand = self.extend_candidate(
                base_cand, 
                tok_id, 
                extensions[ext_idx][1][0, tok_id], 
                sub_word_difficulties, 
                log_probs.device)
            
            key_cand = tuple(tok_id for tok_id in new_cand.decoded_ids.tolist() if tok_id not in self.special_token_ids) 

            if key_cand not in added:
                cands.append(new_cand)
                added.add(key_cand)
            
        return cands
        
    def score(self, cand: CandidateExercise, target_difficulty: float, is_finished: bool) -> float:
        avg_ll = cand.likelihood / max(1, cand.length)
        projected_diff = cand.difficulty if is_finished else cand.difficulty + cand.future_diff
        return self.factor_likelihood * avg_ll - self.factor_diff * abs(projected_diff - target_difficulty)
        
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,  # (1, T)
        attention_mask: torch.Tensor,  # (1, T)
        target_difficulty: float,  
        target_keywords: list[str],
        student_state: torch.Tensor,  # (1, V)
        max_length: int = 30,
    ) -> list[CandidateExercise]:
        
        device = input_ids.device
        sub_word_difficulties = (student_state[:, self.sub_word_ids] * self.oov_mask).squeeze(0)  # (1, S)

        start_ids = torch.tensor(
            [self.tokenizer.eos_token_id, self.tokenizer.bos_token_id],
            dtype=torch.long, 
            device=device
        )

        diff_tens = torch.tensor([[target_difficulty]], device=device)

        beams = [CandidateExercise(decoded_ids=start_ids)]
        finished: list[CandidateExercise] = []
        finished_keys: set[tuple[int, ...]] = set()


        max_steps = max_length - len(start_ids)
        assert max_steps > 0, "Max length must be greater than reserved start tokens"

        for _ in range(max_steps):
            
            extensions: list[tuple[CandidateExercise, torch.Tensor]] = []
            
            for cand in beams:
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    student_state=student_state,
                    difficulty=diff_tens,
                    decoder_input_ids=cand.decoded_ids.unsqueeze(0),  # (1, D)
                )
                logits = output.logits[:, -1, :]  # (1, Vocab)
                log_probs = torch.log_softmax(logits, dim=-1)  # (1, Vocab)

                extensions.append((cand, log_probs))

            assert extensions, "No beam extensions generated; check model output and beam search implementation"

            candidates = self.top_k(extensions, sub_word_difficulties) # list[CandidateExercise]

            r_input_ids = input_ids.repeat(len(candidates), 1)  # (B, T)
            r_attention_mask = attention_mask.repeat(len(candidates), 1)  # (B, T)
            r_student_state = student_state.repeat(len(candidates), 1)  # (B, V)
            r_difficulty = diff_tens.repeat(len(candidates), 1)  # (B, 1)

            input_embds = self.model.get_input_embeddings(
                input_ids=r_input_ids,
                difficulty=r_difficulty,
                student_state=r_student_state,
            ) # (B, T, E)


            assert len({cand.decoded_ids.shape[0] for cand in candidates}) == 1, "All candidates must have the same decoded_ids length for batch processing"
            decoder_input_ids = torch.stack([cand.decoded_ids for cand in candidates], dim=0)  # (B, T_cur)

            gen_outputs = self.model.model.generate(
                inputs_embeds=input_embds,
                attention_mask=r_attention_mask,
                decoder_input_ids=decoder_input_ids,
                num_beams=1,
                max_new_tokens=self.lookahead_steps,
                return_dict_in_generate=True,
                output_scores=True,
                num_return_sequences=1
            ) 

            assert gen_outputs.scores is not None, "Model did not return scores; ensure model is configured to return scores during generation"

            score_tensor = torch.softmax(
                torch.stack(gen_outputs.scores), # scores[lookahead] = [(B, V)], -> (lookahead, B, V) 
                dim=-1
            ).permute(1, 0, 2)  # (B, lookahead, V)

            future_Ediff = torch.matmul(
                score_tensor,  # (B, lookahead, V)
                sub_word_difficulties.unsqueeze(-1) # (V, 1)
            ) # (B, L, 1)
            
            sum_future_Ediff = future_Ediff.squeeze(-1).sum(dim=1)  # (B, )
            
            for (cand, future) in zip(candidates, sum_future_Ediff.tolist()):
                cand.future_diff = future
                cand.final_score = self.score(
                    cand, 
                    target_difficulty, 
                    is_finished=False
                )

            candidates.sort(key=lambda c: c.final_score, reverse=True) # sort by projected final score
            beams = candidates[:self.beam_size]  # Keep top beams for next step

            for cand in beams:
                if int(cand.decoded_ids[-1].item()) == self.tokenizer.eos_token_id:
                    key = tuple(tok_id for tok_id in cand.decoded_ids.tolist() if tok_id not in self.special_token_ids)
                    if key not in finished_keys:
                        finished.append(cand)
                        finished_keys.add(key)
                
            if len(finished) >= self.beam_size:
                break

        pool = finished or beams # If no finished candidates, fall back to current beams

        for cand in pool:
            cand.final_score = self.score(cand, target_difficulty, is_finished=True)

        best_cand = max(pool, key=lambda c: c.final_score)
        return best_cand.decoded_ids.unsqueeze(0)  # (1, L)
    
