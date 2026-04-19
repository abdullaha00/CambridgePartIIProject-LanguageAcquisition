from dataclasses import dataclass
import math
from random import random
from typing import Callable, Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm

from models.modular_qg.common.tokens import TOK_A, TOK_BOS, TOK_EOS, TOK_G, TOK_N, TOK_Q, TOK_Y
from models.modular_qg.common.data import history_text

def resize_prompt(prompt:str, length = 200) -> str:
    split_prompt = prompt.split(" ")
    if len(split_prompt) <= length:
        return prompt
    
    new_prompt = split_prompt[-(length-1):]
    if TOK_Q in new_prompt:
        q_idx = new_prompt.index(TOK_Q)
        new_prompt = new_prompt[q_idx:]
    
    new_line = " ".join(new_prompt)
    return new_line

@dataclass
class QGExample:
    input_ids: torch.Tensor #(T,)
    attention_mask: torch.Tensor #(T,)
    labels: torch.Tensor #(T,), with -100 for ignore
    difficulty: torch.Tensor #(1,)

class QGDataset:
    """
    QG training examples are of the form:
    prefix + difficulty_val + <G> + <Q> question_text <A>
    where the difficulty assigned to question_text is computed by a frozen LMKTmodel,
    and question_text is held out from the student's history used to train the LMKTmodel.
    """
    def __init__(
        self,
        histories: Dict[str, List[Tuple[str, int]]],
        held_out_qs: List[str],
        tokenizer,
        difficulty_fn,
        max_enc_length: int = 1022,
    ):
        
        # Ensure <G> is present
        g_id = tokenizer.convert_tokens_to_ids(TOK_G)
        self.y_id = tokenizer.convert_tokens_to_ids(TOK_Y)
        self.n_id = tokenizer.convert_tokens_to_ids(TOK_N)
        unk = tokenizer.unk_token_id
        assert unk not in [g_id, self.y_id, self.n_id], f"Special tokens not found in tokenizer."

        # TODO: Shuffle, check trained history
        # chosen_qs = held_out_qs[:100]  # for now, limit to first 100 held-out questions

        self.examples: List[QGExample] = []

        for _uid, hist in tqdm(histories.items(), desc="Building QG examples", leave=False):
            
            t = len(hist)
            
            pref_text = history_text(hist[:-1])
            
            assert len(held_out_qs) > 0, "No held-out questions available for QG example generation."
            n_sample = min(len(held_out_qs), 5)
            sampled_qs = np.random.choice(held_out_qs, size=n_sample, replace=False)  # sample up to 5 questions for this user]
            
            prefix_texts = []
            
            for q_text in sampled_qs:
                full_prompt = f"{pref_text} {TOK_Q} {q_text} {TOK_A}".strip()
                full_prompt = resize_prompt(full_prompt)

                last_q = full_prompt.rfind(TOK_Q)
                assert last_q != -1, f"Could not find {TOK_Q} in full prompt: {full_prompt}"

                prefix_prompt = full_prompt[:last_q].strip()
                assert prefix_prompt.endswith(TOK_Y) or prefix_prompt.endswith(TOK_N), f"Prefix prompt should end with {TOK_Y} or {TOK_N}: {prefix_prompt}"

                prefix_texts.append(prefix_prompt)
            
            # Compute difficulty batched
            with torch.no_grad():
                diffs = difficulty_fn(prefix_texts, list(sampled_qs))  # (n_sample,)

            for pref_text, q_text, diff in zip(prefix_texts, sampled_qs, diffs):
                p_y = math.floor(diff.item() * 1000) / 1000  # format to 3 decimal places
                scaled_diff = p_y * 100 # scale up

                target_text = f"{TOK_G} {q_text} {TOK_EOS}"
                target_enc = tokenizer.encode(target_text, add_special_tokens=False, truncation=False)

                prefix_enc = tokenizer.encode(pref_text, add_special_tokens=False, truncation=False)
                max_pref_len = max_enc_length - len(target_enc)
                if len(prefix_enc) > max_pref_len:
                    prefix_enc = prefix_enc[-max_pref_len:]
                
                enc_ids_full = torch.tensor(prefix_enc + target_enc)
                assert g_id in enc_ids_full, f"Generated input does not contain {TOK_G}: {enc_ids_full}"
                
                # Mask everything up to and including <G> with -100
                # Only train on tokens AFTER <G> (the generated question)
                labels = enc_ids_full.clone()
                g_positions = (enc_ids_full == g_id).nonzero(as_tuple=False)
                assert g_positions.numel() > 0, f"No <G> token found in enc_ids_full"
                g_idx = g_positions[0].item()
                labels[:g_idx + 1] = -100

                self.examples.append(QGExample(
                    input_ids=enc_ids_full,
                    attention_mask=torch.ones_like(enc_ids_full),
                    labels=labels,
                    difficulty=torch.tensor(scaled_diff, dtype=torch.float32)
                ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
def qg_collate(batch: List[QGExample], pad_token_id: int) -> Dict[str, torch.Tensor]:

    B = len(batch)
    T_max = max(x.input_ids.numel() for x in batch)

    input_ids_pad = torch.full((B, T_max), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T_max), dtype=torch.long)
    labels = torch.full((B, T_max), -100, dtype=torch.long)
    difficulties = torch.zeros((B, 1), dtype=torch.float32)

    for i, ex in enumerate(batch):
        T = ex.input_ids.numel()
        input_ids_pad[i, :T] = ex.input_ids
        attention_mask[i, :T] = 1

        labels[i, :T] = ex.labels
        # mask padding 
        labels[i, T:] = -100
        difficulties[i] = ex.difficulty

    return {
        "input_ids": input_ids_pad,
        "attention_mask": attention_mask,
        "labels": labels,
        "difficulty": difficulties,
    }
