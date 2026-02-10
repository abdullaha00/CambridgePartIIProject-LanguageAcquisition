from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import torch
from tqdm import tqdm

from models.text_kt.common.tokens import TOK_A, TOK_G, TOK_N, TOK_Q, TOK_Y
from models.text_kt.common.data import history_text

@dataclass
class QGExample:
    input_ids: torch.Tensor #(T,)
    attention_mask: torch.Tensor #(T,)
    labels: torch.Tensor #(T,), with -100 for ignore
    difficulty: torch.Tensor #(1,)

class QGDataset:
    """
    QG training examples are of the form:
    prefix + <G> + <Q> question_text <A>
    where the difficulty assigned to question_text is computed by a frozen LMKTmodel,
    and question_text is held out from the student's history used to train the LMKTmodel.
    """
    def __init__(
        self,
        histories: Dict[str, List[Tuple[str, int]]],
        held_out_qs: List[str],
        tokenizer,
        difficulty_fn: Callable[[str, str], float],
        max_enc_length: int = 1024,
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
            
            full_text = history_text(hist)

            enc_ids_full = tokenizer.encode(
                full_text,
                add_special_tokens=False,
                truncation=False,
                return_tensors="pt",
            ) #(1 , T_full)

            #TRUNCUATION: we keep the last max_enc_length tokens
            if len(enc_ids_full[0]) > max_enc_length:
                enc_ids_full = enc_ids_full[:, -max_enc_length:]
            
            #prompt_ids = enc_ids_full + 

            # we train on sequences of the form
            # "<Q>_0 question_text_0 <A>_0 <Y/N>_0.... <Y/N>_{t-1}"
            labels = enc_ids_full.clone()
            yn_mask = (enc_ids_full == self.y_id) | (enc_ids_full == self.n_id)
            labels[~yn_mask] = -100  # only compute loss on <Y/N> tokens to match paper

            self.examples.append(
                QGExample(
                    input_ids=enc_ids_full.squeeze(0), #(T, )
                    attention_mask=torch.ones_like(enc_ids_full).squeeze(0),
                    labels=labels.squeeze(0),
                    difficulty=torch.tensor([0.0], dtype=torch.float),  # Placeholder difficulty
                )
            )

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
        difficulties[i] = ex.difficulty

    return {
        "input_ids": input_ids_pad,
        "attention_mask": attention_mask,
        "labels": labels,
        "difficulty": difficulties,
    }

