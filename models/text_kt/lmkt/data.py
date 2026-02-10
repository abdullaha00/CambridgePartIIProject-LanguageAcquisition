from dataclasses import dataclass
from functools import partial
import random
from typing import Dict, List, Tuple, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

from transformers import AutoTokenizer

from data_processing.data_parquet import load_train_and_eval_df
from models.text_kt.common.data import collapse_to_exercise, build_user_sequences_text, history_text
from models.text_kt.common.tokens import TOK_N, TOK_Y

logger = logging.getLogger(__name__)

class SeqDatasetLMKT(Dataset):
    """
    seqs[idx] holds encoded tok_ids for the full hitory of user idx
    """

    def __init__(self, histories: dict, tokenizer, max_length: int = 1024):
        
        self.seqs: List[torch.Tensor] = []
        # TOKENISE HISTORIES
        for _, history in tqdm(histories.items(), desc="Tokenizing histories", leave=False):
            text = history_text(history)
            token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False) 

            if len(token_ids) <= max_length:
                    self.seqs.append(torch.tensor(token_ids, dtype=torch.long))
                    continue
            else:
                # Use a windowing approach
                # OPTIONS: sample window randomly
                # USE SLIDING WINDOW
                # mode = "sliding" or "random" 
                mode = "truncate-left"
                if mode == "truncate-left": # keep the most recent tokens
                    window_ids = token_ids[-max_length:]
                    self.seqs.append(torch.tensor(window_ids, dtype=torch.long))
                elif mode == "truncate-right": # keep the earliest tokens
                    window_ids = token_ids[:max_length]
                    self.seqs.append(torch.tensor(window_ids, dtype=torch.long))
                elif mode == "sliding":
                    # WE USE A SLIDING WINDOW DUE TO CONTEXT WINDOW LENGTHS
                    WINDOW = 1024
                    STRIDE = 256
                    for start in range(0, len(token_ids), STRIDE):
                        window_ids = token_ids[start:start+WINDOW]
                        self.seqs.append(torch.tensor(window_ids, dtype=torch.long))
                elif mode == "random":
                    start = random.randint(0, len(token_ids) - max_length)
                    window_ids = token_ids[start:start+max_length]
                    self.seqs.append(torch.tensor(window_ids, dtype=torch.long))

        logger.info(f"Built LMKTDataset: {len(self.seqs)} sequences.")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]

def lmkt_collate(batch, pad_token_id: int, y_id: int, n_id: int) -> Dict[str, torch.Tensor]:
    # Batch: list[Tensor(seq_len)]

    T_max = max(x.numel() for x in batch)
    B = len(batch)

    seqs_padded = torch.full((B, T_max), pad_token_id, dtype=torch.long)
    mask = torch.zeros((B, T_max), dtype=torch.long) 

    for i, seq in enumerate(batch):
        T = seq.numel()
        seqs_padded[i, :T] = seq
        mask[i, :T] = 1

    #===== -100 signifies ignore for LM loss =====
    # We ignore everything but <Y/N> labels

    labels = torch.full_like(seqs_padded, -100)
    
    yn_mask = (seqs_padded == y_id) | (seqs_padded == n_id)
    labels[yn_mask] = seqs_padded[yn_mask]
    
    return {"input_ids": seqs_padded, "attention_mask": mask, "labels": labels}

