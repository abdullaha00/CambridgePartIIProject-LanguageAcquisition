import pandas as pd
from typing import Dict, List, Sequence, Tuple
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import logging 

logger = logging.getLogger(__name__)

class SeqDataset(Dataset):
    def __init__(self, seqs: dict):
        # Convert dict to list of (uid, (q_ids, correct_list)) 
        self.seqs = list(seqs.items())

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        uid, (q_ids, correct_list) = self.seqs[idx]
        return uid, torch.from_numpy(q_ids.copy()), torch.from_numpy(correct_list.copy())

def batch_pad(batch):
    # batch: list of (uid, q, a)
    
    uids, q_ids, correct_list = zip(*batch)

    T_max = max(len(q_seq) for q_seq in q_ids)
    B = len(q_ids)

    q_ids_padded = torch.zeros((B, T_max), dtype=torch.long)
    correct_list_padded = torch.zeros((B, T_max), dtype=torch.long)
    mask = torch.zeros((B, T_max), dtype=torch.bool)

    for i, (q, a) in enumerate(zip(q_ids, correct_list)):
        T = len(q)
        q_ids_padded[i, :T] = q
        correct_list_padded[i, :T] = a
        mask[i, :T] = 1
    
    return uids, q_ids_padded, correct_list_padded, mask

def build_user_sequences(df: pd.DataFrame, qid_map: dict) -> dict:
    """
    
    """
    
    seqs = {} 
    
    for uid, df_user in df.groupby("user_id", sort=False):
        q_ids = df_user["question_id"].to_numpy()
        correct_list = df_user["correct"].to_numpy()

        seqs[uid] = (q_ids, correct_list)

    return seqs 

#====== LM - KT HELPERS ======

def build_user_sequences_text(df_ex: pd.DataFrame) -> dict:
    """
    Converts exercise-level dataframe to per ordered histories
    Returns:
        histories[user_id] = [(ref_ans+text, correct01), ...] in time-order
    """
    
    histories: Dict[str, List[Tuple[str, int]]] = {}

    for uid, g in tqdm(df_ex.groupby("user_id", sort=False), desc="Building user histories", leave=False):
        ref_ans_list = g["ref_ans"].tolist()
        correct_list = g["correct"].tolist()

        histories[uid] = list(zip(ref_ans_list, correct_list))
    
    return histories

def history_text(history: Sequence[Tuple[str, int]]) -> str:
    """
    Composes history text from sequence of (ref_ans, correct01)
    """
    
    out = []

    for text, correct in history:
        out.append(f"<Q>: {text} <A>: {'<Y>' if correct == 1 else '<N>'}")
    return " ".join(out)

def lmkt_batch_pad(batch, pad_token_id: int = 0):
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
    labels = seqs_padded.clone()
    labels[mask == 0] = -100
    
    return {"input_ids": seqs_padded, "attention_mask": mask, "labels": labels}

class SeqDatasetLMKT(Dataset):
    def __init__(self, histories: dict, tokenizer, max_length: int = 1024):
        
        examples: List[Dict[str, torch.Tensor]] = []
        # TOKENISE HISTORIES
        for _, history in tqdm(histories.items(), desc="Tokenizing histories", leave=False):
            text = history_text(history)
            token_ids = tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True) 

            #TODO: use sliding windows to get around context length limits (1024 for gpt2)
            examples.append(torch.tensor(token_ids, dtype=torch.long))

        self.seqs = examples
        logger.info(f"Built LMKTDataset: {len(self.seqs)} sequences.")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]