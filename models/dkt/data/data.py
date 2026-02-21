import pandas as pd
from typing import Dict

import numpy as np
import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset

BATCH_SIZE = 32

def build_user_sequences_tok(df: pd.DataFrame) -> dict:
    seqs = {} 
    
    for uid, df_user in df.groupby("user_id", sort=False):
        item_ids = df_user["item"].to_numpy()
        correct_list = (df_user["label"]==0).to_numpy()

        seqs[uid] = (item_ids, correct_list)

    return seqs

def build_user_sequences_qid(df: pd.DataFrame, qid_map: dict) -> dict:    
    seqs = {} 
    
    for uid, df_user in df.groupby("user_id", sort=False):
        q_ids = df_user["question_id"].to_numpy()
        correct_list = df_user["correct"].to_numpy()

        seqs[uid] = (q_ids, correct_list)

    return seqs

def generate_qid_map(df: pd.DataFrame) -> Dict[str, int]:
    unique_qs = df["ref_ans"].unique()
    qid_map = {q: i for i, q in enumerate(unique_qs)}
    return qid_map

def apply_qid_map(df: pd.DataFrame, qid_map: dict, drop_unk=True) -> pd.DataFrame:
    df["question_id"] = df["ref_ans"].map(qid_map)

    # We can either drop if drop_unk is true
    if drop_unk:
        df = df.dropna(subset=["question_id"])
    else:
        # replace final qid value 
        unk_qid = len(qid_map)
        df["question_id"] = df["question_id"].fillna(unk_qid)
    return df

def embed_sentence_matrix(sentence_list: List[str], model_name: str = "distilbert-base-uncased") -> np.ndarray:

    all_embs = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    with torch.no_grad():
        
        for i in range(0, len(sentence_list), BATCH_SIZE):
            batch_texts = sentence_list[i:i+BATCH_SIZE]

            enc = tokenizer(
                batch_texts,
                padding=True,
                return_tensors="pt"
            )

            out = model(**enc)

            last_hidden_state = out.last_hidden_state
            
            # Pooling
            sentence_emb = last_hidden_state.mean(dim=1)

            all_embs.append(sentence_emb.cpu().numpy())

    return np.vstack(all_embs) # (num_sentences, embedding_dim)

class DKTSeqDataset(Dataset):
    def __init__(self, seqs: dict):
        # Convert dict to list of (uid, (q_ids, correct_list)) 
        self.seqs = list(seqs.items())

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        uid, (q_ids, correct_list) = self.seqs[idx]
        return uid, torch.from_numpy(q_ids.copy()), torch.from_numpy(correct_list.copy())

def collate_dkt(batch):
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
