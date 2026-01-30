import pandas as pd
from torch.utils.data import Dataset
import torch

class SeqDataset(Dataset):
    def __init__(self, seqs: dict):
        # Convert dict to list of (uid, (q_ids, correct_list)) 
        self.seqs = list(seqs.items())

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):

        uid, (q_ids, correct_list) = self.seqs[idx]
        return uid, torch.from_numpy(q_ids), torch.from_numpy(correct_list)

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
    
    seqs = {} 
    
    for uid, df_user in df.groupby("user_id", sort=False):
        q_ids = df_user["question_id"].to_numpy()
        correct_list = df_user["correct"].to_numpy()

        seqs[uid] = (q_ids, correct_list)

    return seqs 
