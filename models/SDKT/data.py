
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SDKTVocabs:
    token_vocab: dict[str, int]
    meta_vocabs: dict[str, dict[str, int]]

    @property
    def num_tokens(self) -> int:
        return len(self.token_vocab)
    
    @property
    def meta_vocab_sizes(self) -> dict[str, int]:
        return {col: len(vocab) for col, vocab in self.meta_vocabs.items()}


def build_vocab(series: pd.Series) -> dict[str, int]:

    vocab = {x: i+1 for i, x in enumerate(series.unique())}
    return vocab

def build_meta_vocabs(df: pd.DataFrame, meta_cols: list[str]) -> dict[str, dict[str, int]]:
    
    meta_vocabs = {}
    for col in meta_cols:
        meta_vocabs[col] = build_vocab(df[col])
    return meta_vocabs

def build_user_sequences(
    df: pd.DataFrame,
    token_vocab: dict[str, int],
    meta_vocabs: dict[str, dict[str, int]],
) -> dict[str, tuple[list[int], dict[str, list[int]], list[int]]]:

    user_seqs = {}
    unk_tok_id = len(token_vocab) + 1

    for uid, df_u in df.groupby("user_id"):

        
        lemma_seq = (df_u["lemma"]
                     .map(token_vocab)
                     .fillna(unk_tok_id)
                     .to_numpy(dtype=np.int64))
        
        # NOTE: we flip labels so that 1 = correct and 0 = incorrect, since DKT typically models P(correct)
        label_seq = (1-df_u["label"]).to_numpy()

        meta_ids = {}

        for col in meta_vocabs:
            vocab = meta_vocabs[col]
            unk_id = len(vocab) + 1
            meta_ids[col] = (
                df_u[col].map(vocab)
                .fillna(unk_id).
                to_numpy(dtype=np.int64)
                )
        
        user_seqs[uid] = (lemma_seq, meta_ids, label_seq)

    return user_seqs

#===== DATASETS

class SDKTTrainDataset(Dataset):
    def __init__(self, user_seqs: dict[str, tuple[list[int], dict[str, list[int]], list[int]]]):
        self.user_seqs = list(user_seqs.values())
    
    def __len__(self):
        return len(self.user_seqs)
    
    def __getitem__(self, idx):
        lemma_seq, meta_ids, label_seq = self.user_seqs[idx]
        T = len(lemma_seq)

        split_r = np.random.uniform(0.4, 0.9)
        k = int(T * split_r)
        k = max(k, 5) # ensure at least 1 token in input seq
        k = min(k, T-5) # ensure at least 1 token in label seq

        return {
            "enc_q": torch.from_numpy(lemma_seq[:k]),
            "enc_m": {col: torch.from_numpy(ids[:k]) for col, ids in meta_ids.items()},
            "enc_a": torch.from_numpy(label_seq[:k]),
            "dec_q": torch.from_numpy(lemma_seq[k-1:-1]), # teacher
            "dec_m": {col: torch.from_numpy(ids[k-1:-1]) for col, ids in meta_ids.items()},
            "dec_a": torch.from_numpy(label_seq[k:]),
            "enc_last_a": torch.tensor(label_seq[k-1], dtype=torch.long)
        }
    
class SDKTEvalDataset(Dataset):
    def __init__(self, train_seqs: dict, eval_seqs: dict):
        self.data = []
        for uid in eval_seqs:
            if uid not in train_seqs:
                logger.warning(f"User {uid} in eval set not found in train set. This user will be skipped in evaluation.")
                continue
            tr_q, tr_m, tr_l = train_seqs[uid]
            ev_q, ev_m, ev_l = eval_seqs[uid]

            self.data.append((tr_q, tr_l, tr_m, ev_q, ev_l, ev_m))
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        tr_q, tr_l, tr_m, ev_q, ev_l, ev_m = self.data[idx]
        return {
            "enc_q": torch.from_numpy(tr_q),
            "enc_a": torch.from_numpy(tr_l),
            "enc_m": {col: torch.from_numpy(ids) for col, ids in tr_m.items()},
            "dec_q": torch.from_numpy(ev_q),
            "dec_a": torch.from_numpy(ev_l),
            "dec_m": {col: torch.from_numpy(ids) for col, ids in ev_m.items()},
            "enc_last_a": torch.tensor(tr_l[-1], dtype=torch.long)
        }
    

PAD_ID = 0

def collate_sdkt(batch: list[dict]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    # batch is a list of dicts with keys: enc_q, enc_a, enc_m, dec_q, dec_a, dec_m

    B = len(batch)

    Tmax_enc = max(x["enc_q"].numel() for x in batch)
    Tmax_dec = max(x["dec_q"].numel() for x in batch)

    m_keys = batch[0]["enc_m"].keys()

    enc_q_batch = torch.full((B, Tmax_enc), PAD_ID, dtype=torch.long)
    enc_a_batch = torch.full((B, Tmax_enc), PAD_ID, dtype=torch.long)
    enc_m_batch = {col: torch.full((B, Tmax_enc), PAD_ID, dtype=torch.long) for col in m_keys}
    enc_mask_batch = torch.zeros((B, Tmax_enc), dtype=torch.bool)

    dec_q_batch = torch.full((B, Tmax_dec), PAD_ID, dtype=torch.long)
    dec_a_batch = torch.full((B, Tmax_dec), PAD_ID, dtype=torch.long)
    dec_m_batch = {col: torch.full((B, Tmax_dec), PAD_ID, dtype=torch.long) for col in m_keys}
    dec_mask_batch = torch.zeros((B, Tmax_dec), dtype=torch.bool)

    enc_last_a_batch = torch.zeros(B, dtype=torch.long)

    for i, x in enumerate(batch):
        T_enc = x["enc_q"].numel()
        enc_q_batch[i, :T_enc] = x["enc_q"]
        enc_a_batch[i, :T_enc] = x["enc_a"]
        enc_mask_batch[i, :T_enc] = True
        for m in m_keys:
            enc_m_batch[m][i, :T_enc] = x["enc_m"][m]
        
        T_dec = x["dec_q"].numel()
        dec_q_batch[i, :T_dec] = x["dec_q"]
        dec_a_batch[i, :T_dec] = x["dec_a"]
        dec_mask_batch[i, :T_dec] = True
        for m in m_keys:
            dec_m_batch[m][i, :T_dec] = x["dec_m"][m]
        
        enc_last_a_batch[i] = x["enc_last_a"]
    
    return {
        "enc_q": enc_q_batch,
        "enc_a": enc_a_batch,
        "enc_m": enc_m_batch,
        "enc_mask": enc_mask_batch,
        "dec_q": dec_q_batch,
        "dec_a": dec_a_batch,
        "dec_m": dec_m_batch,
        "dec_mask": dec_mask_batch,
        "enc_last_a": enc_last_a_batch
    }
