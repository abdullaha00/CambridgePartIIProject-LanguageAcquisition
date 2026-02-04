import pandas as pd
from typing import Dict, List, Sequence, Tuple
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import logging 

# DKT SEQUENCE BUILDER
def build_user_sequences_qid(df: pd.DataFrame, qid_map: dict) -> dict:    
    seqs = {} 
    
    for uid, df_user in df.groupby("user_id", sort=False):
        q_ids = df_user["question_id"].to_numpy()
        correct_list = df_user["correct"].to_numpy()

        seqs[uid] = (q_ids, correct_list)

    return seqs


# LMKT SEQUENCE BUILDER
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