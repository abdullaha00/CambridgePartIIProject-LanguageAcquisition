import pandas as pd
import numpy as np
import pandas as pd
from typing import Dict, List, Sequence, Tuple
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import logging 
from typing import Sequence, Tuple
from .tokens import TOK_Q, TOK_A, TOK_Y, TOK_N

def collapse_to_exercise(df: pd.DataFrame) -> pd.DataFrame:
    
    # CHECK IF LABELS ARE AVAILABLE
    if df["label"].isna().any():
        raise ValueError("Some labels are missing; cannot collapse to exercise level.")

    return (

        df.groupby(["ex_instance_id", "user_id"], sort=False).agg(
            ref_ans=("tok", " ".join),
            # exercise is correct if ALL tokens are correct (label 0)
            correct=("label", lambda x: int(np.any(x == 0)))
        ).reset_index()

    )

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
    (ref_ans, correct01) -> <Q> ref_ans <A> <Y/N> <Q> ref_ans <A> <Y/N> ..."

    """
    
    out = []

    for text, correct in history:
        out.append(f"{TOK_Q} {text} {TOK_A} {TOK_Y if correct == 1 else TOK_N}")
    return " ".join(out)