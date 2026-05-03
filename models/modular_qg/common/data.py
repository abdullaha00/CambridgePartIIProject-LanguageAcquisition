import pandas as pd
import numpy as np
import pandas as pd
from typing import Dict, List, Sequence, Tuple
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import logging 
from typing import Sequence, Tuple

from .tokens import TOK_Q, TOK_A, TOK_Y, TOK_N, TOK_BOS

def collapse_to_exercise(df: pd.DataFrame) -> pd.DataFrame:
    
    # CHECK IF LABELS ARE AVAILABLE
    if df["label"].isna().any():
        raise ValueError("Some labels are missing.")

    if "ex_key" not in df.columns:
        ex_key = df["tok_id"].str.slice(0, 10)
        df["ex_key"] = ex_key

    return (

        df.groupby(["ex_key", "user_id"], sort=False).agg(
            tok_text=("tok", " ".join),
            # exercise is correct if ALL tokens are correct (label 0)
            correct=("label", lambda x: int(np.all(x == 0)))
        ).reset_index()
    )

def merge_with_prompts(df_ex: pd.DataFrame, df_prompt: pd.DataFrame) -> pd.DataFrame:

    df = df_ex.merge(df_prompt, on="ex_key", how="left")

    if df["prompt"].isna().any():
        missing = df[df["prompt"].isna()]["ex_inst_idx"].unique()
        raise ValueError(f"Missing prompts for exercise ids: {missing}")

    return df

def build_user_sequences_text(df_ex: pd.DataFrame) -> Dict[str, List[Tuple[str, int]]]:
    """
    Converts exercise-level dataframe to per ordered histories
    Returns:
        histories[user_id] = [(prompt+text, correct01), ...] in time-order
    """
    
    histories: Dict[str, List[Tuple[str, int]]] = {}

    for uid, g in tqdm(df_ex.groupby("user_id", sort=False), desc="Building user histories", leave=False):
        prompt_list = g["prompt"].tolist()
        correct_list = g["correct"].tolist()

        histories[str(uid)] = list(zip(prompt_list, correct_list))
    
    return histories

def history_text(history: Sequence[Tuple[str, int]], compact: bool = False) -> str:
    """
    Composes history text from sequence of (prompt, correct01)
    (prompt, correct01) -> <BOS> <Q> prompt <A> <Y/N> <Q> prompt <A> <Y/N> ...

    """
    assert history

    if compact:
        body = "".join(f"{TOK_Q}{text}{TOK_A}{TOK_Y if correct == 1 else TOK_N}" for text, correct in history)
        return f"{TOK_BOS}{body}"

    body = " ".join(f"{TOK_Q} {text} {TOK_A} {TOK_Y if correct == 1 else TOK_N}" for text, correct in history)
    return f"{TOK_BOS} {body}".strip()
