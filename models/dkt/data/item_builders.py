from typing import Dict, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from models.text_kt.common.data import collapse_to_exercise

@dataclass(frozen=True)
class SeqBundle:
    seqs: Dict[str, Tuple[np.ndarray, np.ndarray]]
    item_map: Dict[str, int]
    item_level: str 

def build_item_map(train_items: pd.Series) -> Dict[str, int]:
    unique_items = train_items.unique()
    item_map = {item: i for i, item in enumerate(unique_items)}
    return item_map

def apply_item_map(df: pd.DataFrame, item_col: str, item_map: dict, drop_unk: bool) -> pd.DataFrame:
    df["item_id"] = df[item_col].map(item_map)
    
    if drop_unk:
        df = df.dropna(subset=["item_id"])
    else:
        unk_item_id = len(item_map)
        df["item_id"] = df["item_id"].fillna(unk_item_id)

    return df

def group_user_seqs(df: pd.DataFrame, item_col: str, outcome_col: str) -> dict:
    seqs = {} 
    
    for uid, df_user in df.groupby("user_id", sort=False):
        item_ids = df_user[item_col].to_numpy()
        correct_list = df_user[outcome_col].to_numpy()
        seqs[uid] = (item_ids, correct_list)

    return seqs

def build_tok_sequences(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    item_col: str = "lemma",
    drop_unk: bool = True
) -> SeqBundle:
    
    item_map = build_item_map(df_train[item_col])
    
    df_train = apply_item_map(df_train, item_col, item_map, drop_unk=drop_unk)
    df_eval = apply_item_map(df_eval, item_col, item_map, drop_unk=drop_unk)

    train_seqs = group_user_seqs(df_train, "item_id", "label")
    eval_seqs = group_user_seqs(df_eval, "item_id", "label")

    return SeqBundle(
        seqs={"train": train_seqs, "eval": eval_seqs},
        item_map=item_map,
        item_level="token"
    )

def build_ex_sequences(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    item_col: str = "ref_ans",
    drop_unk: bool = True
) -> SeqBundle:
    
    df_train_ex = collapse_to_exercise(df_train)
    df_eval_ex = collapse_to_exercise(df_eval)

    item_map = build_item_map(df_train_ex[item_col])
    
    df_train_ex = apply_item_map(df_train_ex, item_col, item_map, drop_unk=drop_unk)
    df_eval_ex = apply_item_map(df_eval_ex, item_col, item_map, drop_unk=drop_unk)

    train_seqs = group_user_seqs(df_train_ex, "item_id", "correct")
    eval_seqs = group_user_seqs(df_eval_ex, "item_id", "correct")

    return SeqBundle(
        seqs={"train": train_seqs, "eval": eval_seqs},
        item_map=item_map,
        item_level="exercise"
    )


