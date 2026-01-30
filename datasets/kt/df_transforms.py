import pandas as pd
import numpy as np

def collapse_to_exercise(df: pd.DataFrame) -> pd.DataFrame:
    
    # CHECK IF LABELS ARE AVAILABLE

    if df["label"].isna().any():
        raise ValueError("Some labels are missing; cannot collapse to exercise level.")

    return (

        df.groupby(["ex_instance_id", "user_id"], sort=False).agg(

            ref_ans=("tok", " ".join),
            correct=("label", lambda x: int(np.all(x == 1)))
        ).reset_index()

    )
    
    # SLOWER IMPLEMENTATION; can be augmented to include progress bar
    # def block_op(df_block: pd.DataFrame) -> pd.DataFrame:
    #     toks = df_block["tok"].tolist()
    #     ans = " ".join(toks)

    #     correct = int(np.all(df_block["label"] == 1))

    #     return pd.Series({
    #         "ref_ans": ans,
    #         "correct": correct,
    #     })

    # df_e = df.groupby(
    #     ["ex_instance_id", "user_id"]
    # ).apply(block_op).reset_index()

    # return df_e

def generate_qid_map(df: pd.DataFrame) -> dict:
    unique_qs = df["ref_ans"].unique()
    qid_map = {q: i for i, q in enumerate(unique_qs)}
    return qid_map

def apply_qid_map(df: pd.DataFrame, qid_map: dict) -> pd.DataFrame:
    df["question_id"] = df["ref_ans"].map(qid_map)

    # WE DROP UNSEEEN QUESTION
    df = df.dropna(subset=["question_id"])
    return df



