import pandas as pd
from typing import Dict

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



