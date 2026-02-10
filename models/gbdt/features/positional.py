import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def add_positional_features(df: pd.DataFrame) -> pd.DataFrame:

    df["prev_tok"] = df.groupby("ex_instance_id")["tok"].shift(1).fillna("<NONE>")
    df["next_tok"] = df.groupby("ex_instance_id")["tok"].shift(-1).fillna("<NONE>")

    # NOT IN NYU PAPER
    # df["prev_root"] = df.groupby("ex_instance_id")["lemma"].shift(1)
    # df["next_root"] = df.groupby("ex_instance_id")["lemma"].shift(-1)

    df["prev_pos"] = df.groupby("ex_instance_id")["pos"].shift(1).fillna("<NONE>")
    df["next_pos"] = df.groupby("ex_instance_id")["pos"].shift(-1).fillna("<NONE>")

    # ========= DEPENDENCY ROOT + POS

    rt_tok = np.full(len(df), "<NONE>")
    rt_pos = np.full(len(df), "<NONE>")

    for _, g in tqdm(df.groupby("ex_instance_id", sort=False), desc="Positional features"):
        global_idxs = g.index
        root_idxs = g["rt"] - 1

        valid = (root_idxs >= 0) & (root_idxs < len(g))

        rt_tok[global_idxs[valid]] = g["tok"].iloc[root_idxs[valid]]
        rt_pos[global_idxs[valid]] = g["pos"].iloc[root_idxs[valid]]
    
    df["rt_tok"] = rt_tok
    df["rt_pos"] = rt_pos

    return df
