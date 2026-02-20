import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def add_positional_features(df: pd.DataFrame) -> pd.DataFrame:

    df["prev_tok"] = df.groupby("ex_instance_id")["tok"].shift(1).fillna("<NONE>")
    df["next_tok"] = df.groupby("ex_instance_id")["tok"].shift(-1).fillna("<NONE>")
    df["prev_pos"] = df.groupby("ex_instance_id")["pos"].shift(1).fillna("<NONE>")
    df["next_pos"] = df.groupby("ex_instance_id")["pos"].shift(-1).fillna("<NONE>")

    # NOT IN NYU PAPER
    # df["prev_root"] = df.groupby("ex_instance_id")["lemma"].shift(1)
    # df["next_root"] = df.groupby("ex_instance_id")["lemma"].shift(-1)

    # ========= DEPENDENCY ROOT + POS
    
    pos_in_ex = df.groupby("ex_instance_id", sort=False).cumcount().add(1)

    # (ex_instance_id, pos_in_ex)
    k = pd.MultiIndex.from_arrays([df["ex_instance_id"], pos_in_ex])
    # (ex_instance_id, rt), where rt is the 1-idxed position of the root token within ex 
    rt_k = pd.MultiIndex.from_arrays([df["ex_instance_id"], df["rt"]])

    tok_map = pd.Series(df["tok"].values, index=k) #(ex_instance_id, pos_in_ex) -> tok
    pos_map = pd.Series(df["pos"].values, index=k) #(ex_instance_id, pos_in_ex) -> pos

    df["rt_tok"] = tok_map.reindex(rt_k).fillna("<NONE>").to_numpy()
    df["rt_pos"] = pos_map.reindex(rt_k).fillna("<NONE>").to_numpy()
    
    return df



