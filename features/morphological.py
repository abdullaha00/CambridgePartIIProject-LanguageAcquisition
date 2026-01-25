import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

def add_morph_features(df: pd.DataFrame) -> pd.DataFrame:

    meta = df["meta"].fillna("")

    split_meta = meta.str.split("|")
    meta_expl = split_meta.explode()

    meta_expl = meta_expl[meta_expl.ne("")]

    kv = meta_expl.str.split("=", expand=True)
    kv = kv.dropna()

    keys = kv[0]
    vals = kv[1]

    feat = "morph_feature:" + keys + "_" + vals

    one_hot = pd.crosstab(feat.index, feat)

    one_hot_row = one_hot.reindex(df.index, fill_value=0)

    df = df.join(one_hot_row)
    
    return df