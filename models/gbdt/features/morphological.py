import pandas as pd
from tqdm.auto import tqdm

def add_morph_features(df: pd.DataFrame) -> pd.DataFrame:

    steps = tqdm(total=5, desc="morph feats")

    meta = df["meta"].fillna("")

    split_meta = meta.str.split("|")
    meta_expl = split_meta.explode(); steps.update(1)

    meta_expl = meta_expl[meta_expl.ne("")]

    kv = meta_expl.str.split("=", expand=True); steps.update(1)
    kv = kv.dropna()

    keys = kv[0]
    vals = kv[1]

    feat = "morph__" + keys + "_" + vals

    one_hot = pd.crosstab(feat.index, feat); steps.update(1)

    one_hot_row = one_hot.reindex(df.index, fill_value=0); steps.update(1)

    df = df.join(one_hot_row); steps.update(1)
    
    return df