import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tqdm.auto import tqdm

def exercise_view(df: pd.DataFrame) -> pd.DataFrame:
    df["ex_id"] = df["tok_id"].str.slice(0,10).copy()
    ex = (
        df[["user_id", "ex_id", "days"]]
        .drop_duplicates(["user_id", "ex_id"])
    )
    return ex

def add_bursts(ex: pd.DataFrame) -> pd.DataFrame:
    ex["dt_hours"] = ex.groupby("user_id")["days"].diff() * 24
    ex["new_burst"]  = ex["dt_hours"].isna() | (ex["dt_hours"] >= 1)
    ex["burst_id"] = ex.groupby("user_id")["new_burst"].cumsum() - 1
    return ex

def burst_stats(ex: pd.DataFrame) -> pd.DataFrame:
    burst_sizes = (
        ex.groupby(["user_id", "burst_id"], sort=False)
        .size()
        .rename("burst_size")
        .reset_index()
    )

    return (
        burst_sizes.groupby("user_id", sort=False)["burst_size"]
        .agg(["mean", "median", "count"])
        .rename(
            columns = {
                "mean": "burst_mean",
                "median": "burst_median",
                "count": "burst_count",
            }
        )
        .reset_index()
    )

def tod_entropy(ex: pd.DataFrame) -> pd.DataFrame:
    tod = ex["days"].mod(1.0)
    bins = (tod*72).astype(int).clip(upper=71).rename("bin")

    df_bin = (

        ex.groupby(["user_id", bins])
        .size()
        .rename("cnt")
        .reset_index()

    )

    bin_sums = df_bin.groupby("user_id", sort=False)["cnt"].transform("sum")
    probs = df_bin["cnt"] / bin_sums

    df_bin["-p_log_p"] = - probs * np.log(probs)

    out = (

        df_bin.groupby("user_id", sort=False)["-p_log_p"]
        .sum()
        .rename("tod_entropy")
        .reset_index()

    )

    return out

def add_user_feats(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:

    steps = tqdm(total=6, desc="user feats")

    # == Tag and merge into one stream
    
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train["is_test"] = 0
    df_test["is_test"] = 1
    
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    #==========
    
    ex = exercise_view(df); steps.update(1)
    ex = add_bursts(ex); steps.update(1)

    bstats = burst_stats(ex); steps.update(1)
    ent = tod_entropy(ex); steps.update(1)

    user_feats = bstats.merge(ent, on="user_id"); steps.update(1)
    
    # =========
    
    df = df.merge(user_feats, on="user_id"); steps.update(1)
    df_train_out = df[df["is_test"] == 0].reset_index(drop=True)
    df_test_out = df[df["is_test"] == 1].reset_index(drop=True)
    
    return df_train_out, df_test_out

