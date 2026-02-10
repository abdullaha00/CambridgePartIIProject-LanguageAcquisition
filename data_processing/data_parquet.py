import pandas as pd
import numpy as np
from pathlib import Path
from .path import BASE
from pandas.api.types import is_integer_dtype, is_float_dtype

# Columns that should always be stored as category dtype
STORED_CAT_COLS = {
    "user_id", "tok", "lemma", "pos", "meta", "type",
    "countries", "client", "session", "format",
    "prev_tok", "next_tok", "rt_tok", "prev_pos", "next_pos", "rt_pos",
    "track", "translation",
}

def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
   
    for col in df.columns:
        dtype = df[col].dtype

        # Cast cats
        if col in STORED_CAT_COLS:
                df[col] = df[col].astype("category")
        
        # Downcast integers
        elif is_integer_dtype(dtype):
            col_min, col_max = df[col].min(), df[col].max()

            for t in [np.int8, np.int16, np.int32]:
                # check if values fall within min, max range of type
                if col_min >= np.iinfo(t).min and col_max <= np.iinfo(t).max:
                    df[col] = df[col].astype(t)
                    break

        # float downcasting
        elif is_float_dtype(dtype) and dtype != np.float32:
            df[col] = df[col].astype(np.float32)

    return df

def parquet_exists(track: str = "en_es", split: str = "train", variant: str = "reprocessed") -> bool:
    path = BASE / "parquet" / track / variant / f"{track}_{split}_{variant}.parquet"
    return path.exists()

def get_parquet(track: str = "en_es", split: str = "train", variant: str = "reprocessed", subset=None, tag_split=False) -> pd.DataFrame:
    path = BASE / "parquet" / track / variant / f"{track}_{split}_{variant}.parquet"
    
    if subset is not None:
        # Read only user_id column first to determine which users to keep
        users = pd.read_parquet(path, columns=["user_id"])["user_id"].drop_duplicates().iloc[:subset]
        df = pd.read_parquet(path, filters=[("user_id", "in", users.tolist())])
    else:
        df = pd.read_parquet(path)

    if tag_split:
        df["split"] = split

    df = downcast_df(df)
    return df

def save_parquet(df, track: str, split: str, variant: str):
    
    # ensure folder exists
    parent_path = BASE / "parquet" / track / variant
    parent_path.mkdir(parents=True, exist_ok=True)
    
    # path to save parquet
    path = parent_path / f"{track}_{split}_{variant}.parquet"
    
    df.to_parquet(path, index=False)

def load_train_and_eval_df(track: str, variant: str, train_with_dev: bool, subset=None):
    df_train_data = get_parquet(track, "train", variant, subset=subset)    
    df_dev_data = get_parquet(track, "dev", variant, subset=subset)
    
    if not train_with_dev:
        df_train = df_train_data
        df_eval = df_dev_data
    else:
        df_test_data = get_parquet(track, "test", variant, subset=subset)
        df_train = pd.concat([df_train_data, df_dev_data])
        df_eval = df_test_data

    return df_train, df_eval
