import pandas as pd
from pathlib import Path

BASE = Path("/home/abdullah/Documents/Projects/CambridgePartIIProject-LanguageAcquisition/")

def get_parquet(track: str = "en_es", split: str = "train", variant: str = "reprocessed") -> pd.DataFrame:
    path = BASE / "parquet" / track / variant / f"{track}_{split}_{variant}.parquet"
    
    df = pd.read_parquet(path)
    
    # Label df with split
    df = df.assign(split=split)
    
    return df

def save_parquet(df, track: str, split: str, variant: str):
    
    # ensure folder exists
    parent_path = BASE / "parquet" / track / variant
    parent_path.mkdir(parents=True, exist_ok=True)
    
    # path to save parquet
    path = parent_path / f"{track}_{split}_{variant}.parquet"
    
    df.to_parquet(path, index=False)