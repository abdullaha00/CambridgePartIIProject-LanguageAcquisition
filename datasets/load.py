import pandas as pd
from pathlib import Path

BASE = Path("/home/abdullah/Documents/Projects/CambridgePartIIProject-LanguageAcquisition/")

def get_parquet(track: str, split: str, typ: str) -> pd.DataFrame:
    path = BASE / "parquet" / track / typ / f"{track}_{split}_{typ}.parquet"
    return pd.read_parquet(path)

