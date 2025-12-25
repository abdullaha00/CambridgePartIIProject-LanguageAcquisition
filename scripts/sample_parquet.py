import pandas 

track = "en_es"
split = "train"


def sample_parquet(full=False):
    df = pd.read_parquet(f"parquet/{track}/minimal/{track}_{split}_minimal.parquet")
    if not full:
        return df[:1000]
    return df

