import pandas as pd
import numpy as np
from collections import Counter, defaultdict

rng = np.random.default_rng(seed=42)

ALPHA_ERR = [0.3, 0.1, 0.03, 0.01]
TEST_PATH = "/home/abdullah/Documents/Projects/CambridgePartIIProject-LanguageAcquisition/parquet/en_es/minimal/en_es_train_minimal.parquet"


def n_test_per_user():
    df = pd.read_parquet(TEST_PATH)
    return df.groupby("user_id")["sentence_id"].nunique().to_dict()


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:

    N = len(df)
    K = len(ALPHA_ERR)

    user_test_n = n_test_per_user()
    
    ex_seen = np.zeros(N, dtype=np.int32)
    tok_seen = np.zeros(N, dtype=np.int32)
    root_seen = np.zeros(N, dtype=np.int32)

    err_tok = np.zeros((N,K))
    err_root = np.zeros((N,K))

    for uid, g in df.groupby("user_id", sort=False):
        #g = g.sort_values("days")

        ex_idx = 0  

        ex_count = Counter()
        tok_count = Counter()
        root_count = Counter()

        err_tok_map = defaultdict(lambda: np.zeros(K))
        err_root_map = defaultdict(lambda: np.zeros(K))
        
        n = user_test_n[uid]
        r_ignore = rng.integers(0, n)
        
        for eidx, ex_df in g.groupby("sentencetime_id", sort=False):
            
            row_idxs = ex_df.index
            toks = tuple(ex_df["tok"])

            # --- FEATURE READ ---

            ex_seen[row_idxs] = ex_count[toks]

            for r, tok, root in zip(row_idxs, ex_df["tok"], ex_df["lemma"]):

                tok_seen[r] = tok_count[tok]
                root_seen[r] = root_count[root]

                err_tok[r, :] = err_tok_map[tok]
                err_root[r, :] = err_root_map[root]

            utoks = ex_df["tok"].unique()
            uroots = ex_df["lemma"].unique()

            # --- TRAIN-TIME MASKING

            
            # --- INTERNAL COUNTERS

            ex_count[toks] += 1

            for utok in utoks:
                tok_count[utok] += 1
                root_count[uroot] += 1

                label_mean = ex_df[ex_df["tok"] == utok]["label"].mean()
                error_tok[utok, :] += ALPHA_ERR * (label_mean - error_tok[utok, i]) 

            
            
    return df