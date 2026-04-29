import pandas as pd
import numpy as np
from pathlib import Path
from .data import FAVocabs, FASequence, add_positional_num_features

POS_NUM_COLS = [
    "pos_ex_len_log",
    "pos_index_log",
    "pos_from_end_log",
    "pos_rel",
    "pos_is_first",
    "pos_is_last",
    "tok_len_log",
    "tok_is_title",
    "tok_is_upper",
    "tok_has_digit",
    "tok_has_apostrophe",
    "tok_has_punct"
]

META_SOURCE_COLS = ["days", "time", "rt"]

def add_positional_num_features(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:   

    for df in (df_train, df_eval):
        
        # === helper cols
        g = df.groupby("ex_key")
        pos_in_ex = g.cumcount() + 1
        ex_len = g.transform("size")
        assert ex_len.min() > 0, "Invalid ex"
        assert pos_in_ex.min() >= 0, "Invalid position"
        # ===

        # == EXERCISE POS FEATS

        # np default is float64, switch to 32 to save memory
        df["pos_ex_len_log"] = np.log(ex_len).astype(np.float32)
        df["pos_index_log"] = np.log(pos_in_ex).astype(np.float32)
        df["pos_from_end_log"] = np.log((ex_len - pos_in_ex - 1).clip(lower=0)).astype(np.float32)
        df["pos_rel"] = (pos_in_ex / (ex_len - 1)).astype(np.float32)
        df["pos_is_first"] = (pos_in_ex == 1).astype(np.float32)
        df["pos_is_last"] = (pos_in_ex == ex_len).astype(np.float32)

        # == Tok features

        tok = df["tok"]

        df["tok_len_log"] = np.log(tok.str.len().astype(np.float32)).astype(np.float32)
        df["tok_is_title"] = tok.str.istitle().astype(np.float32)
        df["tok_is_upper"] = tok.str.isupper().astype(np.float32)
        df["tok_has_digit"] = tok.str.contains(r"\d", regex=True).astype(np.float32)
        df["tok_has_apostrophe"] = tok.str.contains("'", regex=False).astype(np.float32)
        df["tok_has_punct"] = tok.str.contains(r"[^\w\s']", regex=True).astype(np.float32)

    return df_train, df_eval, tuple(POS_NUM_COLS)

def key(tok_series, suffix_series):
    return tok_series + "|" + suffix_series

def add_user_history_num_features(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    global_label_mean = df_train["label"].mean()

    train_users = df_train["user_id"]
    eval_users = df_eval["user_id"]
    
    # groups = [(out_name, suffix)]
    # where we group by "user_id | suffix"
    # and output columns including the out_name and suffix, e.g. hist_user_tok_log_count

    groups = [ 
        ("user", df_train["user_id"], df_eval["user_id"]), # out_name, train_key, eval_key
        ("user_token", key(df_train["tok"], df_train["user_id"]), key(df_eval["tok"], df_eval["user_id"])),
        ("user_format", key(df_train["tok"], df_train["format"]), key(df_eval["tok"], df_eval["format"])),
        ("user_country", key(df_train["tok"], df_train["countries"]), key(df_eval["tok"], df_eval["countries"]))
    ]

    num_cols = []
    for name, train_key, eval_key in groups:
        #===== TRAIN ROWS: we can use labels here
        train_counts = df_train.groupby(train_key, sort=False).cumcount()
        train_errors = df_train.groupby(train_key, sort=False)["label"].cumsum() - df_train["label"] # previous errs

        #===== EVAL ROWS: no labels; just copy aggregate train data
        train_stats = df_train.groupby(train_key, sort=False)["label"].agg(["count", "sum"])

        eval_counts = eval_key.map(train_stats["count"]).astype(np.float32)
        eval_errors = eval_key.map(train_stats["sum"]).astype(np.float32)

        #=====
        count_col = f"hist_{name}_log_count"
        error_rate_col = f"hist_{name}_error_rate"
        num_cols.extend([count_col, error_rate_col])

        df_train[count_col] = np.log1p(train_counts).astype(np.float32)
        df_train[error_rate_col] = ((train_errors + ALPHA*global_label_mean) / (train_counts + ALPHA)).astype(np.float32)

        df_eval[count_col] = np.log1p(eval_counts).astype(np.float32)
        df_eval[error_rate_col] = ((eval_errors + ALPHA*global_label_mean) / (eval_counts + ALPHA)).astype(np.float32)

    return df_train, df_eval, tuple(num_cols)

def add_global_history_num_features(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_label_mean = df_train["label"].mean()

    # We group by token | x 
    # and calculate counts/errors

    train_pos_in_ex = df_train.groupby("ex_key").cumcount().clip(upper=12)
    eval_pos_in_ex = df_eval.groupby("ex_key").cumcount().clip(upper=12)

    groups = [
        ("token", df_train["tok"], df_eval["tok"]),
        ("format", df_train["format"], df_eval["format"]),
        ("country", df_train["countries"], df_eval["countries"]),
        ("token_format", key(df_train["tok"], df_train["format"]), key(df_eval["tok"], df_eval["format"])),
        ("format_pos", key(df_train["format"], train_pos_in_ex), key(df_eval["format"], eval_pos_in_ex)),
        ("token_pos", key(df_train["tok"], train_pos_in_ex), key(df_eval["tok"], eval_pos_in_ex)),
    ]

    num_cols = []
    for name, train_key, eval_key in groups:
        #===== TRAIN ROWS: we can use labels here
        train_counts = df_train.groupby(train_key, sort=False).cumcount()
        train_errors = df_train.groupby(train_key, sort=False)["label"].cumsum() - df_train["label"] # previous errs

        #===== EVAL ROWS: no labels; just copy aggregate train data
        train_stats = df_train.groupby(train_key, sort=False)["label"].agg(["count", "sum"])

        eval_counts = eval_key.map(train_stats["count"]).astype(np.float32)
        eval_errors = eval_key.map(train_stats["sum"]).astype(np.float32)

        #=====
        count_col = f"hist_{name}_log_count"
        error_rate_col = f"hist_{name}_error_rate"
        num_cols.extend([count_col, error_rate_col])

        df_train[count_col] = np.log1p(train_counts).astype(np.float32)
        df_train[error_rate_col] = ((train_errors + ALPHA*global_label_mean) / (train_counts + ALPHA)).astype(np.float32)

        df_eval[count_col] = np.log1p(eval_counts).astype(np.float32)
        df_eval[error_rate_col] = ((eval_errors + ALPHA*global_label_mean) / (eval_counts + ALPHA)).astype(np.float32)

    return df_train, df_eval, tuple(num_cols)

def add_metadata_num_features(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    num_cols = []
    
    for source_col in META_SOURCE_COLS:
        num_log_col = f"meta_{source_col}_log"
        num_missing_col = f"meta_{source_col}_missing"
        num_cols.extend([num_log_col, num_missing_col])

        for df in (df_train, df_eval):
            vals = df[source_col].astype(np.float32)
            missing = vals.isna()

            df[num_log_col] = np.log1p(vals)
            df[num_missing_col] = missing.astype(np.float32)

    for df in (df_train, df_eval):

        days = df["days"].fillna(0).astype(np.float32)
        day_phase = (days % 1.0).to_numpy()

        df["meta_day_phase_sin"] = np.sin(2 * np.pi * day_phase).astype(np.float32)
        df["meta_day_phase_cos"] = np.cos(2 * np.pi * day_phase).astype(np.float32)

    num_cols.extend(["meta_day_phase_sin", "meta_day_phase_cos"])

    return df_train, df_eval, tuple()

def normalise_numeric_features(df_train: pd.DataFrame, df_eval: pd.DataFrame, num_cols: list[str]) -> list[str]:
    # Normalise based on train only 

    for col in num_cols:
        mean = df_train[col].mean()
        std = df_train[col].std()

        if pd.isna(std) or std == 0:
            logger.warning(f"Standard deviation for column {col} is zero or na; setting to 1.0")
            std = 1.0

        df_train[col] = ((df_train[col] - mean) / std).astype(np.float32)
        df_eval[col] = ((df_eval[col] - mean) / std).astype(np.float32)

    return df_train, df_eval