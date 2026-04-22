import re
import pandas as pd
from models.gbdt.params import CAT_FEATS, DROP

santitise_pattern = re.compile(r'[\[\]\{\}":,]')

def sanitise_lightgbm_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = [santitise_pattern.sub("_", col_name) for col_name in df.columns]

    assert len(set(new_cols)) == len(new_cols), "collision after sanitise"

    if new_cols == list(df.columns):
        return df

    df = df.copy()
    df.columns = new_cols
    return df


def prepare_xy_lightgbm(df_train, df_test, track: str):
    # align
    df_train, df_test = df_train.align(df_test, join="left", axis=1, fill_value=0)
    df_train = sanitise_lightgbm_feature_names(df_train)
    df_test = sanitise_lightgbm_feature_names(df_test)

    feat_cols = [col for col in df_train.columns if col not in DROP]
    cat_cols = [col for col in CAT_FEATS if col in feat_cols]

    if track == "all":
        cat_cols.append("track")
    else:
        # ensure track is not included as a feature for individual track models
        if "track" in feat_cols:
            feat_cols.remove("track")

    X_train = df_train[feat_cols].copy()
    X_test = df_test[feat_cols].copy()
    y_train = df_train["label"]
    y_test = df_test["label"]

    for col in cat_cols:
        unique_vals = pd.concat([X_train[col], X_test[col]], ignore_index=True).dropna().unique()
        X_train[col] = pd.Categorical(X_train[col], categories=unique_vals)
        X_test[col] = pd.Categorical(X_test[col], categories=unique_vals)

    return X_train, y_train, X_test, y_test, feat_cols, cat_cols

def align_train_test(df_train, df_test):
    df_train, df_test = df_train.align(df_test, join="left", axis=1, fill_value=0)
    return df_train, df_test
