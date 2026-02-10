

from models.gbdt.params import CAT_FEATS, DROP


def prepare_xy_lightgbm(df_train, df_test, track: str):
    # align
    df_train, df_test = df_train.align(df_test, join="left", axis=1, fill_value=0)

    feat_cols = [col for col in df_train.columns if col not in DROP]
    cat_cols = [col for col in CAT_FEATS if col in feat_cols]

    if track == "all":
        cat_cols.append("track")
    else:
        # ensure track is not included as a feature for individual track models
        if "track" in feat_cols:
            feat_cols.remove("track")

    X_train = df_train[feat_cols]
    X_test = df_test[feat_cols]
    y_train = df_train["label"]
    y_test = df_test["label"]

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    
    return X_train, y_train, X_test, y_test, feat_cols, cat_cols

def align_train_test(df_train, df_test):
    df_train, df_test = df_train.align(df_test, join="left", axis=1, fill_value=0)
    return df_train, df_test
