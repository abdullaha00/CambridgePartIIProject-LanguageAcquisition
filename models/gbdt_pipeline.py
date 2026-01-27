
import pandas as pd
import numpy as np
import lightgbm as lgb
from datasets.data_parquet import get_parquet, save_parquet
from features.lexical import add_lexical_feats
from features.temporal import add_temporal_features
from features.positional import add_positional_features
from features.user import add_user_feats
from features.morphological import add_morph_features
from sklearn.metrics import roc_auc_score, log_loss

#==== CONFIG

TRACK = "en_es"
train_with_dev = False
SUBSET = 1000
SAVE_FEATS = True

def build_features(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    df_train = df_train.copy()
    df_test = df_test.copy()

    # Temporal + user done with train/test together
    df_train, df_test = add_temporal_features(df_train, df_test)
    df_train, df_test = add_user_feats(df_train, df_test)
    
    # Combine for rest of features
    df_train["is_test"] = 0
    df_test["is_test"] = 1
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # Lexical
    df_all = add_lexical_feats(df_all)

    # Morphological one-hot encode
    df_all = add_morph_features(df_all)

    # Positional
    df_all = add_positional_features(df_all)

    df_train = df_all[df_all.is_test == 0].reset_index(drop=True)
    df_test = df_all[df_all.is_test == 1].reset_index(drop=True)

    if SAVE_FEATS:
        save_parquet(df_train, TRACK, "train", "features")
        save_parquet(df_test, TRACK, "dev" if not train_with_dev else "test", "features")

    return df_train, df_test


if __name__ == "__main__":
        
    #==== LOAD DATA

    df_train_data = get_parquet(TRACK, "train", "reprocessed", SUBSET)    
    
    df_dev_data = get_parquet(TRACK, "dev", "reprocessed", SUBSET)
    
    df_test_data = get_parquet(TRACK, "test", "reprocessed", SUBSET)

    if not train_with_dev:
        df_train = df_train_data
        df_test = df_dev_data
    else:
        df_train = pd.concat([df_train_data, df_dev_data])
        df_test = df_test_data

    #====== FEATURES =====

    df_train, df_test = build_features(df_train, df_test)

    #===== TRAIN GDBT

    DROP = {
        "label", "is_test",
        "ex_instance_id", "tok_id",
        "meta", "ex_id",  
    }
    CAT_FEATS = [
    "user_id", "session",
    "tok", "lemma",
    "prev_tok", "next_tok", "rt_tok",
    "pos", "prev_pos", "next_pos", "rt_pos",
    "type", "format", "client", "countries", "translation"
    ]

    df_train, df_test = df_train.align(df_test, join="left", axis=1, fill_value=0)
    feat_cols = [col for col in df_train.columns if col not in DROP]
    cat_cols = [col for col in CAT_FEATS if col in feat_cols]

    X_train = df_train[feat_cols].copy()
    X_test = df_test[feat_cols].copy()

    # for col in cat_cols:
    #     X_train[col] = X_train[col].astype("category")
    #     X_test[col] = X_test[col].astype("category")

    for c in cat_cols:
        cats = pd.Index(X_train[c].fillna("<NA>").astype(str).unique())
        X_train[c] = pd.Categorical(X_train[c].fillna("<NA>").astype(str), categories=cats)
        X_test[c]  = pd.Categorical(X_test[c].fillna("<NA>").astype(str),  categories=cats)

    y_train = df_train["label"]
    y_test = df_test["label"]

    #--- train model

    model = lgb.LGBMClassifier( 
        n_estimators=650,
        learning_rate=0.05,
        num_leaves=512,
        random_state=42
    )

    model.fit(X_train, y_train, categorical_feature=cat_cols)

    #===== EVALUATE

    score = model.score(X_test, y_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc}")
    print(f"Score: {score}")


"""
Scores with default feats.
AUC: 0.8122643673091771
Score: 0.8719738547243749"""


