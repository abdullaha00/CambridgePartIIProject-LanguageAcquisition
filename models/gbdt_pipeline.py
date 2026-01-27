
import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
from datasets.data_parquet import get_parquet, parquet_exists, save_parquet
from features.build_features import build_features
from sklearn.metrics import roc_auc_score, log_loss

parser = argparse.ArgumentParser()
parser.add_argument("--subset", type=int, default=None)
parser.add_argument("--track", type=str, default="en_es")
parser.add_argument("--train_with_dev", action="store_true")
parser.add_argument("--save_feats", action="store_true")
args = parser.parse_args()

#==== CONFIG

TRACK = args.track
train_with_dev = args.train_with_dev
SUBSET = args.subset
SAVE_FEATS = args.save_feats

if __name__ == "__main__":
        
    #==== LOAD DATA

    df_train_data = get_parquet(TRACK, "train", "reprocessed", subset=SUBSET)    
    
    df_dev_data = get_parquet(TRACK, "dev", "reprocessed", subset=SUBSET)
    
    if not train_with_dev:
        df_train = df_train_data
        df_test = df_dev_data
    else:
        df_test_data = get_parquet(TRACK, "test", "reprocessed", subset=SUBSET)
        
        df_train = pd.concat([df_train_data, df_dev_data])
        df_test = df_test_data

    #====== FEATURES =====
    
    if parquet_exists(TRACK, "train", "features_dev") and parquet_exists(TRACK, "test", "features_dev"):
        print("Loading precomputed features...")
        df_train = get_parquet(TRACK, "train", "features_dev")
        df_test = get_parquet(TRACK, "test", "features_dev")
    else:
        df_train, df_test = build_features(df_train, df_test, train_with_dev, save_feats=True)

    #===== TRAIN GDBT

    DROP = {
        "label", "is_test", "translation",
        "ex_instance_id", "tok_id",
        "meta", "ex_id",  
    }
    CAT_FEATS = [
    "user_id", "session",
    "tok", "lemma",
    "prev_tok", "next_tok", "rt_tok",
    "pos", "prev_pos", "next_pos", "rt_pos",
    "type", "format", "client", "countries"
    ]

    df_train, df_test = df_train.align(df_test, join="left", axis=1, fill_value=0)
    feat_cols = [col for col in df_train.columns if col not in DROP]
    cat_cols = [col for col in CAT_FEATS if col in feat_cols]

    X_train = df_train[feat_cols].copy()
    X_test = df_test[feat_cols].copy()

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    y_train = df_train["label"]
    y_test = df_test["label"]

    #--- train model

    model = lgb.LGBMClassifier(
        n_estimators=650,
        learning_rate=0.05,
        num_leaves=512,
        min_child_samples=100,     
        subsample=1.0,
        colsample_bytree=0.7, 
        cat_smooth=200,
        max_cat_threshold=32,
        random_state=42,
        n_jobs=-1
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


