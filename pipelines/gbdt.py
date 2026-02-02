
import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
from datasets.data_parquet import get_parquet, parquet_exists, load_train_and_eval_df
from features.build_features import build_features
from sklearn.metrics import roc_auc_score, f1_score
from experiments.log_db import log_run

#==== PARSING ARGS

def parse_gdbt_args(gdbt_args=None):
    # PARSE GDBT SPECIFIC FLAGS
    p = argparse.ArgumentParser(description="GBDT Pipeline Args")
    p.add_argument("--no_save_feats", action="store_true")
    
    args = p.parse_args(gdbt_args)
    return args

#=======

# TRACK -> LIGHTGBM PARAMS
NYU_LGBM_PARAMS = {
    "fr_en": dict(
        num_leaves=256,
        learning_rate=0.05,
        min_child_samples=100,     # min_data_in_leaf
        n_estimators=750, # num_boost_round
        cat_smooth=200,
        colsample_bytree=0.7, # feature_fraction
        max_cat_threshold=32,
        objective="binary",
        ),
    "en_es": dict(
        num_leaves=512,
        learning_rate=0.05,
        min_child_samples=100,
        n_estimators=650,
        cat_smooth=200,
        colsample_bytree=0.7,
        max_cat_threshold=32,
        objective="binary",
    ),
    "es_en": dict(
        num_leaves=512,
        learning_rate=0.05,
        min_child_samples=100,
        n_estimators=600,
        cat_smooth=200,
        colsample_bytree=0.7,
        max_cat_threshold=32,
        objective="binary",
    ),
    "all": dict(
        num_leaves=1024,
        learning_rate=0.05,
        min_child_samples=100,
        n_estimators=750,
        cat_smooth=200,
        colsample_bytree=0.7,
        max_cat_threshold=64,
        objective="binary",
    ),
}

#========== GDBT PIPELINE

def run_gbdt_pipeline(TRACK="en_es",SUBSET=None,  train_with_dev=False, SAVE_FEATS=True):
    
    #====== FEATURES =====
    
    if SUBSET is None and parquet_exists(TRACK, "train", "features_dev") and parquet_exists(TRACK, "test", "features_dev"):
        print("Loading precomputed features...")
        df_train = get_parquet(TRACK, "train", "features_dev")
        df_test = get_parquet(TRACK, "test", "features_dev")
    else:
        if SUBSET is None and SAVE_FEATS:
            print("Building features and saving...")
        df_train, df_test = load_train_and_eval_df(TRACK, "reprocessed", train_with_dev, subset=SUBSET)
        df_train, df_test = build_features(df_train, df_test, train_with_dev, save_feats=SAVE_FEATS, TRACK=TRACK)

    #===== TRAIN GDBT

    DROP = {"label", "is_test", "translation",
        "ex_instance_id", "tok_id",
        "meta", "ex_id"}
    
    CAT_FEATS = ["user_id", "session","tok", "lemma",
    "prev_tok", "next_tok", "rt_tok","pos", "prev_pos", 
    "next_pos", "rt_pos", "type", "format", "client", "countries"]

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
        **NYU_LGBM_PARAMS[TRACK]
    )

    model.fit(X_train, y_train, categorical_feature=cat_cols)

    #===== EVALUATE

    score = model.score(X_test, y_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print(f"AUC: {auc}")
    print(f"Accuracy: {score}")
    print(f"F1 Score: {f1}")

    #====== RETURN METRICS

    return {
        "auc": auc,
        "accuracy": score,
        "f1": f1,
    }

"""
Scores with default feats.
AUC: 0.8122643673091771
"""
