from datasets.data_parquet import load_train_and_eval_df
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from experiments.log_db import log_run

def run_lr_pipeline(TRACK="en_es", SUBSET=None, train_with_dev=False):

    #=== LOAD DATA
    df_train, df_test = load_train_and_eval_df(TRACK, "reprocessed", train_with_dev, subset=SUBSET)

    X_train, Y_train = df_train.iloc[:, :-1], df_train['label']
    X_test, Y_test = df_test.iloc[:, :-1], df_test['label']

    # === ENCODE DATA

    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    encodedData = enc.fit_transform(X_train[['user_id', 'tok', 'pos', 'type']])
    encoded_test = enc.transform(X_test[['user_id', 'tok', 'pos', 'type']])

    # === 

    model = LogisticRegression()
    model.fit(encodedData, Y_train)

    # === 

    score = model.score(encoded_test, Y_test)
    y_proba = model.predict_proba(encoded_test)

    auc = roc_auc_score(Y_test, y_proba[:, 1])

    print(f"model score: {score}")
    print(f"auc: {auc}")

    #=== RETURN LOG METRICS

    return {
        "auc": auc,
        "accuracy": score,
    }