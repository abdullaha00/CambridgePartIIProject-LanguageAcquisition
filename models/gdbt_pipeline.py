
import pandas as pd
import numpy as np
import lightgbm as lgb
from datasets.data_parquet import get_parquet, save_parquet
from features.lexical import add_lexical_feats
from features.temporal import add_temporal_features
from features.user import add_user_feats
from features.morphological import add_morph_features
from sklearn.metrics import roc_auc_score, log_loss

#==== CONFIG

TRACK = "en_es"
train_with_dev = False
SUBSET = 1000

#==== LOAD DATA

df_train_data = get_parquet(TRACK, "train", "reprocessed")[:SUBSET]
df_dev_data = get_parquet(TRACK, "dev", "reprocessed")[:SUBSET]
df_test_data = get_parquet(TRACK, "test", "reprocessed")[:SUBSET]

if not train_with_dev:
    df_train = df_train_data
    df_test = df_dev_data
else:
    df_train = pd.concat([df_train_data, df_dev_data])
    df_test = df_test_data

#====== FEATURES =====

df_train = add_lexical_feats(df_train)
df_test = add_lexical_feats(df_test)

df_train, df_test = add_temporal_features(df_train, df_test)

df_train, df_test = add_user_feats(df_train, df_test)

df_train = add_morph_features(df_train)
df_test = add_morph_features(df_test)

#===== TRAIN GDBT

CAT_FEATS = ["user_id", "tok", "lemma", "pos", "type", "format", "client"]
NUM_FEATS = ["days", "time", "rt"]
ALL_FEATS = CAT_FEATS + NUM_FEATS

X_train = df_train[ALL_FEATS]
y_train = df_train["label"]

X_test = df_test[ALL_FEATS]
y_test = df_test["label"]

#--- convert categoricals for LightGBM
for col in CAT_FEATS:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

#--- train model

model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train, categorical_feature=CAT_FEATS)

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


