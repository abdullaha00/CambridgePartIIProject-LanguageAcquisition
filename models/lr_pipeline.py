import pandas as pd
import numpy as np
from datasets.load import get_parquet
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


#=== LOAD DATA
TRACK = "es_en"

df_train = get_parquet(TRACK, "train", "original")
df_dev = get_parquet(TRACK, "dev", "original")

X_train, Y_train = df_train.iloc[:, :-1], df_train['label']
X_dev, Y_dev = df_dev.iloc[:, :-1], df_dev['label']

# === ENCODE DATA

enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
encodedData = enc.fit_transform(X_train[['user_id', 'tok', 'pos', 'type']])
encodeddev = enc.transform(X_dev[['user_id', 'tok', 'pos', 'type']])

# === 

model = LogisticRegression()
model.fit(encodedData, Y_train)

# === 

score = model.score(encodeddev, Y_dev)
y_proba = model.predict_proba(encodeddev)

auc = roc_auc_score(Y_dev, y_proba[:, 1])

print(f"model score: {score}")
print(f"auc: {auc}")