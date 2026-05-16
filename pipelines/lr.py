from data_processing.data_parquet import load_train_and_eval_df_strict
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import logging
from db.log_db import MetricRecord
from pipelines.common.evaluation import save_binary_eval_predictions

logger = logging.getLogger(__name__)

CAT_FEATURES = ["user_id", "format", "tok", "pos", "deprel"]
LR_EVAL_EXTRA_COLS = ["user_id", "tok_id"]

def run_lr_pipeline(TRACK="en_es", SUBSET=None, train_with_dev=False, tag=None):

    #=== LOAD DATA
    df_train, df_test, test_labels = load_train_and_eval_df_strict(TRACK, "reprocessed", train_with_dev, subset=SUBSET)

    X_train, Y_train = df_train.iloc[:, :-1], df_train['label']
    X_test, Y_test = df_test.iloc[:, :], test_labels

    # === ENCODE DATA

    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")

    morph = CountVectorizer(tokenizer=lambda x: [kv.split("=")[0] for kv in x.split("|")], 
                            token_pattern=None, 
                            lowercase=False, 
                            binary=True)
    
    encodedData = hstack(
        [enc.fit_transform(X_train[CAT_FEATURES]), morph.fit_transform(X_train["meta"].fillna(""))],
        format="csr",
    )
    encoded_test = hstack(
        [enc.transform(X_test[CAT_FEATURES]), morph.transform(X_test["meta"].fillna(""))],
        format="csr",
    )

    # === 

    model = LogisticRegression()
    model.fit(encodedData, Y_train)

    # === 

    score = model.score(encoded_test, Y_test)
    y_proba = model.predict_proba(encoded_test)

    auc = roc_auc_score(Y_test, y_proba[:, 1])

    logger.info(f"model score: {score}")
    logger.info(f"auc: {auc}")

    #=== RETURN LOG METRICS
    rec = MetricRecord(
        model="lr",
        track=TRACK,
        subset=SUBSET,
        train_with_dev=train_with_dev,
        auc=auc,
        acc=score,
        f1=float("nan"),
        tag=tag,
    )
    pred_path = save_binary_eval_predictions(
        rec,
        y_true=Y_test,
        probs=y_proba[:, 1],
        extra_cols={col: df_test[col].to_numpy() for col in LR_EVAL_EXTRA_COLS},
    )
    logger.info(f"Saved LR evaluation predictions to {pred_path}")
    return [rec]
