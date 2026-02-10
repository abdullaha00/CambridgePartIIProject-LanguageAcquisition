from dataclasses import dataclass
import gc
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from models.gbdt.gbdt_model import GBDTModel
from models.gbdt.params import NYU_LGBM_PARAMS
from models.gbdt.utils import prepare_xy_lightgbm
from config.consts import ALL_TRACK, TRACKS
import logging

logger = logging.getLogger(__name__)

def combine_probs(p_tr, p_all, alpha=0.5):

    logit = lambda p: np.log(p / (1 - p))
    inv_logit = lambda l: 1 / (1 + np.exp(-l))

    return inv_logit(alpha * logit(p_tr) + (1 - alpha) * logit(p_all))

@dataclass(frozen=True)
class EnsembleOutputs:
    per_track_metrics: dict[str, dict[str, float]]
    combined_metrics: dict[str, dict[str, float]]

class GBDTEnsemble:
    def __init__(self):
        self.feat_cols = None
        self.cat_cols = None

        self.model_all: GBDTModel | None = None
        self.models_tr: dict[str, GBDTModel] = {}

    def fit_individual_tracks(self, per_track_train_tests: dict[str, tuple[pd.DataFrame, pd.DataFrame]]):
        
        track_test = {}
        for tr, (df_tr_train, df_tr_test) in per_track_train_tests.items():
            model_tr = GBDTModel(track=tr)
            X_test, y_test = model_tr.fit(df_tr_train, df_tr_test)
            self.models_tr[tr] = model_tr
            track_test[tr] = (X_test, y_test)

        return track_test

    def fit_all_model(self, df_all_train: pd.DataFrame, df_all_test: pd.DataFrame):

        # ensure track column is present (marked)

        # Fit all-track model
       
        self.model_all = GBDTModel(track="all")
        return self.model_all.fit(df_all_train, df_all_test)
    
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        
        assert self.model_all is not None, "Model has not been trained yet. Call fit() before predict_proba()."
        assert self.feat_cols is not None, "Feature columns not set. Call fit() before predict_proba()."
        assert self.cat_cols is not None, "Categorical columns not set. Call fit() before predict_proba()."

        # ensure all features are present in the input, and in the same order as during training
        X = X.reindex(columns=self.feat_cols, fill_value=0)

        for col in self.cat_cols:
            if not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")

        # retain original index
        return pd.Series(self.model_all.predict_proba(X)[:, 1], index=X.index)

    def evaluate(self, per_track_tests: dict[str, tuple[pd.DataFrame, pd.Series]]) -> dict:
        assert self.model_all is not None, "Model has not been trained yet. Call fit() before evaluate()."

        per_track_metrics = {}
        combined_metrics = {}

        for track, (X_test, y_test) in per_track_tests.items():

            if track == ALL_TRACK:
                m_tr = self.model_all
            else:
                m_tr = self.models_tr[track] 

            p_tr = m_tr.predict_proba(X_test)

            p_all = self.model_all.predict_proba(X_test)
            
            per_track_metrics[track] = m_tr.evaluate(X_test, y_test)

            if track != ALL_TRACK:
                p_comb = combine_probs(p_tr, p_all)

                # Compute ALL metrics
                y_pred = (p_comb >= 0.5)
                
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, p_comb)
                f1 = f1_score(y_test, y_pred)
                
                combined_metrics[track] = {
                    "accuracy": acc,
                    "auc": auc,
                    "f1": f1
                }
            
        return EnsembleOutputs(per_track_metrics=per_track_metrics, combined_metrics=combined_metrics)
