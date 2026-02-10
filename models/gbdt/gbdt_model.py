from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from models.gbdt.params import NYU_LGBM_PARAMS
from models.gbdt.utils import prepare_xy_lightgbm
import logging

logger = logging.getLogger(__name__)

class GBDTModel:
    def __init__(self, track):
        self.track = track
        self.model = None
        self.feat_cols = None
        self.cat_cols = None    

        self.X_test = None
        self.y_test = None        


    def fit(self, df_train, df_test):
        X_train, y_train, X_test, y_test, feat_cols, cat_cols = prepare_xy_lightgbm(df_train, df_test, self.track)
        
        self.feat_cols = feat_cols
        self.cat_cols = cat_cols

        self.model = LGBMClassifier(
            **NYU_LGBM_PARAMS[self.track],
        )

        self.model.fit(X_train, y_train, categorical_feature=self.cat_cols)

        return X_test, y_test
        
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        
        assert self.model is not None, "Model has not been trained yet. Call fit() before predict_proba()."
        assert self.feat_cols is not None, "Feature columns not set. Call fit() before predict_proba()."
        assert self.cat_cols is not None, "Categorical columns not set. Call fit() before predict_proba()."

        # ensure all features are present in the input, and in the same order as during training
        X = X.reindex(columns=self.feat_cols, fill_value=0)

        for col in self.cat_cols:
            if not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")

        # retain original index
        return pd.Series(self.model.predict_proba(X)[:, 1], index=X.index)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
        p = self.predict_proba(X_test)
        y_pred = (p >= 0.5).astype(int)

        acc = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, p)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"AUC: {auc}")
        logger.info(f"Accuracy: {acc}")
        logger.info(f"F1 Score: {f1}")
        
        return {
            "auc": auc,
            "accuracy": acc,
            "f1": f1,
        }