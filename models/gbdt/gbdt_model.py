from lightgbm import LGBMClassifier
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from models.gbdt.params import NYU_LGBM_PARAMS
from models.gbdt.utils import prepare_xy_lightgbm
from tqdm.auto import tqdm
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

        n_estimators = NYU_LGBM_PARAMS[self.track]["n_estimators"]

        self.model = LGBMClassifier(
            **NYU_LGBM_PARAMS[self.track],
            verbose=1,
        )

        logger.info(f"Fitting GBDT for track={self.track}"
                    f"({X_train.shape[0]:,} rows, {len(feat_cols)} feats, "
                    f"{len(cat_cols)} cat, {n_estimators} rounds)")

        with tqdm(total=n_estimators, desc=f"GBDT [{self.track}]", unit="round") as pbar:
            
            #CALLBACK for progress bar
            def callback(lgb_env):
                pbar.update(1)
                if lgb_env.evaluation_result_list:
                    # returns [(dataset, metric, value, is_higher_better), ...]
                    _, metric, val, _ = lgb_env.evaluation_result_list[0]
                    pbar.set_postfix_str(f"{self.track}_{metric}={val:.4f}")
                    # if (lgb_env.iteration +1) % 25 == 0 or lgb_env.iteration == 0:
                    #     logger.info(f"[iter {lgb_env.iteration + 1:>4d}] eval {self.track}_{metric}={val:.4f}")
            
            self.model.fit(
                X_train, y_train,
                categorical_feature=self.cat_cols,
                eval_set=[(X_test, y_test)],
                eval_names=["eval"],
                eval_metric="auc",
                callbacks=[
                    callback,
                    lgb.early_stopping(stopping_rounds=100, first_metric_only=True, verbose=False),
                ],
            )

        if getattr(self.model, "best_iteration_", None):
            logger.info("Best iteration for %s: %s", self.track, self.model.best_iteration_)

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
