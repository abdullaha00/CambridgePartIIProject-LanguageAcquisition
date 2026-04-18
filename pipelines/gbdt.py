
import logging
import os
import threading
import time
import lightgbm as lgb
import argparse

import numpy as np
import psutil
from config.consts import ALL_TRACK, TRACKS
from db.log_db import MetricRecord
from models.gbdt.ensemble import GBDTEnsemble
from models.gbdt.features.lesions import LESIONS, apply_lesion
from models.gbdt.gbdt_model import GBDTModel 
from data_processing.data_parquet import get_parquet, parquet_exists, load_train_and_eval_df
from models.gbdt.features.feature_load_store import get_feature_dfs
from models.gbdt.features.build_features import build_features
from sklearn.metrics import roc_auc_score, f1_score
from models.gbdt.ensemble import combine_probs
import pandas as pd
import gc
from models.gbdt.params import CAT_FEATS
from models.gbdt.utils import prepare_xy_lightgbm
from pipelines.common.checkpointing import save_lgbm
from pipelines.common.common import mk_record
from pipelines.common.evaluation import save_binary_eval_predictions

logger = logging.getLogger(__name__)

GBDT_EVAL_EXTRA_COLS = [
        "user_id",
        "tok_id",
    ]


#===
def monitor_memory(interval=5):
    process = psutil.Process(os.getpid())
    while True:
        mem = process.memory_info().rss / (1024**3)
        with open("memory.log", "a") as f:
            f.write(f"{time.time()},{mem:.2f} GB\n")
        time.sleep(interval)

threading.Thread(target=monitor_memory, daemon=True).start()

#==== PARSING ARGS

def parse_gdbt_args(gdbt_args=None):
    # PARSE GDBT SPECIFIC FLAGS
    p = argparse.ArgumentParser(description="GBDT Pipeline Args")
    p.add_argument("--disable-save", action="store_true", 
                   help="Disable saving computed features for disk when full dataset is used" \
                   "(smaller subsets are never saved)")
    p.add_argument("--clean-build", action="store_true", 
                   help="Force clean feature build by ignoring existing parquet files", default=True)
    p.add_argument("--lesion", type=str, default=None, choices = list(LESIONS.keys()), help="Lesion to apply to featureset")


    args = p.parse_args(gdbt_args)
    return args

#========== GDBT PIPELINE

def cast_cats(df_train, df_test, cat_cols=CAT_FEATS):
    for col in cat_cols:
        df_train[col] = df_train[col].astype("category")
        cats = df_train[col].cat.categories # (indexed [item0, item1, ...])
        df_test[col] = pd.Categorical(df_test[col], categories=cats)
    return df_train, df_test

def run_gbdt_pipeline(track="en_es",SUBSET=None,  train_with_dev=False, next_args=None, tag=None):

    logger.info(f"Running GBDT pipeline for track {track} with train_with_dev={train_with_dev} and SUBSET={SUBSET}")

    gbdt_args = parse_gdbt_args(next_args)
    SAVE_FEATS = not gbdt_args.disable_save
    CLEAN_BUILD = gbdt_args.clean_build
    LESION = gbdt_args.lesion

    if track != ALL_TRACK:
        #====== FEATURES =====
        df_train, df_test = get_feature_dfs(track, subset=SUBSET, train_with_dev=train_with_dev, save_feats=SAVE_FEATS, clean_build=CLEAN_BUILD)
        #df_train, df_test = cast_cats(df_train, df_test)
        #====== LESION =====
        if gbdt_args.lesion is not None:
            df_train = apply_lesion(df_train, gbdt_args.lesion)
            df_test = apply_lesion(df_test, gbdt_args.lesion)
        #===== TRAIN GDBT =====
        model = GBDTModel(track=track)
        X_test, y_test = model.fit(df_train, df_test)
        #===== EVALUATE =====
        metrics = model.evaluate(X_test, y_test, return_detailed=True)
        records = [mk_record(
            model_name="gbdt",
            track=track,
            subset=SUBSET,
            train_with_dev=train_with_dev,
            metrics=metrics,
            variant=LESION,
            tag=tag,
            )]

        pred_path = save_binary_eval_predictions(
            records[0],
            y_true=metrics["targets"],
            probs=metrics["preds"],
            extra_cols={col: df_test[col].to_numpy() for col in GBDT_EVAL_EXTRA_COLS},
        )
        logger.info(f"Saved evaluation predictions to {pred_path}")
    
        #== SAVING
        if SUBSET is None:
            save_lgbm(model.model, records[0])
            
        return records
    
    else:

        logger.info("Running GBDT ensemble pipeline for all tracks")

        tr_dfs = {}
        for tr in TRACKS:
            logger.info(f"Building features for {tr}...")
            df_train, df_test = get_feature_dfs(tr, subset=SUBSET, train_with_dev=train_with_dev, save_feats=SAVE_FEATS, clean_build=CLEAN_BUILD)
            
            # LESION
            if gbdt_args.lesion is not None:    
                df_train = apply_lesion(df_train, gbdt_args.lesion)
                df_test = apply_lesion(df_test, gbdt_args.lesion)
            
            df_train["track"] = tr
            df_test["track"] = tr
            
            tr_dfs[tr] = (df_train, df_test)
             
        ens_model = GBDTEnsemble()
        tests = ens_model.fit_individual_tracks(tr_dfs)
        
        df_all_train = pd.concat([tr_dfs[tr][0] for tr in TRACKS], axis=0, ignore_index=True)
        df_all_test = pd.concat([tr_dfs[tr][1] for tr in TRACKS], axis=0, ignore_index=True)
        
        del tr_dfs
        gc.collect()
        
        tests[ALL_TRACK] = ens_model.fit_all_model(df_all_train, df_all_test)
        
        ens_out = ens_model.evaluate(tests)

        per_track_metrics = ens_out.per_track_metrics
        combined_metrics = ens_out.combined_metrics
        
        out_records = []
        evals_to_write = {}

        for tr in TRACKS:
            
            # === PREDICT in pipeline for eval saving
            X_test, y_test = tests[tr]
            p_tr = ens_model.models_tr[tr].predict_proba(X_test)
            p_all = ens_model.model_all.predict_proba(X_test)
            p_comb = combine_probs(p_tr, p_all)
            extra_cols = {col: tr_dfs[tr][1][col].to_numpy() for col in GBDT_EVAL_EXTRA_COLS}

            tr_mets = per_track_metrics[tr]
            comb_mets = combined_metrics[tr]

            # == track_record
            tr_record = mk_record(
                model_name="gbdt",
                track=tr,
                subset=SUBSET,
                train_with_dev=train_with_dev,
                metrics=tr_mets,
                variant=LESION,
                tag=tag,
            )
            out_records.append(tr_record)
            evals_to_write[("gbdt", tr)] = {
                "y_true": y_test.to_numpy(),
                "probs": p_tr.to_numpy(),
                "extra_cols": extra_cols,
            }

            comb_record = mk_record(
                model_name="gbdt_ens",
                track=tr,
                subset=SUBSET,
                train_with_dev=train_with_dev,
                metrics=comb_mets,
                variant=LESION,
                tag=tag,
            )
            out_records.append(comb_record)
            evals_to_write[("gbdt_ens", tr)] = {
                "y_true": y_test.to_numpy(),
                "probs": np.asarray(p_comb),
                "extra_cols": extra_cols,
            }

        if per_track_metrics.get(ALL_TRACK) is not None:
            X_all_test, y_all_test = tests[ALL_TRACK]
            p_all = ens_model.model_all.predict_proba(X_all_test)
            all_mets = per_track_metrics[ALL_TRACK]
            all_record = mk_record(
                model_name="gbdt_all",
                track=ALL_TRACK,
                subset=SUBSET,
                train_with_dev=train_with_dev,
                variant=LESION,
                metrics=all_mets,
                tag=tag,
            )
            out_records.append(all_record)
            evals_to_write[("gbdt_all", ALL_TRACK)] = {
                "y_true": y_all_test.to_numpy(),
                "probs": p_all.to_numpy(),
                "extra_cols": {col: df_all_test[col].to_numpy() for col in GBDT_EVAL_EXTRA_COLS},
            }

        for rec in out_records:
            pred_path = save_binary_eval_predictions(
                rec,
                y_true=evals_to_write[(rec.model, rec.track)]["y_true"],
                probs=evals_to_write[(rec.model, rec.track)]["probs"],
                extra_cols=evals_to_write[(rec.model, rec.track)]["extra_cols"],
            )
            logger.info(f"Saved evaluation predictions to {pred_path}")

            if SUBSET is None:
                if rec.model == "gbdt":
                    save_lgbm(ens_model.models_tr[rec.track].model, rec)
                elif rec.model == "gbdt_all":
                    save_lgbm(ens_model.model_all.model, rec)
        return out_records

# """
# Scores with default feats.
# AUC: 0.8122643673091771
# """
