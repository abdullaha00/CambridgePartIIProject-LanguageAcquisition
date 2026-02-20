
import logging
import lightgbm as lgb
import argparse

import numpy as np
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

logger = logging.getLogger(__name__)

#==== PARSING ARGS

def parse_gdbt_args(gdbt_args=None):
    # PARSE GDBT SPECIFIC FLAGS
    p = argparse.ArgumentParser(description="GBDT Pipeline Args")
    p.add_argument("--disable-save", action="store_true", 
                   help="Disable saving computed features for disk when full dataset is used" \
                   "(smaller subsets are never saved)")
    p.add_argument("--clean-build", action="store_true", 
                   help="Force clean feature build by ignoring existing parquet files")
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

def run_gbdt_pipeline(track="en_es",SUBSET=None,  train_with_dev=False, next_args=None):

    logger.info(f"Running GBDT pipeline for track {track} with train_with_dev={train_with_dev} and SUBSET={SUBSET}")

    gbdt_args = parse_gdbt_args(next_args)
    SAVE_FEATS = not gbdt_args.disable_save
    CLEAN_BUILD = gbdt_args.clean_build
    LESION = gbdt_args.lesion

    if track != ALL_TRACK:
        #====== FEATURES =====
        df_train, df_test = get_feature_dfs(track, subset=SUBSET, train_with_dev=train_with_dev, save_feats=SAVE_FEATS, clean_build=CLEAN_BUILD)
        df_train, df_test = cast_cats(df_train, df_test)
        #====== LESION =====
        if gbdt_args.lesion is not None:
            df_train = apply_lesion(df_train, gbdt_args.lesion)
            df_test = apply_lesion(df_test, gbdt_args.lesion)
        #===== TRAIN GDBT =====
        model = GBDTModel(track=track)
        X_test, y_test = model.fit(df_train, df_test)
        #===== EVALUATE =====
        metrics = model.evaluate(X_test, y_test)
        records = [MetricRecord(
            model="gbdt",
            track=track,
            subset=SUBSET,
            train_with_dev=train_with_dev,
            variant=None,
            auc=metrics.get("auc"),
            acc=metrics.get("accuracy"),
            f1=metrics.get("f1"),
        )]
        return records
    

    else:
        logger.info("Running GBDT ensemble pipeline for all tracks")

        tr_dfs = {}
        for tr in TRACKS:
            logger.info(f"Building features for {tr}...")
            df_train, df_test = get_feature_dfs(tr, subset=SUBSET, train_with_dev=train_with_dev, save_feats=SAVE_FEATS, clean_build=CLEAN_BUILD)
            
            #LESION
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

        for tr in TRACKS:
            tr_mets = per_track_metrics[tr]
            comb_mets = combined_metrics[tr]
            out_records.append(MetricRecord(
                model="gbdt",
                track=tr,
                subset=SUBSET,
                train_with_dev=train_with_dev,
                variant=LESION, 
                auc=tr_mets.get("auc"),
                acc=tr_mets.get("accuracy"),
                f1=tr_mets.get("f1"),
            ))
            out_records.append(MetricRecord(
                model="gbdt_ens",
                track=tr,
                subset=SUBSET,
                train_with_dev=train_with_dev,
                variant=LESION,
                auc=comb_mets.get("auc"),
                acc=comb_mets.get("accuracy"),
                f1=comb_mets.get("f1"),
            ))

        if per_track_metrics.get(ALL_TRACK) is not None:
            all_mets = per_track_metrics[ALL_TRACK]
            out_records.append(MetricRecord(
                model="gbdt_ens",
                track=ALL_TRACK,
                subset=SUBSET,
                train_with_dev=train_with_dev,
                variant=LESION,
                auc=all_mets.get("auc"),
                acc=all_mets.get("accuracy"),
                f1=all_mets.get("f1"),
            ))

        return out_records

# """
# Scores with default feats.
# AUC: 0.8122643673091771
# """