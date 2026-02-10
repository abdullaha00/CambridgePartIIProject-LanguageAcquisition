

import logging

from data_processing.data_parquet import get_parquet, load_train_and_eval_df, parquet_exists
from models.gbdt.features.build_features import build_features

logger = logging.getLogger(__name__)

def get_feature_dfs(track: str, subset=None, train_with_dev=False, save_feats=True, clean_build=False) -> tuple:

    eval_track = "test" if train_with_dev else "dev"

    can_use_cache = (
        not clean_build
        and parquet_exists(track, "train", f"features_{eval_track}")
        and parquet_exists(track, "eval", f"features_{eval_track}")
    )

    if can_use_cache:
        logger.info(f"Loading precomputed features for track {track}...")
        df_train = get_parquet(track, "train", f"features_{eval_track}", subset=subset)
        df_test = get_parquet(track, "eval", f"features_{eval_track}", subset=subset)
        logger.info("Loaded features with shapes: train %s, test %s", df_train.shape, df_test.shape)
        logger.info("Containing %d users in train and %d users in test", df_train["user_id"].nunique(), df_test["user_id"].nunique())
    else:
        #NOTE: we only save features if SUBSET is none and SAVE_FEATS is true
        save_feats = save_feats and subset is None
        logger.info(f"Computing features with saving={save_feats} and SUBSET={subset}...")
        df_train, df_test = load_train_and_eval_df(track, "reprocessed", train_with_dev, subset=subset)
        df_train, df_test = build_features(df_train, df_test, train_with_dev, save_feats=save_feats, TRACK=track)

    return df_train, df_test