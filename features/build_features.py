import pandas as pd
from features.lexical import add_lexical_feats
from features.temporal import add_temporal_features_stream
from features.positional import add_positional_features
from features.user import add_user_feats_stream
from features.morphological import add_morph_features
from datasets.data_parquet import save_parquet

def build_features(df_train: pd.DataFrame, df_test: pd.DataFrame, train_with_dev, save_feats, TRACK) -> tuple[pd.DataFrame, pd.DataFrame]:

    df_all = pd.concat([df_train.assign(is_test=0), df_test.assign(is_test=1)], ignore_index=True)

    # Temporal + user done with train/test together
    df_all = add_temporal_features_stream(df_all)
    df_all = add_user_feats_stream(df_all)

    # Lexical
    df_all = add_lexical_feats(df_all)

    # Morphological one-hot encode
    df_all = add_morph_features(df_all)

    # Positional
    df_all = add_positional_features(df_all)

    df_train = df_all[df_all.is_test == 0].reset_index(drop=True)
    df_test = df_all[df_all.is_test == 1].reset_index(drop=True)

    if save_feats:
        variant_name = "features_test" if train_with_dev else "features_dev"
        save_parquet(df_train, TRACK, "train", variant_name)
        save_parquet(df_test, TRACK, "test", variant_name)

    return df_train, df_test