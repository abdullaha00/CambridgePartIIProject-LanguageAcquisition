import pandas as pd
from models.gbdt.features.lexical import add_lexical_feats
from models.gbdt.features.temporal import add_temporal_features_stream
from models.gbdt.features.positional import add_positional_features
from models.gbdt.features.user import add_user_feats_stream
from models.gbdt.features.morphological import add_morph_features
from data_processing.data_parquet import downcast_df, save_parquet

def build_features(df_train: pd.DataFrame, df_test: pd.DataFrame, train_with_dev, save_feats, track) -> tuple[pd.DataFrame, pd.DataFrame]:

    df_all = pd.concat([df_train.assign(is_test=0), df_test.assign(is_test=1)], ignore_index=True)

    # Convert categorical columns to plain strings so downstream .fillna() / assignment works
    for col in df_all.select_dtypes("category").columns:
        df_all[col] = df_all[col].astype("object")

    # Temporal + user done with train/test together
    df_all = add_temporal_features_stream(df_all)
    df_all = add_user_feats_stream(df_all)

    # Lexical
    df_all = add_lexical_feats(df_all, track)

    # Morphological one-hot encode
    df_all = add_morph_features(df_all)

    # Positional
    df_all = add_positional_features(df_all)

    df_train = df_all[df_all.is_test == 0].reset_index(drop=True)
    df_test = df_all[df_all.is_test == 1].reset_index(drop=True)

    if save_feats:
        variant_name = "features_test" if train_with_dev else "features_dev"
        save_parquet(df_train, track, "train", variant_name)
        save_parquet(df_test, track, "eval", variant_name)

    df_train, df_test = downcast_df(df_train), downcast_df(df_test)

    return df_train, df_test