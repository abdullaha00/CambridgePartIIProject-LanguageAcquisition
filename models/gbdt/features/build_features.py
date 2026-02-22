import pandas as pd
from models.gbdt.features.lexical import add_lexical_feats
from models.gbdt.features.temporal import add_temporal_features_stream
from models.gbdt.features.positional import add_positional_features
from models.gbdt.features.user import add_user_feats_stream
from models.gbdt.features.morphological import add_morph_features
from data_processing.data_parquet import downcast_df, save_parquet
from models.gbdt.params import CAT_FEATS

def mark_ids_with_lang(df, src_lang):
    # Mark identifiers with language to prevent collisions in combined track model
    id_cols = {"tok", "lemma", "prev_tok", "next_tok", "rt_tok", "user_id"} 
    for col in id_cols:

        if col in {"prev_tok", "next_tok", "rt_tok"}:
            # these may have <NONE> values which we don't want to mark
            mask_none = df[col] == "<NONE>"
            df[col] = df[col].where(mask_none, src_lang + "_" + df[col])
        df[col] = src_lang + "_" + df[col].astype(str)
        
    return df

def build_features(df_train: pd.DataFrame, df_test: pd.DataFrame, train_with_dev, save_feats, track) -> tuple[pd.DataFrame, pd.DataFrame]:

    df_all = pd.concat([df_train.assign(is_test=0), df_test.assign(is_test=1)], ignore_index=True)
    
    # NOTE "ex_instance_id" is specific to each split,
    # so we remove this to prevent errors
    df_all = df_all.drop(columns=["ex_instance_id"], errors="ignore")
    
    # and use a global exercise key
    df_all["ex_key"] = df_all["tok_id"].str.slice(0, 10)

    # Convert categorical columns to plain strings
    for col in df_all.select_dtypes("category").columns:
        df_all[col] = df_all[col].astype("object")

    # lowercase

    # Temporal + user done with train/test together
    df_all = add_temporal_features_stream(df_all)
    df_all = add_user_feats_stream(df_all)

    # Lexical
    df_all = add_lexical_feats(df_all, track)

    # Morphological one-hot encode
    df_all = add_morph_features(df_all)

    # Positional
    df_all = add_positional_features(df_all)

    # Mark with language
    # src_lang, _ = track.split("_")
    # df_all = mark_ids_with_lang(df_all, src_lang)

    df_train = df_all[df_all.is_test == 0].reset_index(drop=True)
    df_test = df_all[df_all.is_test == 1].reset_index(drop=True)

    if save_feats:
        variant_name = "features_test" if train_with_dev else "features_dev"
        save_parquet(df_train, track, "train", variant_name)
        save_parquet(df_test, track, "eval", variant_name)

    df_train, df_test = downcast_df(df_train), downcast_df(df_test)

    
    return df_train, df_test