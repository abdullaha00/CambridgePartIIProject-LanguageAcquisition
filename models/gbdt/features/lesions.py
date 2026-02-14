import logging
import pandas as pd
logger = logging.getLogger(__name__)

# === THIS DEFINES THE DIFFERENT FEATURE GROUPS

USER_ID = {"user_id"}

WORD_IDS = {
    "tok",
    "lemma",
    "user_id",
    "prev_tok",
    "next_tok",
}

WORD_OTHER = {
    "pos",
    "prev_pos",
    "next_pos",
}

EXTERNAL = {
    "translation",
    "src_freq",
    "dst_freq",
    "lev_distance",
}

USER_OTHER = {
    "burst_mean",
    "burst_median",
    "burst_count",
    "tod_entropy",
}

EXERCISE = {
    "client",
    "session",
    "format",
    "days",
    "time",
    "countries",
    "type",
}

MORPH_PREFIX = "morph__"
TEMPORAL_PREFIXES = (
    "err_tok_",
    "err_root_",
)

TEMPORAL_EXACT = {
    "ex_seen",
    "tok_seen",
    "root_seen",
    "tok_seen_lab",
    "root_seen_lab",
    "tok_seen_unlab",
    "root_seen_unlab",
    "tok_tslast",
    "root_tslast",
    "tok_tslast_lab",
    "root_tslast_lab",
    "tok_first",
    "root_first",
}

NEIGHBORS = {
    "prev_tok", "next_tok", "rt_tok",
    "prev_pos", "next_pos", "rt_pos",
}    

FEATURE_GROUPS = {
    "user_id": USER_ID,
    "word_ids": WORD_IDS,
    "word_other": WORD_OTHER,
    "external": EXTERNAL,
    "user_other": USER_OTHER,
    "exercise": EXERCISE,
    "temporal_exact": TEMPORAL_EXACT,
    "temporal_prefixes": TEMPORAL_PREFIXES,
    "neighbors": NEIGHBORS,
}

PREFIX_GROUPS = {
    "morph": MORPH_PREFIX,
    "temporal_prefixes": TEMPORAL_PREFIXES,
}

LESIONS = {
    "none": [],

    "neighbors": ["neighbors"],

    "word_ids": ["word_ids"],
    "word_other": ["word_other", "morph"], 
    "word": ["word_ids", "word_other", "morph"],

    "external": ["external"],

    "user_id": ["user_id"],
    "user_other": ["user_other"],
    "user": ["user_id", "user_other"],

    "temporal": ["temporal_exact", "temporal_err"],

    "exercise": ["exercise"],
}

def lesion_to_drop_set(df_cols, lesion: str) -> dict:
    if lesion not in LESIONS:
        raise ValueError(f"Invalid lesion name: {lesion}")

    drop_set = set()

    for group in LESIONS[lesion]:

        if group in FEATURE_GROUPS:
            # exact match
            drop_set |= FEATURE_GROUPS[group] # union
        elif group in PREFIX_GROUPS:
            # prefix match
            prefix = PREFIX_GROUPS[group]
            drop_set |= {col for col in df_cols if col.startswith(prefix)}
        else:
            raise ValueError(f"Invalid feature group in lesion: {group}")

    return drop_set

def apply_lesion(df: pd.DataFrame, lesion: str) -> pd.DataFrame:
    if lesion is None or lesion == "none":
        return df

    drop_set = lesion_to_drop_set(df.columns, lesion)
    logger.info(f"Applying lesion '{lesion}' by dropping {len(drop_set)} features: {drop_set}")
    return df.drop(columns=drop_set)