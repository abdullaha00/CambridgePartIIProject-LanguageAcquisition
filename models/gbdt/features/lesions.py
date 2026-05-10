import logging
import pandas as pd
logger = logging.getLogger(__name__)

# === THIS DEFINES THE DIFFERENT FEATURE GROUPS

USER_ID = {"user_id"}

WORD_IDS = {
    "tok",
    "lemma",
    "prev_tok",
    "next_tok",
    "rt_tok",
}

WORD_LINGUISTIC = {
    "pos",
    "prev_pos",
    "next_pos",
    "rt_pos",
    "rt",
    "deprel",
    "tok_len",
}

EXTERNAL_LEXICAL = {
    "src_freq",
    "dst_freq",
    "lev_distance",
    "aoa",
}

USER_OTHER = {
    "burst_mean",
    "burst_median",
    "burst_count",
    "tod_entropy",
}

EXERCISE_CONTEXT = {
    "ex_inst_idx",
    "exercise_num",
    "exercise_length",
    "client",
    "session",
    "format",
    "days",
    "time",
    "countries",
}

METADATA_EXTRA = {"countries"}

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
    "word_linguistic": WORD_LINGUISTIC,
    "word_other": WORD_LINGUISTIC,
    "external_lexical": EXTERNAL_LEXICAL,
    "external": EXTERNAL_LEXICAL,
    "user_other": USER_OTHER,
    "exercise_context": EXERCISE_CONTEXT,
    "exercise": EXERCISE_CONTEXT,
    "metadata_extra": METADATA_EXTRA,
    "temporal_exact": TEMPORAL_EXACT,
    "neighbors": NEIGHBORS,
}

PREFIX_GROUPS = {
    "morph": MORPH_PREFIX,
    "temporal_prefixes": TEMPORAL_PREFIXES,
}

LESIONS = {
    "none": [],

    "exercise_context": ["exercise_context"],
    "user": ["user_id", "user_other"],
    "word": ["word_ids", "word_linguistic", "morph"],
    "word_ids": ["word_ids"],
    "word_linguistic": ["word_linguistic", "morph"],
    "external_lexical": ["external_lexical"],
    "temporal": ["temporal_exact", "temporal_prefixes"],
    "neighbors": ["neighbors"],
    "metadata_extra": ["metadata_extra"],

    "exercise": ["exercise_context"],
    "word_other": ["word_linguistic", "morph"],
    "external": ["external_lexical"],
    "user_id": ["user_id"],
    "user_other": ["user_other"],
}

EVAL_LESIONS = [
    "none",
    "exercise_context",
    "user",
    "word",
    "word_ids",
    "word_linguistic",
    "external_lexical",
    "temporal",
    "neighbors",
    "metadata_extra",
]

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
