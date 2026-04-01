from collections import defaultdict
from dataclasses import dataclass
import logging
import math
import random 
import pandas as pd
from typing import List, Dict

from tqdm import tqdm

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 1024

@dataclass
class ExerciseRecordDF:
    ex_key: str
    user_id: int
    text: str
    tokens: List[str]
    labels: List[int]
    pos_tags: List[str]
    split: int

@dataclass
class ExerciseRecord:
    text: str
    tokens: list[str]
    pos_tags: list[str]
    labels: list[int]
    split: int
    state_position: int
    adaptive_difficulty: float
    non_adaptive_difficulty: float
    keywords: list[str]

@dataclass  
class UserData:
    user_id: str
    word_ids: list[int]
    labels: list[int]
    split_ids: list[int]
    interaction_ids: list[int]
    exercises: list[ExerciseRecord]

def collapse_with_labels(df: pd.DataFrame, split_label: int) -> List[ExerciseRecordDF]:
    """
    Collapses token-level dataframe to exercise-level, using labels to determine correctness.
    Exercise is correct if ALL tokens are correct (label 0).
    """
    
    # CHECK IF LABELS ARE AVAILABLE
    if df["label"].isna().any():
        raise ValueError("Some labels are missing; cannot collapse to exercise level.")

    ex_key = df["tok_id"].str.slice(0, 10)
    df["ex_key"] = ex_key

    grouped = (
        df.groupby(["ex_key", "user_id"], sort=False).agg(
            text=("tok", " ".join),
            tokens=("tok", list),
            labels=("label", list),
            pos_tags=("pos", list)
        )
    ).reset_index() 

    out = []

    for row in tqdm(grouped.itertuples(index=False), desc="Collapsing to exercise level", leave=False):
        ex_key, user_id, text, tokens, labels, pos_tags = row

        out.append(ExerciseRecordDF(
            ex_key=ex_key,
            user_id=user_id,
            text=text,
            tokens=tokens,
            labels=labels,
            pos_tags=pos_tags,
            split=split_label
        ))

    return out

def build_word_vocab(train_df: pd.DataFrame) -> Dict[str, int]:
    """
    Builds a word vocabulary from the training data, mapping each unique token to a unique integer ID.
    """
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    for token in tqdm(train_df["tok"].unique(), desc="Building word vocabulary"):
        token = token.lower()
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

def compute_difficulty(df_train: pd.DataFrame) -> dict[str, float]:
    """
    Computes non-adaptive difficulty for each exercise based on training data.
    Difficulty is defined as the proportion of incorrect attempts (label 1) among all attempts for that exercise.
    """

    # ensure toks are lowercase
    df_train["tok"] = df_train["tok"].str.strip().str.lower()

    difficulty = df_train.groupby("tok")["label"].mean().to_dict()

    return difficulty

def sample_keywords(toks: List[str], pos_tags: List[str], rate: float) -> List[str]:
    
    KEYWORD_SAMPLE_PRIORITY = (
        {"NOUN", "VERB"},
        {"ADJ", "ADV"},
        {"PUNCT", "SYM", "X", "ADP", "AUX", "INTJ", "CCONJ", "DET", "PROPN", "NUM", "PART", "SCONJ", "PRON"},
    )
    assert len(toks) == len(pos_tags), "Tokens and POS tags must have the same length."
        
    # round half up
    target_count = math.floor(len(toks) * rate + 0.5)
    if target_count <= 0:
        #logger.warning(f"Target keyword count is {target_count}, given rate {rate} and {len(toks)} tokens. Returning empty keyword list.")
        return []

    tok_lists = [[] for _ in KEYWORD_SAMPLE_PRIORITY]
    for tok, pos_tag in zip(toks, pos_tags):
        
        # defensive
        tok = tok.lower()

        if tok in {"am", "is", "are"}: # treat as lowest priority
            tok_lists[-1].append(tok)
            continue
        
        matched=False
        for i, pos_group in enumerate(KEYWORD_SAMPLE_PRIORITY):
            if pos_tag in pos_group:
                tok_lists[i].append(tok)
                matched=True
                break
        
        assert matched, f"POS tag '{pos_tag}' did not match any priority group for token '{tok}'."

    sampled: set[str] = set()
    for tok_list in tok_lists:
        num_to_sample = target_count - len(sampled)

        if 1 <= num_to_sample <= len(tok_list):
            sampled.update(random.sample(tok_list, k=num_to_sample))

        if len(tok_list) >= target_count:
            break

    sampled_toks = list(sampled)
    # shuffle to avoid leaking priority
    random.shuffle(sampled_toks)

    return sampled_toks

def load_user_data(df: pd.DataFramem) -> pd.DataFrame:
    """
    Loads user data from token-level dataframe, collapsing to exercise-level and including labels.
    """

    # assume marked
    train_df = df[df["split"] == "train"]
    dev_df = df[df["split"] == "dev"]
    test_df = df[df["split"] == "test"]

    
    logger.info(f"Train samples: {len(train_df)}, Dev samples: {len(dev_df)}, Test samples: {len(test_df)}")
    train_recs = collapse_with_labels(train_df, 1)
    dev_recs = collapse_with_labels(dev_df, 2)
    test_recs = collapse_with_labels(test_df, 3)

    logger.info(f"Collapsed to exercises - Train: {len(train_recs)}, Dev: {len(dev_recs)}, Test: {len(test_recs)}")
    # RANDOM 80% sample for "seen"
    all_unique_texts = sorted(set(rec.text for recs in (train_recs, dev_recs, test_recs) for rec in recs))
    seen_texts = set(pd.Series(all_unique_texts).sample(frac=0.8, random_state=42))
    
    user_records: Dict[int, List[ExerciseRecord]] = defaultdict(list)

    for rec in train_recs + dev_recs + test_recs:
        if rec.user_id not in user_records:
            user_records[rec.user_id] = []
        user_records[rec.user_id].append(rec)

    # Compute non-adaptive difficulty from training data
    word_difficulty = compute_difficulty(train_df + dev_df) 
    
    # Build word vocab from training data
    word_vocab = build_word_vocab(train_df + dev_df)

    unk_id, bos_id, eos_id = word_vocab["<unk>"], word_vocab["<bos>"], word_vocab["<eos>"]
    users: List[UserData] = []

    no_diff_count = 0
    total_raw_tok_count = 0

    seq_len_budget = MAX_SEQ_LEN - 2

    for user_id, recs in tqdm(sorted(user_records.items()), desc="Processing users"):
        
        assert recs, "User {user_id} has no records after collapsing" 

        raw_tok_count = sum(len(rec.tokens) for rec in recs)

        if raw_tok_count > 2500: # NOTE: skip users with too much data (reference)
            continue
            
        if not any(rec.split == 3 for rec in recs): # skip users with no test data
            logger.warning(f"User {user_id} has no test data; skipping.")
            continue
        
        total_raw_tok_count += raw_tok_count
        # Truncuate-left
        num_items = 0
        tok_count = 0

        for rec in recs[::-1]:  # reverse to truncate from the left
            if tok_count + len(rec.tokens) <= seq_len_budget:
                num_items += 1
                tok_count += len(rec.tokens)
            else:
                break
        
        trunc_recs = recs[-num_items:]


        user_word_ids: List[int] = [bos_id]
        user_labels: List[int] = [-100]
        user_split_ids: List[int] = [0]
        user_interaction_ids: List[int] = [-1]
        user_exercises: List[ExerciseRecord] = []

        for interaction_id, rec in enumerate(trunc_recs):

            non_adaptive_difficulty = 0.0

            # non-adaptive difficulty
            for tok in rec.tokens:
                if tok.lower() not in word_difficulty:
                    no_diff_count += 1
                non_adaptive_difficulty += word_difficulty.get(tok.lower(), 0.0)

            state_position = len(user_word_ids)-1 # last token idx before exercise 

            user_exercises.append(ExerciseRecord(
                text=rec.text,
                tokens=rec.tokens,
                pos_tags=rec.pos_tags,
                labels=rec.labels,
                split=rec.split,
                state_position=state_position,
                non_adaptive_difficulty=non_adaptive_difficulty,
                adaptive_difficulty=sum(rec.labels), 
                keywords=sample_keywords(rec.tokens, rec.pos_tags, rate=0.3)
            ))

            for tok, lab in zip(rec.tokens, rec.labels):
                user_word_ids.append(word_vocab.get(tok.lower(), unk_id))
                user_labels.append(lab)
                user_split_ids.append(rec.split)
                user_interaction_ids.append(interaction_id)

        user_word_ids.append(eos_id)  # add EOS token at the end of user's sequence
        user_labels.append(-100)  # -100 for EOS token to ignore in loss
        user_split_ids.append(0)
        user_interaction_ids.append(-1)

        assert len(user_word_ids) <= MAX_SEQ_LEN, \
        f"User {user_id} exceeds MAX_SEQ_LEN: {len(user_word_ids)} > {MAX_SEQ_LEN}"

        users.append(UserData(
            user_id=user_id,
            word_ids=user_word_ids,
            labels=user_labels,
            split_ids=user_split_ids,
            interaction_ids=user_interaction_ids,
            exercises=user_exercises
        ))


    logger.warning(
        f"Tokens not found in word difficulty dict: {no_diff_count} / {total_raw_tok_count} "
        f"({no_diff_count/total_raw_tok_count:.2%})"
    )

    return users, seen_texts
