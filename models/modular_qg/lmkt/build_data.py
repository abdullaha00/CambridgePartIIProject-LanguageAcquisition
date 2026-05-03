from dataclasses import dataclass
from typing import Dict, List, Tuple
from functools import partial
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
from data_processing.data_parquet import load_train_and_eval_df
from models.modular_qg.common.data import build_user_sequences_text
from models.modular_qg.common.tokens import TOK_N, TOK_Y
from models.modular_qg.common.data import collapse_to_exercise
from models.modular_qg.lmkt.data import SeqDatasetLMKT, lmkt_collate

logger = logging.getLogger(__name__)

@dataclass
class LMKTDataBundle:
    train_dataloader: DataLoader
    eval_histories: Dict[str, List[Tuple[str, int]]]
    pref_ns: Dict[str, int]
    train_seen_prompts: set[str]
    tokenizer: AutoTokenizer
    compact_serialisation: bool

def build_lmkt_dataloaders(
    track: str,
    variant: str,
    subset: int | None,
    train_with_dev: bool,
    tokenizer: AutoTokenizer,
    batch_size: int = 64,
    shuffle_train: bool = True,
    yn_loss_only: bool = True,
    reverse_translate_only: bool = True,
    sliding_window: bool = False,
    compact_serialisation: bool = False,
    ) -> LMKTDataBundle:
    
    #======= LOAD DATA


    df_train, df_eval = load_train_and_eval_df(
        track, variant, train_with_dev, subset=subset
    )
    logger.info("Loading prompt data!")

    dft_prompts, dfe_prompts = load_train_and_eval_df(
        track, "prompt", train_with_dev, subset=subset
    )

    if reverse_translate_only:
        logger.info("Filtering to reverse_translate format only")
        df_train = df_train[df_train["format"] == "reverse_translate"]
        df_eval = df_eval[df_eval["format"] == "reverse_translate"]
    else:
        logger.info("Keeping both reverse_translate and reverse_tap formats")
        df_train = df_train[(df_train["format"] == "reverse_translate") | (df_train["format"] == "reverse_tap")]
        df_eval = df_eval[(df_eval["format"] == "reverse_translate") | (df_eval["format"] == "reverse_tap")]



    #======= Collapse data
    logger.info("Collapsing data")

    df_train = collapse_to_exercise(df_train)
    df_eval = collapse_to_exercise(df_eval)

    assert "ex_key" in df_train.columns, "ex_key not found in collapsed training dataframe"
    assert "ex_key" in df_eval.columns, "ex_key not found in collapsed eval"

    # Merge with prompts to get prompt text
    df_train = df_train.merge(dft_prompts[["ex_key", "prompt"]], on="ex_key", how="left")
    df_eval = df_eval.merge(dfe_prompts[["ex_key", "prompt"]], on="ex_key", how="left")

    # Sanity check to make sure all prompts were filled
    missing_train = df_train["prompt"].isna().sum()
    assert missing_train == 0, f"Missing prompts in training data after merge: {missing_train}"
    missing_eval = df_eval["prompt"].isna().sum()
    assert missing_eval == 0, f"Missing prompts in eval data: {missing_eval}"

    # seen/unseen prompts
    train_seen_prompts = set(df_train["prompt"].tolist())

    #==== Build sequences
    logger.info("Building sequences")

    train_histories = build_user_sequences_text(df_train)
    eval_histories = build_user_sequences_text(df_eval)



    # We want to prepend each user's training history as context

    eval_histories_prepended = {}
    pref_ns = {}

    for uid, eval_history in tqdm(eval_histories.items(), 
                                  desc="Prepending training history to eval histories", leave=False):
        
        assert uid in train_histories, f"User {uid} in eval set not found in train set"
        train_history = train_histories.get(uid, [])
        eval_histories_prepended[uid] = train_history + eval_history
        pref_n = len(train_history) # COUNT OF EXERCISES
        pref_ns[uid] = pref_n
        
    #==== Build dataset
    train_mode = "sliding" if sliding_window else "truncate-left"
    logger.info(f"Building LMKT training dataset with mode={train_mode}, compact_serialisation={compact_serialisation}")
    train_ds = SeqDatasetLMKT(
        train_histories,
        tokenizer=tokenizer,
        mode=train_mode,
        compact_serialisation=compact_serialisation
    )

    #==== Build dataloaders
    y_id = tokenizer.convert_tokens_to_ids(TOK_Y)
    n_id = tokenizer.convert_tokens_to_ids(TOK_N)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, collate_fn=
                          partial(lmkt_collate, pad_token_id=tokenizer.pad_token_id,
                                  y_id=y_id, n_id=n_id, yn_loss_only=yn_loss_only))
    return LMKTDataBundle(train_dataloader=train_dl, 
                          eval_histories=eval_histories_prepended, 
                          pref_ns=pref_ns, 
                          train_seen_prompts=train_seen_prompts,
                          tokenizer=tokenizer,
                          compact_serialisation=compact_serialisation)
