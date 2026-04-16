

from dataclasses import dataclass
from typing import Dict

import numpy as np
from config.consts import ITEM_EX, ITEM_TOK
from models.dkt.data.item_builders import SeqBundle, build_ex_sequences, build_tok_sequences
from models.modular_qg.common.data import collapse_to_exercise
from models.dkt.data.data import apply_qid_map, generate_qid_map
from models.dkt.data.data import DKTSeqDataset, collate_dkt
from models.dkt.data.data import build_user_sequences_qid
from data_processing.data_parquet import load_train_and_eval_df
import logging
from torch.utils.data import DataLoader
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class DKTDataBundle:
    train_dataset: DataLoader
    eval_dataset: DataLoader
    item_map: Dict[str, int]

def truncuate_seqs(seqs: dict, max_len: int) -> dict:
    return {
        uid: (q_ids[-max_len:], correct[-max_len:])
        for uid, (q_ids, correct) in seqs.items()
    }

def prepend_train_history(train_seqs: dict, eval_seqs: dict) -> dict:
    # eval_seqs is dict of uid -> (q_ids, correct_list)
    # train_seqs is dict of uid -> (q_ids, correct_list)

    full_eval_seqs = {}
    pref_lens = {}

    for uid, (eval_q, eval_a) in eval_seqs.items():
        train_q, train_a = train_seqs[uid]
        pref_lens[uid] = len(train_q)
    
        full_q = np.concatenate([train_q, eval_q])
        full_a = np.concatenate([train_a, eval_a])

        full_eval_seqs[uid] = (full_q, full_a)

    return full_eval_seqs, pref_lens

def build_dkt_dataloaders(
        track: str,
        variant: str,
        subset: int | None,
        item_level: str,
        train_with_dev: bool,
        batch_size: int = 32,
        shuffle_train: bool = True,
        max_seq_len: int = None,
        use_prompts: bool = False
    ) -> DKTDataBundle:

    #======= LOAD DATA (needed cols)

    logger.info("Loading dataframes for track %s, variant %s, subset %s, train_with_dev=%s, item_level=%s",
                track, variant, subset, train_with_dev, item_level)

    if item_level == ITEM_TOK:
        DF_COLS = ["user_id", "lemma", "label"]
    elif item_level == ITEM_EX:
        DF_COLS = ["user_id", "tok_id", "tok", "label", "format"]
    else:
        raise ValueError(f"Invalid item_level {item_level}")
    
    df_train, df_eval = load_train_and_eval_df(
        track, variant, train_with_dev, subset=subset, columns=DF_COLS
    )

    #======= Get correct seq bundle

    if item_level == ITEM_TOK:
        bundle: SeqBundle = build_tok_sequences(df_train, df_eval, item_col="lemma", drop_unk=False)
    elif item_level == ITEM_EX:
        
        dft_prompts, dfe_prompts = None, None
        if use_prompts:
            # We restrict to reverse_translate tasks, since those have prompts only
            logger.info("Restricting to reverse_translate format for exercise-level DKT to use prompts")
            df_train = df_train[df_train["format"] == "reverse_translate"]
            df_eval = df_eval[df_eval["format"] == "reverse_translate"]

            
            dft_prompts, dfe_prompts = load_train_and_eval_df(
                track, "prompt", train_with_dev, subset=subset
            )

        item_col = "prompt" if use_prompts else "tok_text"
        bundle: SeqBundle = build_ex_sequences(df_train, df_eval, item_col=item_col, dft_prompts=dft_prompts, dfe_prompts=dfe_prompts, drop_unk=False)
    
    train_seqs = bundle.seqs["train"]
    eval_seqs = bundle.seqs["eval"]

    eval_full_seqs, pref_lens = prepend_train_history(train_seqs, eval_seqs)

    #==== Truncate long sequences to cap memory during training
    if max_seq_len is not None:
        train_seqs = truncuate_seqs(train_seqs, max_seq_len)

    #==== Build datasets
    train_ds = DKTSeqDataset(train_seqs, prefix_lens={})
    test_ds = DKTSeqDataset(eval_full_seqs, prefix_lens=pref_lens)

    #==== Build dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, collate_fn=collate_dkt)
    eval_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_dkt)   

    return DKTDataBundle(train_dataset=train_dl, eval_dataset=eval_dl, item_map=bundle.item_map)
