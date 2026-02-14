

from dataclasses import dataclass
from typing import Dict
from config.consts import ITEM_EX, ITEM_TOK
from models.dkt.data.item_builders import SeqBundle, build_ex_sequences, build_tok_sequences
from models.text_kt.common.data import collapse_to_exercise
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
    """Truncate sequences to at most max_len, keeping the most recent interactions."""
    return {
        uid: (q_ids[-max_len:], correct[-max_len:])
        for uid, (q_ids, correct) in seqs.items()
    }

def build_dkt_dataloaders(
    track: str,
    variant: str,
    subset: int | None,
    item_level: str,
    train_with_dev: bool,
    batch_size: int = 64,
    shuffle_train: bool = True,
    max_seq_len: int = None,
    ) -> DKTDataBundle:

    #======= LOAD DATA (needed cols)

    logger.info("Loading dataframes for track %s, variant %s, subset %s, train_with_dev=%s, item_level=%s",
                track, variant, subset, train_with_dev, item_level)

    if item_level == ITEM_TOK:
        DF_COLS = ["user_id", "lemma", "label"]
    elif item_level == ITEM_EX:
        DF_COLS = ["user_id", "ex_instance_id", "tok", "label"]
    else:
        raise ValueError(f"Invalid item_level {item_level}")
    
    df_train, df_eval = load_train_and_eval_df(
        track, variant, train_with_dev, subset=subset
    )

    #======= Get correct seq bundle

    if item_level == ITEM_TOK:
        bundle: SeqBundle = build_tok_sequences(df_train, df_eval, item_col="lemma", drop_unk=True)
    elif item_level == ITEM_EX:
        bundle: SeqBundle = build_ex_sequences(df_train, df_eval, item_col="ref_ans", drop_unk=True)
    
    train_seqs = bundle.seqs["train"]
    eval_seqs = bundle.seqs["eval"]

    #==== Truncate long sequences to cap memory during training
    if max_seq_len is not None:
        train_seqs = truncuate_seqs(train_seqs, max_seq_len)

    #==== Build datasets
    train_ds = DKTSeqDataset(train_seqs)
    test_ds = DKTSeqDataset(eval_seqs)

    #==== Build dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, collate_fn=collate_dkt)
    eval_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_dkt)   

    return DKTDataBundle(train_dataset=train_dl, eval_dataset=eval_dl, item_map=bundle.item_map)
