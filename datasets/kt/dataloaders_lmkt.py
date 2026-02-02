
from dataclasses import dataclass
from typing import Dict, List, Tuple

from transformers import AutoTokenizer
from datasets.kt.df_transforms import collapse_to_exercise
from datasets.kt.seq_dataset import SeqDataset, SeqDatasetLMKT, build_user_sequences_text, lmkt_batch_pad
from datasets.data_parquet import load_train_and_eval_df
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

TOK_Q = "<Q>"
TOK_A = "<A>"
TOK_Y = "<Y>"
TOK_N = "<N>"

SPECIAL_TOKS = [TOK_Q, TOK_A, TOK_Y, TOK_N]

@dataclass
class LMKTDataBundle:
    train_dataset: SeqDataset
    eval_histories: Dict[str, List[Tuple[str, int]]]
    tokenizer: AutoTokenizer

def build_lmkt_dataloaders(
    track: str,
    variant: str,
    subset: int | None,
    train_with_dev: bool,
    tokenizer: any,
    batch_size: int = 64,
    shuffle_train: bool = True,
    ) -> LMKTDataBundle:
    
    #======= LOAD DATA

    df_train, df_eval = load_train_and_eval_df(
        track, variant, train_with_dev, subset=subset
    )

    #======= Collapse data

    logger.info("Collapsing data")

    df_train = collapse_to_exercise(df_train)
    df_eval = collapse_to_exercise(df_eval)


    #==== Build sequences
    logger.info("Building sequences")

    train_histories = build_user_sequences_text(df_train)
    eval_histories = build_user_sequences_text(df_eval)

    #==== Build dataset

    train_ds = SeqDatasetLMKT(train_histories, tokenizer=tokenizer)

    #==== Build dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, collate_fn=lmkt_batch_pad)
    return LMKTDataBundle(train_dataset=train_dl, eval_histories=eval_histories, tokenizer=tokenizer)



