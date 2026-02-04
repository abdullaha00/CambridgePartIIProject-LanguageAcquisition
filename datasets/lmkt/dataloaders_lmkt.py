
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Tuple

from transformers import AutoTokenizer
from datasets.common.df_utils import collapse_to_exercise
from datasets.common.sequence_builders import build_user_sequences_text
from datasets.lmkt.lmkt_dataset import SeqDatasetLMKT
from datasets.common.collate import lmkt_collate
from data.data_parquet import load_train_and_eval_df
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class LMKTDataBundle:
    train_dataset: SeqDatasetLMKT
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
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, collate_fn=
                          partial(lmkt_collate, pad_token_id=tokenizer.pad_token_id))
    return LMKTDataBundle(train_dataset=train_dl, eval_histories=eval_histories, tokenizer=tokenizer)



