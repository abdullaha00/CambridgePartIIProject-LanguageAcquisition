from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
from functools import partial
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging

from data_processing.data_parquet import load_train_and_eval_df
from models.text_kt.common.data import build_user_sequences_text
from models.text_kt.common.tokens import TOK_N, TOK_Y
from models.text_kt.common.data import collapse_to_exercise
from models.text_kt.lmkt.data import SeqDatasetLMKT, lmkt_collate

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
    tokenizer: AutoTokenizer,
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
    y_id = tokenizer.convert_tokens_to_ids(TOK_Y)
    n_id = tokenizer.convert_tokens_to_ids(TOK_N)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, collate_fn=
                          partial(lmkt_collate, pad_token_id=tokenizer.pad_token_id,
                                  y_id=y_id, n_id=n_id))
    return LMKTDataBundle(train_dataset=train_dl, eval_histories=eval_histories, tokenizer=tokenizer)


