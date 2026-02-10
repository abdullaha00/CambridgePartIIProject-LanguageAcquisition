

from dataclasses import dataclass
from typing import Dict
from models.text_kt.common.data import collapse_to_exercise
from models.dkt.data.data import apply_qid_map, generate_qid_map
from models.dkt.data.data import DKTSeqDataset, collate_dkt
from models.dkt.data.data import build_user_sequences_qid
from data_processing.data_parquet import load_train_and_eval_df
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

@dataclass
class DKTDataBundle:
    train_dataset: DKTSeqDataset
    eval_dataset: DKTSeqDataset
    qid_map: Dict[str, int]

def build_dkt_dataloaders(
    track: str,
    variant: str,
    subset: int | None,
    train_with_dev: bool,
    batch_size: int = 64,
    shuffle_train: bool = True,
    ) -> DKTDataBundle:
    
    #======= LOAD DATA

    logger.info("Loading dataframes for track %s, variant %s, subset %s, train_with_dev=%s",
                track, variant, subset, train_with_dev)

    df_train, df_eval = load_train_and_eval_df(
        track, variant, train_with_dev, subset=subset
    )

    #======= Collapse data

    logger.info("Collapsing data")

    df_train = collapse_to_exercise(df_train)
    df_eval = collapse_to_exercise(df_eval)

    # === Generate qid map from train and apply to both

    logger.info("Mapping questions to qids")

    qid_map_train = generate_qid_map(df_train)

    df_train = apply_qid_map(df_train, qid_map_train)
    df_eval = apply_qid_map(df_eval, qid_map_train) 

    #==== Build sequences
    logger.info("Building sequences")

    train_seqs = build_user_sequences_qid(df_train, qid_map_train)
    eval_seqs = build_user_sequences_qid(df_eval, qid_map_train)

    #==== Build datasets

    train_ds = DKTSeqDataset(train_seqs)
    test_ds = DKTSeqDataset(eval_seqs)

    #==== Build dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, collate_fn=collate_dkt)
    eval_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_dkt)   

    return DKTDataBundle(train_dataset=train_dl, eval_dataset=eval_dl, qid_map=qid_map_train)