from datasets.kt.seq_dataset import batch_pad
from datasets.kt.seq_dataset import build_user_sequences
from datasets.kt.seq_dataset import SeqDataset
from datasets.data_parquet import get_parquet
import numpy as np
import pandas as pd
import logging
import argparse
from datasets.kt.df_transforms import collapse_to_exercise, generate_qid_map, apply_qid_map
from torch.utils.data import DataLoader     
import torch
from models.DKT.DKT import DKT

parser = argparse.ArgumentParser()
parser.add_argument("--subset", type=int, default=None)
parser.add_argument("--track", type=str, default="en_es")
parser.add_argument("--train_with_dev", action="store_true")
parser.add_argument("--no_save_feats", action="store_true")
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

#==== CONFIG

torch.manual_seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRACK = args.track
train_with_dev = args.train_with_dev
SUBSET = args.subset
SAVE_FEATS = not args.no_save_feats
EPOCHS = args.epochs

if __name__ == "__main__":
        
    #==== LOAD DATA

    logger.info(f"Loading parquet data for track {TRACK}")

    df_train_data = get_parquet(TRACK, "train", "original", subset=SUBSET)    
    df_dev_data = get_parquet(TRACK, "dev", "original", subset=SUBSET)
    
    if not train_with_dev:
        df_train = df_train_data
        df_test = df_dev_data
    else:
        df_test_data = get_parquet(TRACK, "test", "original", subset=SUBSET)
        df_train = pd.concat([df_train_data, df_dev_data])
        df_test = df_test_data
    
    #=== Collapse data

    logger.info("Collapsing data")

    df_train = collapse_to_exercise(df_train)
    df_test = collapse_to_exercise(df_test)

    #=== Generate qid map from train and apply to both

    logger.info("Mapping questions to qids")

    qid_map_train = generate_qid_map(df_train)

    df_train = apply_qid_map(df_train, qid_map_train)
    df_test = apply_qid_map(df_test, qid_map_train)

    #=== Build sequences
    logger.info("Building sequences")

    train_seqs = build_user_sequences(df_train, qid_map_train)
    test_seqs = build_user_sequences(df_test, qid_map_train)

    #==== Build datasets

    train_ds = SeqDataset(train_seqs)
    test_ds = SeqDataset(test_seqs)
    
    #==== Build dataloaders
    
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=batch_pad)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=batch_pad)

    #==== Build model

    model = DKT(len(qid_map_train), emb_dim=128, head_dim=256)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ==== Train

    for epoch in range(EPOCHS):
        loss = model.train_epoch(train_dl, opt)
        logger.info(f"Epoch {epoch} loss: {loss}")
    
    #==== Evaluate

    test_auc = model.evaluate_auc(test_dl)
    logger.info(f"Test AUC: {test_auc}")