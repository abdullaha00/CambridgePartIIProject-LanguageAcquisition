from models.DKT.BertDKT import BertDKT
from datasets.kt.seq_dataset import batch_pad
from datasets.kt.seq_dataset import build_user_sequences
from datasets.kt.seq_dataset import SeqDataset
from datasets.data_parquet import get_parquet, load_train_and_eval_df
import numpy as np
import pandas as pd
import logging
import argparse
from datasets.kt.df_transforms import collapse_to_exercise, generate_qid_map, apply_qid_map
from torch.utils.data import DataLoader     
import torch
from models.DKT.DKT import DKT
from datasets.kt.text_embeddings import embed_sentence_matrix

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

    logger.info(f"Loading parquet data for track {TRACK}, subset {SUBSET}, train_with_dev={train_with_dev}")
    df_train, df_test = load_train_and_eval_df(TRACK, "reprocessed", train_with_dev, subset=SUBSET)
    
    #=== Collapse data

    logger.info("Collapsing data")
    df_train = collapse_to_exercise(df_train)
    df_test = collapse_to_exercise(df_test)

    #=== Generate qid map from train and apply to both

    logger.info("Mapping questions to qids")

    qid_map_train = generate_qid_map(df_train)
    df_train = apply_qid_map(df_train, qid_map_train)
    df_test = apply_qid_map(df_test, qid_map_train)

    #=== Generate text embeddings

    logger.info("Generating text embeddings")

    train_emb_matrix = embed_sentence_matrix(list(qid_map_train.keys()))

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

    model = BertDKT(len(qid_map_train), emb_dim=128, head_dim=256, emb_matrix=train_emb_matrix)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ==== Train

    for epoch in range(EPOCHS):
        loss = model.train_epoch(train_dl, opt)
        logger.info(f"Epoch {epoch} loss: {loss}")
    
    #==== Evaluate

    test_auc = model.evaluate_auc(test_dl)
    logger.info(f"Test AUC: {test_auc}")