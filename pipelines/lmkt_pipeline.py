from datasets.kt.dataloaders_dkt import build_dkt_dataloaders
from datasets.kt.dataloaders_lmkt import build_lmkt_dataloaders
from models.dkt.BertDKT import BertDKT
import numpy as np
import pandas as pd
import logging
import argparse
from datasets.kt.df_transforms import generate_qid_map, apply_qid_map
from torch.utils.data import DataLoader     
import torch
from models.dkt.DKT import DKT
from datasets.kt.text_embeddings import embed_sentence_matrix
from tqdm import tqdm

from models.dkt.lmkt import LMKTModel

logger = logging.getLogger(__name__)

def parse_lmkt_args(dkt_args=None):
    # PARSE SPECIFIC FLAGS
    p = argparse.ArgumentParser(description="LMKT Pipeline Args")
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args(dkt_args)
    return args

def run_lmkt_pipeline(TRACK,SUBSET,train_with_dev, EPOCHS):

    logger.info(f"Running DKT pipeline for LM-KT")

    logger.info(f"Building dataloaders for track {TRACK}, subset {SUBSET}, train_with_dev={train_with_dev}")
    
    #==== Build model

    model = LMKTModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    #==== BUILD DATALOADER
    lmkt_data = build_lmkt_dataloaders(
        track=TRACK,
        variant="minimal",
        subset=SUBSET,
        train_with_dev=train_with_dev,
        tokenizer=model.tokenizer,
        batch_size=1,
        shuffle_train=True
    )

    # ==== Train
    
    for epoch in tqdm(range(EPOCHS), desc="DKT Training Epochs"):
        loss = model.train_one_epoch(lmkt_data.train_dataset, opt)
        logger.info(f"Epoch {epoch} loss: {loss}")
    
    #==== Evaluate

    metrics = model.evaluate_metrics(lmkt_data.eval_histories)
    logger.info(f"Test Metrics | AUC=%.5f | Accuracy=%.5f | F1=%.5f", 
                metrics["auc"], metrics["accuracy"], metrics["f1"])

    return metrics