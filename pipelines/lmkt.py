from numpy import record
from models.text_kt.lmkt.build_data import build_lmkt_dataloaders
import logging
import argparse
import torch
from tqdm import tqdm

from db.log_db import MetricRecord
from models.text_kt.lmkt.lmkt import LMKTModel

logger = logging.getLogger(__name__)

def parse_lmkt_args(dkt_args=None):
    # PARSE SPECIFIC FLAGS
    p = argparse.ArgumentParser(description="LMKT Pipeline Args")
    p.add_argument("-e", "--epochs", type=int, default=5)
    args = p.parse_args(dkt_args)
    return args

def run_lmkt_pipeline(TRACK,SUBSET,train_with_dev, EPOCHS):

    logger.info("Running DKT pipeline for LM-KT")

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
        batch_size=2,
        shuffle_train=True
    )

    # ==== Train
    
    for epoch in tqdm(range(EPOCHS), desc="LMKT Training Epochs"):
        loss = model.train_one_epoch(lmkt_data.train_dataset, opt)
        logger.info(f"Epoch {epoch} loss: {loss}")
    
    #==== Evaluate

    metrics = model.evaluate_metrics(lmkt_data.eval_histories)
    logger.info("Test Metrics | AUC=%.5f | Accuracy=%.5f | F1=%.5f", 
                metrics["auc"], metrics["accuracy"], metrics["f1"])

    return [MetricRecord(
        model="lmkt",
        track=TRACK,
        subset=SUBSET,
        train_with_dev=train_with_dev,
        variant=None,
        auc=metrics.get("auc"),
        acc=metrics.get("accuracy"),
        f1=metrics.get("f1"),
        epochs=EPOCHS
    )]