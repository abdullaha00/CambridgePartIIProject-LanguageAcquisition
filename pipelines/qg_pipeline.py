from datasets.lmkt.dataloaders_lmkt import build_lmkt_dataloaders
import logging
import argparse
import torch
from tqdm import tqdm

from models.lmkt.lmkt import LMKTModel

logger = logging.getLogger(__name__)

def parse_qg_args(qg_args=None):
    # PARSE SPECIFIC FLAGS
    p = argparse.ArgumentParser(description="QG Pipeline Args")
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args(qg_args)
    return args

def run_qg_pipeline(TRACK,SUBSET,train_with_dev, EPOCHS):

    logger.info("Running QG pipeline")

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
    logger.info("Test Metrics | AUC=%.5f | Accuracy=%.5f | F1=%.5f", 
                metrics["auc"], metrics["accuracy"], metrics["f1"])


    #======================== FREEZE MODEL

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    

    return metrics