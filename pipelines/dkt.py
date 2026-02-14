from models.dkt.data.build_data import build_dkt_dataloaders
from db.log_db import MetricRecord
from models.dkt.BertDKT import BertDKT
import logging
import argparse
import torch
from models.dkt.DKT import DKT
from models.dkt.data.data import embed_sentence_matrix
from tqdm import tqdm

logger = logging.getLogger(__name__)

def parse_dkt_args(dkt_args=None):
    # PARSE DKT SPECIFIC FLAGS
    p = argparse.ArgumentParser(description="GBDT Pipeline Args")    
    args = p.parse_args(dkt_args)
    return args
    

def run_dkt_pipeline(model_name, TRACK,SUBSET,train_with_dev, ITEM_LEVEL, epochs, eval_every, next_args):

    dkt_args = parse_dkt_args(next_args)

    logger.info(f"Running DKT pipeline for model {model_name}")

    logger.info(f"Building dataloaders for track {TRACK}, subset {SUBSET}, train_with_dev={train_with_dev}")
    #==== BUILD DATALOADER
    dkt_data = build_dkt_dataloaders(
        track=TRACK,
        variant="reprocessed",
        subset=SUBSET,
        item_level=ITEM_LEVEL,
        train_with_dev=train_with_dev,
        batch_size=32,
        shuffle_train=True
    )
    #==== Build model

    if model_name == "dkt":
        model = DKT(len(dkt_data.item_map), emb_dim=128, head_dim=256)
    else:
        train_emb_matrix = embed_sentence_matrix(list(dkt_data.item_map.keys()))
        model = BertDKT(len(dkt_data.item_map), emb_dim=128, head_dim=256, emb_matrix=train_emb_matrix)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ==== Train

    records = []
    loss_history = []

    pbar = tqdm(range(epochs), desc="DKT Training Epochs")
    for epoch in pbar:
        loss = model.train_epoch(dkt_data.train_dataset, opt)
        loss_history.append(loss)

        pbar.set_postfix(loss=f"{loss:.4f}")
    
        #==== Evaluate

        if (epoch == epochs-1) or ((epoch+1) % eval_every == 0):
            metrics = model.evaluate_metrics(dkt_data.eval_dataset)
            logger.info("Test Metrics | AUC=%.5f | Accuracy=%.5f | F1=%.5f", 
                        metrics["auc"], metrics["accuracy"], metrics["f1"])

            record = MetricRecord(
                model=model_name,
                track=TRACK,
                subset=SUBSET,
                train_with_dev=train_with_dev,
                variant=ITEM_LEVEL,
                auc=metrics.get("auc"),
                acc=metrics.get("accuracy"),
                f1=metrics.get("f1"),
                epochs=epochs
            )
            records.append(record)

    logger.info("DKT loss history: %s", loss_history)

    return records