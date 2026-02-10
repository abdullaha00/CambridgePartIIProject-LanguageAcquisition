from models.dkt.data.build_data import build_dkt_dataloaders
from db.records import MetricRecord
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
    p.add_argument("-e", "--epochs", type=int, default=5)
    
    args = p.parse_args(dkt_args)
    return args

def run_dkt_pipeline(model_name, TRACK,SUBSET,train_with_dev, EPOCHS):

    logger.info(f"Running DKT pipeline for model {model_name}")

    logger.info(f"Building dataloaders for track {TRACK}, subset {SUBSET}, train_with_dev={train_with_dev}")
    #==== BUILD DATALOADER
    dkt_data = build_dkt_dataloaders(
        track=TRACK,
        variant="minimal",
        subset=SUBSET,
        train_with_dev=train_with_dev,
        batch_size=32,
        shuffle_train=True
    )

    #==== Build model

    if model_name == "dkt":
        model = DKT(len(dkt_data.qid_map), emb_dim=128, head_dim=256)
    else:
        train_emb_matrix = embed_sentence_matrix(list(dkt_data.qid_map.keys()))
        model = BertDKT(len(dkt_data.qid_map), emb_dim=128, head_dim=256, emb_matrix=train_emb_matrix)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ==== Train

    loss_history = []

    pbar = tqdm(range(EPOCHS), desc="DKT Training Epochs")
    for epoch in pbar:
        loss = model.train_epoch(dkt_data.train_dataset, opt)
        loss_history.append(loss)

        pbar.set_postfix(loss=f"{loss:.4f}")
    
    logger.info("DKT loss history: %s", loss_history)
    
    #==== Evaluate

    metrics = model.evaluate_metrics(dkt_data.eval_dataset)
    logger.info("Test Metrics | AUC=%.5f | Accuracy=%.5f | F1=%.5f", 
                metrics["auc"], metrics["accuracy"], metrics["f1"])

    record = MetricRecord(
        model=model_name,
        track=TRACK,
        subset=SUBSET,
        train_with_dev=train_with_dev,
        variant=None,
        auc=metrics.get("auc"),
        acc=metrics.get("accuracy"),
        f1=metrics.get("f1"),
    )
    return [record]