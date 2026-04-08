from numpy import record
from models.modular_qg.lmkt.build_data import build_lmkt_dataloaders
import logging
import argparse
import torch
from tqdm import tqdm

from db.log_db import MetricRecord
from models.modular_qg.lmkt.lmkt import LMKTModel
from pipelines.common.checkpointing import save_torch

logger = logging.getLogger(__name__)

# Add any LM-KT specific arguments here, future use (none currently)
def parse_lmkt_args(dkt_args=None):
    p = argparse.ArgumentParser(description="LM-KT Pipeline Args")
    args = p.parse_args(dkt_args)
    return args

def run_lmkt_pipeline(TRACK, SUBSET, train_with_dev, EPOCHS, eval_every: int = 1, save_every: int | None = None):

    if save_every is None:
        save_every = eval_every

    logger.info("Running LM-KT pipeline")

    logger.info(f"Building dataloaders for track {TRACK}, subset {SUBSET}, train_with_dev={train_with_dev}")
    
    #==== Build model

    model = LMKTModel()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

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
    records = []
    
    for epoch in tqdm(range(1, EPOCHS + 1), desc="LMKT Training Epochs"):
        loss = model.train_one_epoch(lmkt_data.train_dataloader, opt)
        logger.info(f"Epoch {epoch} loss: {loss}")

        if epoch % eval_every == 0:
            metrics = model.evaluate_metrics(lmkt_data.eval_histories, lmkt_data.pref_ns)
            logger.info("Epoch %d | AUC=%.5f | Accuracy=%.5f | F1=%.5f",
                        epoch, metrics["auc"], metrics["accuracy"], metrics["f1"])

            rec = MetricRecord(
                model="lmkt",
                track=TRACK,
                subset=SUBSET,
                train_with_dev=train_with_dev,
                variant=None,
                auc=metrics.get("auc"),
                acc=metrics.get("accuracy"),
                f1=metrics.get("f1"),
                epochs=epoch,
            )
            records.append(rec)

            if save_every and epoch % save_every == 0 or epoch == EPOCHS:
                save_path = save_torch(model, opt, rec)
                logger.info(f"Checkpoint at epoch {epoch} saved to {save_path}")

    return records