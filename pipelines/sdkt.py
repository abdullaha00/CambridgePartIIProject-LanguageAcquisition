# type: ignore

from __future__ import annotations

import argparse
import logging
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from db.log_db import MetricRecord
from models.SDKT.build_data import build_sdkt_dataloaders
from models.SDKT.SDKT_model import SDKTModel

logger = logging.getLogger(__name__)

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            result[k] = {k2: v2.to(device) for k2, v2 in v.items()}
        else:
            result[k] = v.to(device)
    return result

def parse_sdkt_args(sdkt_args=None):
    p = argparse.ArgumentParser(description="seqDKT Pipeline Args")
    p.add_argument("--variant", type=str, default="reprocessed")
    p.add_argument("--emb-dim", type=int, default=300)
    p.add_argument("--hid-dim", type=int, default=128)
    p.add_argument("--meta-emb-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    return p.parse_args(sdkt_args)

def run_sdkt_pipeline(
     model_name: str,
        TRACK: str,
        SUBSET: Optional[int],
        train_with_dev: bool,
        EPOCHS: int,
        eval_every: int,
        next_args: Optional[list[str]] = None
) -> list[MetricRecord]:
    
    sdkt_args = parse_sdkt_args(next_args)
    logger.info(f"Running seqDKT with args: {sdkt_args}")

    data_bundle = build_sdkt_dataloaders(
        track=TRACK,
        subset=SUBSET,
        train_with_dev=train_with_dev,
        variant=sdkt_args.variant,
        batch_size=sdkt_args.batch_size
    )

    model = SDKTModel(
        num_toks=data_bundle.vocabs.num_tokens,
        meta_vocab_sizes=data_bundle.vocabs.meta_vocab_sizes,
        emb_dim=sdkt_args.emb_dim,
        hid_dim=sdkt_args.hid_dim,
        meta_emb_dim=sdkt_args.meta_emb_dim,
        dropout=sdkt_args.dropout
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=sdkt_args.lr)
    records = []

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(data_bundle.train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
            batch = move_batch_to_device(batch, model.device)
            optimizer.zero_grad()
            logits, targets, mask = model(batch, teacher_forcing=True)
            loss = model.loss_fn(logits, targets, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_bundle.train_dl)
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

        if epoch % eval_every == 0:
            metrics = model.evaluate(data_bundle.eval_dl)
            logger.info(f"Epoch {epoch} - Eval Metrics: {metrics}")
            records.append(MetricRecord(
                model=model_name,
                track=TRACK,
                subset=SUBSET,
                epochs=epoch,
                train_with_dev=train_with_dev,
                auc=metrics.get("auc"),
                acc=metrics.get("accuracy"),
                f1=metrics.get("f1")
            ))

    return records
