from __future__ import annotations

import argparse
import logging
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
from db.log_db import MetricRecord
from models.fa_bilstm.build_data import build_fab_dataloaders
from models.fa_bilstm.model import FABModel
from pipelines.common.checkpointing import load_torch_ckpt, save_torch
from pipelines.common.evaluation import binary_metrics_score, save_binary_eval_predictions

logger = logging.getLogger(__name__)

# ==== CONFIG 
DATA_VARIANT = "reprocessed"
FEATURE_SET = "exercise"
BATCH_SIZE = 64

TOKEN_EMB_DIM = 750
FEATURE_EMB_DIM = 32
FEATURE_MERGE = "concat"
HIDDEN_DIM = 750
HIDDEN_DIM_IS_TOTAL = False
NUM_LAYERS = 1
DROPOUT = 0.2
INPUT_DROPOUT = None

USE_HISTORY_FEATURES = True
USE_RICH_HISTORY_FEATURES = True
USE_POSITION_FEATURES = True
USE_NUMERIC_METADATA = True
NORMALIZE_NUMERIC_FEATURES = True

POSITIVE_CLASS_WEIGHT = 2.5
CLASSIFIER_HIDDEN_DIM = 128
CLASSIFIER_DROPOUT = 0.1

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 5.0


def parse_fab_args(sdkt_args=None):
    p = argparse.ArgumentParser(description="FA-BiLSTM Pipeline Args")
    p.add_argument("--variant", type=str, default=DATA_VARIANT)
    p.add_argument("--feature-set", type=str, default=FEATURE_SET)
    p.add_argument("--emb-dim", type=int, default=TOKEN_EMB_DIM)
    p.add_argument("--hid-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--meta-emb-dim", type=int, default=FEATURE_EMB_DIM)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--history-features", action="store_true", default=USE_HISTORY_FEATURES)
    p.add_argument("--rich-history-features", action="store_true", default=USE_RICH_HISTORY_FEATURES)
    p.add_argument("--position-features", action="store_true", default=USE_POSITION_FEATURES)
    p.add_argument("--numeric-metadata", action="store_true", default=USE_NUMERIC_METADATA)
    p.add_argument("--normalize-numeric-features", action="store_true", default=NORMALIZE_NUMERIC_FEATURES)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(sdkt_args)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        # handle dict of tensors (e.g. feat_ids)
        elif isinstance(value, dict):
            out[key] = {k: v.to(device) for k, v in value.items()}
        else:
            out[key] = value
    return out

def train_epoch(model, dataloader, opt, device, grad_clip=None, show_progress=True):
    model.train()

    sum_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False, disable=not show_progress):
        batch = move_batch_to_device(batch, device)
        
        opt.zero_grad()
        loss = model.loss(batch)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sum_loss += loss.item()

    return sum_loss / max(len(dataloader), 1)

def evaluate(model, dataloader, device, show_progress=True) -> dict:
    model.eval()
    all_probs = []
    all_labels = []

    detailed = {"user_id": [], "tok_id": [], "ex_key": [], "target_pos": []}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False, disable=not show_progress):
            batch_device = move_batch_to_device(batch, device)
            probs = model.predict(batch_device)

            mask = batch_device["mask"]
            all_probs.extend(probs[mask].cpu().numpy())
            all_labels.extend(batch_device["labels"][mask].cpu().numpy())

            # === APPEND DETAILED INFO
            for i, seq_mask in enumerate(mask):
                T = int(seq_mask.sum().item())
                detailed["user_id"].extend([batch["user_ids"][i]] * T)
                detailed["ex_key"].extend([batch["ex_keys"][i]] * T)
                detailed["tok_id"].extend(batch["tok_ids"][i][:T].tolist())
                detailed["target_pos"].extend(batch["target_pos"][i][:T].tolist())

    probs_np = np.asarray(all_probs)
    labels_np = np.asarray(all_labels)

    metrics = binary_metrics_score(labels_np, probs_np)
    metrics["preds"] = probs_np
    metrics["targets"] = labels_np
    for k, v in detailed.items():
        metrics[k] = np.asarray(v)

    return metrics  

def run_fa_bilstm_pipeline(
    track: str,
    subset: Optional[int],
    train_with_dev: bool,
    epochs: int,
    eval_every: int,
    next_args: Optional[list[str]] = None,
    tag: Optional[str] = None,
    save_every: Optional[int] = None,
    resume_from: Optional[str] = None,
) -> list[MetricRecord]:

    args = parse_fab_args(next_args)

    logger.info(f"Running FA-Bilstm with args: {args}")
    logger.info(f"Building dataloaders for track {track}, subset {subset}, train_with_dev={train_with_dev}")

    #==== DATALOADERS
    data = build_fab_dataloaders(
        track=track,
        variant=args.variant,
        subset=subset,
        train_with_dev=train_with_dev,
        feature_set=args.feature_set,
        batch_size=args.batch_size,
        user_history_features=args.history_features,
        global_history_features=args.rich_history_features,
        position_features=args.position_features,
        numeric_metadata=args.numeric_metadata,
        normalise_numeric_features=args.normalize_numeric_features,
    )

    #=== MODEL and opt
    model = FABModel(
        tok_vocab_size=data.vocabs.vocab_size,
        feat_vocab_sizes=data.vocabs.feature_vocab_sizes,
        tok_emb_dim=args.emb_dim,
        feat_emb_dim=args.meta_emb_dim,
        hidden_dim=args.hid_dim,
        num_layers=NUM_LAYERS,
        dropout=args.dropout,
        positive_class_weight=POSITIVE_CLASS_WEIGHT,
        numeric_feat_dim=len(data.vocabs.numeric_feature_cols),
        classifier_hidden_dim=CLASSIFIER_HIDDEN_DIM,
        classifier_dropout=CLASSIFIER_DROPOUT,
    )

    device = model.device
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    start_epoch = 1
    
    # === LOAD FROM CHECKPOINT (if resuming)
    if resume_from is not None:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        ckpt = load_torch_ckpt(resume_from) 

        # ensure model aligns with checkpoint
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict") is not None:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        stored_ep = ckpt.get("epoch")
        if stored_ep is not None:       
            start_epoch = stored_ep + 1

        # restore rng state
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            torch.random.set_rng_state(rng_state["torch"].cpu())
            np.random.set_state(rng_state["numpy"])

            if rng_state.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng_state["cuda"]])
    #==

    records = []
    rec_variant = f"fa_bilstm_{args.variant}_{args. feature_set.replace(',', '-')}"

    #==== TRAIN
    for epoch in range(start_epoch, epochs + 1):
        loss = train_epoch(model, data.train_dl, opt, device, GRAD_CLIP, not args.no_progress)
        logger.info(f"Epoch {epoch} - Train Loss: {loss:.4f}")
        if epoch != epochs and epoch % eval_every != 0:
            continue

        metrics = evaluate(model, data.eval_dl, device, not args.no_progress)
        logger.info("Epoch %d - Eval AUC=%.5f Accuracy=%.5f F1=%.5f", epoch, metrics["auc"], metrics["accuracy"], metrics["f1"])
        rec = MetricRecord(
            model="fa-bilstm",
            track=track,
            subset=subset,
            train_with_dev=train_with_dev,
            variant=rec_variant,
            epochs=epoch,
            auc=metrics["auc"],
            acc=metrics["accuracy"],
            f1=metrics["f1"],
            tag=tag,
        )
        records.append(rec)
        pred_path = save_binary_eval_predictions(
            rec,
            y_true=metrics["targets"],
            probs=metrics["preds"],
            extra_cols={
                "user_id": metrics["user_id"],
                "tok_id": metrics["tok_id"],
                "ex_key": metrics["ex_key"],
                "target_pos": metrics["target_pos"],
            },
        )
        logger.info("Saved FA BiLSTM evaluation predictions to %s", pred_path)
        if save_every and (epoch % save_every == 0 or epoch == epochs):
            ckpt_path = save_torch(model, opt, rec)
            logger.info("Checkpoint at epoch %d saved to %s", epoch, ckpt_path)

    return records
