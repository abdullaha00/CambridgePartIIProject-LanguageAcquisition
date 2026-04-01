import argparse
import logging

import pandas as pd
import torch
from transformers import get_linear_schedule_with_warmup

from data_processing.data_parquet import get_parquet
from db.log_db import MetricRecord
from models.adaptive_qg.adaptive_data import MAX_SEQ_LEN, build_word_vocab, load_user_data
from models.adaptive_qg.aqg_kt import (
    DKT,
    HIDDEN_SIZE,
    NUM_LAYERS,
    WARMUP_RATE,
    evaluate_adaptive_qg_dkt,
    train_dkt_epoch,
)
from pipelines.common.checkpointing import save_torch
from pipelines.common.common import mk_record

logger = logging.getLogger(__name__)

def parse_aqg_kt_args(aqg_args=None):
    p = argparse.ArgumentParser(description="Adaptive QG KT Pipeline Args")
    p.add_argument("--variant", type=str, default="reprocessed") # original/reprocessed
    p.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-rate", type=float, default=WARMUP_RATE)
    p.add_argument("--positive-weight", type=float, default=3.0)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--epochs", type=int, default=5)
    return p.parse_args(aqg_args)

def run_aqg_dkt_pipeline(
        model_name: str,
        track: str,
        subset: int | None,
        train_with_dev: bool,
        EPOCHS: int,
        eval_every: int,
        next_args: list[str] | None = None,
        tag: str | None = None,
        save_every: int | None = None,
    ) -> list[MetricRecord]:
        
    aqg_args = parse_aqg_kt_args(next_args)
    if save_every is None:
        save_every = eval_every
    
    # === GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting Adaptive QG KT Pipeline with args: %s", vars(aqg_args))
    
    # ==== DATA LOADING AND PREPARATION
    train_df = get_parquet(track, "train", aqg_args.variant, subset=subset)
    train_users = train_df["user_id"].unique()

    dev_df = get_parquet(track, "dev", aqg_args.variant, user_filter=train_users)
    test_df = get_parquet(track, "test", aqg_args.variant, user_filter=train_users)

    train_df["split"] = "train"
    dev_df["split"] = "dev"
    test_df["split"] = "test"

    combined_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    train_splits = [1, 2] 
    eval_splits = [3]

    user_data_list, seen_texts = load_user_data(combined_df)
    word_vocab = build_word_vocab(train_df)

    assert user_data_list, "No user data found after loading. Please check the input data and preprocessing steps." 

    # === MODEL 
    dkt = DKT(len(word_vocab), hidden_size=aqg_args.hidden_size, num_layers=aqg_args.num_layers).to(device)
    
    opt = torch.optim.AdamW(dkt.parameters(), lr=aqg_args.lr) # Weighted Adam with decoupled weight decay NOTE

    total_steps = (EPOCHS * len(user_data_list))
    warmup_steps = int(aqg_args.warmup_rate * total_steps) # 10% of total steps for warmup

    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    records: list[MetricRecord] = []

    # === TRAINING LOOP
    for epoch in range(1, EPOCHS+1):
        
        avg_loss = train_dkt_epoch(
            dkt=dkt,
            user_data_list=user_data_list,
            opt=opt,
            scheduler=scheduler,
            device=device,
            target_split=train_splits,
            max_length=aqg_args.max_seq_len,
            positive_weight=aqg_args.positive_weight,
        )

        logger.info("Epoch %d | train_loss=%.4f", epoch, avg_loss/len(user_data_list))

        # === EVALUATION

        metrics = evaluate_adaptive_qg_dkt(
            dkt=dkt,
            user_data_list=user_data_list,
            target_split=eval_splits,
            device=device,
            max_length=aqg_args.max_seq_len,
        )

        tok_metrics = metrics["token_level"]
        q_metrics = metrics["question_level"]

        logger.info("users: %d | tokens: %d | questions: %d", metrics["users"], tok_metrics["count"], q_metrics["count"])

        logger.info("TOKEN-LEVEL EVAL:"
            "Epoch %d | eval_loss=%.4f | AUC=%.5f | AUC (seen)=%.5f | AUC (unseen)=%.5f | Accuracy=%.5f | F1=%.5f",
            epoch,
            avg_loss,
            tok_metrics["auc_last_w"],
            tok_metrics["seen_auc_last_w"],
            tok_metrics["unseen_auc_last_w"],
            tok_metrics["acc_last_w"],
            tok_metrics["f1_last_w"],
        )

        logger.info("QUESTION-LEVEL EVAL:"
            "Epoch %d | AUC=%.5f | AUC (seen)=%.5f | AUC (unseen)=%.5f | Accuracy=%.5f | F1=%.5f",
            epoch,
            q_metrics["auc_last_q"],
            q_metrics["seen_auc_last_q"],
            q_metrics["unseen_auc_last_q"],
            q_metrics["acc_last_q"],
            q_metrics["f1_last_q"],
        )

        r1 = MetricRecord(
            model=model_name + "_token_level",
            track=track,
            subset=subset if subset is not None else -1,
            train_with_dev=train_with_dev,
            variant="",
            tag=tag,
            epochs=epoch,
            auc=tok_metrics["auc_last_w"],
            acc=tok_metrics["acc_last_w"],
            f1=tok_metrics["f1_last_w"],
        )

        r2 = MetricRecord(
            model=model_name + "_question_level",
            track=track,
            subset=subset if subset is not None else -1,
            train_with_dev=train_with_dev,
            variant="",
            tag=tag,
            epochs=epoch,
            auc=q_metrics["auc_last_q"],
            acc=q_metrics["acc_last_q"],
            f1=q_metrics["f1_last_q"],
        )
        
        records.append(r1)
        records.append(r2)

    # == SAVING
    if subset is None and (save_every and epoch % save_every == 0 or epoch == EPOCHS):
        save_path = save_torch(
            model=dkt,
            opt=opt,
            rec=r1,
            extra={
                "word_vocab": word_vocab,
                "seen_texts": sorted(seen_texts),
                "config": vars(aqg_args),
                "eval_metrics": metrics,
            },
        )
        logger.info("Checkpoint at epoch %d saved to %s", epoch, save_path)

    return records