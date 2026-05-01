from models.dkt.data.build_data import build_dkt_dataloaders
from config.consts import ITEM_EX, ITEM_TOK
from db.log_db import MetricRecord
from models.dkt.BertDKT import BertDKT
import logging
import argparse
import numpy as np
import torch
from models.dkt.DKT import DKT
from models.dkt.data.data import embed_sentence_matrix
from pipelines.common.checkpointing import save_torch, load_torch_ckpt
from pipelines.common.evaluation import save_binary_eval_predictions
from tqdm import tqdm

logger = logging.getLogger(__name__)

def parse_dkt_args(dkt_args=None):
    p = argparse.ArgumentParser(description="DKT Pipeline Args")
    p.add_argument("--slam-eval", action="store_true", default=False)
    p.add_argument("--use-prompts", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(dkt_args)
    
def run_dkt_pipeline(model_name, TRACK, SUBSET, train_with_dev, ITEM_LEVEL, epochs, eval_every, next_args, tag, save_every: int | None, resume_from: str | None):

    dkt_args = parse_dkt_args(next_args)

    logger.info(f"Running DKT pipeline for model {model_name}")
    logger.info(f"Running DKT with args: {dkt_args}")

    logger.info(f"Building dataloaders for track {TRACK}, subset {SUBSET}, train_with_dev={train_with_dev}")
    logger.info(f"ITEM_LEVEL: {ITEM_LEVEL}, use_prompts: {dkt_args.use_prompts}")
    #==== BUILD DATALOADER
    dkt_data = build_dkt_dataloaders(
        track=TRACK,
        variant="reprocessed",
        subset=SUBSET,
        item_level=item_level,
        train_with_dev=train_with_dev,
        batch_size=32,
        shuffle_train=True,
        use_prompts=dkt_args.use_prompts
    )
    
    #==== Build model

    if model_name == "dkt":
        model = DKT(len(dkt_data.item_map), emb_dim=128, head_dim=256)
    else:
        train_emb_matrix = embed_sentence_matrix(list(dkt_data.item_map.keys()))
        model = BertDKT(len(dkt_data.item_map), emb_dim=128, head_dim=256, emb_matrix=train_emb_matrix)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            torch.random.set_rng_state(rng_state["torch"].cpu())
            np.random.set_state(rng_state["numpy"])

            if rng_state.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng_state["cuda"]])
        

    # ==== Train

    rec_variant = "item_tok" if ITEM_LEVEL == ITEM_TOK else (
        "item_ex_" + f"{"prompts" if dkt_args.use_prompts else "slam_toks"}"
    )

    records = []
    loss_history = []

    pbar = tqdm(range(start_epoch, epochs + 1), desc="DKT Training Epochs")
    for epoch in pbar:
        loss = model.train_epoch(dkt_data.train_dataset, opt)
        loss_history.append(loss)

        pbar.set_postfix(loss=f"{loss:.4f}")

        #==== Evaluate

        if epoch == epochs or epoch % eval_every == 0:
            metrics = model.evaluate_metrics(dkt_data.eval_dataset, teacher_forcing= not dkt_args.slam_eval, return_detailed=True)
            logger.info(
                "Test Metrics | AUC=%s | AUC (seen)=%s | AUC (unseen)=%s | Accuracy=%.5f | F1=%.5f | n_seen=%d | n_unseen=%d",
                metrics["auc"],
                metrics["auc_seen"],
                metrics["auc_unseen"],
                metrics["accuracy"],
                metrics["f1"],
                metrics["n_seen"],
                metrics["n_unseen"],
            )

            record = MetricRecord(
                model=model_name,
                track=TRACK,
                subset=SUBSET,
                train_with_dev=train_with_dev,
                variant=rec_variant,
                auc=metrics.get("auc"),
                acc=metrics.get("accuracy"),
                f1=metrics.get("f1"),
                auc_seen=metrics.get("auc_seen"),
                auc_unseen=metrics.get("auc_unseen"),
                epochs=epoch,
                tag=tag
            )
            records.append(record)
        
            pred_path = save_binary_eval_predictions(
                record,
                y_true=metrics["targets"],
                probs=metrics["preds"],
                extra_cols={
                    "seen": metrics["seen"].astype(np.int8),
                    "uid": metrics["uid"],
                    "target_pos": metrics["target_pos"],
                },
            )
            logger.info(f"Saved evaluation predictions to {pred_path}")

            if save_every and epoch % save_every == 0 or epoch == epochs:
                ckpt_path = save_torch(model, opt, record)
                logger.info(f"Checkpoint at epoch {epoch} saved to {ckpt_path}")

    logger.info("DKT loss history: %s", loss_history)

    return records
