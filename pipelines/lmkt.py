from models.modular_qg.lmkt.build_data import build_lmkt_dataloaders
import logging
import argparse
import numpy as np
import torch
from tqdm import tqdm

from db.log_db import MetricRecord
from models.modular_qg.lmkt.lmkt import LMKTModel
from pipelines.common.checkpointing import load_torch_ckpt, save_torch
from pipelines.common.evaluation import save_binary_eval_predictions

logger = logging.getLogger(__name__)

# Add any LM-KT specific arguments here, future use (none currently)
def parse_lmkt_args(dkt_args=None):
    p = argparse.ArgumentParser(description="LM-KT Pipeline Args")
    args = p.parse_args(dkt_args)
    return args

def run_lmkt_pipeline(TRACK, SUBSET, train_with_dev, EPOCHS, eval_every: int = 1, save_every: int | None = None, next_args=None, resume_from=None, tag=None) -> list[MetricRecord]:

    if save_every is None:
        save_every = eval_every

    logger.info("Running LM-KT pipeline")

    logger.info(f"Building dataloaders for track {TRACK}, subset {SUBSET}, train_with_dev={train_with_dev}")
    
    #==== Build model

    model = LMKTModel()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
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
    
    for epoch in tqdm(range(start_epoch, EPOCHS + 1), desc="LMKT Training Epochs"):
        loss = model.train_one_epoch(lmkt_data.train_dataloader, opt)
        logger.info(f"Epoch {epoch} loss: {loss}")

        if epoch == EPOCHS or epoch % eval_every == 0:
            metrics = model.evaluate_metrics(
                lmkt_data.eval_histories, 
                lmkt_data.pref_ns,
                lmkt_data.train_seen_prompts,
                return_detailed=True,
            )
            
            logger.info("Epoch %d | AUC=%.5f | AUC (seen)=%s | AUC (unseen)=%s | Accuracy=%.5f | F1=%.5f",
                        epoch, metrics["auc"], metrics["auc_seen"], metrics["auc_unseen"], metrics["accuracy"], metrics["f1"])

            rec = MetricRecord(
                model="lmkt",
                track=TRACK,
                subset=SUBSET,
                train_with_dev=train_with_dev,
                variant=None,
                auc=metrics.get("auc"),
                acc=metrics.get("accuracy"),
                f1=metrics.get("f1"),
                auc_seen=metrics.get("auc_seen"),
                auc_unseen=metrics.get("auc_unseen"),
                epochs=epoch,
                tag=tag,
            )
            records.append(rec)

            pred_path = save_binary_eval_predictions(
                rec,
                y_true=metrics["targets"],
                probs=metrics["preds"],
                pred_labels=metrics["pred_labels"],
                extra_cols={
                    "prob_n": metrics["preds_n"],
                    "seen": metrics["seen"].astype(np.int8),
                    "uid": metrics["uid"],
                    "target_pos": metrics["target_pos"],
                    "prompt_text": metrics["prompt_text"],
                },
            )
            logger.info(f"Saved evaluation predictions to {pred_path}")

            if save_every and epoch % save_every == 0 or epoch == EPOCHS:
                save_path = save_torch(model, opt, rec)
                logger.info(f"Checkpoint at epoch {epoch} saved to {save_path}")

    return records
