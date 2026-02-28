

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
from models.SDKT.VDKT import VDKTModel
from models.SDKT.fasttext import load_fasttext_vecs
from pipelines.common.checkpointing import save_torch, load_torch_ckpt
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
    p.add_argument("--no-fasttext", action="store_true")

    return p.parse_args(sdkt_args)

def run_sdkt_pipeline(
    model_name: str,
    TRACK: str,
    SUBSET: Optional[int],
    train_with_dev: bool,
    EPOCHS: int,
    eval_every: int,
    next_args: Optional[list[str]] = None,
    tag: Optional[str] = None,
    save_every: Optional[int] = None,
    resume_from: Optional[str] = None,
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

    if not sdkt_args.no_fasttext:
        logger.info("Loading FastText embeddings...")
        emb_mat = load_fasttext_vecs(
            lang=TRACK.split("_")[0],
            vocab=data_bundle.vocabs.token_vocab,
            emb_dim=sdkt_args.emb_dim,
            out_dir="./fasttext",
            subset=SUBSET,
            cache=True
        )

    # == MODEL
    if model_name == "sdkt":
        model = SDKTModel(
            num_toks=data_bundle.vocabs.num_tokens,
            meta_vocab_sizes=data_bundle.vocabs.meta_vocab_sizes,
            emb_dim=sdkt_args.emb_dim,
            hid_dim=sdkt_args.hid_dim,
            meta_emb_dim=sdkt_args.meta_emb_dim,
            dropout=sdkt_args.dropout,
            emb_matrix=emb_mat if not sdkt_args.no_fasttext else None
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    elif model_name == "vdkt":
        model = VDKTModel(
            num_toks=data_bundle.vocabs.num_tokens,
            meta_vocab_sizes=data_bundle.vocabs.meta_vocab_sizes,
            emb_dim=sdkt_args.emb_dim,
            hid_dim=sdkt_args.hid_dim,
            meta_emb_dim=sdkt_args.meta_emb_dim,
            dropout=sdkt_args.dropout,
            emb_matrix=emb_mat if not sdkt_args.no_fasttext else None
        ).to("cuda" if torch.cuda.is_available() else "cpu")


    optimizer = torch.optim.Adam(model.parameters(), lr=sdkt_args.lr)
    start_epoch = 1

   
    # === LOAD FROM CHECKPOINT (if resuming)
    if resume_from is not None:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        ckpt = load_torch_ckpt(resume_from)

        # ensure model aligns with checkpoint
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        stored_ep = ckpt.get("epoch")
        if stored_ep is not None:       
            start_epoch = stored_ep + 1 
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            torch.random.set_rng_state(rng_state["torch"])
            np.random.set_state(rng_state["numpy"])
            if rng_state.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            
        global_step = ckpt.get("extra", {}).get("final_global_step", 0)
        total_steps = ckpt.get("extra", {}).get("total_steps", EPOCHS * len(data_bundle.train_dl))

        logger.info(f"Resumed from epoch {start_epoch - 1}, continuing from epoch {start_epoch}")
    else:
        global_step = 0
        total_steps = EPOCHS * len(data_bundle.train_dl)
    
    records = []

    for epoch in range(start_epoch, EPOCHS+1):  
        model.train()
        total_loss = 0.0

        for batch in tqdm(data_bundle.train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
            batch = move_batch_to_device(batch, model.device)
            optimizer.zero_grad()

            if model_name == "sdkt":
                logits, targets, mask = model(batch, teacher_forcing=True)
                loss = model.loss_fn(logits, targets, mask)
            else:
                preds, targets, mask, kl_loss = model.forward_vdkt(batch, teacher_forcing=True)

                kl_weight = model.kl_annealing_weight(
                    step=global_step, 
                    total_steps=total_steps,
                    alpha_r=0.5,
                    beta_r=0.15,
                    max_weight=1.0
                )
                loss = model.elbo_loss(preds, targets, mask, kl_loss, weight=kl_weight)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            global_step += 1

        avg_loss = total_loss / len(data_bundle.train_dl)
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

        if epoch % eval_every == 0:
            metrics = model.evaluate(data_bundle.eval_dl)
            logger.info(f"Epoch {epoch} - Eval Metrics: {metrics}")
            rec = MetricRecord(
                model=model_name,
                track=TRACK,
                subset=SUBSET,
                epochs=epoch,
                train_with_dev=train_with_dev,
                auc=metrics.get("auc"),
                acc=metrics.get("accuracy"),
                f1=metrics.get("f1"),
                variant=sdkt_args.variant,
                tag=tag,
            )
            records.append(rec)

            if SUBSET is None and (save_every and epoch % save_every == 0 or epoch == EPOCHS):
                if model_name == "vdkt":
                    extra_data = {
                        "total_steps": total_steps,
                        "final_global_step": global_step
                    }
                else:
                    extra_data = None
                    
                ckpt_path = save_torch(model, optimizer, rec, extra=extra_data)
                logger.info(f"Checkpoint at epoch {epoch} saved to {ckpt_path}")


    return records
