from functools import partial
from data_processing.data_parquet import load_train_and_eval_df
from models.text_kt.common.data import collapse_to_exercise, build_user_sequences_text, history_text
from models.text_kt.lmkt.build_data import build_lmkt_dataloaders
import logging
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.text_kt.qg.data import QGDataset

from models.text_kt.qg.data import qg_collate
from models.text_kt.lmkt.lmkt import LMKTModel
from models.text_kt.qg.qg import LMKTQG

logger = logging.getLogger(__name__)

def parse_qg_args(qg_args=None):
    # PARSE SPECIFIC FLAGS
    p = argparse.ArgumentParser(description="QG Pipeline Args")
    args = p.parse_args(qg_args)
    return args

def run_qg_pipeline(TRACK,SUBSET,train_with_dev, EPOCHS):

    logger.info("Running QG pipeline")

    # ===== TRAIN LMKT

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
    
    for epoch in tqdm(range(EPOCHS), desc="LMKT Training Epochs"):
        loss = model.train_one_epoch(lmkt_data.train_dataset, opt)
        logger.info(f"Epoch {epoch} loss: {loss}")
    
    #==== Evaluate

    metrics = model.evaluate_metrics(lmkt_data.eval_histories)
    logger.info("Test Metrics | AUC=%.5f | Accuracy=%.5f | F1=%.5f", 
                metrics["auc"], metrics["accuracy"], metrics["f1"])

    #--- FREEZE MODEL

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    #============ BUILD QG

    logger.info("Building QG Model")

    df_train, df_eval = load_train_and_eval_df(TRACK, "minimal", train_with_dev, subset=SUBSET)
    df_train_ex, df_eval_ex = collapse_to_exercise(df_train), collapse_to_exercise(df_eval)

    train_histories = build_user_sequences_text(df_train_ex)
    held_out_qs = df_eval_ex["ref_ans"].unique().tolist()

    def difficulty_fn(prefix_text: str, q_text: str) -> float:
        return model.p_y_given_question(prefix_text, q_text)
    
    # Generator model

    qg_model = LMKTQG(model_name = "gpt2")
    tok = qg_model.tokenizer

    qg_dataset = QGDataset(
        histories=train_histories,
        held_out_qs=held_out_qs,
        tokenizer=tok,
        difficulty_fn=difficulty_fn
    )

    qg_dataloader = DataLoader(
        qg_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=partial(qg_collate, pad_token_id=tok.pad_token_id)
    )

    # ====== TRAIN generator

    logger.info("Training QG Model")
    opt = torch.optim.Adam(qg_model.parameters(), lr=1e-4)

    qg_model.train()

    for ep in tqdm(range(EPOCHS), desc="QG Training Epochs"):
        total_loss = 0.0
        for batch in tqdm(qg_dataloader, desc="QG Batches"):
            opt.zero_grad()
            
            out = qg_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                difficulty=batch["difficulty"]
            )

            loss = out.loss
            loss.backward()
            opt.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(qg_dataloader)
        logger.info(f"Epoch {ep} QG Loss: {avg_loss}")

