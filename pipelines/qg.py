from functools import partial


from data_processing.data_parquet import load_train_and_eval_df
from db.log_db import MetricRecord
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
from models.text_kt.qg.evaluate import run_qg_evaluation
from models.text_kt.qg.qg import LMKTQG
from pipelines.common.checkpointing import load_torch_ckpt

logger = logging.getLogger(__name__)

def parse_qg_args(qg_args=None):
    # PARSE SPECIFIC FLAGS
    p = argparse.ArgumentParser(description="QG Pipeline Args")
    p.add_argument("-l", "--load-path", type=str, default=None, help="Path to load LMKT model from (if any)")
    p.add_argument("-e", "--epochs", type=int, default=2)
    args = p.parse_args(qg_args)
    return args

def run_qg_pipeline(TRACK,SUBSET,train_with_dev, EPOCHS, extra_args=None):
    
    logger.info("Running QG pipeline")
    args = parse_qg_args(extra_args)

    # ===== TRAIN LMKT

    logger.info(f"Building dataloaders for track {TRACK}, subset {SUBSET}, train_with_dev={train_with_dev}")


    if args.load_path is not None:
        logger.info(f"Loading LMKT model from {args.load_path}")
        ckpt = load_torch_ckpt(args.load_path)
        model = LMKTModel()
        
        # ==== THE LMKT CKPT IS NOT TRAINED WITH <G> TOKEN, SO
        ckpt_vocab = ckpt["model_state_dict"]["model.transformer.wte.weight"].shape[0]
        if len(model.tokenizer) != ckpt_vocab:
            logger.warning("Vocab mismatch: checkpoint=%d, current=%d — resizing to match checkpoint",
                           ckpt_vocab, len(model.tokenizer))
            model.model.resize_token_embeddings(ckpt_vocab)

        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("LMKT model loaded successfully (epoch=%s, auc=%.4f)",
                     ckpt.get("epoch", "?"), ckpt.get("metrics", {}).get("auc", float("nan")))
    else:
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
            loss = model.train_one_epoch(lmkt_data.train_dataloader, opt)
            logger.info(f"Epoch {epoch} loss: {loss}")
        
        #==== Evaluate (commented out)
        # metrics = model.evaluate_metrics(lmkt_data.eval_histories, lmkt_data.pref_ns)
        # logger.info("Test Metrics | AUC=%.5f | Accuracy=%.5f | F1=%.5f", 
        #             metrics["auc"], metrics["accuracy"], metrics["f1"])

    #--- FREEZE MODEL & MOVE TO CPU to save GPU memory for QG
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.cpu()
    torch.cuda.empty_cache()

    #============ BUILD QG
    logger.info("Building QG Model")

    df_train, df_held = load_train_and_eval_df(TRACK, "minimal", train_with_dev, subset=SUBSET)
    dft_prompts, dfh_prompts = load_train_and_eval_df(TRACK, "prompt", train_with_dev, subset=SUBSET)

    # Restrict to reverse_translate tasks
    df_train = df_train[df_train["format"] == "reverse_translate"]
    df_held = df_held[df_held["format"] == "reverse_translate"]

    df_train_ex, df_held_ex = collapse_to_exercise(df_train), collapse_to_exercise(df_held)

    # Merge prompt text into collapsed DataFrames
    df_train_ex = df_train_ex.merge(dft_prompts[["ex_key", "prompt"]], on="ex_key", how="left")
    df_held_ex = df_held_ex.merge(dfh_prompts[["ex_key", "prompt"]], on="ex_key", how="left")

    # ==== WE use df_held as a held-out set 

    train_histories = build_user_sequences_text(df_train_ex)
    held_histories = build_user_sequences_text(df_held_ex)

    held_out_qs = df_held_ex["prompt"].unique().tolist()

    # === LOGGING
    train_users = set(df_train_ex["user_id"].unique())
    held_users = set(df_held_ex["user_id"].unique())
    logger.info(
        "QG held-out split=%s | held-out users=%d | overlap_with_lmkt_train=%d | held-out questions=%d",
        "test" if train_with_dev else "dev",
        len(held_users),
        len(train_users & held_users),
        len(held_out_qs),
    )

    @torch.no_grad()
    def difficulty_fn(prefix_texts: list, q_texts: list) -> torch.Tensor:
        
        # Move model to GPU (if used), then back to CPU after inference to save memory
        model.to(model.device)
        result = model.p_y_given_question_batch(prefix_texts, q_texts)
        model.cpu()
        torch.cuda.empty_cache()
        return result
    
    # ==== BUILD AND TRAIN QG MODEL

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
        batch_size=2,
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

            # gradient clipping
            
            grad = torch.nn.utils.clip_grad_norm_(qg_model.parameters(), max_norm=1.0)
            
            if grad > 1.0:
                logger.warning(f"Gradient norm {grad:.2f} exceeds max norm, clipping applied.")

            opt.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(qg_dataloader)
        logger.info(f"Epoch {ep} QG Loss: {avg_loss}")

    # === EVALUATE

    logger.info("Evaluating QG model (generation + difficulty targeting)")
    qg_model.eval()

    # evaluation handles gpu/cpu switching internally
    qg_model.cpu()
    torch.cuda.empty_cache()

    qg_metrics = run_qg_evaluation(
        qg_model=qg_model,
        lmkt_model=model,
        eval_histories=held_histories,
        train_questions=df_train_ex["prompt"].unique().tolist(),
    )

    logger.info("QG Evaluation | Diff MAE=%.4f | Diff Corr=%.4f | Distinct-1=%.4f | Distinct-2=%.4f | Novelty=%.4f | N=%d",
                qg_metrics.get("d_mae", float("nan")),
                qg_metrics.get("d_pearson_corr", float("nan")),
                qg_metrics.get("distinct_1", 0),
                qg_metrics.get("distinct_2", 0),
                qg_metrics.get("novelty", 0),
                qg_metrics.get("n_generated_questions", 0))
    
    #TODO: augment DB to accept QG metrics
    return []

    
