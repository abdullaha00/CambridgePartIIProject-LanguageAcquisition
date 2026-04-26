from functools import partial

from data_processing.data_parquet import get_parquet
from db.log_db import GenerationRecord
from models.modular_qg.common.data import collapse_to_exercise, build_user_sequences_text
from models.modular_qg.lmkt.build_data import build_lmkt_dataloaders
import logging
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.modular_qg.qg.data import QGDataset

from models.modular_qg.qg.data import qg_collate
from models.modular_qg.lmkt.lmkt import LMKTModel
from models.modular_qg.qg.evaluate import run_qg_evaluation
from models.modular_qg.qg.qg import LMKTQG
from pipelines.common.checkpointing import load_torch_ckpt

logger = logging.getLogger(__name__)

def parse_qg_args(qg_args=None):
    # PARSE SPECIFIC FLAGS
    p = argparse.ArgumentParser(description="QG Pipeline Args")
    p.add_argument("-l", "--load-path", type=str, default=None, help="Path to load LMKT model from (if any)")
    p.add_argument("-e", "--epochs", type=int, default=2)
    args = p.parse_args(qg_args)
    return args

def get_reverse_translate_ex(track: str, split: str, subset=None, user_filter=None):
    df = get_parquet(track, split, "minimal", subset=subset, user_filter=user_filter)
    df_prompts = get_parquet(track, split, "prompt")

    df = df[df["format"] == "reverse_translate"]
    df_ex = collapse_to_exercise(df)
    df_ex = df_ex.merge(df_prompts[["ex_key", "prompt"]], on="ex_key", how="left")

    missing = df_ex["prompt"].isna().sum()
    assert missing == 0, f"Missing prompts in {split} split after merge: {missing}"

    return df_ex


def run_qg_pipeline(track,SUBSET,train_with_dev, EPOCHS, extra_args=None, tag=None):
    
    logger.info("Running QG pipeline")
    args = parse_qg_args(extra_args)

    # ===== TRAIN LMKT

    logger.info(f"Building dataloaders for track {track}, subset {SUBSET}, train_with_dev={train_with_dev}")

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
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

        #==== BUILD DATALOADER
        lmkt_data = build_lmkt_dataloaders(
            track=track,
            variant="minimal",
            subset=SUBSET,
            train_with_dev=False,
            tokenizer=model.tokenizer,
            batch_size=2,
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

    df_lmkt_train_ex = get_reverse_translate_ex(track, "train", subset=SUBSET)
    train_users = df_lmkt_train_ex["user_id"].unique()

    if train_with_dev:
        logger.info("train-with-dev is default for QG!;" \
        "Intended splits: LMKT_train: train, QG_train: dev", "QG_dev: test")

    qg_train_split = "dev"
    qg_eval_split = "test"

    df_qg_train_ex = get_reverse_translate_ex(track, qg_train_split, user_filter=train_users)
    df_qg_eval_ex = get_reverse_translate_ex(track, qg_eval_split, user_filter=train_users)

    qg_train_histories = build_user_sequences_text(df_qg_train_ex)
    qg_eval_histories = build_user_sequences_text(df_qg_eval_ex)

    held_out_qs = df_qg_train_ex["prompt"].unique().tolist() # used to train 

    # === LOGGING
    qg_train_users = set(df_qg_train_ex["user_id"].unique())
    qg_eval_users = set(df_qg_eval_ex["user_id"].unique())
    logger.info(
        "QG train split=%s | QG eval split=%s | QG train users=%d | QG eval users=%d | "
        "overlap_with_lmkt_train=%d | held-out questions=%d",
        qg_train_split,
        qg_eval_split,
        len(qg_train_users),
        len(qg_eval_users),
        len(set(train_users) & qg_train_users),
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
        histories=qg_train_histories,
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
    opt = torch.optim.AdamW(qg_model.parameters(), lr=5e-5)

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
            
            # if grad > 1.0:
            #     logger.warning(f"Gradient norm {grad:.2f} exceeds max norm, clipping applied.")

            opt.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(qg_dataloader)
        logger.info(f"Epoch {ep} QG Loss: {avg_loss}")

    # === EVALUATE

    qg_eval_qs = df_qg_eval_ex["prompt"].unique().tolist()
    qg_eval_dataset = QGDataset(
        histories=qg_eval_histories,
        held_out_qs=qg_eval_qs,
        tokenizer=tok,
        difficulty_fn=difficulty_fn,
    )
    qg_eval_dataloader = DataLoader(
        qg_eval_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=partial(qg_collate, pad_token_id=tok.pad_token_id)
    )

    # === GET ALL DATASET QUESTIONS

    all_prompts: list[str] = []
    for split in ["train", "dev", "test"]:
        df_min = get_parquet(track, split, "minimal", columns=["tok_id", "format"])
        
        df_min = df_min[df_min["format"] == "reverse_translate"].copy()
        df_min["ex_key"] = df_min["tok_id"].str.slice(0, 10)
        assert df_min["ex_key"].isna().sum() == 0, f"Missing ex_key in {split} min data."

        revt_keys = set(df_min["ex_key"].dropna())

        df_prompts = get_parquet(track, split, "prompt")
        df_prompts = df_prompts[df_prompts["ex_key"].isin(revt_keys)]

        assert df_prompts["prompt"].isna().sum() == 0, f"Missing prompts in {split} after filtering."
        all_prompts.extend(df_prompts["prompt"])

    # keep ordering but deduplicate
    all_prompts = list(dict.fromkeys(all_prompts))

    logger.info("Evaluating QG model (generation + difficulty targeting)")
    qg_model.eval()

    # evaluation handles gpu/cpu switching internally
    qg_model.cpu()
    torch.cuda.empty_cache()

    qg_metrics = run_qg_evaluation(
        qg_model=qg_model,
        lmkt_model=model,
        eval_histories=qg_eval_histories,
        reference_questions=all_prompts,
        qg_eval_dataloader=qg_eval_dataloader,
    )

    logger.info("QG Evaluation | Diff MAE=%.4f | Diff Corr=%.4f | Distinct-1=%.4f | Distinct-2=%.4f | Novelty=%.4f | N=%d",
                qg_metrics.get("d_mae", float("nan")),
                qg_metrics.get("d_pearson_corr", float("nan")),
                qg_metrics.get("distinct_1", 0),
                qg_metrics.get("distinct_2", 0),
                qg_metrics.get("novelty", 0),
                qg_metrics.get("n_generated_questions", 0))
    
    rec = GenerationRecord(
        model="qg",
        track=track,
        subset=SUBSET,
        train_with_dev=train_with_dev,
        d_mae=qg_metrics.get("d_mae"),
        d_rmse=qg_metrics.get("d_rmse"),
        d_pearson_corr=qg_metrics.get("d_pearson_corr"),
        distinct_1=qg_metrics.get("distinct_1"),
        distinct_2=qg_metrics.get("distinct_2"),
        unique_q_ratio=qg_metrics.get("unique_q_ratio"),
        novelty=qg_metrics.get("novelty"),
        perplexity=qg_metrics.get("perplexity"),
        n_generated_questions=qg_metrics.get("n_generated_questions"),
        epochs=EPOCHS,
        tag=tag,
    )
    return [rec]
