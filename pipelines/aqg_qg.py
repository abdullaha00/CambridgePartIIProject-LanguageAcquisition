import argparse
import logging
import math
import random

import nltk
import pandas as pd
from sacrebleu import corpus_bleu
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_processing.data_parquet import get_parquet
from db.log_db import MetricRecord
from models.adaptive_qg.aqg.aqg import DCDecoder, ExerciseGenerator, build_qg_batch_user, tokenize_qg_input, tokenize_qg_output
from models.adaptive_qg.aqg.aqg_data import build_subword_mapping
from models.adaptive_qg.aqg.nlp import ensure_nltk
from models.adaptive_qg.aqg_dkt.adaptive_data import MAX_SEQ_LEN, build_word_vocab, compute_difficulty, load_user_data
from models.adaptive_qg.aqg_dkt.aqg_kt import (
    DKT,
    HIDDEN_SIZE,
    NUM_LAYERS,
    WARMUP_RATE,
    evaluate_adaptive_qg_dkt,
    kt_objective,
    kt_tensors,
    train_dkt_epoch,
)
from pipelines.aqg_kt import run_aqg_dkt_pipeline
from pipelines.common.checkpointing import load_torch_ckpt, save_torch
from pipelines.common.common import mk_record

MAX_TRAIN_USERS = 100
JOINT_START = 3
MINIBATCH_SIZE = 64
TEMPERATURE = 2.0
MIN_HIST = 5
INC_EX_OFFSET = MIN_HIST
INCONSISTENCY_WEIGHT = 0.8
MAX_INPUT_LENGTH = 15
MAX_OUTPUT_LENGTH = 30
NUM_BEAMS = 4
LOOKAHEAD_STEPS = 2

MAX_EVAL_EXERCISES = 100

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
    p.add_argument("--dkt-path", type=str, default=None, help="Path to a .ckpt file to initialize DKT model from")
    return p.parse_args(aqg_args)

def run_aqg_qg_pipeline(
        track: str,
        subset: int | None,
        train_with_dev: bool,
        EPOCHS: int,
        eval_every: int,
        next_args: list[str] | None = None,
        tag: str | None = None,
        save_every: int | None = None,
        qg_model_name: str = "facebook/bart-base",
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
    comb_train_df = pd.concat([train_df, dev_df], ignore_index=True)

    train_splits = [1, 2]
    eval_splits = [3]

    user_data_list, seen_texts = load_user_data(train_df, dev_df, test_df)
    #word_vocab = build_word_vocab(comb_train_df)

    assert user_data_list, "No user data found after loading. Please check the input data and preprocessing steps."
    
    assert aqg_args.dkt_path is not None # memory

    # EITHER LOAD DKT OR TRAIN FROM SCRATCH
    if aqg_args.dkt_path:
        logger.info(f"Resuming from checkpoint: {aqg_args.dkt_path}")
        ckpt = load_torch_ckpt(aqg_args.dkt_path)

        assert "word_vocab" in ckpt and "model_state_dict" in ckpt, f"Checkpoint at {aqg_args.dkt_path} is missing required keys. Found keys: {ckpt.keys()}"

        word_vocab = ckpt["word_vocab"]
        dkt_state = ckpt["model_state_dict"]

        dkt = DKT(
            vocab_size=len(word_vocab),
            hidden_size=aqg_args.hidden_size,
            num_layers=aqg_args.num_layers,
        ).to(device)

        dkt.load_state_dict(dkt_state)

        logger.info("Loaded DKT model from checkpoint with state dict keys: %s", dkt_state.keys())
    
    else:
        _, dkt = run_aqg_dkt_pipeline(
            track=track,
            subset=subset,
            train_with_dev=train_with_dev,
            EPOCHS=aqg_args.epochs,
            eval_every=eval_every,
            next_args=next_args,
            tag=tag,
            save_every=save_every,
        )


    # shuffle user data
    random.shuffle(user_data_list)
    user_data_list = user_data_list[:MAX_TRAIN_USERS]
    
    logger.info(f"Loaded user data - Users: {len(user_data_list)}, Unique texts: {len(word_vocab)}, Seen texts: {len(seen_texts)}")

    # === MODEL

    gen_model = ExerciseGenerator(qg_model_name, vocab_size=len(word_vocab)).to(device)
    tokenizer = gen_model.tokenizer
    
    sw_map, oov_mask = build_subword_mapping(word_vocab, tokenizer, device)

    # === OPT

    opt = torch.optim.AdamW([
        {"params": gen_model.parameters()},
        {"params": dkt.parameters()}, 
    ], lr=aqg_args.lr)

    total_steps = (EPOCHS * len(user_data_list))
    warmup_steps = int(aqg_args.warmup_rate * total_steps)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    records: list[MetricRecord] = []

    # === TRAINING LOOP
    for epoch in range(1, EPOCHS+1):

        epoch_loss = 0.0
        epoch_user_count = 0
        
        gen_model.train()
        is_joint = epoch >= JOINT_START

        if is_joint:
            dkt.train()
            if epoch == JOINT_START:
                logger.info("Starting joint training of DKT and QG models.")
        else:
            dkt.eval() # freeze DKT initially

        for user_data in user_data_list:

            train_exs = [rec for rec in user_data.exercises if rec.split == 1]
            if len(train_exs) < MIN_HIST: # skip users with too little data
                continue

            kt_inputs = kt_tensors(user_data, max_length=aqg_args.max_seq_len, device=device)

            if is_joint:
                kt_logits = dkt(kt_inputs["word_ids"], kt_inputs["labels"])
                knowledge_states = torch.sigmoid(kt_logits).squeeze(0) # (T, V)

                state_positions = torch.tensor(
                    [rec.state_position for rec in user_data.exercises], dtype=torch.long, device=device
                )

                kt_loss = kt_objective(kt_logits, kt_inputs["word_ids"], kt_inputs["labels"],
                                    kt_inputs["split_ids"], kt_inputs["interaction_ids"], state_positions, target_split=[1],)
            
            else:
                with torch.no_grad():
                    kt_logits = dkt(kt_inputs["word_ids"], kt_inputs["labels"])
                    knowledge_states = torch.sigmoid(kt_logits).squeeze(0) # (T, V)
            
            # === QG Training Step

            opt.zero_grad()
            num_batch = math.ceil(len(train_exs) / MINIBATCH_SIZE)
            user_loss_val = 0.0

            inc_ex_count = max(0, len(train_exs) - INC_EX_OFFSET)
            if inc_ex_count < 0:
                logger.warning(f"User {user_data.user_id} has only {len(train_exs)} training exercises, which is less than the inconsistency offset of {INC_EX_OFFSET}. Skipping inconsistency loss for this user.")
                inc_ex_count = 0
            
            for start in range(0, len(train_exs), MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                batch_exs = train_exs[start:end]
                
                processed = build_qg_batch_user(
                    exercises=batch_exs,
                    knowledge_states=knowledge_states,
                    difficulties=None,
                    model=gen_model,
                    tokenizer=tokenizer,
                    sub_word_ids=sw_map,
                    oov_mask=oov_mask,
                    word_vocab=word_vocab,
                    word_error_rates=None,
                    use_extra_feats=False,
                    max_in_length=MAX_INPUT_LENGTH,
                    max_out_length=MAX_OUTPUT_LENGTH,
                )

                (batch_input_ids, batch_attn_mask, student_states, 
                input_difficulties, batch_dec_ids, batch_labels, 
                sub_word_difficulties) = processed

                forward_out = gen_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attn_mask,
                    student_state=student_states,
                    difficulty=input_difficulties,
                    decoder_input_ids=batch_dec_ids,
                    labels=batch_labels,
                )

                batch_loss = forward_out.loss / num_batch
                user_loss_val += forward_out.loss.item() / num_batch

                # === INCONSISTENCY LOSS

                if inc_ex_count > 0:

                    inc_start = max(INC_EX_OFFSET - start, 0) # start idx in batch for inconsistency loss
                    if inc_start < len(batch_exs) and inc_ex_count > 0:
                        probs = torch.softmax(forward_out.logits[inc_start:] / TEMPERATURE, dim=-1)  # (B, T_dec, Vocab)

                        gen_difficulty = torch.bmm(
                            probs,  # (B, T_dec, S)
                            sub_word_difficulties[inc_start:].unsqueeze(-1) # (B, S, 1)
                        ) # (B, T_dec, 1)

                        E_diff = gen_difficulty.squeeze(-1).sum(1)  # (B, T_dec)

                        chunk_inc_count = len(batch_exs) - inc_start
                        weight = chunk_inc_count / inc_ex_count

                        inc_loss = INCONSISTENCY_WEIGHT * weight * nn.functional.l1_loss(
                            input_difficulties[inc_start:].squeeze(1), # (B, 1) -> (B,)
                            E_diff # (B, )
                        )

                        batch_loss += inc_loss
                
                # == add KT loss to last chunk to share backward pass (joint)
                is_last_chunk = (start + MINIBATCH_SIZE) >= len(train_exs)
                if is_last_chunk and is_joint:
                    assert kt_loss is not None, "KT loss should be computed for joint training epochs"

                    batch_loss += kt_loss
                
                retain_graph = is_joint and not is_last_chunk  # retain graph for all but last chunk in joint training
                batch_loss.backward(retain_graph=retain_graph)
            
            opt.step()
            scheduler.step()
            epoch_loss+= user_loss_val
            epoch_user_count += 1
    
        logger.info(f"Epoch {epoch} - Loss/user: {epoch_loss/epoch_user_count:.4} (over {epoch_user_count} users)")
        
    # ======= EVALUATION

    dkt.eval()
    gen_model.eval()
    ensure_nltk()

    decoder = DCDecoder(
        model=gen_model,
        tokenizer=tokenizer,
        sub_word_ids=sw_map,
        oov_mask=oov_mask,
        beam_size = NUM_BEAMS,
        lookahead_steps= LOOKAHEAD_STEPS,
    )

    new_accum = lambda: {
        "ref_exercises": [],
        "gen_exercises": [],
        "meteors": [],
        "dmae": [],
        "invalid": [],
        "kc_cover_count": 0,
        "kc_target_count": 0
    }

    accum = {"seen": new_accum(), "unseen": new_accum()}
    evaluated_ex_count = 0

    with torch.no_grad():
        for user_data in tqdm(user_data_list, desc="Evaluating users"):
            eval_exs = [rec for rec in user_data.exercises if rec.split in eval_splits]
            if not eval_exs:
                logger.warning(f"User {user_data.user_id} has no evaluation exercises; skipping.")
                continue
            
            # === compute knowledge states for user (dkt)
            kt_inputs = kt_tensors(user_data, max_length=aqg_args.max_seq_len, device=device)
            kt_logits = dkt(kt_inputs["word_ids"], kt_inputs["labels"])
            knowledge_states = torch.sigmoid(kt_logits).squeeze(0) # (T, V)

            for ex in eval_exs:
                assert 0 <= ex.state_position < knowledge_states.size(0), f"Invalid state position {ex.state_position} for exercise {ex.exercise_id} of user {user_data.user_id}"
                state_pos = ex.state_position # token idx right before exercise
                student_state = knowledge_states[state_pos].unsqueeze(0) # (1, V)

                input_ids, attn_mask = tokenize_qg_input(
                    ex.keywords, 
                    tokenizer=tokenizer, 
                    max_length=MAX_INPUT_LENGTH,
                    prefix_reserve=2, # reserve for special tokens
                )

                input_ids = input_ids.to(device) # (1, T_in)
                attn_mask = attn_mask.to(device) # (1, T_in)
                
                ref_labels = tokenize_qg_output(
                    ex.tokens,  # use exercise tokens as reference for evaluation 
                    tokenizer=tokenizer, 
                    max_length=MAX_OUTPUT_LENGTH, # reserve for special tokens
                    ).to(device) # (1, T_out)
                
                ref_ids = ref_labels.clone() # (1, T_out)
                ref_ids[ref_ids == -100] = tokenizer.pad_token_id # for torch.gather

                sub_word_difficulties = student_state[:, sw_map] * oov_mask # (1, S)
                target_difficulty = torch.sum(
                    torch.gather(sub_word_difficulties, dim=1, index=ref_ids), # (1, T_out)
                    dim=-1 # (1,)
                ).item() 

                # USE CONSTRAINED DECODING
                output_ids = decoder.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    target_difficulty=target_difficulty,
                    target_keywords=ex.keywords,
                    student_state=student_state,
                    max_length=MAX_OUTPUT_LENGTH,
                )

                generated_difficulty = torch.sum(
                    torch.gather(sub_word_difficulties, dim=1, index=output_ids), # (1, T_out)
                    dim=-1
                ).item()

                generated_text = tokenizer.decode(output_ids.squeeze(), skip_special_tokens=True)
                gen_toks = nltk.word_tokenize(generated_text)
                ref_toks = nltk.word_tokenize(ex.text)
                
                split = "seen" if ex.text in seen_texts else "unseen"
                bucket = accum[split]

                logger.info(f"User {user_data.user_id} - Ref: {ex.text} | Gen: {generated_text} | Target Diff: {target_difficulty:.4f}")

                bucket["ref_exercises"].append([ref_toks]) # list of lists for corpus_bleu
                bucket["gen_exercises"].append(gen_toks)
                bucket["meteors"].append(
                    nltk.translate.meteor_score.meteor_score([ref_toks], gen_toks)
                )
                bucket["dmae"].append(abs(target_difficulty - generated_difficulty))
                bucket["invalid"].append(len(gen_toks) == 0)

                keyword_toks = nltk.word_tokenize(" ".join(ex.keywords))

                gen_toks_set = set(gen_toks)
                bucket["kc_cover_count"] += sum(1 for kw in keyword_toks if kw in gen_toks_set)
                bucket["kc_target_count"] += len(keyword_toks)
                evaluated_ex_count += 1

                if evaluated_ex_count >= MAX_EVAL_EXERCISES:
                    logger.info(f"Reached maximum evaluation exercise count of {MAX_EVAL_EXERCISES}. Stopping evaluation.")
                    break

        for split, bucket in accum.items():
            bleu = nltk.translate.bleu_score.corpus_bleu(
                bucket["ref_exercises"],
                bucket["gen_exercises"],
                weights=[0.25, 0.25, 0.25, 0.25]
            ) * 100.0 # sacrebleu returns score in [0, 1], convert to percentage

            metrics = {
                "bleu": bleu,
                "meteor": sum(bucket["meteors"]) / len(bucket["meteors"]) * 100           if bucket["meteors"] else 0.0,
                "dmae": sum(bucket["dmae"]) / len(bucket["dmae"])                         if bucket["dmae"] else 0.0,
                "invalid_rate": sum(bucket["invalid"]) / len(bucket["invalid"]) * 100     if bucket["invalid"] else 0.0,
                "kc_coverage": bucket["kc_cover_count"] / bucket["kc_target_count"] * 100 if bucket["kc_target_count"] > 0 else 0.0,
            }

            print(f"=== {split.upper()} EXERCISES ===")
            print(metrics)
                

    return []

        
        