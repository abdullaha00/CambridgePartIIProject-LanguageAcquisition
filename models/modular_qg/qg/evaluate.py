import math
import logging
from collections import Counter, defaultdict
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import sacrebleu
from tqdm import tqdm

from models.modular_qg.common.data import history_text

logger = logging.getLogger(__name__)

def eval_bleu(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """
    Evaluate BLEU score between predicted questions and reference questions.
    Uses sacrebleu for standard BLEU computation.
    """

    assert len(preds) == len(refs), "Number of predictions and references must match for BLEU evaluation."

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return {
        "bleu": bleu.score
    }


def evaluate_difficulty_targets(
    target_difficulties: List[float],
    achieved_difficulties: List[float],      
) -> Dict[str, float]:
    
    """
    For each target difficulty, we generate questions and measure actual probability 
    from frozen LMKT model
    """

    targets = np.array(target_difficulties)
    achieved = np.array(achieved_difficulties)

    # Compute metrics
    mae = np.mean(np.abs(targets - achieved))
    rmse = np.sqrt(np.mean((targets - achieved) ** 2))

    # Pearson correlation
    if len(targets) > 1 and np.std(targets) > 0 and np.std(achieved) > 0:
        corr = np.corrcoef(targets, achieved)[0, 1] 
    else:
        logger.warning(f"Not enough data points or zero variance to compute Pearson correlation."
                        f" {{targets={targets}}}, {{achieved={achieved}}}")
        corr = float("nan")

    return {
        "d_mae": mae,
        "d_rmse": rmse,
        "d_pearson_corr": corr
    }

def evaluate_diversity(
    generated_questions: List[str]) -> Dict[str, float]:
    """Evaluate diversity of generated questions using metrics like distinct-n and self-BLEU.
    """

    assert len(generated_questions) > 0, "No generated questions to evaluate diversity."

    all_unigrams: List[str] = []
    all_bigrams: List[Tuple[str, str]] = []

    for q in generated_questions:
        tokens = q.lower().split()
        all_unigrams.extend(tokens)
        all_bigrams.extend(zip(tokens, tokens[1:]))

    n_uni = len(all_unigrams)
    n_bi = len(all_bigrams)

    distinct_1 = len(set(all_unigrams)) / n_uni if n_uni > 0 else 0.0
    distinct_2 = len(set(all_bigrams)) / n_bi if n_bi > 0 else 0.0
    unique_q_ratio = len(set(generated_questions)) / len(generated_questions)

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "unique_q_ratio": unique_q_ratio
    }

def evaluate_novelty(
        generated_questions: List[str],
        training_questions: List[str]
) -> Dict[str, float]:
    """Evaluate novelty as the fraction of generated questions that do not appear
    in training set using exact match, case insensitive
    """

    assert len(generated_questions) > 0, "No generated questions to evaluate novelty."
    assert len(training_questions) > 0, "No training questions provided."

    training_set = set(q.lower().strip() for q in training_questions)
    novel_count = sum(1 for q in generated_questions if q.lower().strip() not in training_set)

    novelty_ratio = novel_count / len(generated_questions) if generated_questions else 0.0

    return {
        "novelty": novelty_ratio
    }

@torch.no_grad()
def evaluate_perplexity(
        generated_questions: List[str],
        tokenizer,
        lm_model,
        max_length: int = 512
):
    """
    Evaluate perplexity of generated questions using a frozen language model.
    """

    lm_model.eval()
    # Use the inner HuggingFace model for perplexity (lm_model may be an LMKTModel wrapper)
    hf_model = lm_model.model if hasattr(lm_model, 'model') else lm_model
    total_loss = 0.0
    total_tokens = 0

    for q_txt in generated_questions:
        if not q_txt.strip():
            logger.warning("Skipping empty question text for perplexity evaluation.")
            continue
        
        input_toks = tokenizer(q_txt, return_tensors="pt", truncation=True, max_length=max_length) # (1, seq_len)
        input_ids = input_toks["input_ids"].to(lm_model.device) # (1, seq_len)
        if input_ids.size(1) <= 1:
            logger.warning("Skipping single-token question for perplexity: '%s'", q_txt)
            continue

        outputs = hf_model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        total_loss += loss * input_ids.size(1)  # loss is averaged over tokens
        total_tokens += input_ids.size(1) - 1  # we predict next token, so count is seq_len - 1
    
    if total_tokens == 0:
        logger.warning("No valid tokens for perplexity evaluation.")
        return {"perplexity": float("nan")}
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": perplexity
    }

def _gpu(m):
    if torch.cuda.is_available():
        m.to("cuda")

def _cpu(m):
    m.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_qg_evaluation(
    qg_model,
    lmkt_model,
    eval_histories: Dict[str, List[Tuple[str, int]]], 
    train_questions: List[str], 
    target_difficulties: List[float] =  np.linspace(0.1, 0.9, 9).tolist(),
    num_users: int = 50,
    num_samples_per_difficulty: int = 30,
) -> Dict[str, float]:
    
    """
    For each (prompt, target_difficulty) pair, generate samples using qg_model 
    For each sample, evaluate P(Y) using LMKT
    For each difficulty, we can report (target, mean_achieved, std_achieved)
    """

    qg_model.eval()
    lmkt_model.eval()

    uids = list(eval_histories.keys())
    if len(uids) > num_users:
        logger.info(f"Limiting evaluation to {num_users} users out of {len(uids)} total. (random sampling)")
        uids = random.sample(uids, num_users)
    
    all_gens = []
    all_targ_diffs = []
    all_achieved_diffs = []

    diff_achieved = defaultdict(list) # target_diff -> list of achieved diffs

    logger.info(f"Evaluating QG on {len(uids)} users x {len(target_difficulties)} difficulty levels "
                f"x {num_samples_per_difficulty} samples")
    
    for uid in tqdm(uids, desc="Evaluating QG"):
        hist = eval_histories[uid]
        assert len(hist) > 0, f"User {uid} has empty history, skipping."

        prefix_text = history_text(hist[:-1])  # structured prefix with <BOS> <Q> ... <A> <Y/N>

        for target_diff in target_difficulties:
            _gpu(qg_model)
            gen_questions = qg_model.generate(
                history_prefix=prefix_text,
                target_diff=target_diff,
                num_gen_seqs=num_samples_per_difficulty,
                max_new_toks=20
            )
            _cpu(qg_model)

            gen_qs = [q.strip() for q in gen_questions if q.strip()]
            if len(gen_qs) == 0:
                logger.warning(f"No valid generated questions for user {uid} at target difficulty {target_diff}, skipping.")
                continue

            all_gens.extend(gen_qs)
            
            # USE LMKT TO EVALUATE DIFFICULTY OF GENERATED QUESTIONS
            _gpu(lmkt_model)
            pref_texts = [prefix_text] * len(gen_qs)
            achieved_diffs = lmkt_model.p_y_given_question_batch(pref_texts, gen_qs)
            achieved_diffs = achieved_diffs.detach().cpu().tolist()  # convert CUDA tensor to list of floats
            _cpu(lmkt_model)

            for q, ach in zip(gen_qs, achieved_diffs):
                all_targ_diffs.append(target_diff)
                all_achieved_diffs.append(ach)
                diff_achieved[target_diff].append(ach)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Per difficulty breakdown (AFTER all users have been evaluated)

    per_diff_stats = {}

    for key in sorted(diff_achieved.keys()):
        achs = diff_achieved[key]
        per_diff_stats[key] = {
            "target_diff": key,
            "mean_achieved": np.mean(achs),
            "std_achieved": np.std(achs),
            "num_samples": len(achs)
        }
        logger.info(f"Target Diff {key:.2f} | Achieved Diff Mean {per_diff_stats[key]['mean_achieved']:.4f} | "
                    f"Achieved Diff Std {per_diff_stats[key]['std_achieved']:.4f} | Num Samples {per_diff_stats[key]['num_samples']}")

    # Metrics

    diff_metrics = evaluate_difficulty_targets(all_targ_diffs, all_achieved_diffs)
    diversity_metrics = evaluate_diversity(all_gens)
    novelty_metrics = evaluate_novelty(all_gens, train_questions)
    _gpu(lmkt_model)
    perplexity_metrics = evaluate_perplexity(all_gens, qg_model.tokenizer, lmkt_model)
    
    _cpu(qg_model)
    _cpu(lmkt_model)

    metrics = {
        **diff_metrics,
        **diversity_metrics,
        **novelty_metrics,
        **perplexity_metrics,
        "n_generated_questions": len(all_gens),
        "per_diff_stats": per_diff_stats
    }

    logger.info("\nQG Evaluation Results:")
    for k, v in metrics.items():
        if k == "per_diff_stats":
            continue
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics

