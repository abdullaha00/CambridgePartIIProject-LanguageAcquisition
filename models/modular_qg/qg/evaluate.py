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
        reference_questions: List[str]
) -> Dict[str, float]:
    """Evaluate novelty as the fraction of generated questions that do not appear
    in reference question set using exact match, case insensitive
    """

    assert len(generated_questions) > 0, "No generated questions to evaluate novelty."
    assert len(reference_questions) > 0, "No reference questions provided."

    reference_set = set(q.lower().strip() for q in reference_questions)
    novel_count = sum(1 for q in generated_questions if q.lower().strip() not in reference_set)

    novelty_ratio = novel_count / len(generated_questions) if generated_questions else 0.0

    return {
        "novelty": novelty_ratio
    }

@torch.no_grad()
def evaluate_qg_perplexity(qg_model, dataloader) -> Dict[str, float]:
    """Evaluate QG perplexity from conditional QG loss on held-out examples."""

    qg_model.eval()
    device = next(qg_model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating QG perplexity", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        n_tokens = int((batch["labels"] != -100).sum().item())
        assert n_tokens != 0, "No valid tokens in batch"
        
        out = qg_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            difficulty=batch["difficulty"],
        )
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
    
    if total_tokens == 0:
        logger.warning("No valid tokens for QG perplexity evaluation.")
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
    reference_questions: List[str],
    qg_eval_dataloader: torch.utils.data.DataLoader,
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

        prefix_text = history_text(hist)  # structured prefix with <BOS> <Q> ... <A> <Y/N>

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
    novelty_metrics = evaluate_novelty(all_gens, reference_questions)
    _gpu(qg_model)
    perplexity_metrics = evaluate_qg_perplexity(qg_model, qg_eval_dataloader)
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
