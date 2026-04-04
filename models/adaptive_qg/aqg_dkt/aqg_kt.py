from dataclasses import dataclass
import logging
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import torch
from typing import Dict
from .adaptive_data import MAX_SEQ_LEN, UserData
import torch.nn as nn

logger = logging.getLogger(__name__)

#==== CONFIG

HIDDEN_SIZE = 100
NUM_LAYERS = 3

WARMUP_RATE = 0.03

LOSS_CURR_WEIGHT = 0.0
LOSS_LAST_WEIGHT = 1.0
LOSS_LAST_QUESTION_WEIGHT = 0.5
L1_WEIGHT = 0.1
L2_WEIGHT = 0.1

#====
class DKT(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        
        self.num_words = vocab_size
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.label_embeddings = nn.Embedding(3, hidden_size, padding_idx=2) # 0, 1, and 2 for pad #NOTE 

        self.encoder = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bias=True
        )

        self.ffn = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, word_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        word_ids: (B, T)
        labels: (B, T) with values in {0, 1} for correct/incorrect and -100 for padding
        """
        word_embeds = self.word_embeddings(word_ids)

        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 2 # map -100 to the padding index for labels
        label_embeds = self.label_embeddings(safe_labels)

        x = word_embeds + label_embeds # (B, T, H)
        
        hidden_states, _ = self.encoder(x) # (B, T, H), _ = (h_n, c_n) where h_n is (num_layers, B, H)
        output = self.ffn(hidden_states) # (B, T, num_words)

        return output

# ==== TRAINING DKT
    
def pad_list(values: list[int], max_length: int, pad_value: int = 0) -> list[int]:
    """
    Pads a list of integers to a specified maximum length with a given padding value.
    UserData should already be truncated.
    """
    if len(values) > max_length:
        raise ValueError(
            f"Sequence length {len(values)} exceeds max_length={max_length}. "
        )
    
    return values + [pad_value] * (max_length - len(values)) # pad with the specified value

def kt_tensors(user_data: UserData, max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Converts UserData into tensors.
    Each value in dict is a tensor of shape (max_length,).
    """
    return {
        "word_ids": torch.tensor(pad_list(user_data.word_ids, max_length, pad_value=0), dtype=torch.long, device=device).unsqueeze(0), # add batch dimension
        "labels": torch.tensor(pad_list(user_data.labels, max_length, pad_value=-100), dtype=torch.long, device=device).unsqueeze(0), 
        "split_ids": torch.tensor(pad_list(user_data.split_ids, max_length, pad_value=0), dtype=torch.long, device=device).unsqueeze(0),  
        "interaction_ids": torch.tensor(pad_list(user_data.interaction_ids, max_length, pad_value=-1), dtype=torch.long, device=device).unsqueeze(0) 
    }

def kt_predictions(
        logits: torch.Tensor, # (B, T, V) 
        word_ids: torch.Tensor, # (B, T)
        interaction_ids: torch.Tensor, # (B, T)
        state_positions: torch.Tensor # [N_ex]  one pre-exercise position for each interaction id
    ) -> torch.Tensor:

        word_inds = word_ids.unsqueeze(-1) # (B, T, 1)
        logits_cur_step = logits.gather(-1, word_inds).squeeze(-1) # (B, T)

        shift_logits = torch.roll(logits, shifts=1, dims=1) # (B, T)
        logits_last_step = shift_logits.gather(-1, word_inds).squeeze(-1) # (B, T)

        valid_interaction_ids = interaction_ids.clone() # (1, T)
        valid_interaction_ids[valid_interaction_ids == -1] = 0 # treat padded interactions as valid for masking purposes

        state_indices = state_positions[valid_interaction_ids.squeeze(0)] # (T, ) interaction id -> state position (pos before exercise start)

        logits_last_question = logits[0, state_indices, :] # (T, V) int -> logit distribution for prev 

        w = word_ids.squeeze(0).unsqueeze(-1) # (T, 1)
        logits_last_question = logits_last_question.gather(-1, w).squeeze(-1).unsqueeze(0) # (1, T) with the logits for the last question word

        return {
            "logits_cur_step": logits_cur_step,  # (B, T)
            "logits_last_step": logits_last_step,  # (B, T)
            "logits_last_question": logits_last_question # (B, T)
        }


def kt_objective(
        logits: torch.Tensor, # (B, T, V)
        word_ids: torch.Tensor, # (B, T)
        labels: torch.Tensor, # (B, T)
        split_ids: torch.Tensor, # (B, T)
        interaction_ids: torch.Tensor, # (B, T)
        state_positions: torch.Tensor, # (B, T)
        target_split: list[int] = [1,2],
        positive_weight: float = 3.0
    ) -> torch.Tensor:

        # PREDICTIONS
        predictions = kt_predictions(
            logits=logits,
            word_ids=word_ids,
            interaction_ids=interaction_ids,
            state_positions=state_positions
        )

        loss_fn = nn.BCEWithLogitsLoss(reduction="none") 

        split_mask = sum([split_ids == split for split in target_split]) # (B, T) boolean mask for target splits

        targets = labels.long().clone() # (B, T) convert to long for cross-entropy
        
        target_mask = (targets != -100) & split_mask # only consider positions with valid labels and in target splits
        targets[labels == -100] = 0 # map -100 to 0 for loss calculation, will be ignored due to weight

        weights = torch.where(targets == 1, positive_weight, 1.0) * target_mask
        denom = weights.sum().clamp(min=1.0) 

        loss_current = torch.sum(
             loss_fn(predictions["logits_cur_step"].float(), targets.float()) * weights
        ) / denom

        loss_last = torch.sum(
            loss_fn(predictions["logits_last_step"].float(), targets.float()) * weights
        ) / denom

        loss_last_question = torch.sum(
            loss_fn(predictions["logits_last_question"].float(), targets.float()) * weights
        ) / denom

        # We apply smoothness regularisation 

        reg_positions = (split_ids != 0).float().unsqueeze(-1) # (B, T, 1) only apply regularization to non-padded positions
        probs = torch.sigmoid(logits) # (B, T, V)
        shift_probs = torch.roll(probs, shifts=1, dims=1) # (B, T, V)
        
        # Average over (timesteps, vocab size)
        reg_denom = reg_positions.sum().clamp(min=1.0) * logits.size(-1)
        l1_reg = torch.sum(torch.abs(probs - shift_probs) * reg_positions) / reg_denom
        l2_reg = torch.sum(torch.square((probs - shift_probs)) * reg_positions) / reg_denom

        total_loss = (
            loss_current * LOSS_CURR_WEIGHT 
            + loss_last * LOSS_LAST_WEIGHT
            + loss_last_question * LOSS_LAST_QUESTION_WEIGHT
            + l1_reg * L1_WEIGHT
            + l2_reg * L2_WEIGHT
                    )

        return total_loss

def train_dkt_epoch(
        dkt: DKT,
        user_data_list: list[UserData],
        opt: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        device: torch.device,
        max_length: int,
        positive_weight: float,
        target_split: list[int]
    ) -> float:

    dkt.train()

    epoch_loss = 0.0
    epoch_users = 0

    for user_data in user_data_list:
        kt_inputs = kt_tensors(user_data, max_length=max_length, device=device) 

        state_positions = torch.tensor([ex.state_position for ex in user_data.exercises], dtype=torch.long, device=device) # (N_ex,)

        logits = dkt(kt_inputs["word_ids"], kt_inputs["labels"]) # (1, T, V)

        loss = kt_objective(
            logits=logits,
            word_ids=kt_inputs["word_ids"],
            labels=kt_inputs["labels"],
            split_ids=kt_inputs["split_ids"],
            interaction_ids=kt_inputs["interaction_ids"],
            state_positions=state_positions,
            target_split=target_split, # train using train, dev
            positive_weight=positive_weight
        )

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_users += 1
    
    return epoch_loss / epoch_users 
            
# === EVALUATION

def collapse_question_predictions(
    word_ids: np.ndarray,
    labels: np.ndarray,
    split_ids: np.ndarray,
    interaction_ids: np.ndarray,
    probs_last_q: np.ndarray,
    ) -> dict[str, np.ndarray]:
     
    seen_qs = set()
    q_labels: list[int] = []
    q_probs: list[float] = []
    q_split_ids: list[int] = []
    q_seen_bool: list[bool] = []

    current_toks: list[int] = []
    current_labs: list[int] = []
    current_probs: list[float] = []
    current_splits: list[int] = []

    prev_interaction_id = None

    for i, interaction_id in enumerate(interaction_ids):
        if interaction_id == -1:
            continue # skip padded interactions
        
        if prev_interaction_id is not None and interaction_id != prev_interaction_id:
            toktup = tuple(current_toks) # signature for question text

            q_labels.append(max(current_labs)) # label = 1 on any token (mistake) means label = 1 for the question
            q_probs.append(max(current_probs)) # take max predicted prob across question tokens as the question-level prediction
            q_split_ids.append(current_splits[0]) 
            q_seen_bool.append(toktup in seen_qs) 

            seen_qs.add(toktup)
            current_toks = []
            current_labs = []
            current_probs = []
            current_splits = []

        prev_interaction_id = interaction_id
        current_toks.append(word_ids[i])
        current_labs.append(labels[i])
        current_probs.append(probs_last_q[i])
        current_splits.append(split_ids[i])
    
    if current_toks: # handle last question
        toktup = tuple(current_toks)
        q_labels.append(max(current_labs))
        q_probs.append(max(current_probs))
        q_split_ids.append(current_splits[0])
        q_seen_bool.append(toktup in seen_qs)
    
    return {
        "q_labels": np.array(q_labels), 
        "q_pred_probs": np.array(q_probs), 
        "q_split_ids": np.array(q_split_ids), 
        "q_seen_bool": np.array(q_seen_bool)
    }
            
@torch.no_grad()
def evaluate_adaptive_qg_dkt(
        dkt: DKT,
        user_data_list: list,
        target_split: list[int],
        device: torch.device,
        max_length: int
    ) -> dict[str, object]: 

    assert len(target_split) == 1, "Evaluation only supports evaluating on a single split (e.g. dev or test)"
    
    dkt = dkt.to(device)
    dkt.eval()

    total_users = 0

    # token-level predictions
    tlev_all_labels = []
    tlev_all_p_err_cur_w = []
    tlev_all_p_err_last_w = []
    tlev_all_seen_bool = []

    # question-level predictions
    qlev_all_labels = []
    qlev_all_p_err_last_q = []
    qlev_all_seen_bool = []

    for user_data in user_data_list:

        # === PREPARE INPUTS
        kt_inputs = kt_tensors(user_data, max_length=max_length, device=device) 

        word_ids = kt_inputs["word_ids"].squeeze(0).detach().cpu().numpy() # (T, )
        labels = kt_inputs["labels"].squeeze(0).cpu().detach().numpy() # (T, )
        split_ids = kt_inputs["split_ids"].squeeze(0).cpu().detach().numpy() # (T, )
        interaction_ids = kt_inputs["interaction_ids"].squeeze(0).cpu().detach().numpy() # (T, )

        state_positions = torch.tensor([ex.state_position for ex in user_data.exercises], dtype=torch.long, device=device) # (N_ex,)

        logits = dkt(kt_inputs["word_ids"], kt_inputs["labels"]) # (1, T, V)

        preds = kt_predictions(
            logits=logits,
            word_ids=kt_inputs["word_ids"],
            interaction_ids=kt_inputs["interaction_ids"],
            state_positions=state_positions
        )

        probs_cur_w = torch.sigmoid(preds["logits_cur_step"]).detach().cpu().squeeze(0) # (T, V)
        probs_last_w = torch.sigmoid(preds["logits_last_step"]).detach().cpu().squeeze(0) # (T, V)
        probs_last_q = torch.sigmoid(preds["logits_last_question"]).detach().cpu().squeeze(0) # (T,)

        # === SEEN WORD LABELS
        seen_words = set()
        
        seen_word_bool = np.zeros(kt_inputs["word_ids"].shape[1], dtype=bool) # (T,)
        for i, wid in enumerate(kt_inputs["word_ids"].squeeze(0).cpu().detach().numpy()):
            seen_word_bool[i] = True if wid in seen_words else False
            seen_words.add(wid)
        
        # === COLLAPSE TO QUESTION-LEVEL PREDICTIONS 

        q_data = collapse_question_predictions(
            word_ids=word_ids,
            labels=labels,
            split_ids=split_ids,
            interaction_ids=interaction_ids,
            probs_last_q=probs_last_q.numpy()
        )

        # ==== token-level

        valid_mask = (split_ids == target_split[0]) # only consider valid labels in target split
        if valid_mask.sum() == 0:
            logger.warning(f"No valid labels for user {user_data.user_id} in target split {target_split[0]}")
            continue

        tlev_all_labels.append(labels[valid_mask])
        tlev_all_p_err_cur_w.append(probs_cur_w.numpy()[valid_mask])
        tlev_all_p_err_last_w.append(probs_last_w.numpy()[valid_mask])
        tlev_all_seen_bool.append(seen_word_bool[valid_mask])

        # ==== question-level

        valid_mask = (q_data["q_split_ids"] == target_split[0])
        if valid_mask.sum() == 0:
            logger.warning(f"No valid question-level labels for user {user_data.user_id} in target split {target_split[0]}")
            continue

        qlev_all_labels.append(q_data["q_labels"][valid_mask])
        qlev_all_p_err_last_q.append(q_data["q_pred_probs"][valid_mask])
        qlev_all_seen_bool.append(q_data["q_seen_bool"][valid_mask])
    
        total_users += 1
    
    labels = np.concatenate(tlev_all_labels)
    p_err_cur_w = np.concatenate(tlev_all_p_err_cur_w)
    p_err_last_w = np.concatenate(tlev_all_p_err_last_w)
    seen_bool = np.concatenate(tlev_all_seen_bool)

    q_labels = np.concatenate(qlev_all_labels)
    p_err_last_q = np.concatenate(qlev_all_p_err_last_q)
    q_seen_bool = np.concatenate(qlev_all_seen_bool)

    return {
        "users": total_users,
        "token_level": {
            "count": len(labels),

            "auc_cur_w": roc_auc_score(labels, p_err_cur_w),
            
            "auc_last_w": roc_auc_score(labels, p_err_last_w),
            "seen_auc_last_w": roc_auc_score(labels[seen_bool], p_err_last_w[seen_bool]),
            "unseen_auc_last_w": roc_auc_score(labels[~seen_bool], p_err_last_w[~seen_bool]),

            "f1_last_w": f1_score(labels, p_err_last_w > 0.5),
            "f1_cur_w": f1_score(labels, p_err_cur_w > 0.5),
            
            "acc_cur_w": ((p_err_cur_w > 0.5) == labels).mean(),
            "acc_last_w": ((p_err_last_w > 0.5) == labels).mean(),

            
        },
        "question_level": {
            "count": len(q_labels),

            "auc_last_q": roc_auc_score(q_labels, p_err_last_q),
            "seen_auc_last_q": roc_auc_score(q_labels[q_seen_bool], p_err_last_q[q_seen_bool]),
            "unseen_auc_last_q": roc_auc_score(q_labels[~q_seen_bool], p_err_last_q[~q_seen_bool]),
            
            "f1_last_q": f1_score(q_labels, p_err_last_q > 0.5),

            "acc_last_q": ((p_err_last_q > 0.5) == q_labels).mean(),
        }
    }
            
