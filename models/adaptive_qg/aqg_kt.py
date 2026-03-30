from dataclasses import dataclass
import torch
from typing import Dict
from .adaptive_data import MAX_SEQ_LEN, UserData
import torch.nn as nn

#==== CONFIG

HIDDEN_SIZE = 128
NUM_LAYERS = 2

#====
class DKT(nn.Module):
    def __init__(self, num_words: int, hidden_size: int, num_layers: int):
        super().__init__()
        
        self.num_words = num_words
        self.word_embeddings = nn.Embedding(num_words, hidden_size, padding_idx=0)
        self.label_embeddings = nn.Embedding(3, hidden_size, padding_idx=2) # 0, 1, and 2 for pad #NOTE 

        self.encoder = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bias=True
        )

        self.ffn = nn.Linear(hidden_size, num_words)
        
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
    If the list is longer than max_length, it will be truncated from the left.
    """
    if len(values) > max_length:
        return values[-max_length:] # keep the most recent interactions
    
    return values + [pad_value] * (max_length - len(values)) # pad with the specified value

def kt_tensor(user_data: UserData, max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
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

        shift_logits = torch.roll(logits, shifts=-1, dims=1) # (B, T)
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
            "logits_last_question": logits_last_question
        }


def kt_objective(
        logits: torch.Tensor, # (B, T, V)
        word_ids: torch.Tensor, # (B, T)
        labels: torch.Tensor, # (B, T)
        split_ids: torch.Tensor, # (B, T)
        interaction_ids: torch.Tensor, # (B, T)
        state_positions: torch.Tensor, # (B, T)
        target_split: int | list[int] = 1,
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
        shift_probs = torch.roll(probs, shifts=-1, dims=1) # (B, T, V)
        
        reg_denom = reg_positions.sum().clamp(min=1.0) # number of valid positions for regularization
        l1_reg = torch.sum(torch.abs(probs - shift_probs) * reg_positions) / reg_denom
        l2_reg = torch.sum(torch.square((probs - shift_probs) ** 2) * reg_positions) / reg_denom

        total_loss = loss_current + loss_last + loss_last_question + 0.01 * l1_reg + 0.01 * l2_reg

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
        kt_inputs = kt_tensor(user_data, max_length=max_length, device=device) 

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
    
    return epoch_loss
        
            
# === EVALUATION

@torch.no_grad()
def evaluate_adaptive_qg_dkt(
        dkt: DKT,
        user_data_list: list,
        target_split: list[int],
        device: torch.device,
        positive_weight: float,
        max_length: int
    ) -> dict[str, object]: 
    
    dkt = dkt.to(device)
    dkt.eval()

    total_loss = 0.0
    total_users = 0

    for user_data in user_data_list:
        kt_inputs = kt_tensor(user_data, max_length=max_length, device=device) 

        state_positions = torch.tensor([ex.state_position for ex in user_data.exercises], dtype=torch.long, device=device) # (N_ex,)

        logits = dkt(kt_inputs["word_ids"], kt_inputs["labels"]) # (1, T, V)

        loss = kt_objective(
            logits=logits,
            word_ids=kt_inputs["word_ids"],
            labels=kt_inputs["labels"],
            split_ids=kt_inputs["split_ids"],
            interaction_ids=kt_inputs["interaction_ids"],
            state_positions=state_positions,
            target_split=target_split # evaluate on specified split
        )

        total_loss += loss.item()
        total_users += 1

    avg_loss = total_loss / total_users if total_users else float("nan")

    return {"loss": avg_loss, "users": total_users}