from time import time
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
import logging

from datasets.kt.seq_dataset import history_text

logger = logging.getLogger(__name__)

TOK_Q = "<Q>"
TOK_A = "<A>"
TOK_Y = "<Y>"
TOK_N = "<N>"

SPECIAL_TOKS = [TOK_Q, TOK_A, TOK_Y, TOK_N]

class LMKTModel(torch.nn.Module):
    def __init__(self, model_name: str = "gpt2",) -> None:
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKS}
        )

        # GPT2 has no pad token; we can use eos token as pad token as a replacement
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Resize token embeddings for new special tokens
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.to(self.device)
    
    def forward(self, **batch):
        out = self.model(**batch)
        return out.loss, out.logits

    def train_one_epoch(self, dataloader, optimizer):
        
        self.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            #clear grad before running forward
            optimizer.zero_grad()
            loss, _ = self(**batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def evaluate_metrics(self, histories: Dict[str, List[Tuple[str, int]]]):
        """
        Evaluate AUC on eval_dl
        """
        self.eval()
        self.to(self.device)

        tok = self.tokenizer

        y_id = tok.convert_tokens_to_ids(TOK_Y)
        n_id = tok.convert_tokens_to_ids(TOK_N)

        assert y_id > 0 and n_id > 0, \
            f"Could not find special tokens: {TOK_Y}, {TOK_N}"

        all_labels = []
        all_preds_y = []

        with torch.no_grad():
            for uid, hist in tqdm(histories.items(), desc="Evaluating", leave=False):
                # hist = [(q1, y1), ...]

                if len(hist) == 0:
                    logger.warning(f"Empty history for user {uid}")

                text = history_text(hist)
                ids = tok.encode(text, add_special_tokens=False, truncation=True, max_length=1024, return_tensors="pt").to(self.device)
                
                label_pos = [i for i, t in enumerate(ids[0]) if t in [y_id, n_id]]

                loss, logits = self(input_ids=ids) # (1, T, V)

                probs = torch.softmax(logits[0, :, :], dim=-1) # (T, V)

                # Extract predictions for label positions
                for i, pos in enumerate(label_pos):
                    label = ids[0, pos].item()
                    
                    targ = 1 if label == y_id else 0
                    p_y = float(probs[pos-1, y_id].item())
                    
                    all_labels.append(targ)
                    all_preds_y.append(p_y)

            #===== METRIC CALCULATION =====

            auc = roc_auc_score(all_labels, all_preds_y)
            preds = (torch.tensor(all_preds_y) >= 0.5).long()
            accuracy = accuracy_score(torch.tensor(all_labels).cpu().numpy(), preds.cpu().numpy())
            f1 = f1_score(torch.tensor(all_labels).cpu().numpy(), preds.cpu().numpy())

            return {
                "auc": auc,
                "accuracy": accuracy,
                "f1": f1,
            }

