import re
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import logging

from models.text_kt.common.data import history_text

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

        self.model.config.loss_type = "ForCausalLMLoss"

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
        Evaluate AUC on histories
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
                # hist = [(q1, y1), (q2, y2), ...]

                if len(hist) == 0:
                    logger.warning(f"Empty history for user {uid}")
                    continue
                
                # GENERATE FULL HISTORY ENCDOING WITH NO TRUNCUATION
                text = history_text(hist)
                ids = tok.encode(text, add_special_tokens=False, truncation=False, return_tensors="pt").to(self.device)
                
                label_pos = [i for i, t in enumerate(ids[0]) if t in [y_id, n_id]]

                # SCORE USING PREVIOUS 1024 tokens

                for pos in label_pos:
                    assert pos > 0

                    start = max(0, pos-1024) 
                    window_ids = ids[:, start:pos]

                    _, logits = self(input_ids=window_ids) # (1, w, V)

                    label = ids[0, pos].item()
                    targ = 1 if label == y_id else 0

                    logit_y = logits[0, -1, y_id]
                    logit_n = logits[0, -1, n_id]

                    # We care about the conditional P(Y | Y or N)

                    p_y = torch.softmax(
                        torch.stack([logit_y, logit_n]),
                        dim=0
                    )[0].item()
                    
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

# ===== QUESTION GENERATION FUNCTIONS for lmkt

    def p_y_given_question_batch(self, history_prefixes: List[str], question_texts: List[str]) -> torch.Tensor:
        self.eval()
        
        tok = self.tokenizer
        y_id = tok.convert_tokens_to_ids(TOK_Y)
        n_id = tok.convert_tokens_to_ids(TOK_N)

        assert y_id > 0 and n_id > 0, \
            f"Could not find special tokens: {TOK_Y}, {TOK_N}"
        
        prompts = []

        for hp, qt in zip(history_prefixes, question_texts):
            
            # Warn if q_text empty
            if not (qt and qt.strip()):
                logger.warning(f"Empty question_text passed to p_y_given_question")
            
            if hp and hp.strip():
                prompts.append(f"{hp} {TOK_Q} {qt} {TOK_A}")
            else:
                prompts.append(f"{TOK_Q} {qt} {TOK_A}")
        
        enc_out = tok(
            prompts, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=1024,
            return_tensors="pt").to(self.device)

        # use FP16 on GPUs for speedup
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            logits = self.model(**enc_out).logits # (B, T, V)

        B = logits.size(0)
        T_batch = enc_out["attention_mask"].sum(dim=1)-1 # (B, )

        # We want logits[[0,1,...B-1], [T_0, T_1, ...T_{B-1}], y_id/n_id]
        
        batch_idx = torch.arange(B, device=self.device)

        # [ logits[b_0, T_0], logits[b_1, T_1], ... logits[b_{B-1}, T_{B-1}] ] -> (B, V)
        last_logits = logits[batch_idx, T_batch, :] # (B, V)
        logit_yn = last_logits[:, [y_id, n_id]] # (B, 2)

        p_y = torch.softmax(
           logit_yn,
            dim=1
        )[:, 0] # (B,) holding probs

        return p_y

#===================================== EXPERIMENTAL QG FUNCTIONS

    # Complete sequence; used for testing!
    def generate_candidate_questions( 
        self,
        history_prefix: str,
        num_candidates: int = 10,
        max_new_toks: int = 40
    ):
        self.eval()
        tok = self.tokenizer

        cands=[]

        prompt = f"{history_prefix} {TOK_Q}"
        enc_prompt_ids = tok.encode(prompt, add_special_tokens=False, truncation=True, max_length=1024, return_tensors="pt").to(self.device)

        gen = self.model.generate(
            input_ids=enc_prompt_ids,
            max_new_tokens=max_new_toks,
            num_return_sequences=num_candidates,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        ) # (num_candidates, seq_len)

        for seq in gen:
            new_seq = seq[enc_prompt_ids.shape[1]:] # remove prompt
            text = tok.decode(new_seq, skip_special_tokens=True)

            if TOK_A in text:
                question_text = text.split(TOK_A)[0]
            else:
                question_text = text

            if True:
                question_text = re.sub(r"\s+", " ", question_text).strip()
                #question_text = question_text.replace("<Q>", "").replace("</Q>", "").strip()
            
            cands.append(question_text.strip())

        return cands
    
    def selection_question(
        self,
        history_prefix: str, 
        candidates: list[str],
        target_prob: float = 0.6,
    ) -> dict:
        
        scored = []

        for q in candidates:
            p_y = self.p_y_given_question(history_prefix, q)       
            scored.append((q, p_y))

        scored.sort(key=lambda x: abs(x[1] - target_prob))

        best_q, best_p = scored[0] if scored else (None, None)

        return {
            "question": best_q,
            "predicted_p_y": best_p
        }                



        

        
