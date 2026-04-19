import re
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import logging

from models.modular_qg.common.data import history_text
from models.modular_qg.common.tokens import TOK_BOS, TOK_EOS, TOK_N, TOK_Y, TOK_Q, TOK_A, TOK_PAD

logger = logging.getLogger(__name__)

BODY_TOKS = [TOK_Q, TOK_A, TOK_Y, TOK_N]

class LMKTModel(torch.nn.Module):
    def __init__(self, model_name: str = "gpt2",) -> None:
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")          

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.config.loss_type = "ForCausalLMLoss"

        added = self.tokenizer.add_special_tokens(
            {
                "bos_token": TOK_BOS,
                "eos_token": TOK_EOS,
                "pad_token": TOK_PAD, # GPT2 has no pad token so we use our own
                "additional_special_tokens": BODY_TOKS}
        )

        # Resize token embeddings for new special tokens
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        model_max = getattr(self.model.config, "max_position_embeddings", 1024)
        tok_max = self.tokenizer.model_max_length
        self.max_length = min(model_max, tok_max)

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

    def evaluate_metrics(
            self, 
            histories: Dict[str, List[Tuple[str, int]]],
            eval_pref_ns: Dict[str, int],
            train_seen_prompts: set[str],
            return_detailed: bool = True,
        ) -> dict[str, float]:

        """
        Evaluate LMKT on histories
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
        all_preds_n = []

        all_seen_mask = []
        assert train_seen_prompts is not None
        det_uids = []
        det_target_pos = []
        det_prompt_texts = []

        with torch.no_grad():
            for uid, hist in tqdm(histories.items(), desc="Evaluating", leave=False):
                # hist = [(q1, y1), (q2, y2), ...]

                if len(hist) == 0:
                    logger.warning(f"Empty history for user {uid}")
                    continue
                
                # GENERATE FULL HISTORY ENCDOING WITH NO TRUNCUATION
                text = history_text(hist)
                ids = tok.encode(text, add_special_tokens=False, truncation=False, return_tensors="pt").to(self.device)
                
                # We do not evaluate on the train prefix of the sequence

                assert uid in eval_pref_ns, f"User {uid} not in eval_pref_ns"
                pref_n = eval_pref_ns[uid]

                all_label_pos = [i for i, t in enumerate(ids[0]) if t in [y_id, n_id]]

                # The first pref_n exercises are for training; each exercise contributes one label 
                # so we skip the first pref_n labels

                eval_label_idxs = all_label_pos[pref_n:]
                eval_prompts = [p for p, _ in hist[pref_n:]]
                eval_target_pos = range(pref_n, pref_n + len(eval_label_idxs))

                assert len(eval_label_idxs) == len(eval_prompts)
                                
                # SCORE USING MODEL CONTEXT WINDOW

                for pos, prompt_text, target_pos in zip(
                    eval_label_idxs,
                    eval_prompts,
                    eval_target_pos,
                ):
                    assert pos > 0

                    is_seen = prompt_text in train_seen_prompts

                    start = max(0, pos-self.max_length) 
                    window_ids = ids[:, start:pos]

                    _, logits = self(input_ids=window_ids) # (1, w, V)

                    label = ids[0, pos].item()
                    targ = 1 if label == y_id else 0

                    last_tok_sm = torch.softmax(logits[0, -1, :], dim=0)

                    # logit_y = logits[0, -1, y_id]
                    # logit_n = logits[0, -1, n_id]

                    # # We care about the conditional P(Y | Y or N)

                    # p_y = torch.softmax(
                    #     torch.stack([logit_y, logit_n]),
                    #     dim=0
                    # )[0].item()
                    
                    all_labels.append(targ)
                    all_preds_y.append(last_tok_sm[y_id].item())
                    all_preds_n.append(last_tok_sm[n_id].item())

                    all_seen_mask.append(is_seen)
                    if return_detailed:
                        det_uids.append(uid)
                        det_target_pos.append(target_pos)
                        det_prompt_texts.append(prompt_text)

            #===== METRIC CALCULATION =====
            
            # USE FULL SOFTMAX DISTRIBUTION
            auc = roc_auc_score(all_labels, all_preds_y)
            preds = (torch.tensor(all_preds_y) >= 0.5).long()

            # USE P(Y) > P(N) to predict Y
            preds = (torch.tensor(all_preds_y) > torch.tensor(all_preds_n)).long()
            pred_labels = preds.cpu().numpy().astype(np.int8)
            labels = np.asarray(all_labels, dtype=np.int8)
            accuracy = accuracy_score(labels, pred_labels)
            f1 = f1_score(labels, pred_labels)

            # labels and preds for seen vs unseen analysis
            seen_labels, unseen_labels = [], []
            seen_preds_y, unseen_preds_y = [], []
            for l, p, seen in zip(all_labels, all_preds_y, all_seen_mask):
                if seen:
                    seen_labels.append(l)
                    seen_preds_y.append(p)
                else:
                    unseen_labels.append(l)
                    unseen_preds_y.append(p)
            
            if len(set(seen_labels)) <= 1:
                logger.warning("Only one class present in seen_labels; cannot compute AUC here")
            if len(set(unseen_labels)) <= 1:
                logger.warning("Only one class present in unseen_labels; cannot compute AUC here")

            metrics = {
                "auc": auc,
                "accuracy": accuracy,
                "f1": f1,
                "auc_seen": roc_auc_score(seen_labels, seen_preds_y) if len(set(seen_labels)) > 1 else None,
                "auc_unseen": roc_auc_score(unseen_labels, unseen_preds_y) if len(set(unseen_labels)) > 1 else None,
                "n_seen": len(seen_labels),
                "n_unseen": len(unseen_labels)
            }

            if return_detailed:
                metrics["preds"] = np.asarray(all_preds_y, dtype=np.float64)
                metrics["preds_n"] = np.asarray(all_preds_n, dtype=np.float64)
                metrics["pred_labels"] = pred_labels
                metrics["targets"] = labels
                metrics["seen"] = np.asarray(all_seen_mask, dtype=bool)
                metrics["uid"] = np.asarray(det_uids, dtype=object)
                metrics["target_pos"] = np.asarray(det_target_pos, dtype=np.int64)
                metrics["prompt_text"] = np.asarray(det_prompt_texts, dtype=object)

            return metrics

# ===== QUESTION GENERATION FUNCTIONS for LMKT

    def p_y_given_question_batch(self, history_prefixes: List[str], question_texts: List[str]) -> torch.Tensor:
        self.eval()
        
        tok = self.tokenizer
        y_id = tok.convert_tokens_to_ids(TOK_Y)
        n_id = tok.convert_tokens_to_ids(TOK_N)

        assert y_id > 0 and n_id > 0, \
            f"Could not find special tokens: {TOK_Y}, {TOK_N}"
        
        if len(history_prefixes) != len(question_texts):
            raise ValueError(f"Length mismatch: {len(history_prefixes)} != {len(question_texts)}")
        
        prompts = []

        for hp, qt in zip(history_prefixes, question_texts):
            
            # Warn if q_text empty
            if not (qt and qt.strip()):
                logger.warning(f"Empty question_text passed to p_y_given_question")
            
            if hp and hp.strip():
                prompts.append(f"{hp}{TOK_Q}{qt}{TOK_A}")
            else:
                prompts.append(f"{TOK_Q}{qt}{TOK_A}")
        
        
        enc_out = tok(
            prompts, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=self.max_length,
            padding=True,
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
        # logit_yn = last_logits[:, [y_id, n_id]] # (B, 2)

        # p_y = torch.softmax(
        #    logit_yn,
        #     dim=1
        # )[:, 0] # (B,) holding probs

        # We use a full vocab softmax

        last_tok_sm = torch.softmax(last_logits, dim=1) # (B, V)
        p_y = last_tok_sm[:, y_id] # (B,)

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

        prompt = f"{history_prefix}{TOK_Q}"
        enc_prompt_ids = tok.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

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



        

        
