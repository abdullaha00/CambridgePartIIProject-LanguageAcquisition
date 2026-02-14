from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)

class DKTBase(nn.Module):

    def __init__(self, num_q, emb_dim=128, head_dim=64):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnn = nn.LSTM(emb_dim, head_dim, num_layers=2, batch_first=True)

        self.q_head_w = nn.Embedding(num_q, head_dim)
        self.q_head_b = nn.Embedding(num_q, 1)
    
    @abstractmethod
    def encode_input(self, q_ids, correct_list) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, q_ids, correct_list) -> torch.Tensor:
        # Map questions to embeddings
        
        # encodes what kind of update it should apply to learner state
        
        emb = self.encode_input(q_ids, correct_list) # (B, T, emb_dim)

        # Pass through RNN to get sequence of learner latent state
        h, _ = self.rnn(emb)

        return h #(B, T, H)

    def predict_next(self, h: torch.Tensor, q_next: torch.Tensor) -> torch.Tensor:
        # h: (B, T-1, H)
        
        w = self.q_head_w(q_next) # (B, T-1, H)
        b = self.q_head_b(q_next).squeeze(-1) # (B, T-1)

        logits = (h * w).sum(dim=-1) + b

        return torch.sigmoid(logits)
    
    def next_loss(self, Q, A, mask):
        Q_in, A_in = Q[:, :-1], A[:, :-1] # (q_0, q_1, ..., q_{T-2})
        Q_targ, A_targ = Q[:, 1:], A[:, 1:] # (q_1, q_2, ..., q_{T-1})

        h = self(Q_in, A_in) # (B, T-1, H) for states (s_0, s_1, ..., s_{T-2}) after observing q_i
        p = self.predict_next(h, Q_targ) # (B, T-1) for (p_1, p_2, ..., p_{T-1}) 

        #(Q[i], A{i}) -> predict p_{i+1}
        #mask[:, 1:] ignores padding in predicted timesteps, leaving us with p = (p_1, p_2, ..., p_{k}) and similar for A_targ

        seq_mask = mask[:, 1:]

        loss = F.binary_cross_entropy(p[seq_mask], A_targ[seq_mask].float())

        return loss

    def train_epoch(self, dl: DataLoader, opt: torch.optim.Optimizer):  
        self.train()

        total, n = 0.0, 0
        for uids, Q, A, mask in dl:

            Q = Q.to(self.device)
            A = A.to(self.device)
            mask = mask.to(self.device)
           
            opt.zero_grad()
            loss = self.next_loss(Q, A, mask)
            loss.backward()
            opt.step()

            total += loss.item()
            n+=1
        
        return total / n
            
    def evaluate_metrics(self, dl: DataLoader) -> float:
        self.eval()

        all_preds = []
        all_targs = []

        with torch.no_grad():
            for uids, Q, A, mask in dl:
                Q = Q.to(self.device)
                A = A.to(self.device)
                mask = mask.to(self.device)

                Q_in, A_in = Q[:, :-1], A[:, :-1]

                h = self(Q_in, A_in)
                probs = self.predict_next(h, Q[:, 1:])

                targ_mask = mask[:, 1:]

                all_preds.append(probs[targ_mask])
                all_targs.append(A[:, 1:][targ_mask])

        all_preds = torch.cat(all_preds)
        all_targs = torch.cat(all_targs)

        if len(torch.unique(all_targs)) < 2:
            logger.warning("Warning: only one class present in y_true. AUC is not defined in this case.")
            return float('nan')  # AUC is not defined in this case

        auc = roc_auc_score(all_targs.cpu().numpy(), all_preds.cpu().numpy())
        
        preds = (all_preds >= 0.5).long()
        accuracy = accuracy_score(all_targs.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(all_targs.cpu().numpy(), preds.cpu().numpy())

        return {
            "auc": auc,
            "accuracy": accuracy,
            "f1": f1,
        }