import torch
import torch.nn as nn
import logging
import torch.nn.functional as F

from models.utils import compute_metrics

logger = logging.getLogger(__name__)

class SDKTModel(nn.Module):
    def __init__(self, num_toks, meta_vocab_sizes, emb_dim=128, hid_dim=256, meta_emb_dim=16, num_layers=1, dropout=0.2, emb_matrix=None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok_emb = nn.Embedding(num_toks, emb_dim)

        if emb_matrix is not None:
            assert emb_matrix.shape == (num_toks + 2, emb_dim), f"Embedding matrix shape {emb_matrix.shape} does not match expected shape {(num_toks + 2, emb_dim)}"
            self.t_emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix, dtype=torch.float), freeze=False, padding_idx=0)
        else:
            self.t_emb = nn.Embedding(num_toks + 2, emb_dim, padding_idx=0) # +2 for UNK and padding
        self.a_emb = nn.Embedding(2, emb_dim)

        self.meta_embs = nn.ModuleDict()

        for name, size in meta_vocab_sizes.items():
            self.meta_embs[name] = nn.Embedding(size+2, meta_emb_dim, padding_idx=0) # +2 for UNK and padding
        
        total_meta = len(meta_vocab_sizes)

        # ENCODER:
        # Input: [t, a, f] tok, ans, one embedding per feature
        
        enc_input_dim = emb_dim + emb_dim + total_meta * meta_emb_dim # [t, a, f] tok, ans, one embedding per feature
        self.encoder_bilstm = nn.LSTM(
            input_size=enc_input_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.encoder_lstm = nn.LSTM(
            input_size=2*hid_dim, # biLSTM output
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # We have H after final output latyer

        dec_input_dim = emb_dim + emb_dim + meta_emb_dim * total_meta  # [t, pred_a, m] tok, ans_prev, meta
        self.decoder_lstm = nn.LSTM(
            input_size=dec_input_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.pred_input_dim = emb_dim + meta_emb_dim * total_meta + hid_dim  # [t, m, pred_a] tok, ans_prev, meta
        self.pred_layer = nn.Sequential(
            nn.Linear(self.pred_input_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)
    
    def embed_meta(self, meta_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # meta_dict is a dict of {feat_name: tensor} where tensor is (B, T)
        # returns a tensor of shape (B, T, total_meta_emb_dim)
        meta_embs = []
        for name, emb in self.meta_embs.items():
            meta_embs.append(emb(meta_dict[name]))
        
        assert len(meta_embs) > 0, "No meta features provided"
        return torch.cat(meta_embs, dim=-1)

    def encode(
        self,
        q_ids: torch.Tensor, # (B, T)
        a_ids: torch.Tensor, # (B, T)
        meta_dict: dict[str, torch.Tensor], # dict of (B, T)
        mask: torch.Tensor # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        t_emb = self.t_emb(q_ids) # (B, T, emb_dim)
        a_emb = self.a_emb(a_ids) # (B, T, emb_dim)
        m_emb = self.embed_meta(meta_dict) # (B, T, total_meta_emb_dim)

        x = torch.cat([t_emb, a_emb, m_emb], dim=-1) # (B, T, enc_input_dim)
        x = self.dropout(x)

        T_batch = mask.sum(dim=1) # (B,)
        assert (T_batch > 0).all(), "All sequences must have at least one valid token"

        # forward padding for pack_padded_sequence
        packed = nn.utils.rnn.pack_padded_sequence(x, T_batch.cpu(), batch_first=True, enforce_sorted=False)
        bi_out, _ = self.encoder_bilstm(packed) # PackedSequence with data of shape (sum(T_batch), 2*hid_dim)
        bi_out, _ = nn.utils.rnn.pad_packed_sequence(bi_out, batch_first=True) # (B, Tmax, 2*hid_dim)

        packed2 = nn.utils.rnn.pack_padded_sequence(bi_out, T_batch.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, c) = self.encoder_lstm(packed2) # PackedSequence with data
        return h, c

    def decode(
        self,
        dec_q: torch.Tensor, # (B, T_dec)
        dec_a: torch.Tensor, # (B, T_dec)
        dec_m: dict[str, torch.Tensor], # dict of (B, T_dec)
        enc_h: torch.Tensor, # (num_layers, B, hid_dim)
        enc_c: torch.Tensor, # (num_layers, B, hid_dim)
        enc_mask: torch.Tensor, # (B, T_enc)
        last_enc_a: torch.Tensor, # (B,)
        teacher_forcing: bool = True
    ) -> torch.Tensor:
        
        B, T_dec = dec_q.shape
    
        t_emb = self.t_emb(dec_q) # (B, T_dec, emb_dim)
        m_emb = self.embed_meta(dec_m) # (B, T_dec, total_meta_emb_dim)
    
        prev_a = last_enc_a # (B,) 

        preds = []

        h, c = enc_h, enc_c
         
        for t in range(T_dec):
            prev_a_emb = self.a_emb(prev_a) # (B, emb_dim)
            x_t = torch.cat([t_emb[:, t, :], m_emb[:, t, :], prev_a_emb], dim=-1) # (B, dec_input_dim)

            # step with sequence length = 1
            out, (h,c) = self.decoder_lstm(x_t.unsqueeze(1), (h, c)) # out: (B, 1, hid_dim)
            h_t = out.squeeze(1) # (B, hid_dim)
            
            pred_input = torch.cat([t_emb[:, t, :], m_emb[:, t, :], h_t], dim=-1) # (B, pred_input_dim)
            logit_t = self.pred_layer(pred_input)
            p_t = torch.sigmoid(logit_t).squeeze(-1) # (B,)
            
            preds.append(p_t)

            if teacher_forcing:
                prev_a = dec_a[:, t] # (B,)
            else:
                prev_a = (p_t > 0.5).long() # (B,)
    
        return torch.stack(preds, dim=1) # (B, T_dec)

    def forward(
        self,
        batch_data: dict,
        teacher_forcing: bool = True):
        
        enc_state = self.encode(
            q_ids=batch_data["enc_q"],
            a_ids=batch_data["enc_a"],
            meta_dict=batch_data["enc_m"],
            mask=batch_data["enc_mask"]
        )

        preds = self.decode(
            dec_q=batch_data["dec_q"],
            dec_a=batch_data["dec_a"],
            dec_m=batch_data["dec_m"],
            enc_h=enc_state[0],
            enc_c=enc_state[1],
            enc_mask=batch_data["enc_mask"],
            last_enc_a=batch_data["enc_last_a"],
            teacher_forcing=teacher_forcing
        )

        return preds, batch_data["dec_a"], batch_data["dec_mask"]

    @staticmethod
    def loss_fn(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # preds and targets are (B, T), mask is (B, T)
        assert preds.shape == targets.shape == mask.shape, "Preds, targets, and mask must have the same shape"
        assert mask.sum() > 0, "Mask must have at least one valid position"

        return F.binary_cross_entropy(preds[mask], targets[mask].float())

    def evaluate(self, dl, teacher_forcing=True) -> dict[str, float]:
        self.eval()
        all_preds = []
        all_targets = []

        for batch in dl:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else 
                     {k2: v2.to(self.device) for k2, v2 in v.items()} if isinstance(v, dict) else 
                     v for k, v in batch.items()}
            
            with torch.no_grad():
                preds, targets, mask = self(batch, teacher_forcing=teacher_forcing)

                assert preds.shape == targets.shape == mask.shape, "Preds, targets, and mask must have the same shape"
                assert mask.sum() > 0, "Mask must have at least one valid position"

                all_preds.append(preds[mask].cpu())
                all_targets.append(targets[mask].cpu())

        p = torch.cat(all_preds, dim=0)
        t = torch.cat(all_targets, dim=0)

        metrics = compute_metrics(p.numpy(), t.numpy())

        return metrics


        



        

        
        



