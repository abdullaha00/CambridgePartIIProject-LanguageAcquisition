from typing import Tuple, override

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from models.SDKT.SDKT_model import SDKTModel
from models.utils import compute_metrics

class VDKTModel(SDKTModel):
    def __init__(self, 
        num_toks, 
        meta_vocab_sizes,
        emb_dim=128,
        hid_dim=256,
        meta_emb_dim=16,
        num_layers=1,
        dropout=0.2, 
        emb_matrix=None,
        latent_dim=128
        ):

        super().__init__(
            num_toks=num_toks,
            meta_vocab_sizes=meta_vocab_sizes,
            emb_dim=emb_dim,
            hid_dim=hid_dim,
            meta_emb_dim=meta_emb_dim,
            num_layers=num_layers,
            dropout=dropout,
            emb_matrix=emb_matrix
        )

        self.prior_mean = nn.Linear(self.pred_input_dim, latent_dim)
        self.prior_logvar = nn.Linear(self.pred_input_dim, latent_dim)

        self.posterior_mean = nn.Linear(self.pred_input_dim + emb_dim, latent_dim)
        self.posterior_logvar = nn.Linear(self.pred_input_dim + emb_dim, latent_dim)

        self.vpred_layer = nn.Sequential(
            nn.Linear(self.pred_input_dim + latent_dim, hid_dim),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 1)
        )
    
    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Z = mu + sigma * epsilon, where epsilon sampled from N(0, I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def KL_PQ(self, mu_q, logvar_q, mu_p, logvar_p):
        
        var_p = torch.exp(logvar_p)
        var_q = torch.exp(logvar_q)
        
        # KL divergence between two Gaussians
        kl = 0.5 * (logvar_p - logvar_q + var_q / var_p - 1 + ((mu_q - mu_p)**2 / var_p))
        return kl.sum(-1, keepdim=True)
    
    def decode_vdkt(
        self,
        dec_q: torch.Tensor, # (B, T_dec)
        dec_a: torch.Tensor, # (B, T_dec)
        dec_m: dict[str, torch.Tensor], # dict of (B, T_dec)
        enc_h: torch.Tensor, # (num_layers, B, hid_dim)
        enc_c: torch.Tensor, # (num_layers, B, hid_dim)
        enc_mask: torch.Tensor, # (B, T_enc)
        last_enc_a: torch.Tensor, # (B,)
        teacher_forcing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, T_dec = dec_q.shape
    
        t_emb = self.t_emb(dec_q) # (B, T_dec, emb_dim)
        m_emb = self.embed_meta(dec_m) # (B, T_dec, total_meta_emb_dim)
    
        prev_a = last_enc_a # (B,) 

        preds = []
        kls = []

        h, c = enc_h, enc_c
         
        for t in range(T_dec):
            prev_a_emb = self.a_emb(prev_a) # (B, emb_dim)

            # x_t = [q_emb, m_emb, prev_a_emb]
            x_t = torch.cat([t_emb[:, t, :], m_emb[:, t, :], prev_a_emb], dim=-1) # (B, dec_input_dim)

            # step with sequence length = 1
            out, (h,c) = self.decoder_lstm(x_t.unsqueeze(1), (h, c)) # out: (B, 1, hid_dim)
            h_t = out.squeeze(1) # (B, hid_dim)
            
            # variational input
            o_t = torch.cat([t_emb[:, t, :], m_emb[:, t, :], h_t], dim=-1) # (B, pred_input_dim)

            prior_mu = self.prior_mean(o_t) # (B, latent_dim)
            prior_logvar = self.prior_logvar(o_t) # (B, latent_dim)

            if self.training:
                curr_a = dec_a[:, t]
                true_a_emb = self.a_emb(curr_a) # (B, emb_dim)

                post_in = torch.cat([o_t, true_a_emb], dim=-1) # (B, pred_input_dim + emb_dim)
                post_mu = self.posterior_mean(post_in) # (B, latent_dim)
                post_logvar = self.posterior_logvar(post_in) # (B, latent_dim)

                z_t = self.reparam(post_mu, post_logvar) # (B, latent_dim)
                KL_t = self.KL_PQ(post_mu, post_logvar, prior_mu, prior_logvar) # (B, 1)
            
            else:
                
                z_t = self.reparam(prior_mu, prior_logvar) # (B, latent_dim)
                KL_t = torch.zeros(B, 1, device=o_t.device) # No KL


            pred_input = torch.cat([z_t, o_t], dim=-1) # (B, latent_dim + pred_input_dim)
            logit_t = self.vpred_layer(pred_input)
            p_t = torch.sigmoid(logit_t).squeeze(-1) # (B,)
            
            preds.append(p_t)
            kls.append(KL_t.squeeze(-1)) # [(B,)]
            
            if teacher_forcing:
                prev_a = dec_a[:, t] # (B,)
            else:
                prev_a = (p_t > 0.5).long() # (B,)
    
        return torch.stack(preds, dim=1), torch.stack(kls, dim=1) # (B, T_dec), (B, T_dec)

    def forward_vdkt(self, batch_data: dict, teacher_forcing: bool = True):
        
        enc_h, enc_c = self.encode(
            q_ids=batch_data["enc_q"],
            a_ids=batch_data["enc_a"],
            meta_dict=batch_data["enc_m"],
            mask=batch_data["enc_mask"]
        )

        preds, kls = self.decode_vdkt(
            dec_q=batch_data["dec_q"],
            dec_a=batch_data["dec_a"],
            dec_m=batch_data["dec_m"],
            enc_h=enc_h,
            enc_c=enc_c,
            enc_mask=batch_data["enc_mask"],
            last_enc_a=batch_data["enc_last_a"],
            teacher_forcing=teacher_forcing
        )

        T_batch = batch_data["dec_mask"].float().sum(dim=1) # (B,)
        assert T_batch.min() > 0, "All sequences must have at least one valid token"

        sum_kl = (kls * batch_data["dec_mask"].float()).sum() # (1,)
        total_tokens = T_batch.sum() # (1,)
        avg_kl = sum_kl / total_tokens # scalar
        
        return preds, batch_data["dec_a"], batch_data["dec_mask"], avg_kl

    def forward(self, batch_data: dict, teacher_forcing: bool = True):
        preds, targets, mask, _ = self.forward_vdkt(batch_data, teacher_forcing)
        return preds, targets, mask

    def elbo_loss(self, preds, targets, mask, kl_loss, weight):
        recons = F.binary_cross_entropy(preds[mask].float(), targets[mask].float())
        return recons + weight * kl_loss
    
    @staticmethod
    def kl_annealing_weight(step: int, total_steps: int, alpha_r: float = 0.5, beta_r = 0.15, max_weight: float = 1.0) -> float:
        alpha = total_steps * alpha_r
        beta = total_steps * beta_r

        return float(max_weight *(np.tanh((step - alpha) / beta) + 1.0) / 2.0)
    
