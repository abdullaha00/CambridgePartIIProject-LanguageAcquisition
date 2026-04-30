from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEmbedder(nn.Module):
    def __init__(
        self,
        tok_vocab_size: int,
        feat_vocab_sizes: dict[str, int],
        feat_emb_dim: int,
        tok_emb_dim: int,
        dropout: float        
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.tok_emb = nn.Embedding(tok_vocab_size, tok_emb_dim, padding_idx=0)
        self.feature_embs = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, feat_emb_dim, padding_idx=0)
            for feat, vocab_size in feat_vocab_sizes.items()
        })

        # OUTPUT: [token_emb | feat1_emb | feat2_emb | ...]
        self.output_dim = tok_emb_dim + feat_emb_dim * len(feat_vocab_sizes) # (ET + )

    def forward(self, tok_ids: torch.Tensor, feat_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        tok_emb = self.tok_emb(tok_ids) # (B, T, E_T)
        feat_embs = [emb(feat_ids[feat]) for feat, emb in self.feature_embs.items()]

        combined = torch.cat([tok_emb] + feat_embs, dim=-1)
        return self.dropout(combined)


class FAEncoder(nn.Module):
    def __init__(
        self,
        tok_vocab_size: int,
        feat_vocab_sizes: dict[str, int],
        feat_emb_dim: int,
        tok_emb_dim: int,
        dropout: float,
        hidden_dim: int,
        num_layers: int     
    ):
        super().__init__()
        
        self.embedder = FeatureEmbedder(tok_vocab_size, feat_vocab_sizes, feat_emb_dim, tok_emb_dim, dropout)
        self.blstm = nn.LSTM(
            input_size=self.embedder.output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # bilstm, 2 hidden_dim per direction
        self.output_dim = hidden_dim * 2

    def forward(self, tok_ids: torch.Tensor, feat_ids: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        embedded_seq = self.embedder(tok_ids, feat_ids)
        lengths = mask.sum(dim=1).cpu()
        packed_emb_seq = nn.utils.rnn.pack_padded_sequence(embedded_seq, lengths, batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.blstm(packed_emb_seq)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=tok_ids.size(1))

        return lstm_out

class FABModel(nn.Module):

    def __init__(self,
        tok_vocab_size: int,
        feat_vocab_sizes: dict[str, int],
        tok_emb_dim: int = 750,
        feat_emb_dim: int = 32,
        hidden_dim: int = 750,
        num_layers: int = 1,
        dropout: float = 0.2,
        positive_class_weight: float = 1.0,
        numeric_feat_dim: int = 0,
        classifier_hidden_dim: int = 128,
        classifier_dropout: float = 0.1
    ):

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert positive_class_weight > 0
        
        # use buffer to use same device as model, don't persist
        self.register_buffer(
            "class_weight", torch.tensor([1.0, positive_class_weight], dtype=torch.float32), persistent=False
        )
        self.numeric_feat_dim = numeric_feat_dim
        self.encoder = FAEncoder(tok_vocab_size, feat_vocab_sizes, feat_emb_dim, tok_emb_dim, dropout, hidden_dim, num_layers)

        classifier_input_dim = self.encoder.output_dim + numeric_feat_dim # [BLSTM_OUT | NUMERIC_FEATS]
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.Tanh(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, 2)
        )
        self.to(self.device)

    def forward(self,batch: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.encoder(batch["token_ids"], batch["feature_ids"], batch["mask"]) # (B, T, H*2) (bilstm output)

        if self.numeric_feat_dim > 0:
            numeric_feats = batch["numeric_features"] # (B, T, NUMERIC_FEAT_DIM)
            encoded = torch.cat([encoded, numeric_feats], dim=-1) # (B, H*2 + NUMERIC_FEAT_DIM)

        return self.classifier(encoded)

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(batch) # (B, T, 2)
        labels = batch["labels"] # (B, T)
        mask = batch["mask"]

        # loss fn = cross entropy with class weights, (+ masking)
        return F.cross_entropy(logits[mask], labels[mask], weight=self.class_weight)

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(batch) # (B, T, 2)
        return torch.softmax(logits, dim=-1)[:, :, 1] # positive class prob (B, T)
