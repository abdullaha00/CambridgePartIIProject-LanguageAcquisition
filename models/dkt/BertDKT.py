from asyncio.log import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

from datasets.data_parquet import load_train_and_eval_df
from datasets.kt.df_transforms import collapse_to_exercise
from models.dkt.base import DKTBase

class BertDKT(DKTBase):

    def __init__(self, num_q, emb_dim=128, head_dim=64, emb_matrix: np.ndarray = None):
        super().__init__(num_q, emb_dim, head_dim)

        assert emb_matrix is not None, "emb_matrix must be provided"
        self.register_buffer("emb_matrix", torch.from_numpy(emb_matrix))
        self.lm_dim = emb_matrix.shape[1]
        
        self.proj = nn.Linear(self.lm_dim, emb_dim)
        self.correct_embed = nn.Embedding(2, emb_dim)

    def encode_input(self, q_ids, correct_list):
        x_lm = self.proj(self.emb_matrix[q_ids])
        x = x_lm + self.correct_embed(correct_list)
        return x

  