import torch.nn as nn
from models.dkt.base import DKTBase

class DKT(DKTBase):

    def __init__(self, num_q, emb_dim=128, head_dim=64):
        super().__init__(num_q, emb_dim, head_dim)

        # Use an embedding layer to encode (question_id, correctness) pairs
        self.embed = nn.Embedding(2*num_q, emb_dim)

    def encode_input(self, q_ids, correct_list):
        return self.embed(2*q_ids + correct_list)