import pandas as pd
from typing import Dict, List, Sequence, Tuple
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import logging 

logger = logging.getLogger(__name__)

class DKTSeqDataset(Dataset):
    def __init__(self, seqs: dict):
        # Convert dict to list of (uid, (q_ids, correct_list)) 
        self.seqs = list(seqs.items())

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        uid, (q_ids, correct_list) = self.seqs[idx]
        return uid, torch.from_numpy(q_ids.copy()), torch.from_numpy(correct_list.copy())

