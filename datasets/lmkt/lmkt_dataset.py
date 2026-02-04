from typing import Dict, List
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import logging

from datasets.common.text_format import history_text

logger = logging.getLogger(__name__)


class SeqDatasetLMKT(Dataset):
    def __init__(self, histories: dict, tokenizer, max_length: int = 1024):
        
        examples: List[Dict[str, torch.Tensor]] = []
        # TOKENISE HISTORIES
        for _, history in tqdm(histories.items(), desc="Tokenizing histories", leave=False):
            text = history_text(history)
            token_ids = tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True) 

            #TODO: use sliding windows to get around context length limits (1024 for gpt2)
            examples.append(torch.tensor(token_ids, dtype=torch.long))

        self.seqs = examples
        logger.info(f"Built LMKTDataset: {len(self.seqs)} sequences.")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]