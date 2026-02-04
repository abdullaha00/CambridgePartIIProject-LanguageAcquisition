from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import torch

@dataclass
class QGExample:
    input_ids: torch.Tensor #(T,)
    attention_mask: torch.Tensor #(T,)
    labels: torch.Tensor #(T,), with -100 for ignore
    difficulty: torch.Tensor #(1,)

class QGDataset:
    """
    QG training examples are of the form:
    prefix + <G> + <Q> question_text <A>
    where the difficulty assigned to question_text is computed by a frozen LMKTmodel)
    """
    def __init__(
        self,
        histories: Dict[str, List[Tuple[str, int]]],
        held_out_qs: List[str],
        tokenizer,
        difficulty_fn: Callable[[str, str], float],
        ):

        self.examples = []

        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]