from typing import Sequence, Tuple
from .tokens import TOK_Q, TOK_A, TOK_Y, TOK_N

def history_text(history: Sequence[Tuple[str, int]]) -> str:
    """
    Composes history text from sequence of (ref_ans, correct01)
    """
    
    out = []

    for text, correct in history:
        out.append(f"{TOK_Q} {text} {TOK_A} {TOK_Y if correct == 1 else TOK_N}")
    return " ".join(out)