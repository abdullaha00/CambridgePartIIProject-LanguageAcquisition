import torch
from transformers import AutoTokenizer

def build_subword_mapping(word_vocab: dict, tokenizer: AutoTokenizer, device: torch.device) -> tuple[torch.Tensor, torch.BoolTensor]:
    pad_id = word_vocab["<pad>"]
    mapping_sw = [pad_id] * len(tokenizer)  # subword_id -> word_id, default to pad_id

    wid_to_w = {i: word for word, i in word_vocab.items()}

    whole_match_idxs = set()

    for wid, word in wid_to_w.items():
        if word.startswith("<") and word.endswith(">"):
            continue  # Skip special tokens
        
        formats = (
            word,
            " " + word,
            word.capitalize(),
            " " + word.capitalize()
        )
        
        for fmt in formats:
            subword_ids = tokenizer(fmt, add_special_tokens=False)["input_ids"]
            for sw_id in subword_ids:

                if sw_id not in whole_match_idxs:
                    mapping_sw[sw_id] = wid

                if len(subword_ids) == 1:
                    whole_match_idxs.add(sw_id)

    mapping_sw_tens = torch.tensor(mapping_sw, device=device)
    mask = mapping_sw_tens != pad_id 

    return mapping_sw_tens, mask

