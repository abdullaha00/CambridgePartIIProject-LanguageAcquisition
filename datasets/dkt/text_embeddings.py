import numpy as np
import torch
from typing import List
from transformers import AutoTokenizer, AutoModel

BATCH_SIZE = 32

def embed_sentence_matrix(sentence_list: List[str], model_name: str = "distilbert-base-uncased") -> np.ndarray:

    all_embs = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    with torch.no_grad():
        
        for i in range(0, len(sentence_list), BATCH_SIZE):
            batch_texts = sentence_list[i:i+BATCH_SIZE]

            enc = tokenizer(
                batch_texts,
                padding=True,
                return_tensors="pt"
            )

            out = model(**enc)

            last_hidden_state = out.last_hidden_state
            
            # Pooling
            sentence_emb = last_hidden_state.mean(dim=1)

            all_embs.append(sentence_emb.cpu().numpy())

    return np.vstack(all_embs) # (num_sentences, embedding_dim)

            


    
    