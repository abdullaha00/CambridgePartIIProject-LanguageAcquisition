import os
import shutil
import urllib.request
import fasttext.util
from pathlib import Path
import logging
import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

def load_fasttext_bin(lang: str, out_dir: str | Path = "./fasttext") -> fasttext.FastText:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    model_name = f"cc.{lang}.300.bin"
    model_path = out_dir_p / model_name

    if model_path.exists():
        logger.info(f"FastText model already exists at {model_path}. Skipping download.")
        return fasttext.load_model(str(model_path))
    else:
        logger.info(f"Downloading FastText model for language '{lang}' to {model_path}...")

    fasttext.util.download_model(lang, if_exists='ignore')  # English
    ft = fasttext.load_model(f'cc.{lang}.300.bin')
    
    # == Move into directory

    src_path = Path(model_name)

    if not src_path.exists():
        logger.error(f"Expected FastText model at {src_path} after download, but it was not found.")
        raise FileNotFoundError(f"FastText model not found at {src_path}")
    
    shutil.move(str(src_path), str(model_path))

    return ft

def load_fasttext_vecs(
    lang: str,
    vocab: dict[str, int],
    emb_dim: int = 300,
    out_dir: str | Path = "./fasttext",
    subset: int | None = None,
    cache = True
) -> torch.Tensor:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir_p / f"fasttext_embs_{lang}_dict.pt"
    if cache and cache_path.exists():
        logger.info(f"Loading cached embedding dict from {cache_path}")
        emb_dict = torch.load(cache_path, weights_only=False)
        ft = None
    else:
        logger.info("Loading FastText model and building embedding matrix...")
        ft = load_fasttext_bin(lang, out_dir=out_dir)
        emb_dict = {}

    emb_matrix = np.zeros((len(vocab) + 2, emb_dim)) #0 = PAD, final  = UNK

    matched = 0

    for word, idx in tqdm(vocab.items(), desc="Processing vocabulary"):
        if word in emb_dict:
            emb_matrix[idx] = emb_dict[word]
            matched += 1
        elif ft is not None and word in ft:
            emb_matrix[idx] = ft.get_word_vector(word)
            emb_dict[word] = ft.get_word_vector(word)
            matched += 1
        else:
            emb_matrix[idx] = np.random.normal(scale=0.6, size=(emb_dim,))
    
    logger.info(f"Loaded FastText vectors for language '{lang}'. Matched {matched}/{len(vocab)} tokens in vocab.")

    unk_id = len(vocab) + 1
    emb_matrix[unk_id] = emb_matrix[1:len(vocab)+1].mean(axis=0)  # UNK vector is mean of known token vectors

    tensor = torch.tensor(emb_matrix, dtype=torch.float32)

    if cache and subset is None and not cache_path.exists():  
        torch.save(emb_dict, cache_path.with_name(f"fasttext_embs_{lang}_dict.pt"))
        logger.info(f"Cached embedding dict to {cache_path.with_name(f'fasttext_embs_{lang}_dict.pt')}")

    return tensor

