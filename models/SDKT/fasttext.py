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
    cache=True,
    load_bin_missing_threshold: int = 10,
) -> torch.Tensor:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir_p / f"fasttext_embs_{lang}_dict.pt"
    if cache and cache_path.exists():
        logger.info(f"Loading cached embedding dict from {cache_path}")
        emb_dict = torch.load(cache_path, weights_only=False)

        # == WE pull the actual binary if too many words are not in cache
        missing_words = [word for word in vocab if word not in emb_dict]
        if len(missing_words) > load_bin_missing_threshold:
            logger.info("FastText cache is missing %d/%d vocab entries. Loading binary model.", 
                        len(missing_words), len(vocab))
            ft = load_fasttext_bin(lang, out_dir=out_dir)
        else:
            if missing_words:
                logger.info(
                    "FastText cache is missing %d/%d vocab entries," \
                    "at or below threshold %d. We use random vectors for misses.",
                    len(missing_words),
                    len(vocab),
                    load_bin_missing_threshold,
                )
            ft = None
    else:
        logger.info("Loading FastText model and building embedding matrix...")
        ft = load_fasttext_bin(lang, out_dir=out_dir)
        emb_dict = {}

    emb_matrix = np.zeros((len(vocab) + 2, emb_dim)) #0 = PAD, final  = UNK

    matched = 0
    cache_updated = False

    for word, idx in tqdm(vocab.items(), desc="Processing vocabulary"):
        if word in emb_dict:
            emb_matrix[idx] = emb_dict[word]
            assert emb_matrix[idx].shape == (emb_dim,), \
            f"Cached vector for '{word}' has shape {emb_matrix[idx].shape}, expected {(emb_dim,)}"
            matched += 1
        elif ft is not None and word in ft:
            vec = emb_dict[word]
            assert vec.shape == (emb_dim,), \
            f"Cached vector for '{word}' has shape {emb_matrix[idx].shape}, expected {(emb_dim,)}"
            
            emb_matrix[idx] = vec
            emb_dict[word] = vec

            cache_updated = True
            matched += 1
        else:
            emb_matrix[idx] = np.random.normal(scale=0.6, size=(emb_dim,))
    
    logger.info(f"Loaded FastText vectors for language '{lang}'. Matched {matched}/{len(vocab)} tokens in vocab.")

    unk_id = len(vocab) + 1
    emb_matrix[unk_id] = emb_matrix[1:len(vocab)+1].mean(axis=0)  # UNK vector is mean of known token vectors

    tensor = torch.tensor(emb_matrix, dtype=torch.float32)

    if cache and subset is None and (cache_updated or not cache_path.exists()):  
        torch.save(emb_dict, cache_path.with_name(f"fasttext_embs_{lang}_dict.pt"))
        logger.info(f"Cached embedding dict to {cache_path.with_name(f'fasttext_embs_{lang}_dict.pt')}")

    return tensor
