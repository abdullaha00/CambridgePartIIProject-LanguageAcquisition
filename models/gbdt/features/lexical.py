import pandas as pd
from wordfreq import zipf_frequency
from word2word import Word2word
from rapidfuzz.distance import Levenshtein
from tqdm.auto import tqdm

AOA_PATH = "data/AoA_51715_words.csv"

def get_aoa_map(path=AOA_PATH) -> dict:
    
    aoa_df = pd.read_csv(path, encoding="latin-1")
    lemma_vals = aoa_df["AoA_Kup_lem"]

    aoa_vals = lemma_vals.fillna(aoa_df["AoA_Kup"])

    aoa_map = dict(zip(aoa_df["Word"], aoa_vals))

    return aoa_map

def add_frequency(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    
    utoks = df["tok"].unique()
    freq_map = {x: zipf_frequency(x, lang) for x in utoks}
    df.loc[:, "freq"] = df["tok"].map(freq_map).fillna(0.0)

    return df

def w2w_translate_tok(w2w: Word2word, tok: str, lang: str):

    try:
        dst_toks = w2w(tok)
    except KeyError:
        return None

    dst_tok = max(dst_toks, key=lambda x: zipf_frequency(x.lower(), lang))
    
    return dst_tok.lower()

def compute_lex_maps(utoks, src_lang, dst_lang):

    trans_map = {}
    src_freq_map = {}
    dst_freq_map = {}
    lev_map = {}

    w2w = Word2word(src_lang, dst_lang)    

    for tok in tqdm(utoks, desc="Lexical features"):
        
        src_freq = zipf_frequency(tok, src_lang)
        
        dst_tok = w2w_translate_tok(w2w, tok, dst_lang)

        if dst_tok:
            dst_freq = zipf_frequency(dst_tok, dst_lang)
            lev_frac = Levenshtein.normalized_distance(tok, dst_tok)
        else:
            lev_frac=1.0
            dst_freq=0.0
        
        trans_map[tok] = dst_tok
        src_freq_map[tok] = src_freq
        dst_freq_map[tok] = dst_freq
        lev_map[tok] = lev_frac
    
    aoa_map = get_aoa_map(AOA_PATH)

    return trans_map, src_freq_map, dst_freq_map, lev_map, aoa_map

def add_lexical_feats(df: pd.DataFrame, track: str) -> pd.DataFrame:
    src_lang, dst_lang = track.split("_")

    utoks = df["tok"].unique()

    trans_map, src_freq_map, dst_freq_map, lev_map, aoa_map = compute_lex_maps(utoks, src_lang, dst_lang)

    df["translation"] = df["tok"].map(trans_map)
    df["src_freq"] = df["tok"].map(src_freq_map)
    df["dst_freq"] = df["tok"].map(dst_freq_map)
    df["lev_distance"] = df["tok"].map(lev_map)
    df["aoa"] = df["tok"].map(aoa_map).fillna(0.0)
    df["tok_len"] = df["tok"].str.len()

    return df







