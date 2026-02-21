import logging
from dataclasses import dataclass

from data_processing.data_parquet import load_train_and_eval_df
from models.SDKT.data import SDKTEvalDataset, SDKTTrainDataset, SDKTVocabs, build_user_sequences, build_vocab, build_meta_vocabs, collate_sdkt
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)

META_COLS = ["pos", "format", "session", "client"]

@dataclass(frozen=True)
class SDKTDataBundle:
    train_dl: DataLoader
    eval_dl: DataLoader
    vocabs: SDKTVocabs 

def build_sdkt_dataloaders(track, variant, subset, train_with_dev, batch_size=32) -> SDKTDataBundle:

    logger.info(f"Building DKT dataloaders for track {track}, variant {variant}, subset {subset}, train_with_dev={train_with_dev}")

    df_train, df_eval = load_train_and_eval_df(track, variant, train_with_dev, subset)

    # build vocabs from training data only

    token_vocab = build_vocab(df_train['lemma'])
    meta_vocabs = build_meta_vocabs(df_train, META_COLS)

    sdkt_vocabs = SDKTVocabs(token_vocab=token_vocab, meta_vocabs=meta_vocabs)

    logger.info(f"SDKT vocabs built: token_vocab_size={len(token_vocab)}, meta_vocabs_sizes={dict((col, len(vocab)) for col, vocab in meta_vocabs.items())}")

    train_seqs = build_user_sequences(df_train, token_vocab, meta_vocabs)
    eval_seqs = build_user_sequences(df_eval, token_vocab, meta_vocabs)

    train_ds = SDKTTrainDataset(train_seqs)
    eval_ds = SDKTEvalDataset(train_seqs, eval_seqs)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sdkt)
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sdkt)

    logger.info(f"DKT dataloaders built: train_batches={len(train_dl)}, eval_batches={len(eval_dl)}")

    return SDKTDataBundle(train_dl=train_dl, eval_dl=eval_dl, vocabs=sdkt_vocabs)
