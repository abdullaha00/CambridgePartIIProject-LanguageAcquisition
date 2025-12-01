import pandas as pd
import numpy as np
import stanza
from tqdm import tqdm

df = pd.read_parquet("parquet/en_es_minimal.parquet")
N=len(df)

eng_sents_full = df.groupby("sentence_id")["tok"].apply(" ".join)
sentence_rowidxs = list(df.groupby("sentence_id").indices.values())

fields = ["lemma", "pos", "meta", "type", "rt"]
data = {f: [None]*N for f in fields}

stanza.download("en")

nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma,depparse",
    tokenize_pretokenized=True,
    tokenize_no_ssplit=True
)
eng_sents_full = eng_sents_full

for i in tqdm(range(0, len(eng_sents_full), 256)):
    batch_sents = eng_sents_full[i:i+256]
    batch_idxs = sentence_rowidxs[i:i+256]
    batch_docs = nlp.bulk_process(batch_sents)

    for row_idxs, doc in zip(batch_idxs, batch_docs):
        for row_idx, w in zip(row_idxs, doc.sentences[0].words):
            data["lemma"][row_idx] = w.lemma
            data["pos"][row_idx] = w.upos
            data["meta"][row_idx] = w.feats
            data["type"][row_idx] = w.deprel
            data["rt"][row_idx] = w.head


start_idx = list(df.columns).index("tok") + 1

for idx, name in enumerate(fields):
    df.insert(start_idx + idx, name, data[name])

df.to_parquet("parquet/en_es_reprocessed.parquet", index=False)

