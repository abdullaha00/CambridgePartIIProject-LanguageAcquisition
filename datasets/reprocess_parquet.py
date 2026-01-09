import pandas as pd
import numpy as np
import stanza
from tqdm import tqdm

BATCH = 512

track = "en_es"
split = "train"

df = pd.read_parquet(f"parquet/{track}/minimal/{track}_{split}_minimal.parquet")

N=len(df)

eng_sents_full = []
sentence_rowidxs = []

groups = df.groupby("sentence_id", sort=False)

for sid, g in tqdm(groups, total=groups.ngroups, desc="Processing Sentences"):
  eng_sents_full.append(" ".join(g["tok"].tolist()))
  sentence_rowidxs.append(g.index.to_numpy())

fields = ["lemma", "pos", "meta", "type", "rt"]
data = {f: [None]*N for f in fields}

stanza.download("en")

nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma,depparse",
    tokenize_pretokenized=True,
    tokenize_no_ssplit=True
)

for i in tqdm(range(0, len(eng_sents_full), BATCH)):
    batch_sents = eng_sents_full[i:i+BATCH]
    batch_idxs = sentence_rowidxs[i:i+BATCH]
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

df.to_parquet(f"parquet/{track}/reprocessed/{track}_{split}_reprocessed.parquet", index=False)

