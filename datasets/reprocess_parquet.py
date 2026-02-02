import stanza
from tqdm.auto import tqdm
from data_parquet import get_parquet, save_parquet

BATCH = 256

stanza.download("en")

nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        tokenize_no_ssplit=True,
        use_gpu=True
    )

for track in tqdm(["en_es", "es_en", "fr_en"], desc="Language"):

    for split in tqdm(["train", "dev", "test"], desc=f"{track} track", leave=False):

        df = get_parquet(track, split, "minimal")
        N=len(df)

        eng_sents_full = []
        sentence_rowidxs = []

        groups = df.groupby("ex_instance_id", sort=False)

        for sid, g in tqdm(groups, total=groups.ngroups, desc="Processing Sentences"):
            eng_sents_full.append(" ".join(g["tok"].tolist()))
            sentence_rowidxs.append(g.index.to_numpy())

        fields = ["lemma", "pos", "meta", "type", "rt"]
        data = {f: [None]*N for f in fields}

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

        save_parquet(df, track, split, "reprocessed")

