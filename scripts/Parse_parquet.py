import pandas as pd
import os

def parse(path: str) -> pd.DataFrame:
    hasLabel = path.endswith(".train")

    with open(path) as f:
        out = []
        sentence_idx = -1
        for line in f:
            line = line.strip()
            if not line:
                continue
            elif line.startswith("# p"): # # prompt:...
                sentence_idx += 1
            elif line.startswith("# u"):
                row = list(map(lambda x: x.split(':'), line.split()))
                row = [x[1] for x in row[1:]]
                user, countries, days, client, session, format, time = row

            else:
                r=line.split()
                tokId, tok, pos, meta, typ, rt = r[0:6]

                label = r[-1] if hasLabel else None

                #ex = Exercise(tokId, tok, pos, meta, typ, rt, label)

                out.append({
                    "sentence_id": sentence_idx,
                    "user_id": user,
                    #Token metadata
                    "tok_id": tokId,
                    "tok": tok.lower(),
                    "pos": pos,
                    "meta": meta, #need to process more
                    "type": typ,
                    "rt": rt,
                    #Exercise metadata
                    "countries": countries,
                    "days": days,
                    "client": client,
                    "session": session,
                    "format": format,
                    "time": time,
                    #Label
                    "label": label
                })

        return pd.DataFrame(out)


os.makedirs("parquet", exist_ok=True)

for lang in ["en_es", "es_en", "fr_en"]:
    df = parse(f"data_{lang}/{lang}.slam.20190204.train")
    #df.to_parquet("parquet/{lang}_original.parquet", index=False)
    dfM = df[["sentence_id", "user_id", "tok_id", "tok", "countries", "days", "client", "session", "format", "time", "label"]]
    dfM.to_parquet(f"parquet/{lang}_minimal.parquet", index=False)

