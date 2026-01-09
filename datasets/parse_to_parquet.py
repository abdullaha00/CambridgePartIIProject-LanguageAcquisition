import pandas as pd
import os
from tqdm import tqdm


#schema definition
SCHEMA = {
    #ids
    "sentence_id":     "int64", 
    "user_id":        "string",
    "tok_id":         "string",
    
    #token data
    "tok":            "string",
    "pos":            "string", # dropped in minimal
    "meta":           "string", # dropped in minimal
    "type":           "string", # dropped in minimal
    "rt":             "float64", # dropped in minimal

    #user/exercise data
    "countries":      "string",
    "client":         "string",
    "session":        "string",
    "format":         "string",
    "days":           "float64",
    "time":           "float64",

    #label
    "label":          "Int8",

}

def parse(path: str) -> pd.DataFrame:
    is_train = path.endswith(".train")

    with open(path) as f:
        out = []
        sentence_idx = 0 if not is_train else -1 #initial prompt missing for dev/test
        for line in f:
            line = line.strip()
            if not line:
                continue
            elif line.startswith("# p"): # line looks like "# prompt:..."
                sentence_idx += 1
            elif line.startswith("# u"):
                row = list(map(lambda x: x.split(':'), line.split()))
                row = [x[1] for x in row[1:]]
                user, countries, days, client, session, formt, time = row

            else:
                r=line.split()
                tokId, tok, pos, meta, typ, rt = r[0:6]

                label = r[-1] if is_train else pd.NA

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
                    "format": formt,
                    "time": time,
                    #Label
                    "label": label
                })

        return pd.DataFrame(out)

def parse_key(path:str) -> pd.DataFrame:
    with open(path) as f:
        return pd.read_csv(
            path,
            sep= " ",
            names=["tok_id", "label"]
        )

def merge_with_key(df: pd.DataFrame, df_key: pd.DataFrame) -> pd.DataFrame:
    
    df["label"] = df["tok_id"].map(
        dict(zip(df_key["tok_id"], df_key["label"]))
    )

    return df


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df.replace(to_replace=["null", "", "None"], value=pd.NA, inplace=True)

    for col, dtype in SCHEMA.items():
        assert col in df.columns

        if dtype in ["string", "boolean"]: #non-numeric
            df[col] = df[col].astype(dtype)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        
    return df

for lang in tqdm(["en_es", "es_en", "fr_en"], desc="Language"):

    #create folder for parquet, lang combination
    

    for typ in tqdm(["train", "dev", "test"], desc=f"{lang} split", leave=False):
        os.makedirs(f"parquet/{lang}/minimal", exist_ok=True)
        os.makedirs(f"parquet/{lang}/original", exist_ok=True)

        #------ parse original
        df = parse(f"data_{lang}/{lang}.slam.20190204.{typ}")
        
        # use .key file for dev data
        if typ == "dev":
            df_key = parse_key(f"data_{lang}/{lang}.slam.20190204.{typ}.key")

            df = merge_with_key(df, df_key)

        #------ adjust types
        df = enforce_schema(df)

        #------ 
        dfM = df.drop(columns=["pos", "type", "meta", "rt"])
        
        # save both minimal and original to parquet
        dfM.to_parquet(f"parquet/{lang}/minimal/{lang}_{typ}_minimal.parquet", index=False)
        df.to_parquet(f"parquet/{lang}/original/{lang}_{typ}_original.parquet", index=False)
