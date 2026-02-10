import pandas as pd
from tqdm.auto import tqdm
from config.consts import DEV, SPLITS, TEST, TRACKS
from data_processing.data_parquet import save_parquet

#schema definition
SCHEMA = {
    #ids
    "ex_instance_id": "int32", 
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
        for line in f:
            line = line.strip()
            if not line or line.startswith("# p"):
                continue
            elif line.startswith("# u"):
                row = list(map(lambda x: x.split(':'), line.split()))
                row = [x[1] for x in row[1:]]
                user, countries, days, client, session, formt, time = row

            else:
                r=line.split()
                tok_id, tok, pos, meta, split, rt = r[0:6]

                label = r[-1] if is_train else pd.NA

                #ex = Exercise(tok_id, tok, pos, meta, split, rt, label)

                out.append({
                    "user_id": user,
                    #Token metadata
                    "tok_id": tok_id,
                    "tok": tok.lower(),
                    "pos": pos,
                    "meta": meta, #need to process more
                    "type": split,
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

        df = pd.DataFrame(out)

        # Add ex_instance_id column
        
        ex_id = df["tok_id"].str.slice(0,10).copy()

        df["ex_instance_id"] = (
            pd.factorize(ex_id, sort=False)[0] # factorize returns (codes, uniques)
            .astype("int32")
        ) 

        # re-order to match schema

        df = df[list(SCHEMA.keys())]

        return df

def parse_key(path:str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep= " ",
        names=["tok_id", "label"]
    )

def merge_with_key(df: pd.DataFrame, df_key: pd.DataFrame) -> pd.DataFrame:
    
    df = df.drop(columns=["label"])
    df = df.merge(df_key, on="tok_id", how="left")

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

# ======================

if __name__ == "__main__":

    use_test_key = True

    for track in tqdm(TRACKS, desc="language track"):
        
        #create folder for parquet, track combination
        
        for split in tqdm(SPLITS, desc=f"{track} track", leave=False):
            
            file_path = f"data_{track}/{track}.slam.20190204.{split}"

            #------ parse original
            df = parse(file_path)
            
            # use .key file for dev data
            if split == DEV or (use_test_key and split == TEST):
                df_key = parse_key(f"{file_path}.key")
                df = merge_with_key(df, df_key)

            #------ adjust types
            df = enforce_schema(df)

            #------ 
            dfM = df.drop(columns=["pos", "type", "meta", "rt"])
            
            # save both minimal and original to parquet

            save_parquet(dfM, track, split, "minimal")
            save_parquet(df, track, split, "original")
