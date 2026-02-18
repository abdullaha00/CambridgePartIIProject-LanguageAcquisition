import argparse
import sqlite3
from pathlib import Path
import pandas as pd
from tabulate import tabulate

DB_PATH = Path("db/runs.db")

def read_table(db_path: Path, table: str) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)
    finally:
        con.close()

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found at {DB_PATH}")

df = read_table(DB_PATH, "runs")

# =====

df_f = df[df["subset"].isna()]

# ====

best_auc = (
    df_f.groupby(["model_type", "track"])["auc"].max()
    .reset_index()
)

out = (
    best_auc.pivot(
        index="model_type",
        columns="track",
        values="auc"
    )
)

out = out.drop(columns=["all"], errors="ignore")

#order = ["gbdt", "dkt", "bert_dkt", "lmkt", "lr"]
#out = out.reindex(order)

out = out.sort_values(by="en_es", ascending=False)
out = out.rename_axis(index="Model", columns="Track")

print(tabulate(out, headers="keys", tablefmt="rounded_outline", floatfmt=".4f", showindex=True))



