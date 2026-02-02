from pathlib import Path
import sqlite3
from datetime import datetime

DB_PATH = Path("runs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_type TEXT,
    track TEXT,
    subset INTEGER,
    train_with_dev BOOLEAN,
    auc REAL,
    f1 REAL,
    accuracy REAL,
    runtime_min REAL
);"""


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(SCHEMA)
        conn.commit()

def gen_run_id(model: str, track:str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model}_{track}_{ts}"


def log_run(
        model_name: str,
        track: str,
        subset: int,
        train_with_dev: bool,
        auc: float,
        accuracy: float,
        f1: float,
        runtime_min: float,

):
    
    init_db()
    
    run_id = gen_run_id(model_name, track)

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO runs (run_id, model_type, track, subset, train_with_dev, auc, accuracy, f1, runtime_min)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, model_name, track, subset, train_with_dev, auc, accuracy, f1, runtime_min))
    
        conn.commit()

