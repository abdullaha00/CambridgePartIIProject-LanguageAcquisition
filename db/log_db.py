from pathlib import Path
import sqlite3
from datetime import datetime

from db.records import MetricRecord

DB_PATH = Path("db/runs.db")

RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_type TEXT,
    track TEXT,
    subset INTEGER,
    variant TEXT,
    train_with_dev BOOLEAN,
    auc REAL,
    f1 REAL,
    accuracy REAL,
    runtime_min REAL
);"""


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(RUN_SCHEMA)
        conn.commit()

def gen_run_id(model: str, variant: str, track:str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if variant:
        return f"{model}_{track}_{variant}_{ts}"
    else:
        return f"{model}_{track}_{ts}"


def log_run(
        record: MetricRecord,
        runtime_min: float
):
    
    init_db()
    
    run_id = gen_run_id(record.model, record.variant, record.track)

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO runs (run_id, model_type, track, subset, train_with_dev, variant, auc, accuracy, f1, runtime_min)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, 

            (run_id, record.model, record.track,
              record.subset, record.train_with_dev, record.variant, 
              record.auc, record.acc, record.f1, runtime_min))
    
        conn.commit()

