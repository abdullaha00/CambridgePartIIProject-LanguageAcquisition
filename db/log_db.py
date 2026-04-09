from pathlib import Path
import sqlite3
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

DB_PATH = Path("db/runs.db")

@dataclass(frozen=True)
class MetricRecord:
    model: str
    track: str
    subset: Optional[int]
    train_with_dev: bool
    auc: float
    acc: float
    f1: float
    variant: Optional[str] = None
    epochs: Optional[int] = None
    auc_seen: Optional[float] = None
    auc_unseen: Optional[float] = None
    tag: Optional[str] = None

@dataclass(frozen=True)
class GenerationRecord:
    model: str
    track: str
    subset: Optional[int]
    train_with_dev: bool
    bleu: Optional[float] = None
    meteor: Optional[float] = None
    d_mae: Optional[float] = None
    kc_coverage: Optional[float] = None
    invalid_rate: Optional[float] = None
    d_rmse: Optional[float] = None
    d_pearson_corr: Optional[float] = None
    distinct_1: Optional[float] = None
    distinct_2: Optional[float] = None
    unique_q_ratio: Optional[float] = None
    novelty: Optional[float] = None
    perplexity: Optional[float] = None
    n_generated_questions: Optional[int] = None
    variant: Optional[str] = None
    epochs: Optional[int] = None
    tag: Optional[str] = None

RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_type TEXT,
    track TEXT,
    subset INTEGER,
    variant TEXT,
    tag TEXT,
    epochs INTEGER,
    train_with_dev BOOLEAN,
    auc REAL,
    f1 REAL,
    accuracy REAL,
    auc_seen REAL,
    auc_unseen REAL,
    runtime_min REAL
);"""

GENERATION_RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS generation_runs (
    run_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_type TEXT,
    track TEXT,
    subset INTEGER,
    variant TEXT,
    tag TEXT,
    epochs INTEGER,
    train_with_dev BOOLEAN,
    bleu REAL,
    meteor REAL,
    kc_coverage REAL,
    d_mae REAL,
    invalid_rate REAL,
    d_rmse REAL,
    d_pearson_corr REAL,
    distinct_1 REAL,
    distinct_2 REAL,
    unique_q_ratio REAL,
    novelty REAL,
    perplexity REAL,
    n_generated_questions INTEGER,
    runtime_min REAL
);
"""


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(RUN_SCHEMA)
        c.execute(GENERATION_RUN_SCHEMA)
        conn.commit()

def gen_run_id(model: str, variant: Optional[str], track:str, index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if variant:
        return f"{model}_{track}_{variant}_{ts}_{index}"
    else:
        return f"{model}_{track}_{ts}_{index}"

def log_run_m(
        record: MetricRecord,
        index: int,
        runtime_min: float
):
    
    init_db()
    
    run_id = gen_run_id(record.model, record.variant, record.track, index)

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO runs (run_id, model_type, track, subset, train_with_dev, variant, tag, epochs, auc, accuracy, f1, auc_seen, auc_unseen, runtime_min)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, 

            (run_id, record.model, record.track,
              record.subset, record.train_with_dev, record.variant, record.tag, record.epochs,
              record.auc, record.acc, record.f1, record.auc_seen, record.auc_unseen, runtime_min))

        conn.commit()

def log_run_g(
        record: GenerationRecord,
        index: int,
        runtime_min: float
):
    
    init_db()
    
    run_id = gen_run_id(record.model, record.variant, record.track, index)

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO generation_runs (run_id, model_type, track, subset, train_with_dev, variant, tag, epochs, d_mae, d_rmse, d_pearson_corr, distinct_1, distinct_2, unique_q_ratio, novelty, perplexity, kc_coverage, n_generated_questions, runtime_min)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, 

            (run_id, record.model, record.track,
              record.subset, record.train_with_dev, record.variant, record.tag, record.epochs,
              record.d_mae, record.d_rmse, record.d_pearson_corr, record.distinct_1, record.distinct_2, record.unique_q_ratio, record.novelty, record.perplexity, record.kc_coverage, record.n_generated_questions, runtime_min))
    
        conn.commit()

