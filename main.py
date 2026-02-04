
import argparse
import torch
import numpy as np
from db.log_db import log_run
from pipelines.dkt import run_dkt_pipeline, parse_dkt_args
from pipelines.gbdt import run_gbdt_pipeline, parse_gdbt_args
from time import perf_counter
from pipelines.lmkt_pipeline import parse_lmkt_args, run_lmkt_pipeline
from pipelines.lr import run_lr_pipeline
import logging
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


#==== ARGUMENTS
p = argparse.ArgumentParser()
p.add_argument("model_name", 
               choices=["lr", "gbdt", "dkt", "bert_dkt", "lmkt"])
p.add_argument("--track", type=str, default="en_es", choices=["en_es", "fr_en", "es_en", "all"])
p.add_argument("--train_with_dev", action="store_true")
p.add_argument("--subset", type=int, default=None)
p.add_argument("--no_log", action="store_true")
args, next_args = p.parse_known_args()

model = args.model_name
TRACK = args.track
train_with_dev = args.train_with_dev
SUBSET = args.subset

#===== CONFIG
torch.manual_seed(42)
np.random.seed(42)

#-- start timer

start_time = perf_counter()

#========= DISPATCH

if model == "lr":
    metrics = run_lr_pipeline(
        TRACK=TRACK,
        SUBSET=SUBSET,
        train_with_dev=train_with_dev
    )

elif model == "gbdt":
    g_args = parse_gdbt_args(next_args)
    metrics = run_gbdt_pipeline(
        TRACK=TRACK, 
        train_with_dev=train_with_dev, 
        SUBSET=SUBSET,
        SAVE_FEATS=not g_args.no_save_feats
    )

elif model == "dkt" or model == "bert_dkt":
    d_args = parse_dkt_args(next_args)
    metrics = run_dkt_pipeline(
        model_name=model,
        TRACK=TRACK,
        SUBSET=SUBSET,
        train_with_dev=train_with_dev,
        EPOCHS=d_args.epochs
    )

elif model == "lmkt":
    lmkt_args = parse_lmkt_args(next_args)
    metrics = run_lmkt_pipeline(
        TRACK=TRACK,
        SUBSET=SUBSET,
        train_with_dev=train_with_dev,
        EPOCHS=lmkt_args.epochs
    )

else:
    raise ValueError(f"Unknown model name: {model}")


#=========== LOGGING

if not args.no_log:

    runtime_sec = perf_counter() - start_time
    runtime_min = runtime_sec / 60
    print(f"Total runtime (min): {runtime_min:.2f}")

    log_run(
        model_name=model,
        track=TRACK,
        train_with_dev=train_with_dev,
        subset=SUBSET,
        auc=metrics.get("auc"),
        accuracy=metrics.get("accuracy"),
        f1=metrics.get("f1"),
        runtime_min=runtime_min
    )

    logger.info("Run logged to database.")