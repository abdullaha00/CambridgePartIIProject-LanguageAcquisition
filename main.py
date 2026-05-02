
import argparse
import sys
import torch
import numpy as np
from db.log_db import GenerationRecord, MetricRecord, log_run_g, log_run_m
from time import perf_counter
import logging
from rich.logging import RichHandler
from pipelines.qg import parse_qg_args, run_qg_pipeline
import warnings

# === PIPELINE IMPORTS
from pipelines.gbdt import run_gbdt_pipeline
from pipelines.lmkt import run_lmkt_pipeline
from pipelines.sdkt import run_sdkt_pipeline
from pipelines.qg import run_qg_pipeline
from pipelines.lr import run_lr_pipeline
from pipelines.dkt import run_dkt_pipeline
from pipelines.aqg_kt import run_aqg_dkt_pipeline
from pipelines.aqg_qg import run_aqg_qg_pipeline
from pipelines.sdkt import run_sdkt_pipeline
from pipelines.fa_bilstm import run_fa_bilstm_pipeline

warnings.filterwarnings("ignore", message=".*loss_type.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


#==== ARGUMENTS

p = argparse.ArgumentParser()
p.add_argument("model_name", 
               choices=["lr", "gbdt", "dkt", "bert-dkt", "lmkt", "qg", "sdkt", "vdkt", "aqg-dkt", "aqg-qg", "fa-bilstm"],
               help="Which model to run")
p.add_argument("-t", "--track", type=str, default="en_es", choices=["en_es", "fr_en", "es_en", "all"])
p.add_argument("-d", "--train-with-dev", action="store_true", default=False)
p.add_argument("-s", "--subset", type=int, default=None)
p.add_argument("--no_log", action="store_true")
p.add_argument("--tag", type=str, default=None, help="Tag to label this run")
p.add_argument("-i","--item-level", choices=["token", "exercise"], default="token")
p.add_argument("-e","--epochs", type=int, default=5)
p.add_argument("--eval-every", type=int, default=1)
p.add_argument("--save-every", type=int, default=1, help="Save a checkpoint every N epochs (default: only save final)")
p.add_argument("--resume", type=str, default=None, help="Path to a .ckpt file to resume training from")

args, next_args = p.parse_known_args()

MODEL = args.model_name
TRACK = args.track
TRAIN_WITH_DEV = args.train_with_dev
SUBSET = args.subset
ITEM_LEVEL = args.item_level
EPOCHS = args.epochs
EVAL_EVERY = args.eval_every

if EPOCHS is not None and EVAL_EVERY is None:
    EVAL_EVERY = max(1, EPOCHS // 5)  # default to evaluating 5 times per run

#===== CONFIG
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

#-- start timer

start_time = perf_counter()

#========= DISPATCH

if MODEL == "lr":
    records = run_lr_pipeline(
        TRACK=TRACK,
        SUBSET=SUBSET,
        train_with_dev=TRAIN_WITH_DEV,
        tag=args.tag,
    )

elif MODEL == "gbdt":
    records = run_gbdt_pipeline(
        track=TRACK, 
        train_with_dev=TRAIN_WITH_DEV, 
        SUBSET=SUBSET,
        next_args=next_args,
        tag=args.tag,
    )

elif MODEL == "dkt" or MODEL == "bert-dkt":
    records = run_dkt_pipeline(
        model_name=MODEL,
        TRACK=TRACK,
        SUBSET=SUBSET,
        train_with_dev=TRAIN_WITH_DEV,
        item_level=ITEM_LEVEL,
        epochs=EPOCHS,
        eval_every=EVAL_EVERY,
        save_every=args.save_every,
        resume_from=args.resume,
        next_args=next_args,
        tag=args.tag,
    )

elif MODEL == "lmkt":
    records = run_lmkt_pipeline(
        TRACK=TRACK,
        SUBSET=SUBSET,
        train_with_dev=TRAIN_WITH_DEV,
        EPOCHS=EPOCHS,
        eval_every=EVAL_EVERY,
        save_every=args.save_every,
        resume_from=args.resume,
        tag=args.tag,
    )

elif MODEL == "qg":
    records = run_qg_pipeline(
        track=TRACK,
        SUBSET=SUBSET,
        train_with_dev=TRAIN_WITH_DEV,
        EPOCHS=EPOCHS,
        extra_args=next_args,
        tag=args.tag,
    )

elif MODEL == "sdkt" or MODEL == "vdkt":
    records = run_sdkt_pipeline(
        model_name=MODEL,
        TRACK=TRACK,
        SUBSET=SUBSET,
        train_with_dev=TRAIN_WITH_DEV,
        EPOCHS=EPOCHS,
        eval_every=EVAL_EVERY,
        next_args=next_args,
        tag=args.tag,
        save_every=args.save_every,
        resume_from=args.resume,
    )

elif MODEL == "fa-bilstm":
    records = run_fa_bilstm_pipeline(
        track=TRACK,
        subset=SUBSET,
        train_with_dev=TRAIN_WITH_DEV,
        epochs=EPOCHS,
        eval_every=EVAL_EVERY,
        next_args=next_args,
        tag=args.tag,
        save_every=args.save_every,
        resume_from=args.resume,
    )

elif MODEL == "aqg_dkt":
    records, _ = run_aqg_dkt_pipeline(
        track=TRACK,
        subset=SUBSET,
        train_with_dev=TRAIN_WITH_DEV,
        EPOCHS=EPOCHS,
        eval_every=EVAL_EVERY,
        next_args=next_args,
        tag=args.tag,
        save_every=args.save_every,
    )
elif MODEL == "aqg_qg":
    records = run_aqg_qg_pipeline(
        track=TRACK,
        subset=SUBSET,
        train_with_dev=TRAIN_WITH_DEV,
        EPOCHS=EPOCHS,
        eval_every=EVAL_EVERY,
        next_args=next_args,
        tag=args.tag,
        save_every=args.save_every,
    )

else:
    raise ValueError(f"Unknown model name: {MODEL}")


#=========== LOGGING

if not args.no_log:

    runtime_sec = perf_counter() - start_time
    runtime_min = runtime_sec / 60
    logger.info(f"Total runtime (min): {runtime_min:.2f}")

    for i, rec in enumerate(records):

        if isinstance(rec, MetricRecord):
            log_run_m(rec, index=i, runtime_min=runtime_min)
        else:
            log_run_g(rec, index=i, runtime_min=runtime_min)

    logger.info(f"{len(records)} runs logged to database." if len(records) > 1 else
                "1 run logged to database.")
