
import argparse
import torch
import numpy as np
from db.log_db import log_run
from pipelines.dkt import run_dkt_pipeline, parse_dkt_args
from pipelines.gbdt import run_gbdt_pipeline, parse_gdbt_args
from time import perf_counter
from pipelines.lmkt import parse_lmkt_args, run_lmkt_pipeline
from pipelines.lr import run_lr_pipeline
import logging
from transformers import logging as hf_logging
from rich.logging import RichHandler
from pipelines.qg import parse_qg_args, run_qg_pipeline
import warnings

from pipelines.seqdkt import run_sdkt_pipeline

warnings.filterwarnings("ignore", message=".*loss_type.*")

hf_logging.set_verbosity_error()

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
               choices=["lr", "gbdt", "dkt", "bert_dkt", "lmkt", "qg", "sdkt", "vdkt"],
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
torch.manual_seed(42)
np.random.seed(42)

#-- start timer

start_time = perf_counter()

#========= DISPATCH

if MODEL == "lr":
    metrics = run_lr_pipeline(
        TRACK=TRACK,
        SUBSET=SUBSET,
        train_with_dev=TRAIN_WITH_DEV
    )

elif MODEL == "gbdt":
    records = run_gbdt_pipeline(
        track=TRACK, 
        train_with_dev=TRAIN_WITH_DEV, 
        SUBSET=SUBSET,
        next_args=next_args
    )

elif MODEL == "dkt" or MODEL == "bert_dkt":
    records = run_dkt_pipeline(
        model_name=MODEL,
        TRACK=TRACK,
        SUBSET=SUBSET,
        train_with_dev=TRAIN_WITH_DEV,
        ITEM_LEVEL=ITEM_LEVEL,
        epochs=EPOCHS,
        eval_every=EVAL_EVERY,
        next_args=next_args
    )

elif MODEL == "lmkt":
    records = run_lmkt_pipeline(
        TRACK=TRACK,
        SUBSET=SUBSET,
        train_with_dev=TRAIN_WITH_DEV,
        EPOCHS=EPOCHS
    )

elif MODEL == "qg":
    records = run_qg_pipeline(
        TRACK=TRACK,
        SUBSET=SUBSET,
        train_with_dev=TRAIN_WITH_DEV,
        EPOCHS=EPOCHS
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

else:
    raise ValueError(f"Unknown model name: {MODEL}")


#=========== LOGGING

if not args.no_log:

    runtime_sec = perf_counter() - start_time
    runtime_min = runtime_sec / 60
    logger.info(f"Total runtime (min): {runtime_min:.2f}")

    for i, rec in enumerate(records):
        log_run(rec, i, runtime_min)

    logger.info(f" {len(records)} runs logged to database.")