from pathlib import Path
import os
import runpy
import sys

from data_processing.parse_to_parquet import parse_and_save_all
from data_processing.reprocess_parquet import reprocess_and_save_all

parse_and_save_all()
reprocess_and_save_all()