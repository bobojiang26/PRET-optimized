#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/make_fake_h5_dataset.py \
  --out data/FAKEH5_7CLASS/h5 \
  --slides 70 \
  --patches 96 \
  --dim 64 \
  --classes 7

DATASET_NAME=FAKEH5_7CLASS \
H5_DIR=data/FAKEH5_7CLASS/h5 \
WSI_DIR=data/FAKEH5_7CLASS/images \
ANNO_DIR=data/FAKEH5_7CLASS/anno \
DATASET_INFO=data_info/FAKEH5_7CLASS.json \
DUMP_FEATURES=data/FAKEH5_7CLASS/collected_features \
DUMP_RECORDS=records/FAKEH5_7CLASS_slideLabel_eval.npy \
CLASS_NUM=7 \
EXAMPLE_NUM=1 \
VAL_NUM=21 \
TEST_NUM=21 \
RUNS=1 \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/run_h5_eval.sh
