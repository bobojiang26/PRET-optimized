#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/make_fake_h5_dataset.py \
  --out data/FAKEH5_BINARY/h5 \
  --slides 20 \
  --patches 96 \
  --dim 64 \
  --classes 2

DATASET_NAME=FAKEH5_BINARY \
H5_DIR=data/FAKEH5_BINARY/h5 \
WSI_DIR=data/FAKEH5_BINARY/images \
ANNO_DIR=data/FAKEH5_BINARY/anno \
DATASET_INFO=data_info/FAKEH5_BINARY.json \
DUMP_FEATURES=data/FAKEH5_BINARY/collected_features \
DUMP_RECORDS=records/FAKEH5_BINARY_slideLabel_eval.npy \
CLASS_NUM=1 \
EXAMPLE_NUM=1 \
VAL_NUM=6 \
TEST_NUM=6 \
RUNS=1 \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/run_h5_eval.sh
