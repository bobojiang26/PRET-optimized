#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_NAME="${DATASET_NAME:-MY_H5}"
H5_DIR="${H5_DIR:-data/${DATASET_NAME}/h5}"
WSI_DIR="${WSI_DIR:-data/${DATASET_NAME}/images}"
ANNO_DIR="${ANNO_DIR:-data/${DATASET_NAME}/anno}"
DATASET_INFO="${DATASET_INFO:-data_info/${DATASET_NAME}.json}"
DUMP_FEATURES="${DUMP_FEATURES:-data/${DATASET_NAME}/collected_features}"
DUMP_RECORDS="${DUMP_RECORDS:-records/${DATASET_NAME}_slideLabel_eval.npy}"

CLASS_NUM="${CLASS_NUM:-1}"
EXAMPLE_NUM="${EXAMPLE_NUM:-1}"
RUNS="${RUNS:-1}"
VAL_NUM="${VAL_NUM:-6}"
TEST_NUM="${TEST_NUM:-6}"
TOPK="${TOPK:-3}"
TOP_INSTANCE="${TOP_INSTANCE:-3}"
TEMPERATURE="${TEMPERATURE:-10}"
RELATED_THRESH="${RELATED_THRESH:-0.8}"
SEED="${SEED:-1024}"

"${PYTHON_BIN}" core/main.py \
  --mode eval \
  --topk "${TOPK}" \
  --temperature "${TEMPERATURE}" \
  --related_thresh "${RELATED_THRESH}" \
  --example_num "${EXAMPLE_NUM}" \
  --raw_feature_path "${H5_DIR}" \
  --wsi_path "${WSI_DIR}" \
  --dump_features "${DUMP_FEATURES}" \
  --dataset_info "${DATASET_INFO}" \
  --seed "${SEED}" \
  --top_instance "${TOP_INSTANCE}" \
  --test_num "${TEST_NUM}" \
  --val_num "${VAL_NUM}" \
  --prompt_type slideLabel \
  --prompt_path "${ANNO_DIR}" \
  --ignore 0 \
  --file_min_size 0 \
  --class_num "${CLASS_NUM}" \
  --runs "${RUNS}" \
  --dump_records "${DUMP_RECORDS}"
