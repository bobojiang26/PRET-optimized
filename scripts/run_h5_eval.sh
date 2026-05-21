#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_NAME="${DATASET_NAME:-MY_H5}"
H5_DIR="${H5_DIR:-data/${DATASET_NAME}/h5}"
WSI_DIR="${WSI_DIR:-data/${DATASET_NAME}/images}"
ANNO_DIR="${ANNO_DIR:-data/${DATASET_NAME}/anno}"
DATASET_INFO="${DATASET_INFO:-data_info/${DATASET_NAME}.json}"
DUMP_FEATURES="${DUMP_FEATURES:-data/${DATASET_NAME}/collected_features}"
PROMPT_TYPE="${PROMPT_TYPE:-slideLabel}"
DUMP_RECORDS="${DUMP_RECORDS:-records/${DATASET_NAME}_${PROMPT_TYPE}_eval.npy}"

CLASS_NUM="${CLASS_NUM:-1}"
H5_COORDINATE_MODE="${H5_COORDINATE_MODE:-auto}"
H5_PIXEL_STEP_THRESHOLD="${H5_PIXEL_STEP_THRESHOLD:-16}"
H5_PATCH_SIZE="${H5_PATCH_SIZE:-0}"
SEG="${SEG:-0}"
MULTILABEL="${MULTILABEL:-0}"
EXAMPLE_NUM="${EXAMPLE_NUM:-1}"
RUNS="${RUNS:-1}"
VAL_NUM="${VAL_NUM:-6}"
TEST_NUM="${TEST_NUM:-6}"
TOPK="${TOPK:-3}"
TOP_INSTANCE="${TOP_INSTANCE:-3}"
TEMPERATURE="${TEMPERATURE:-10}"
RELATED_THRESH="${RELATED_THRESH:-0.8}"
SEED="${SEED:-1024}"
REFERENCE_TOKEN_BUDGET="${REFERENCE_TOKEN_BUDGET:-}"
REFERENCE_SPARSIFY_STRATEGY="${REFERENCE_SPARSIFY_STRATEGY:-}"
REFERENCE_ANCHOR_RATIO="${REFERENCE_ANCHOR_RATIO:-}"
REFERENCE_RANDOM_RATIO="${REFERENCE_RANDOM_RATIO:-}"
SIMILARITY_AGGREGATION="${SIMILARITY_AGGREGATION:-}"
SIMILARITY_TEMPERATURE="${SIMILARITY_TEMPERATURE:-}"
ADAPTIVE_MIN_TOPK="${ADAPTIVE_MIN_TOPK:-}"
ADAPTIVE_WINDOW="${ADAPTIVE_WINDOW:-}"
CONTEXT_CENTERING="${CONTEXT_CENTERING:-}"
SPATIAL_SMOOTH_STRENGTH="${SPATIAL_SMOOTH_STRENGTH:-}"
SPATIAL_SMOOTH_RADIUS="${SPATIAL_SMOOTH_RADIUS:-}"
SPATIAL_FEATURE_WEIGHT="${SPATIAL_FEATURE_WEIGHT:-}"
CONFORMAL_ALPHA="${CONFORMAL_ALPHA:-}"
REQUIRE_LABEL="${REQUIRE_LABEL:-0}"

MULTIPLE_ARGS=()
if [[ -n "${MULTIPLE_NUM:-}" ]]; then
  read -r -a MULTIPLE_ARGS <<< "${MULTIPLE_NUM}"
  MULTIPLE_ARGS=(--multiple_num "${MULTIPLE_ARGS[@]}")
fi

SPARSE_ARGS=()
if [[ -n "${REFERENCE_TOKEN_BUDGET}" ]]; then
  SPARSE_ARGS+=(--reference_token_budget "${REFERENCE_TOKEN_BUDGET}")
fi
if [[ -n "${REFERENCE_SPARSIFY_STRATEGY}" ]]; then
  SPARSE_ARGS+=(--reference_sparsify_strategy "${REFERENCE_SPARSIFY_STRATEGY}")
fi
if [[ -n "${REFERENCE_ANCHOR_RATIO}" ]]; then
  SPARSE_ARGS+=(--reference_anchor_ratio "${REFERENCE_ANCHOR_RATIO}")
fi
if [[ -n "${REFERENCE_RANDOM_RATIO}" ]]; then
  SPARSE_ARGS+=(--reference_random_ratio "${REFERENCE_RANDOM_RATIO}")
fi

RESEARCH_ARGS=()
if [[ -n "${SIMILARITY_AGGREGATION}" ]]; then
  RESEARCH_ARGS+=(--similarity_aggregation "${SIMILARITY_AGGREGATION}")
fi
if [[ -n "${SIMILARITY_TEMPERATURE}" ]]; then
  RESEARCH_ARGS+=(--similarity_temperature "${SIMILARITY_TEMPERATURE}")
fi
if [[ -n "${ADAPTIVE_MIN_TOPK}" ]]; then
  RESEARCH_ARGS+=(--adaptive_min_topk "${ADAPTIVE_MIN_TOPK}")
fi
if [[ -n "${ADAPTIVE_WINDOW}" ]]; then
  RESEARCH_ARGS+=(--adaptive_window "${ADAPTIVE_WINDOW}")
fi
if [[ -n "${CONTEXT_CENTERING}" ]]; then
  RESEARCH_ARGS+=(--context_centering "${CONTEXT_CENTERING}")
fi
if [[ -n "${SPATIAL_SMOOTH_STRENGTH}" ]]; then
  RESEARCH_ARGS+=(--spatial_smooth_strength "${SPATIAL_SMOOTH_STRENGTH}")
fi
if [[ -n "${SPATIAL_SMOOTH_RADIUS}" ]]; then
  RESEARCH_ARGS+=(--spatial_smooth_radius "${SPATIAL_SMOOTH_RADIUS}")
fi
if [[ -n "${SPATIAL_FEATURE_WEIGHT}" ]]; then
  RESEARCH_ARGS+=(--spatial_feature_weight "${SPATIAL_FEATURE_WEIGHT}")
fi
if [[ -n "${CONFORMAL_ALPHA}" ]]; then
  RESEARCH_ARGS+=(--conformal_alpha "${CONFORMAL_ALPHA}")
fi

SEG_ARGS=()
if [[ "${SEG}" == "1" || "${SEG}" == "true" || "${SEG}" == "TRUE" ]]; then
  SEG_ARGS+=(--seg)
fi
if [[ "${MULTILABEL}" == "1" || "${MULTILABEL}" == "true" || "${MULTILABEL}" == "TRUE" ]]; then
  SEG_ARGS+=(--multilabel)
fi
if [[ "${REQUIRE_LABEL}" == "1" || "${REQUIRE_LABEL}" == "true" || "${REQUIRE_LABEL}" == "TRUE" ]]; then
  SEG_ARGS+=(--require_label)
fi

EXTRA_ARGS=("$@")

"${PYTHON_BIN}" core/main.py \
  --mode eval \
  --topk "${TOPK}" \
  --temperature "${TEMPERATURE}" \
  --related_thresh "${RELATED_THRESH}" \
  --example_num "${EXAMPLE_NUM}" \
  ${MULTIPLE_ARGS[@]+"${MULTIPLE_ARGS[@]}"} \
  --raw_feature_path "${H5_DIR}" \
  --wsi_path "${WSI_DIR}" \
  --dump_features "${DUMP_FEATURES}" \
  --dataset_info "${DATASET_INFO}" \
  --seed "${SEED}" \
  --top_instance "${TOP_INSTANCE}" \
  --test_num "${TEST_NUM}" \
  --val_num "${VAL_NUM}" \
  --prompt_type "${PROMPT_TYPE}" \
  --prompt_path "${ANNO_DIR}" \
  --ignore 0 \
  --file_min_size 0 \
  --class_num "${CLASS_NUM}" \
  --h5_coordinate_mode "${H5_COORDINATE_MODE}" \
  --h5_pixel_step_threshold "${H5_PIXEL_STEP_THRESHOLD}" \
  --h5_patch_size "${H5_PATCH_SIZE}" \
  --runs "${RUNS}" \
  --dump_records "${DUMP_RECORDS}" \
  ${SEG_ARGS[@]+"${SEG_ARGS[@]}"} \
  ${SPARSE_ARGS[@]+"${SPARSE_ARGS[@]}"} \
  ${RESEARCH_ARGS[@]+"${RESEARCH_ARGS[@]}"} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
