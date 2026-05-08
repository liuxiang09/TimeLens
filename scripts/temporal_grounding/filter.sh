#!/usr/bin/env bash

set -euo pipefail

cleanup() {
  pkill -P $$ 2>/dev/null || true
  exit 130
}
trap cleanup SIGINT SIGTERM

export PYTHONPATH="./:${PYTHONPATH:-}"

dataset="gemini_refined_data"
model_path=""
model_id="qwen3-vl"
processor_path=""
train_jsonl="${TIMELENS_100K_JSONL:-}"
video_root="${TIMELENS_100K_VIDEO_ROOT:-}"
min_tokens=64
total_tokens=14336
fps=2
fps_max_frames=""
pred_root="output/temporal_grounding/filter-data"
seed=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) dataset="$2"; shift 2 ;;
    --model_path) model_path="$2"; shift 2 ;;
    --model_id) model_id="$2"; shift 2 ;;
    --processor_path) processor_path="$2"; shift 2 ;;
    --train_jsonl) train_jsonl="$2"; shift 2 ;;
    --video_root) video_root="$2"; shift 2 ;;
    --min_tokens) min_tokens="$2"; shift 2 ;;
    --total_tokens) total_tokens="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --fps_max_frames) fps_max_frames="$2"; shift 2 ;;
    --pred_root) pred_root="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${model_path}" ]]; then echo "--model_path is required."; exit 1; fi
if [[ -z "${fps_max_frames}" ]]; then fps_max_frames=$((total_tokens / min_tokens * 2)); fi

run_name="${model_id}/FPS-${fps}-maxframes-${fps_max_frames}_TOTALtokens-${total_tokens}_MINtokens-${min_tokens}---$(date +%Y%m%d_%H%M%S)"
pred_path="${pred_root}/${run_name}/${dataset}"
mkdir -p "$(dirname "${pred_path}")"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))}"
CHUNKS=${#GPULIST[@]}
echo "Using GPUs: ${GPULIST[*]}"
echo "Output path: ${pred_path}"

optional_args=()
if [[ -n "${processor_path}" ]]; then optional_args+=(--processor_path "${processor_path}"); fi
if [[ -n "${train_jsonl}" ]]; then optional_args+=(--train_jsonl "${train_jsonl}"); fi
if [[ -n "${video_root}" ]]; then optional_args+=(--video_root "${video_root}"); fi

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.timelens.data.filter_data \
    --dataset "${dataset}" \
    --split train \
    --pred_path "${pred_path}" \
    --model_path "${model_path}" \
    --model_id "${model_id}" \
    "${optional_args[@]}" \
    --chunk "${CHUNKS}" \
    --index "${IDX}" \
    --seed "${seed}" \
    --min_tokens "${min_tokens}" \
    --total_tokens "${total_tokens}" \
    --fps "${fps}" \
    --fps_max_frames "${fps_max_frames}" &
done

wait

shards=( "${pred_path}"_*.jsonl )
if [[ ! -e "${shards[0]}" ]]; then
  echo "No shard outputs found under ${pred_path}_*.jsonl. Please check error logs above."
  exit 1
fi
cat "${pred_path}"_*.jsonl > "${pred_path}.jsonl"
rm -f "${pred_path}"_*.jsonl
echo "Filtered results saved to: ${pred_path}.jsonl"
