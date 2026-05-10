#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

trap 'echo "Stopping evaluation workers..."; pkill -TERM -P $$ 2>/dev/null || true; wait 2>/dev/null || true; exit 130' INT TERM

DATASETS=("charades-timelens" "activitynet-timelens" "qvhighlights-timelens")
if [[ -n "${datasets:-}" ]]; then
  IFS=',' read -ra DATASETS <<< "${datasets}"
fi

model_path=${model_path:-""}
model_id=${model_id:-"qwen3-vl"}
processor_path=${processor_path:-""}
bench_root=${bench_root:-"${TIMELENS_BENCH_ROOT:-}"}
pred_root=${pred_root:-"output/temporal_grounding/eval"}
min_tokens=${min_tokens:-64}
total_tokens=${total_tokens:-14336}
fps=${fps:-2}
fps_max_frames=${fps_max_frames:-""}
seed=${seed:-42}
max_new_tokens=${max_new_tokens:-1024}

IFS=',' read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

if [[ -z "${bench_root}" ]]; then
  echo "bench_root or TIMELENS_BENCH_ROOT is required."
  exit 1
fi
if [[ -z "${model_path}" ]]; then
  echo "model_path is required."
  exit 1
fi

optional_args=()
if [[ -n "${processor_path}" ]]; then optional_args+=(--processor_path "${processor_path}"); fi
if [[ -n "${fps_max_frames}" ]]; then optional_args+=(--fps_max_frames "${fps_max_frames}"); fi

run_tag="$(date +%Y%m%d-%H%M)"
pred_dir="${pred_root}/${model_id}/${run_tag}"
mkdir -p "${pred_dir}"

for dataset in "${DATASETS[@]}"; do
  pred_path="${pred_dir}/${dataset}"
  echo "Evaluating ${dataset}"
  echo "Output path: ${pred_path}.jsonl"

  for IDX in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.timelens.eval.eval_bench \
      --model_path "${model_path}" \
      --model_id "${model_id}" \
      "${optional_args[@]}" \
      --bench_root "${bench_root}" \
      --dataset "${dataset}" \
      --pred_path "${pred_path}" \
      --chunk "${CHUNKS}" \
      --index "${IDX}" \
      --min_tokens "${min_tokens}" \
      --total_tokens "${total_tokens}" \
      --fps "${fps}" \
      --max_new_tokens "${max_new_tokens}" \
      --seed "${seed}" &
  done
  wait

  shards=( "${pred_path}"_*.jsonl )
  if [[ ! -e "${shards[0]}" ]]; then
    echo "No shard outputs found under ${pred_path}_*.jsonl. Please check error logs above."
    exit 1
  fi
  cat "${pred_path}"_*.jsonl > "${pred_path}.jsonl"
  rm -f "${pred_path}"_*.jsonl
  python -m src.timelens.eval.metrics -f "${pred_path}.jsonl"
done
