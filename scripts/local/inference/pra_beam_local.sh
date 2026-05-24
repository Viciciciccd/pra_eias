#!/bin/bash
set -eo pipefail

# ==========================================================================
# Local beam-search launcher (non-SLURM).
#
# Mirrors scripts/slurm/inference/pra_beam.slurm:
#   - split GPUs: client gets first N-1, reward vLLM server gets last one
#   - start vLLM server for reward_model_path
#   - wait for /health
#   - run pra-beam with --reward_model_url injected
#
# Usage:
#   scripts/local/inference/pra_beam_local.sh <config_path> [gpu_csv] [num_shards] [shard_id]
#
#   config_path - YAML consumed by pra-beam --config
#   gpu_csv     - optional GPU ids, e.g. "0,1,2" (default: $CUDA_VISIBLE_DEVICES or 0,1,2)
#   num_shards  - optional; if set with shard_id, passed to pra-beam
#   shard_id    - optional; if set with num_shards, passed to pra-beam
#
# Env (all optional):
#   PRA_DATA_ROOT       - default: ./data
#   PRA_OUTPUT_ROOT     - default: ./outputs
#   PRA_RETRIEVER_INDEX - default: ./data/faiss_index
#   VLLM_PORT           - reward-model server port override
#
# Port rule:
#   - If VLLM_PORT is set, use it.
#   - Else if VLLM_PORT_BASE is set, use $VLLM_PORT_BASE (+ shard_id if sharded).
#   - Else use 8400 (+ shard_id if sharded).
# ==========================================================================

CONFIG_PATH="${1:?Usage: scripts/local/inference/pra_beam_local.sh <config.yaml> [gpu_csv] [num_shards] [shard_id]}"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 2
fi

GPU_CSV="${2:-${CUDA_VISIBLE_DEVICES:-0,1,2}}"
NUM_SHARDS="${3:-}"
SHARD_ID="${4:-}"
if [ -n "$NUM_SHARDS" ] || [ -n "$SHARD_ID" ]; then
  if [ -z "$NUM_SHARDS" ] || [ -z "$SHARD_ID" ]; then
    echo "[ERROR] num_shards and shard_id must be provided together." >&2
    exit 6
  fi
fi

IFS="," read -ra GPU_ARRAY <<< "$GPU_CSV"
NUM_GPUS=${#GPU_ARRAY[@]}
if [ "$NUM_GPUS" -gt 3 ]; then
  echo "[WARN] Beam uses 3 GPUs; trimming \"$GPU_CSV\" to first 3." >&2
  GPU_ARRAY=("${GPU_ARRAY[@]:0:3}")
  NUM_GPUS=3
fi
if [ "$NUM_GPUS" -lt 2 ]; then
  echo "[ERROR] At least 2 GPUs are required (got $NUM_GPUS from \"$GPU_CSV\")." >&2
  exit 3
fi
SERVER_GPU="${GPU_ARRAY[$((NUM_GPUS-1))]}"
CLIENT_GPUS="$(IFS=,; echo "${GPU_ARRAY[*]:0:$((NUM_GPUS-1))}")"

: "${PRA_DATA_ROOT:=./data}"
: "${PRA_OUTPUT_ROOT:=./outputs}"
: "${PRA_RETRIEVER_INDEX:=./data/faiss_index}"
export PRA_DATA_ROOT PRA_OUTPUT_ROOT PRA_RETRIEVER_INDEX

mkdir -p "$PRA_OUTPUT_ROOT/logs/beam"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PRA_OUTPUT_ROOT/logs/beam/local_${RUN_TAG}"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/vllm_server.log"

read_config() {
  python - "$CONFIG_PATH" "$1" <<"PY"
import os, sys, yaml
path, key = sys.argv[1], sys.argv[2]
with open(path) as f:
    cfg = yaml.safe_load(f) or {}
val = cfg.get(key, "")
if isinstance(val, str):
    val = os.path.expandvars(val)
print(val if val is not None else "")
PY
}

REWARD_MODEL_PATH="$(read_config reward_model_path)"
if [ -z "$REWARD_MODEL_PATH" ]; then
  echo "[ERROR] reward_model_path is empty in $CONFIG_PATH" >&2
  exit 4
fi

BASE_PORT="${VLLM_PORT_BASE:-8400}"
if [ -n "${VLLM_PORT:-}" ]; then
  PORT="$VLLM_PORT"
elif [ -n "$SHARD_ID" ]; then
  PORT=$((BASE_PORT + SHARD_ID))
else
  PORT="$BASE_PORT"
fi
VLLM_SERVER_URL="http://localhost:${PORT}"

echo "============================================"
echo "pra-beam local launcher"
echo "  Started:       $(date)"
echo "  Config:        $CONFIG_PATH"
echo "  Client GPUs:   $CLIENT_GPUS"
echo "  Server GPU:    $SERVER_GPU"
echo "  Reward model:  $REWARD_MODEL_PATH"
if [ -n "$NUM_SHARDS" ]; then
  echo "  Sharding:      shard ${SHARD_ID}/${NUM_SHARDS}"
fi
echo "  Server URL:    ${VLLM_SERVER_URL}/v1/completions"
echo "  Logs:          $LOG_DIR"
echo "============================================"
nvidia-smi || true

(
  export CUDA_VISIBLE_DEVICES="$SERVER_GPU"
  python -m vllm.entrypoints.openai.api_server \
    --model "$REWARD_MODEL_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    > "$SERVER_LOG" 2>&1
) &
VLLM_PID=$!

cleanup() {
  echo "[CLEANUP] stopping vLLM (pid=$VLLM_PID)"
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[SERVER] waiting for ${VLLM_SERVER_URL}/health ..."
for _ in $(seq 1 180); do
  if curl -fs "${VLLM_SERVER_URL}/health" > /dev/null 2>&1; then
    echo "[SERVER] ready"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[SERVER] vLLM died before becoming ready. Tail:" >&2
    tail -n 200 "$SERVER_LOG" >&2
    exit 5
  fi
  sleep 2
done

export CUDA_VISIBLE_DEVICES="$CLIENT_GPUS"
if [ -n "$NUM_SHARDS" ]; then
  pra-beam \
    --config "$CONFIG_PATH" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$SHARD_ID" \
    --reward_model_url "${VLLM_SERVER_URL}/v1/completions"
else
  pra-beam \
    --config "$CONFIG_PATH" \
    --reward_model_url "${VLLM_SERVER_URL}/v1/completions"
fi
