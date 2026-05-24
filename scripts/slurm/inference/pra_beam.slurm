#!/bin/bash
#SBATCH --job-name=pra_beam
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:3
#SBATCH --time=36:00:00
# Adjust --partition / --gres GPU type / --account for your cluster.

# ============================================================================
# Beam-search SLURM launcher.
#
# GPU layout (3 GPUs):
#   GPU 0,1 -> policy model (pi_theta) + retriever
#   GPU 2   -> vLLM OpenAI server for reward model (mu_phi)
#
# Usage:
#   sbatch scripts/slurm/inference/pra_beam.slurm <config_path> [num_shards] [shard_id]
#
#   config_path - YAML readable by `pra-beam --config`
#   num_shards  - optional; if set with shard_id, passed to pra-beam
#   shard_id    - optional; if set with num_shards, passed to pra-beam
#                 (or use SLURM_ARRAY_TASK_ID with an array job)
#
# The config_path is the single source of truth for hyperparameters, dataset, and model paths.
# the single source of truth for hyperparameters, dataset, and model paths.
#
# Env (all optional; repo ships with sensible defaults):
#   HF_TOKEN              - HuggingFace token for gated models (required for gated ids)
#   PRA_DATA_ROOT         - expanded inside config paths (default: ./data)
#   PRA_OUTPUT_ROOT       - output root; traces land in $PRA_OUTPUT_ROOT/beam/ (default: ./outputs)
#   PRA_RETRIEVER_INDEX   - expanded inside config paths (default: ./data/faiss_index)
#   PRA_CONDA_ENV         - conda env to activate (default: pra_env)
#   VLLM_PORT              - optional explicit reward-model server port
#   VLLM_PORT_BASE         - base for auto port selection (default: 8400)
# ============================================================================

set -eo pipefail

CONFIG_PATH="${1:?Usage: sbatch pra_beam.slurm <config.yaml> [num_shards] [shard_id]}"
NUM_SHARDS_ARG="${2:-}"
SHARD_ID_ARG="${3:-}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 2
fi

# Environment
source "${HOME}/.bashrc"
conda activate "${PRA_CONDA_ENV:-pra_env}"

# Default the data/output roots so configs that reference $PRA_DATA_ROOT /
# $PRA_OUTPUT_ROOT still resolve on a fresh clone.
: "${PRA_DATA_ROOT:=./data}"
: "${PRA_OUTPUT_ROOT:=./outputs}"
export PRA_DATA_ROOT PRA_OUTPUT_ROOT

# Expand env-var placeholders (${PRA_DATA_ROOT} etc.) once so the Python CLI
# sees fully-resolved paths.
RESOLVED_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$RESOLVED_CONFIG"' EXIT
envsubst < "$CONFIG_PATH" > "$RESOLVED_CONFIG"

# Pull a couple of fields we need here (port calculation, logging tag).
read_config() {
  python - "$RESOLVED_CONFIG" "$1" <<'PY'
import sys, yaml
path, key = sys.argv[1], sys.argv[2]
with open(path) as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get(key, "") if cfg.get(key) is not None else "")
PY
}

CONFIG_NAME="$(read_config config_name)"
CONFIG_NAME="${CONFIG_NAME:-beam}"
REWARD_MODEL_PATH="$(read_config reward_model_path)"

if [ -n "$NUM_SHARDS_ARG" ] || [ -n "$SHARD_ID_ARG" ]; then
  if [ -z "$NUM_SHARDS_ARG" ] || [ -z "$SHARD_ID_ARG" ]; then
    echo "[ERROR] num_shards and shard_id must be provided together." >&2
    exit 5
  fi
  NUM_SHARDS="$NUM_SHARDS_ARG"
  SHARD_ID="$SHARD_ID_ARG"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  SHARD_ID="$SLURM_ARRAY_TASK_ID"
  NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-}"
  if [ -z "$NUM_SHARDS" ]; then
    echo "[ERROR] SLURM array detected but SLURM_ARRAY_TASK_COUNT is empty; pass [num_shards] [shard_id]." >&2
    exit 5
  fi
else
  NUM_SHARDS=""
  SHARD_ID=""
fi

if [ -n "$NUM_SHARDS" ]; then
  if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || ! [[ "$SHARD_ID" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] num_shards/shard_id must be non-negative integers." >&2
    exit 5
  fi
  if [ "$NUM_SHARDS" -lt 1 ] || [ "$SHARD_ID" -lt 0 ] || [ "$SHARD_ID" -ge "$NUM_SHARDS" ]; then
    echo "[ERROR] require num_shards>=1 and 0<=shard_id<num_shards (got ${SHARD_ID}/${NUM_SHARDS})." >&2
    exit 5
  fi
fi

LOG_TAG="${CONFIG_NAME}_${SLURM_JOB_ID}"
if [ -n "$NUM_SHARDS" ]; then
  LOG_TAG="${LOG_TAG}_sh${SHARD_ID}of${NUM_SHARDS}"
fi
LOG_DIR="$PRA_OUTPUT_ROOT/logs/beam/$(date +%Y-%m-%d)/${LOG_TAG}"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/output.log" 2> "$LOG_DIR/error.log"

echo "============================================"
echo "pra-beam: $CONFIG_NAME"
echo "  Job ID:   $SLURM_JOB_ID"
echo "  Node:     $(hostname)"
echo "  Started:  $(date)"
echo "  Config:   $CONFIG_PATH"
echo "  Logs:     $LOG_DIR"
if [ -n "$NUM_SHARDS" ]; then
  echo "  Sharding: shard ${SHARD_ID}/${NUM_SHARDS}"
fi
echo "============================================"
nvidia-smi

# Split allocated GPUs: first N-1 for policy+retriever, last one for vLLM server.
IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_ARRAY[@]}
if [ "$NUM_GPUS" -lt 2 ]; then
  echo "[ERROR] At least 2 GPUs are required (got $NUM_GPUS)." >&2
  exit 3
fi
SERVER_GPU="${GPU_ARRAY[$((NUM_GPUS-1))]}"
CLIENT_GPUS="$(IFS=,; echo "${GPU_ARRAY[*]:0:$((NUM_GPUS-1))}")"

echo "Server GPU:  $SERVER_GPU"
echo "Client GPUs: $CLIENT_GPUS"

# ----------------------------------------------------------------------------
# vLLM server for the reward model
# ----------------------------------------------------------------------------
BASE_PORT="${VLLM_PORT_BASE:-8400}"
if [ -n "${VLLM_PORT:-}" ]; then
  PORT="$VLLM_PORT"
elif [ -n "$SHARD_ID" ]; then
  PORT=$((BASE_PORT + SHARD_ID))
else
  PORT=$((BASE_PORT + SLURM_JOB_ID % 1000))
fi
VLLM_SERVER_URL="http://localhost:${PORT}"
SERVER_LOG="$LOG_DIR/vllm_server.log"

echo ""
echo "[SERVER] Starting vLLM on port $PORT (GPU $SERVER_GPU) for $REWARD_MODEL_PATH"
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
  echo "[CLEANUP] Stopping vLLM (pid=$VLLM_PID)..."
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  rm -f "$RESOLVED_CONFIG"
}
trap cleanup EXIT INT TERM

echo "[SERVER] Waiting for /health..."
for _ in $(seq 1 180); do
  if curl -fs "${VLLM_SERVER_URL}/health" > /dev/null 2>&1; then
    echo "[SERVER] Ready."
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[SERVER] vLLM died before becoming ready. Tail of server log:" >&2
    tail -n 200 "$SERVER_LOG" >&2
    exit 4
  fi
  sleep 2
done

# ----------------------------------------------------------------------------
# Client: pra-beam
# ----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$CLIENT_GPUS"

echo ""
echo "[CLIENT] Running pra-beam..."
if [ -n "$NUM_SHARDS" ]; then
  pra-beam \
    --config "$RESOLVED_CONFIG" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$SHARD_ID" \
    --reward_model_url "${VLLM_SERVER_URL}/v1/completions"
else
  pra-beam \
    --config "$RESOLVED_CONFIG" \
    --reward_model_url "${VLLM_SERVER_URL}/v1/completions"
fi

EXIT_CODE=$?
echo "============================================"
echo "Job completed at $(date), exit code: $EXIT_CODE"
echo "============================================"
exit "$EXIT_CODE"
