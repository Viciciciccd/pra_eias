#!/bin/bash
#SBATCH --job-name=pra_sft
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=7-00:00:00
# Adjust --partition / --gres GPU type / --account for your cluster.

# ============================================================================
# PRA SFT training launcher (DDP via torchrun).
#
# Usage:
#   sbatch scripts/slurm/training/pra_sft.slurm <config_path> [nproc_per_node]
#
#   config_path      - YAML consumed by `pra-sft --config`
#   nproc_per_node   - GPUs to use via torchrun (default: auto-detect from
#                      $CUDA_VISIBLE_DEVICES or SLURM_GPUS_ON_NODE).
#
# Required env:
#   HF_TOKEN          - HuggingFace token for gated models/datasets
#   WANDB_API_KEY     - (optional) W&B key; if unset, training runs in offline mode
#   PRA_OUTPUT_ROOT   - output root (checkpoints go to $PRA_OUTPUT_ROOT/sft/<run>/)
#   PRA_CONDA_ENV     - conda env to activate (default: pra_env)
# ============================================================================

set -eo pipefail

CONFIG_PATH="${1:?Usage: sbatch pra_sft.slurm <config.yaml> [nproc_per_node]}"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 2
fi

NPROC="${2:-}"
if [ -z "$NPROC" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NPROC=${#_GPUS[@]}
  elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NPROC="$SLURM_GPUS_ON_NODE"
  else
    NPROC=1
  fi
fi

source "${HOME}/.bashrc"
conda activate "${PRA_CONDA_ENV:-pra_env}"

# Required so configs/training/*.yaml can reference ${PRA_OUTPUT_ROOT}/sft.
: "${PRA_OUTPUT_ROOT:=./outputs}"
export PRA_OUTPUT_ROOT

CONFIG_TAG="$(basename "$CONFIG_PATH" .yaml)"
LOG_DIR="$PRA_OUTPUT_ROOT/logs/sft/$(date +%Y-%m-%d)/${CONFIG_TAG}_${SLURM_JOB_ID:-local}"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/output.log" 2> "$LOG_DIR/error.log"

echo "============================================"
echo "pra-sft: $CONFIG_TAG"
echo "  Job ID:          ${SLURM_JOB_ID:-local}"
echo "  Node:            $(hostname)"
echo "  Started:         $(date)"
echo "  Config:          $CONFIG_PATH"
echo "  nproc_per_node:  $NPROC"
echo "  Logs:            $LOG_DIR"
echo "============================================"
nvidia-smi

# Single timestamp shared across ranks so every worker writes to the same
# output directory.
export TRAIN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
echo "MASTER_PORT=$MASTER_PORT  TRAIN_TIMESTAMP=$TRAIN_TIMESTAMP"

torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  -m pra.training.sft --config "$CONFIG_PATH"

EXIT_CODE=$?
echo "============================================"
echo "Job completed at $(date), exit code: $EXIT_CODE"
echo "============================================"
exit "$EXIT_CODE"
