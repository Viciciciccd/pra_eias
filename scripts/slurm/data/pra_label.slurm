#!/bin/bash
#SBATCH --job-name=pra_label
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=800G
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
# Adjust --partition / --gres GPU type / --account for your cluster.

# ============================================================================
# Step 2 of the training-data pipeline: per-step reasoning-reward labeling.
#
# Uses a large teacher LLM (Qwen3-235B-A22B-Instruct-2507 FP8 by default) to
# score reasoning steps. Requires 8 GPUs for tensor-parallel inference.
#
# Usage:
#   sbatch scripts/slurm/data/pra_label.slurm <input.json> [teacher_model_path] [output.json]
#
#     input.json          - JSON produced by pra-retrieve
#     teacher_model_path  - HF id / local path of a teacher LLM (default:
#                           Qwen/Qwen3-235B-A22B-Instruct-2507). Any path works;
#                           adjust vLLM settings in src/pra/data/label.py as needed.
#     output.json         - optional explicit output path; defaults to
#                           $PRA_OUTPUT_ROOT/labeled/<teacher_tag>_<input_stem>_<ts>/results_list.json
#
# Required env:
#   PRA_OUTPUT_ROOT - output root for all pipeline artifacts (default: ./outputs)
#   PRA_CONDA_ENV   - conda env (default: pra_env)
# ============================================================================

set -eo pipefail

INPUT="${1:?Usage: sbatch pra_label.slurm <input.json> [teacher_model_path] [output.json]}"
TEACHER_MODEL_PATH="${2:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
OUTPUT="${3:-}"

# Slashes in the HF id would create spurious nested dirs; use the leaf as a tag.
TEACHER_TAG="$(echo "$TEACHER_MODEL_PATH" | awk -F/ '{print $NF}')"

OUTPUT_ROOT="${PRA_OUTPUT_ROOT:-./outputs}"
TS="$(date +%Y%m%d_%H%M%S)"
INPUT_STEM="$(basename "$INPUT" .json)"
if [ -z "$OUTPUT" ]; then
  OUTPUT="$OUTPUT_ROOT/labeled/${TEACHER_TAG}_${INPUT_STEM}_${TS}/results_list.json"
fi
mkdir -p "$(dirname "$OUTPUT")"

source "${HOME}/.bashrc"
conda activate "${PRA_CONDA_ENV:-pra_env}"

LOG_DIR="$OUTPUT_ROOT/logs/label/$(date +%Y-%m-%d)/${TEACHER_TAG}_${SLURM_JOB_ID:-local}"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/output.log" 2> "$LOG_DIR/error.log"

echo "============================================"
echo "pra-label: teacher_model_path=$TEACHER_MODEL_PATH"
echo "  Input:   $INPUT"
echo "  Output:  $OUTPUT"
echo "  Job ID:  ${SLURM_JOB_ID:-local}"
echo "  Node:    $(hostname)"
echo "  Started: $(date)"
echo "============================================"
nvidia-smi

pra-label --input "$INPUT" --output "$OUTPUT" --teacher_model_path "$TEACHER_MODEL_PATH"

EXIT_CODE=$?
echo "============================================"
echo "Job completed at $(date), exit code: $EXIT_CODE"
echo "Wrote: $OUTPUT"
echo "============================================"
exit "$EXIT_CODE"
