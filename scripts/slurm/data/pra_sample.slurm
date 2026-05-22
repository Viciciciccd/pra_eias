#!/bin/bash
#SBATCH --job-name=pra_sample
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
# Adjust --partition / --gres GPU type / --account for your cluster.

# ============================================================================
# Step 0 of the training-data pipeline: sample N reasoning chains per question.
#
# Usage:
#   sbatch scripts/slurm/data/pra_sample.slurm [mode] [model] [n] [input.jsonl] [output.json]
#
#     mode        - "train" (default) | "dev" | "test"  (only used to locate the
#                   default input jsonl and to tag the output filename)
#     model       - policy model (default: Qwen/Qwen3-4B-Instruct-2507)
#     n           - samples per question (default: 32)
#     input.jsonl - path to the split to sample over
#                   (default: $PRA_DATA_ROOT/medqa_4/<mode>.jsonl)
#     output.json - optional explicit output path; defaults to
#                   $PRA_OUTPUT_ROOT/sampled/<model_tag>_<mode>_<n>samples_<ts>.json
#
# Optional env (all have sensible repo-local defaults):
#   PRA_DATA_ROOT   - dataset root (default: ./data; MedQA-4 is in ./data/medqa_4)
#   PRA_OUTPUT_ROOT - output root for all pipeline artifacts (default: ./outputs)
#   PRA_CONDA_ENV   - conda env (default: pra_env)
# ============================================================================

set -eo pipefail

MODE="${1:-train}"
MODEL="${2:-Qwen/Qwen3-4B-Instruct-2507}"
SAMPLES="${3:-32}"
INPUT_JSONL_PATH="${4:-${PRA_DATA_ROOT:-./data}/medqa_4/${MODE}.jsonl}"
OUTPUT="${5:-}"

OUTPUT_ROOT="${PRA_OUTPUT_ROOT:-./outputs}"
MODEL_TAG="$(echo "$MODEL" | awk -F/ '{print $NF}')"
TS="$(date +%Y%m%d_%H%M%S)"
if [ -z "$OUTPUT" ]; then
  OUTPUT="$OUTPUT_ROOT/sampled/${MODEL_TAG}_${MODE}_${SAMPLES}samples_${TS}.json"
fi
mkdir -p "$(dirname "$OUTPUT")"

source "${HOME}/.bashrc"
conda activate "${PRA_CONDA_ENV:-pra_env}"

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | awk -F, 'NF { print NF }')
TP_SIZE="${NUM_GPUS:-1}"

LOG_DIR="$OUTPUT_ROOT/logs/sample/$(date +%Y-%m-%d)/${MODEL_TAG}_${MODE}_${SLURM_JOB_ID:-local}"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/output.log" 2> "$LOG_DIR/error.log"

echo "============================================"
echo "pra-sample"
echo "  Input:   $INPUT_JSONL_PATH"
echo "  Output:  $OUTPUT"
echo "  Model:   $MODEL  (tp=$TP_SIZE)"
echo "  Samples: $SAMPLES per question"
echo "  Job ID:  ${SLURM_JOB_ID:-local}"
echo "  Started: $(date)"
echo "============================================"
nvidia-smi

pra-sample \
  --input_jsonl_path     "$INPUT_JSONL_PATH" \
  --output_path          "$OUTPUT" \
  --policy_model_path    "$MODEL" \
  --samples_per_question "$SAMPLES" \
  --tensor_parallel_size "$TP_SIZE"

EXIT_CODE=$?
echo "============================================"
echo "Job completed at $(date), exit code: $EXIT_CODE"
echo "Wrote: $OUTPUT"
echo "============================================"
exit "$EXIT_CODE"
