#!/bin/bash
#SBATCH --job-name=pra_retrieve
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
# Adjust --partition / --gres GPU type / --account for your cluster.

# ============================================================================
# Step 1 of the training-data pipeline: multi-GPU MedCPT retrieval.
#
# Usage:
#   sbatch scripts/slurm/data/pra_retrieve.slurm <input.json> [version] [output.json]
#
#     input.json    - sampled items produced by pra-sample
#     version       - cumulative | window (last-2 steps, default)
#     output.json   - optional explicit output path; defaults to
#                     $PRA_OUTPUT_ROOT/retrieved/<input_stem>_<version>_<ts>.json
#
# Required env:
#   PRA_RETRIEVER_INDEX  - directory with FAISS *.index and {source}_texts.json (default: ./data/faiss_index)
#   PRA_OUTPUT_ROOT      - output root for all pipeline artifacts (default: ./outputs)
#   PRA_CONDA_ENV        - conda env (default: pra_env)
# ============================================================================

set -eo pipefail

INPUT="${1:?Usage: sbatch pra_retrieve.slurm <input.json> [version] [output.json]}"
VERSION="${2:-window}"
OUTPUT="${3:-}"

INDEX_PATH="${PRA_RETRIEVER_INDEX:-./data/faiss_index}"

OUTPUT_ROOT="${PRA_OUTPUT_ROOT:-./outputs}"
TS="$(date +%Y%m%d_%H%M%S)"
INPUT_STEM="$(basename "$INPUT" .json)"
if [ -z "$OUTPUT" ]; then
  OUTPUT="$OUTPUT_ROOT/retrieved/${INPUT_STEM}_${VERSION}_${TS}.json"
fi
mkdir -p "$(dirname "$OUTPUT")"

source "${HOME}/.bashrc"
conda activate "${PRA_CONDA_ENV:-pra_env}"

LOG_DIR="$OUTPUT_ROOT/logs/retrieve/$(date +%Y-%m-%d)/${VERSION}_${SLURM_JOB_ID:-local}"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/output.log" 2> "$LOG_DIR/error.log"

echo "============================================"
echo "pra-retrieve: version=$VERSION"
echo "  Input:   $INPUT"
echo "  Output:  $OUTPUT"
echo "  Index:   $INDEX_PATH"
echo "  Job ID:  ${SLURM_JOB_ID:-local}"
echo "  Started: $(date)"
echo "============================================"
nvidia-smi

pra-retrieve \
  --input      "$INPUT" \
  --output     "$OUTPUT" \
  --index_path "$INDEX_PATH" \
  --version    "$VERSION"

EXIT_CODE=$?
echo "============================================"
echo "Job completed at $(date), exit code: $EXIT_CODE"
echo "Wrote: $OUTPUT"
echo "============================================"
exit "$EXIT_CODE"
