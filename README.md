<p align="center">
  <img src="media/pra-logo.png" alt="Process Reward Agents" width="910">
</p>

# Process Reward Agents for Steering Knowledge-Intensive Reasoning

<p align="center">
  <a href="https://icml.cc/Conferences/2026" target="_blank"><img src="https://img.shields.io/badge/ICML-2026-blueviolet"></a>
  <a href="https://arxiv.org/abs/2604.09482" target="_blank"><img src="https://img.shields.io/badge/arXiv-2604.09482-red"></a>
  <a href="https://process-reward-agents.github.io/" target="_blank"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
  <a href="https://huggingface.co/process-reward-agents" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20-Model-orange"></a>
  <a href="https://huggingface.co/process-reward-agents" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20-Dataset-orange"></a>
  <a href="https://github.com/eth-medical-ai-lab/pra" target="_blank"><img src="https://img.shields.io/badge/Code-GitHub-brightgreen"></a>
</p>

### News

- **2026** — Process Reward Agents accepted to **ICML 2026** — see you in Seoul!
- **2026** — Process Reward Agents preprint released on arXiv.

## Install

```bash
conda env create -f environment.yml
conda activate pra_env
pip install -e ".[training]"          # inference + training
# or
pip install -e .                      # inference only
```

### Environment variables

The launchers and CLIs read three roots. All three have sensible in-repo
defaults set in `pra/__init__.py`, so a fresh clone works without exporting
anything:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PRA_DATA_ROOT` | `./data` | Input datasets |
| `PRA_OUTPUT_ROOT` | `./outputs` | Pipeline artifacts (traces, checkpoints, logs) |
| `PRA_RETRIEVER_INDEX` | `./data/faiss_index` | MedCPT FAISS index dir |
| `PRA_CONDA_ENV` | `pra_env` | Conda env activated by SLURM wrappers |

Override any of them if you want artifacts to live elsewhere:

```bash
export PRA_DATA_ROOT=/path/to/data
export PRA_OUTPUT_ROOT=/path/to/outputs
export PRA_RETRIEVER_INDEX=/path/to/index
export PRA_CONDA_ENV=pra_env
```

### Default models and data

The repo is wired to the checkpoints and dataset we release. Swap any of
them via CLI flag or YAML key.

| | Default | Override |
| --- | --- | --- |
| Dataset | `data/medqa_4/{dev,test,train}.jsonl` from [`jind11/MedQA`](https://github.com/jind11/MedQA) | `--input_jsonl_path`, or `input_jsonl_path` in the inference YAML |
| Policy model | `Qwen/Qwen3-4B-Instruct-2507` | `--policy_model_path`, or `policy_model_path` in the inference YAML |
| Reward-model backbone | `Qwen/Qwen3-4B-Instruct-2507` | `--reward_model_backbone_path`, or `reward_model_backbone_path` in `configs/training/qwen3_4b.yaml` |
| Reward model | `process-reward-agents/Qwen3-4B-Instruct-2507_SFT_all_docs_bs2x2_lr3e-05_20260420_140000_epoch_3` | `--reward_model_path`, or `reward_model_path` in the inference YAML |
| Teacher (step labeler) | `Qwen/Qwen3-235B-A22B-Instruct-2507` (FP8) | `--teacher_model_path` on `pra-label` |

Every key in the inference YAML is also accepted as a `--<key>` CLI flag on
`pra-beam`; explicit CLI flags take precedence over YAML values.

### Build the retrieval index (BYO corpora)

We do not redistribute the index files due to mixed licenses on the
source corpora. Download the originals from these sources, chunk them
(e.g. with [`RecursiveCharacterTextSplitter`](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)),
embed with [`ncbi/MedCPT-Article-Encoder`](https://huggingface.co/ncbi/MedCPT-Article-Encoder),
and write `<source>.index` (FAISS `IndexFlatIP`, dim 768) and
`<source>_texts.json` (`[{"text": ..., "source": ...}, ...]`) into
`$PRA_RETRIEVER_INDEX/`.

Sources used in the paper:

| Source       | Where to download                              |
| ------------ | ---------------------------------------------- |
| `cpg`        | [`epfl-llm/guidelines`](https://huggingface.co/datasets/epfl-llm/guidelines) (Meditron) |
| `recop`      | [`guan-wang/ReCOP`](https://huggingface.co/datasets/guan-wang/ReCOP) |
| `textbooks`  | [`jind11/MedQA`](https://github.com/jind11/MedQA) |
| `statpearls` | [NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK430685/) |

### Output layout

Each stage writes under `$PRA_OUTPUT_ROOT`:

```
$PRA_OUTPUT_ROOT/
  sampled/    <model>_<mode>_<N>samples_<ts>.json           # pra-sample
  retrieved/  <input_stem>_<version>_<ts>.json              # pra-retrieve
  labeled/    <teacher>_<input_stem>_<ts>/results_list.json # pra-label
  sft/        <run_name>/                                   # pra-sft
  beam/       <config_name>_<job_id>/                       # pra-beam
  logs/       sample|retrieve|label|sft|beam/<date>/<run>/  # SLURM stdout/stderr
```

Each SLURM wrapper takes an optional output path as the last positional
argument. When omitted, it builds a timestamped path under
`$PRA_OUTPUT_ROOT/<stage>/` derived from the input filename, so concurrent
runs never overwrite each other.

## Quick start

The SLURM wrappers under `scripts/slurm/` ship without a `--partition` or
GPU-type filter. Add `--partition=<your_partition>`, change `--gres=gpu:N`
to your cluster's GPU type if needed (e.g. `gpu:H100:N`), and set
`--account=<...>` to match your site before `sbatch`.

### Beam-search inference

Single run (no sharding):

```bash
# SLURM (3 GPUs: 2 for policy+retriever, 1 for the reward vLLM server)
sbatch scripts/slurm/inference/pra_beam.slurm configs/inference/beam_qwen3_4b.yaml
# -> $PRA_OUTPUT_ROOT/beam/<config_name>_<ts>_<job_id>/

# Local (same 2+1 GPU split)
scripts/local/inference/pra_beam_local.sh configs/inference/beam_qwen3_4b.yaml 0,1,2
# -> $PRA_OUTPUT_ROOT/logs/beam/local_<ts>/
```

Sharded run (parallelize the dataset across N workers; each shard needs its
own reward-model vLLM server, on a distinct port):

```bash
# SLURM array (3 shards): SLURM_ARRAY_TASK_ID becomes shard_id
sbatch --array=0-2 scripts/slurm/inference/pra_beam.slurm configs/inference/beam_qwen3_4b.yaml 3

# Or launch shards by hand
pra-beam --config configs/inference/beam_qwen3_4b.yaml --num_shards 3 --shard_id 0
pra-beam --config configs/inference/beam_qwen3_4b.yaml --num_shards 3 --shard_id 1
pra-beam --config configs/inference/beam_qwen3_4b.yaml --num_shards 3 --shard_id 2

# Local wrapper: auto-selects port 8000 + shard_id
scripts/local/inference/pra_beam_local.sh configs/inference/beam_qwen3_4b.yaml 0,1,2 3 0
scripts/local/inference/pra_beam_local.sh configs/inference/beam_qwen3_4b.yaml 3,4,5 3 1
scripts/local/inference/pra_beam_local.sh configs/inference/beam_qwen3_4b.yaml 6,7,8 3 2

# Merge shard outputs and compute accuracy
pra-merge-shards --trace_dir ./outputs/beam --run_tag beam_qwen3_4b_job123 --num_shards 3
```

All inference hyperparameters live in the YAML under `configs/inference/`
(pass any file via `--config`).

If `VLLM_PORT` is set, both launchers use it. Otherwise `pra-beam` resolves
`reward_model_url` from `8000 + shard_id` (sharded) or `8000` / a SLURM
job-id-derived port (non-sharded). The reward prompt uses
`=== REASONING SOLUTION ===` as the reasoning header.

### Reward-model SFT

`pra-sft` renders MedQA options by their original dict keys, matching the
beam-search-time prompt format. To use a different backbone, edit
`reward_model_backbone_path` in `configs/training/qwen3_4b.yaml` (or a
copy).

```bash
# 8 GPUs, DDP via torchrun
sbatch scripts/slurm/training/pra_sft.slurm configs/training/qwen3_4b.yaml
# -> $PRA_OUTPUT_ROOT/sft/<run_name>/
```

### Training-data pipeline

Four stages; each stage's output is the next stage's input. Each stage
auto-names its output under `$PRA_OUTPUT_ROOT`:

```bash
# Stage 0: sample N reasoning chains per MedQA-4 question with the policy
# (input jsonl defaults to $PRA_DATA_ROOT/medqa_4/<mode>.jsonl; pass a 4th arg to override)
sbatch scripts/slurm/data/pra_sample.slurm train Qwen/Qwen3-4B-Instruct-2507 32
# -> $PRA_OUTPUT_ROOT/sampled/Qwen3-4B-Instruct-2507_train_32samples_<ts>.json

# Stage 1: retrieve top-k MedCPT docs for each step (last-2-steps window)
sbatch scripts/slurm/data/pra_retrieve.slurm $PRA_OUTPUT_ROOT/sampled/<...>.json window
# -> $PRA_OUTPUT_ROOT/retrieved/<input_stem>_window_<ts>.json

# Stage 2: label each step with the teacher LLM (Qwen3-235B FP8, 8-GPU TP)
sbatch scripts/slurm/data/pra_label.slurm $PRA_OUTPUT_ROOT/retrieved/<...>.json qwen3-235b
# -> $PRA_OUTPUT_ROOT/labeled/qwen3-235b_<input_stem>_<ts>/results_list.json

# Stage 3: SFT the reward model on the labeled data (see "Reward-model SFT" above)
sbatch scripts/slurm/training/pra_sft.slurm configs/training/qwen3_4b.yaml
# -> $PRA_OUTPUT_ROOT/sft/<run_name>/
```

You can pass an explicit output path as the last positional argument to any
stage wrapper.

## Layout

```
src/pra/
  inference/         # beam.py, merge_shards.py
  training/          # sft.py
  data/              # sample.py, retrieve.py, label.py
  utils/             # args, stages, retriever, scoring, ...
configs/
  inference/         # YAML configs consumed by pra-beam --config
  training/          # YAML configs consumed by pra-sft --config
scripts/slurm/
  inference/ training/ data/
scripts/local/
  inference/
```

### Citation
```bibtex
@inproceedings{sohn2026processrewardagents,
  title     = {Process Reward Agents for Steering Knowledge-Intensive Reasoning},
  author    = {Jiwoong Sohn and Tomasz Sternal and Kenneth Styppa and Torsten Hoefler and Michael Moor},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.09482}
}
```
