#!/usr/bin/env python3
"""Export BioMed-R1-Eval subsets to MedQA-style JSONL for pra-beam.

Source: https://huggingface.co/datasets/zou-lab/BioMed-R1-Eval
Each subset (e.g. ``medmcqa``, ``medqa``) is a separate HF config with a ``test`` split.

Output format (one JSON object per line, same as ``data/medqa_4/*.jsonl``):
  question, answer, options, answer_idx, meta_info

Usage:
  pip install datasets   # if not already installed
  python scripts/preprocess_biomed_r1_eval.py --subset medmcqa
  python scripts/preprocess_biomed_r1_eval.py --subset medmcqa --output ./data/medmcqa/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_options(raw) -> dict[str, str]:
    if isinstance(raw, dict):
        opts = raw
    elif isinstance(raw, str):
        opts = json.loads(raw)
    else:
        raise TypeError(f"options must be str or dict, got {type(raw)}")
    return {str(k).strip().upper(): str(v) for k, v in opts.items()}


def _normalize_answer_idx(raw, options: dict[str, str]) -> str:
    if isinstance(raw, int):
        if 0 <= raw < 26:
            letter = chr(ord("A") + raw)
            if letter in options:
                return letter
        raise ValueError(f"answer_idx int {raw} not in options {list(options)}")
    letter = str(raw).strip().upper()
    if letter not in options:
        raise ValueError(f"answer_idx {letter!r} not in options {list(options)}")
    return letter


def convert_row(row: dict, *, meta_info: str) -> dict:
    options = _parse_options(row["options"])
    answer_idx = _normalize_answer_idx(row["answer_idx"], options)
    return {
        "question": str(row["question"]).strip(),
        "answer": options[answer_idx],
        "options": options,
        "meta_info": meta_info,
        "answer_idx": answer_idx,
    }


def export_subset(
    *,
    repo_id: str,
    subset: str,
    split: str,
    output_path: Path,
    meta_info: str | None = None,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The `datasets` package is required. Install with: pip install datasets"
        ) from e

    ds = load_dataset(repo_id, subset, split=split)
    tag = meta_info or subset
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in ds:
            record = convert_row(row, meta_info=tag)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    default_root = os.environ.get("PRA_DATA_ROOT", "./data")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--repo",
        default="zou-lab/BioMed-R1-Eval",
        help="Hugging Face dataset repo (default: zou-lab/BioMed-R1-Eval)",
    )
    p.add_argument(
        "--subset",
        default="medmcqa",
        help="Dataset config / subset name (default: medmcqa)",
    )
    p.add_argument("--split", default="test", help="Split to export (default: test)")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: $PRA_DATA_ROOT/<subset>/<split>.jsonl)",
    )
    p.add_argument(
        "--meta-info",
        default=None,
        help="Value for meta_info field (default: same as --subset)",
    )
    args = p.parse_args()

    out = Path(args.output) if args.output else Path(default_root) / args.subset / f"{args.split}.jsonl"

    try:
        n = export_subset(
            repo_id=args.repo,
            subset=args.subset,
            split=args.split,
            output_path=out,
            meta_info=args.meta_info,
        )
    except Exception as e:
        print(f"[preprocess] ERROR: {e}", file=sys.stderr)
        return 2

    print(f"[preprocess] Wrote {n} rows to {out}")
    print(f"[preprocess] Run beam search with: pra-beam --input_jsonl_path {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
