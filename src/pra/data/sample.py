from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from pra.utils.prompts import ANSWER_PATTERN

LETTERS = "ABCD"

SYSTEM_PROMPT = (
    "The following is a multiple-choice question about medical knowledge. "
    "Solve this in a step-by-step fashion, prefixed as 'Step 1:', 'Step 2:', etc. "
    "Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; "
    'conclude your response with the phrase "the answer is ([option_id]) [answer_string]".'
)

_ANSWER_SPLIT_RE = re.compile(r"the\s*answer\s*is[^\w]*", re.IGNORECASE)

def _load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _format_question(item: dict) -> str:
    opts = item["options"]
    cs = ""
    if isinstance(opts, dict):
        for letter in LETTERS:
            if letter in opts:
                cs += f"({letter}) {opts[letter]}\n"
    return f"{item['question']}\n{cs}"


def _gt_index(item: dict) -> int:
    ans = str(item["answer_idx"]).strip().upper()
    return LETTERS.index(ans) if ans in LETTERS else -1


def _extract_letter(text: str) -> str:
    m = ANSWER_PATTERN.search(text)
    if m:
        return m.group(1).upper()
    parts = _ANSWER_SPLIT_RE.split(text)
    if len(parts) > 1:
        for ch in parts[1]:
            if ch.upper() in LETTERS:
                return ch.upper()
    return "?"


def _collect_samples(request_output, gt_idx: int) -> Tuple[float, List[int], List[str], List[str]]:
    completions = getattr(request_output, "outputs", None) or [request_output]
    preds: List[int] = []
    outs: List[str] = []
    letters_found: List[str] = []
    for c in completions:
        text = (c.text if hasattr(c, "text") else str(c)).strip()
        outs.append(text)
        letter = _extract_letter(text)
        letters_found.append(letter)
        preds.append(LETTERS.index(letter) if letter in LETTERS else -1)
    acc = sum(1 for p in preds if p == gt_idx) / (len(preds) if preds else 1)
    return acc, preds, outs, letters_found


def _make_chat(question: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def run(
    *,
    input_jsonl_path: str,
    output_path: str,
    policy_model_path: str,
    samples_per_question: int,
    tensor_parallel_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    visible_gpus: Optional[str],
) -> None:
    if visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

    from vllm import LLM, SamplingParams

    items = _load_jsonl(input_jsonl_path)
    questions = [_format_question(it) for it in items]
    gts = [_gt_index(it) for it in items]
    print(f"[INFO] loaded {len(items)} questions from {input_jsonl_path}"
          f"answer_dist={dict(Counter(gts))}")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=top_k,
        top_p=top_p,
        n=samples_per_question,
    )

    print(f"[INFO] loading policy: {policy_model_path}  (tp={tensor_parallel_size})")
    llm = LLM(model=policy_model_path, tensor_parallel_size=tensor_parallel_size)

    chat_messages = [_make_chat(q) for q in questions]
    print(f"[INFO] sampling n={samples_per_question} per question over {len(chat_messages)} questions")
    sampled = llm.chat(chat_messages, sampling_params=sampling_params)

    records: list = []
    for i, req_out in enumerate(sampled):
        gt_idx = gts[i]
        opts = items[i]["options"]
        gt_letter = chr(65 + gt_idx) if 0 <= gt_idx < len(LETTERS) else "?"
        acc, preds, outs, letters_found = _collect_samples(req_out, gt_idx)
        picks_letters = [chr(65 + p) if 0 <= p < len(LETTERS) else "?" for p in preds]
        picks_texts = [opts[ch] for ch in picks_letters]
        records.append(
            {
                "question": items[i]["question"],
                "options": opts,
                "answer_idx": gt_letter,
                "answer": opts[gt_letter],
                "input_messages": chat_messages[i],
                "model_outputs": outs,
                "model_answer_idxs": picks_letters,
                "model_answers": picks_texts,
                "corrects": [p == gt_idx for p in preds],
                "accuracy": acc,
            }
        )
    first_correct = [r["corrects"][0] if r["corrects"] else False for r in records]
    overall_acc = sum(first_correct) / len(first_correct) if first_correct else 0.0
    print(f"\n[DONE] first-sample accuracy: {overall_acc:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[DONE] wrote {len(records)} records -> {output_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample N reasoning samples per question using a policy.")
    default_input = os.path.join(os.environ["PRA_DATA_ROOT"], "medqa_4", "train.jsonl")
    p.add_argument("--input_jsonl_path", default=default_input)
    p.add_argument("--output_path", required=True,
                   help="Where to write the sampled JSON (input to pra-retrieve).")
    p.add_argument("--policy_model_path", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="HF id / local path of the policy model used for sampling.")
    p.add_argument("--samples_per_question", type=int, default=8,
                   help="Number of independent samples per question")
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--visible_gpus", default=None,
                   help="Override CUDA_VISIBLE_DEVICES (e.g. '0,1'). Leave unset inside SLURM.")
    return p.parse_args(argv)


def main_cli() -> None:
    args = parse_args()
    run(
        input_jsonl_path=args.input_jsonl_path,
        output_path=args.output_path,
        policy_model_path=args.policy_model_path,
        samples_per_question=args.samples_per_question,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        visible_gpus=args.visible_gpus,
    )


if __name__ == "__main__":
    main_cli()
