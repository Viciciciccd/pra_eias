from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


_DEFAULT_LLM_KWARGS: Dict[str, Any] = {
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 16384,
}

TEACHER_PRESETS: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {
        "tensor_parallel_size": 4,
        "quantization": "fp8",
    },
}


SYSTEM_INSTRUCTION = (
    "You are a medical expert responsible for evaluating the quality of the last reasoning step in a solution to a medical question.\n"
    "You are provided with relevant documents, the question, and the reasoning trace including prior steps.\n"
    "Your task is to critically assess only the last reasoning step, considering its logical coherence, medical validity, and consistency with the evidence.\n"
    "\n"
    "You must only return one score, and output nothing else:\n"
    "Reasoning Score: Score 1 if the last step is logically coherent, medically sound, and aligns with the provided evidence; otherwise, score 0.\n"
    "\n"
    "Output only your reasoning score in the following format:\n"
    "\n"
    "[REASONING_SCORE]: 1 or 0 (1: correct, 0: incorrect)"
)


_RR_PATTERN = re.compile(
    r"\[?\s*reasoning[\s_]?(?:reward|score)\s*\]?\s*:\s*([01])",
    re.IGNORECASE,
)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_reasoning_score(response: str) -> int:
    matches = list(_RR_PATTERN.finditer(_strip_thinking(response)))
    if not matches:
        raise ValueError("Could not parse reasoning score from response")
    return int(matches[-1].group(1))


def _options_str(options) -> str:
    return "\n".join(f"{k}: {v}" for k, v in options.items()) if options else ""


def _construct_context(related_docs: List[str], question_text: str, correct_answer: str) -> str:
    parts = []
    if related_docs:
        parts.append("=== DOCUMENTS ===\n")
        for i, doc in enumerate(related_docs):
            parts.append(f"Doc {i + 1}: {doc}")
    parts.append("\n\n=== QUESTION ===\n")
    parts.append(question_text)
    if correct_answer:
        parts.append("\n\n=== CORRECT ANSWER ===\n")
        parts.append(correct_answer)
    return "\n".join(parts)


class BatchProcessingState:
    def __init__(self, items: List[Dict[str, Any]]):
        self.active: Dict[str, Dict[str, Any]] = {}
        self.completed: Dict[str, Dict[str, Any]] = {}
        for item in items:
            steps_pairs = item["model_outputs_steps"]
            if not steps_pairs:
                continue
            qid = item["question_id"]
            self.active[qid] = {
                "question_id": qid,
                "question": item["question"],
                "options": item["options"],
                "answer_idx": item["answer_idx"],
                "answer": item["answer"],
                "model_outputs_steps": steps_pairs,
                "current_step": 0,
                "error_count": 0,
                "reasoning_rewards": [],
                "steps_list": [],
            }

    def current_batch(self) -> List[Dict[str, Any]]:
        return [
            {"item_idx": qid, "state": s}
            for qid, s in self.active.items()
            if s["current_step"] < len(s["model_outputs_steps"][0]["steps"])
        ]

    def apply_results(self, batch_results: List[Tuple[str, int, str, str, str]], error_threshold: int):
        for item_idx, reasoning_reward, response_text, sol_text, doc_q_a in batch_results:
            state = self.active[item_idx]
            if reasoning_reward == -1:
                if state["error_count"] < error_threshold:
                    state["error_count"] += 1
                    continue
            state["error_count"] = 0
            cur = state["current_step"]
            state["reasoning_rewards"].append(reasoning_reward)
            related_docs = state["model_outputs_steps"][0]["stepwise_retrievals"][cur]["related_docs"]
            state["steps_list"].append(
                {
                    "reasoning_step_index": cur,
                    "prompt": f"{doc_q_a}\n\n=== REASONING SOLUTION ===\n{sol_text}",
                    "response": response_text,
                    "reasoning_step": state["model_outputs_steps"][0]["steps"][cur],
                    "parsed_scores": reasoning_reward,
                    "reasoning_rewards": state["reasoning_rewards"].copy(),
                    "related_docs": related_docs,
                }
            )
            state["current_step"] = cur + 1
            if state["current_step"] >= len(state["model_outputs_steps"][0]["steps"]):
                self.completed[item_idx] = state
                del self.active[item_idx]

    def stats(self) -> Dict[str, int]:
        return {"active": len(self.active), "completed": len(self.completed)}


def _build_conversation(doc_q_a: str, sol_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"{doc_q_a}\n\n=== REASONING TRACE ===\n{sol_text}"},
    ]


def _process_batch(
    batch: List[Dict[str, Any]],
    llm,
    sampling_params,
) -> List[Tuple[str, int, str, str, str]]:
    conversations = []
    meta = []
    for it in batch:
        state = it["state"]
        cur = state["current_step"]
        retrievals = state["model_outputs_steps"][0]["stepwise_retrievals"][cur]
        doc_q_a = _construct_context(
            retrievals["related_docs"],
            state["question"] + "\n\n" + _options_str(state["options"]),
            f"({state['answer_idx']}): {state['answer']}",
        )
        sol_text = "".join(
            f"Step {idx}: {text}\n"
            for idx, text in state["model_outputs_steps"][0]["steps"][: cur + 1]
        )
        conversations.append(_build_conversation(doc_q_a, sol_text))
        meta.append({"item_idx": it["item_idx"], "sol_text": sol_text, "doc_q_a": doc_q_a})

    outputs = llm.chat(conversations, sampling_params=sampling_params)
    results = []
    for out, m in zip(outputs, meta):
        response = out.outputs[0].text.strip()
        try:
            reward = parse_reasoning_score(response)
        except ValueError:
            reward = -1
        results.append((m["item_idx"], reward, response, m["sol_text"], m["doc_q_a"]))
    return results


def _resolve_llm_kwargs(
    teacher_model_path: str,
    *,
    tensor_parallel_size: Optional[int],
    gpu_memory_utilization: Optional[float],
    max_model_len: Optional[int],
    dtype: Optional[str],
    quantization: Optional[str],
) -> Dict[str, Any]:
    preset = TEACHER_PRESETS.get(teacher_model_path, {})

    def _pick(explicit, key):
        if explicit is not None:
            return explicit
        return preset.get(key, _DEFAULT_LLM_KWARGS[key])

    llm_kwargs: Dict[str, Any] = {
        "model": teacher_model_path,
        "dtype": _pick(dtype, "dtype"),
        "tensor_parallel_size": _pick(tensor_parallel_size, "tensor_parallel_size"),
        "gpu_memory_utilization": _pick(gpu_memory_utilization, "gpu_memory_utilization"),
        "max_model_len": _pick(max_model_len, "max_model_len"),
    }
    if quantization is None:
        quant = preset.get("quantization")
    elif quantization.lower() in ("", "none", "null"):
        quant = None
    else:
        quant = quantization
    if quant:
        llm_kwargs["quantization"] = quant
    return llm_kwargs


def run(
    input_path: str,
    output_path: str,
    *,
    teacher_model_path: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    max_tokens: int = 1024,
    error_threshold: int = 3,
    tensor_parallel_size: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: Optional[int] = None,
    dtype: Optional[str] = None,
    quantization: Optional[str] = None,
) -> None:
    from vllm import LLM, SamplingParams

    with open(input_path) as f:
        items = json.load(f)

    llm_kwargs = _resolve_llm_kwargs(
        teacher_model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        quantization=quantization,
    )
    preset_tag = "preset" if teacher_model_path in TEACHER_PRESETS else "no preset"
    print(f"[label] teacher_model_path={teacher_model_path}  ({preset_tag})")
    print(f"[label] vLLM kwargs: "
          + ", ".join(f"{k}={v!r}" for k, v in llm_kwargs.items() if k != "model"))
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(max_tokens=max_tokens)

    state = BatchProcessingState(items)
    print(f"[label] initialized with {len(state.active)} items")

    step = 0
    while state.active:
        step += 1
        stats = state.stats()
        print(f"[label] step={step}  active={stats['active']}  completed={stats['completed']}")
        batch = state.current_batch()
        if not batch:
            break
        results = _process_batch(batch, llm, sampling_params)
        state.apply_results(results, error_threshold=error_threshold)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(list(state.completed.values()), f)
    print(f"[label] wrote {output_path}  completed={len(state.completed)}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label reasoning steps with a teacher LLM")
    p.add_argument("--input", required=True, help="Input JSON produced by pra-retrieve")
    p.add_argument("--output", required=True, help="Output JSON path (results_list)")
    p.add_argument("--teacher_model_path",
                   default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--error_threshold", type=int, default=3,
                   help="How many parse failures per item before giving up and recording -1")
    p.add_argument("--tensor_parallel_size", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=None)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--dtype", type=str, default=None,
                   help="vLLM dtype (e.g. 'bfloat16', 'float16', 'auto').")
    p.add_argument("--quantization", type=str, default=None,
                   help="vLLM quantization (e.g. 'fp8'). Pass 'none' to disable a preset quantization.")
    return p.parse_args(argv)


def main_cli() -> None:
    args = parse_args()
    run(
        args.input,
        args.output,
        teacher_model_path=args.teacher_model_path,
        max_tokens=args.max_tokens,
        error_threshold=args.error_threshold,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main_cli()
