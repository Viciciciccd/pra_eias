import os
import json
from collections import Counter
from datetime import datetime

from pra.utils.prompts import ANSWER_LAST_STEP_PATTERNS


def extract_predicted_answer(steps: list) -> str | None:
    """Return the answer letter parsed from the last reasoning step, uppercased."""
    if not steps:
        return None
    last = steps[-1] or ""
    for pattern in ANSWER_LAST_STEP_PATTERNS:
        match = pattern.search(last)
        if match:
            return match.group(1).upper()
    return None


def get_correct_answer_letter(answer_idx) -> str:
    if isinstance(answer_idx, int):
        return chr(ord("A") + answer_idx)
    return str(answer_idx).strip().upper()


def select_final_beam(beams: list, method: str = "cumulative", *, require_parsable: bool = True) -> dict | None:
    """Select the final beam by cumulative score.

    When ``require_parsable`` is True (default), only beams with a parsable
    ``the answer is (X)`` are eligible; returns None if none parse.
    """
    if not beams:
        return None
    if method != "cumulative":
        raise ValueError(f"Unknown beam final selection method: {method}")
    pool = beams
    if require_parsable:
        parsable = [b for b in beams if extract_predicted_answer(b.get("steps") or [])]
        if not parsable:
            return None
        pool = parsable
    return max(pool, key=lambda b: b["cumulative_score"])


def _beam_predicted(trace) -> str | None:
    best = trace.get_best_beam()
    return extract_predicted_answer(best["steps"]) if best else None


def compute_live_accuracy(completed: list) -> dict:
    """Compute live accuracy from completed beam traces."""
    if not completed:
        return {"correct": 0, "wrong": 0, "no_answer": 0, "total": 0, "accuracy": 0.0}

    correct = wrong = no_answer = 0
    for t in completed:
        predicted = _beam_predicted(t)
        correct_letter = get_correct_answer_letter(t.q_dict["answer_idx"])
        if predicted is None:
            no_answer += 1
        elif predicted == correct_letter:
            correct += 1
        else:
            wrong += 1

    total = len(completed)
    return {
        "correct": correct,
        "wrong": wrong,
        "no_answer": no_answer,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def format_live_accuracy(completed: list) -> str:
    """Format live accuracy as a compact string for logging."""
    stats = compute_live_accuracy(completed)
    return (
        f"Acc: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']}) | "
        f"✓{stats['correct']} ✗{stats['wrong']} ?{stats['no_answer']}"
    )


def score_results(completed: list) -> dict:
    """Grade completed beam traces and stamp result fields on each trace."""
    correct_count = wrong_count = no_answer_count = 0

    for t in completed:
        predicted = _beam_predicted(t)
        correct_letter = get_correct_answer_letter(t.q_dict["answer_idx"])
        if predicted is None:
            result = "no_answer"
            no_answer_count += 1
        elif predicted == correct_letter:
            result = "correct"
            correct_count += 1
        else:
            result = "wrong"
            wrong_count += 1
        t.predicted_answer = predicted
        t.correct_answer = correct_letter
        t.result = result

    total = len(completed)
    return {
        "total": total,
        "correct": correct_count,
        "wrong": wrong_count,
        "no_answer": no_answer_count,
        "accuracy": correct_count / total if total else 0.0,
    }


def compute_step_distribution(completed: list) -> dict:
    """Compute step count distribution across completed beam traces."""
    step_counts: dict[int, int] = {}
    lens = []
    for t in completed:
        best = t.get_best_beam()
        k = len(best["steps"]) if best else 0
        lens.append(k)
        step_counts[k] = step_counts.get(k, 0) + 1
    avg_steps = sum(lens) / len(lens) if lens else 0
    return {"counts": step_counts, "avg_steps": avg_steps}


def save_results(completed: list, trace_dir: str, config_name: str, test_mode: str, args) -> str:
    """Save results + summary; returns output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    slurm_log_dir = os.environ.get("RSM_LOG_DIR", "")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    policy_model_path = getattr(args, "policy_model_path", None)

    output_name = f"{config_name}_{timestamp}" if config_name else f"{test_mode}_{timestamp}"
    if slurm_job_id and f"job{slurm_job_id}" not in output_name:
        output_name = f"{output_name}_job{slurm_job_id}"

    output_dir = os.path.join(trace_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "all_medqa_input_outputs.json"), "w") as f:
        json.dump([t.to_dict() for t in completed], f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "fsm_state_trace.jsonl"), "w") as f:
        for t in completed:
            for entry in t.history:
                f.write(json.dumps({"j": t.j, **entry}, ensure_ascii=False) + "\n")

    scores = score_results(completed)
    step_dist = compute_step_distribution(completed)
    summary = {
        **scores,
        "avg_steps": step_dist["avg_steps"],
        "step_distribution": {str(k): v for k, v in sorted(step_dist["counts"].items())},
        "args": vars(args),
        "slurm_job_id": slurm_job_id,
        "slurm_log_dir": slurm_log_dir,
        "slurm_array_task_id": slurm_array_task_id,
        "slurm_array_job_id": slurm_array_job_id,
        "policy_model_path": policy_model_path,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Cross-link results <-> SLURM log dir so they're easy to find from either side.
    if slurm_log_dir:
        os.makedirs(slurm_log_dir, exist_ok=True)
        with open(os.path.join(output_dir, "slurm_log_dir.txt"), "w") as f:
            f.write(str(slurm_log_dir).strip() + "\n")
        with open(os.path.join(slurm_log_dir, "results_dir.txt"), "w") as f:
            f.write(str(output_dir).strip() + "\n")

    return output_dir


def _select_beam(trace, method: str):
    require = getattr(trace.config, "beam_require_parsable_answer", True)
    return select_final_beam(trace.beams, method, require_parsable=require)


def compute_beam_accuracy_by_method(completed: list, method: str) -> dict:
    """Compute accuracy using a specific beam selection method."""
    correct = wrong = no_answer = 0
    for t in completed:
        beam = _select_beam(t, method)
        predicted = extract_predicted_answer(beam["steps"]) if beam else None
        correct_letter = get_correct_answer_letter(t.q_dict["answer_idx"])
        if predicted is None:
            no_answer += 1
        elif predicted == correct_letter:
            correct += 1
        else:
            wrong += 1
    total = len(completed)
    return {
        "correct": correct,
        "wrong": wrong,
        "no_answer": no_answer,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def compute_beam_majority_voting(completed: list) -> dict:
    """Compute accuracy using majority voting across all beams."""
    correct = wrong = no_answer = 0
    for t in completed:
        correct_letter = get_correct_answer_letter(t.q_dict["answer_idx"])
        answers = [a for b in t.beams if (a := extract_predicted_answer(b["steps"]))]
        if not answers:
            no_answer += 1
            continue
        predicted = Counter(answers).most_common(1)[0][0]
        if predicted == correct_letter:
            correct += 1
        else:
            wrong += 1
    total = len(completed)
    return {
        "correct": correct,
        "wrong": wrong,
        "no_answer": no_answer,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def compute_beam_oracle(completed: list) -> dict:
    """Oracle upper bound: correct if ANY beam has the right answer."""
    correct = wrong = no_answer = 0
    for t in completed:
        correct_letter = get_correct_answer_letter(t.q_dict["answer_idx"])
        answers = [a for b in t.beams if (a := extract_predicted_answer(b["steps"]))]
        if not answers:
            no_answer += 1
        elif correct_letter in answers:
            correct += 1
        else:
            wrong += 1
    total = len(completed)
    return {
        "correct": correct,
        "wrong": wrong,
        "no_answer": no_answer,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def compute_all_beam_accuracies(completed: list) -> dict:
    """Compute accuracy using all beam selection methods."""
    return {
        "cumulative": compute_beam_accuracy_by_method(completed, "cumulative"),
        "majority_voting": compute_beam_majority_voting(completed),
        "oracle": compute_beam_oracle(completed),
    }


def print_beam_accuracy_comparison(completed: list) -> dict:
    """Print accuracy comparison table for all beam selection methods."""
    results = compute_all_beam_accuracies(completed)

    print(f"\n{'='*70}")
    print("BEAM SELECTION METHOD COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Accuracy':>10} {'Correct':>10} {'Wrong':>10} {'NoAns':>10}")
    print(f"{'-'*70}")

    sorted_methods = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for method, stats in sorted_methods:
        print(
            f"{method:<25} {stats['accuracy']:>10.2%} {stats['correct']:>10} "
            f"{stats['wrong']:>10} {stats['no_answer']:>10}"
        )
    print(f"{'='*70}")

    non_oracle = [(m, s) for m, s in sorted_methods if m != "oracle"]
    if non_oracle:
        best_method, best_stats = non_oracle[0]
        oracle_stats = results["oracle"]
        print(f"\n  Best method: {best_method} ({best_stats['accuracy']:.2%})")
        gap = oracle_stats["accuracy"] - best_stats["accuracy"]
        print(f"  Oracle upper bound: {oracle_stats['accuracy']:.2%} (gap: {gap:.2%})")
    print(f"{'='*70}")
    return results
