from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

from pra.utils.prompts import ANSWER_PATTERN
from pra.utils.scoring import select_final_beam


def extract_predicted_answer(steps: List[str]) -> str | None:
    """Return the last answer letter found in any step, uppercased."""
    for step in reversed(steps or []):
        if not step:
            continue
        m = ANSWER_PATTERN.search(step)
        if m:
            return m.group(1).upper()
    return None


def get_correct_answer_letter(answer_idx) -> str:
    if isinstance(answer_idx, int):
        return chr(ord("A") + answer_idx)
    return str(answer_idx).strip().upper()


def find_latest_result_dir(trace_dir: str, prefix: str) -> str:
    """Pick the latest result directory starting with prefix_YYYYMMDD_HHMMSS."""
    if not os.path.isdir(trace_dir):
        raise FileNotFoundError(f"trace_dir not found: {trace_dir}")
    candidates = sorted(d for d in os.listdir(trace_dir) if d.startswith(prefix + "_"))
    if not candidates:
        raise FileNotFoundError(f"No result directories found under {trace_dir} with prefix {prefix!r}")
    return os.path.join(trace_dir, candidates[-1])


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def infer_shard_id_from_path(path: str) -> int:
    """Infer shard id from a directory/file name containing ``_sh<id>`` or ``_sh<id>ofN``."""
    base = os.path.basename(path.rstrip("/"))
    patterns = (
        r"_sh(\d+)of\d+(?:_|$)",  # beam.py: {name}_sh0of2_{timestamp}
        r"_sh(\d+)(?:_|$)",
    )
    for pat in patterns:
        m = re.search(pat, base) or re.search(pat, path)
        if m:
            return int(m.group(1))
    raise ValueError(f"Could not infer shard id from path (expected `_sh<id>`): {path}")


def _select_beam_by_method(all_beams: list, method: str, *, require_parsable: bool = True) -> dict | None:
    return select_final_beam(all_beams, method, require_parsable=require_parsable)


def _beam_method_stats(merged: list, method: str) -> dict:
    """Accuracy stats for a method using ``all_beams`` in each record."""
    correct = wrong = no_answer = 0

    for rec in merged:
        correct_letter = get_correct_answer_letter(rec["q"]["answer_idx"])
        all_beams = rec["all_beams"]
        beam_answers = [a for b in all_beams if (a := extract_predicted_answer(b["steps"]))]

        if method == "oracle":
            if not beam_answers:
                no_answer += 1
            elif correct_letter in beam_answers:
                correct += 1
            else:
                wrong += 1
            continue

        if method == "majority_voting":
            predicted = Counter(beam_answers).most_common(1)[0][0] if beam_answers else None
        else:
            b = _select_beam_by_method(all_beams, method)
            predicted = extract_predicted_answer(b["steps"]) if b else None

        if predicted is None:
            no_answer += 1
        elif predicted == correct_letter:
            correct += 1
        else:
            wrong += 1

    total = len(merged)
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "no_answer": no_answer,
        "accuracy": correct / total if total else 0.0,
    }


def _compute_step_distribution_from_steps(merged: list) -> dict:
    step_counts: Dict[int, int] = {}
    total = len(merged)
    steps_lens = []
    for rec in merged:
        k = len(rec["steps"])
        steps_lens.append(k)
        step_counts[k] = step_counts.get(k, 0) + 1
    avg_steps = (sum(steps_lens) / total) if total else 0.0
    return {
        "avg_steps": avg_steps,
        "counts": {str(k): v for k, v in sorted(step_counts.items())},
    }


def _compute_search_frequency(merged: list) -> dict:
    """Compute how often beam search triggers SEARCH."""
    decision_total = decision_search = search_events = traces_with_search = 0

    for rec in merged:
        saw_search = False
        for h in rec["history"]:
            action = h["action"]
            if action == "beam_reward_decision":
                decision = h["decision"]
                if decision in ("search", "no_search"):
                    decision_total += 1
                    if decision == "search":
                        decision_search += 1
            elif action == "search":
                search_events += 1
                saw_search = True
        if saw_search:
            traces_with_search += 1

    total_traces = len(merged)
    trace_freq = traces_with_search / total_traces if total_traces else 0.0
    decision_freq = decision_search / decision_total if decision_total else None

    method = "beam_reward_decision" if decision_total > 0 else "trace_has_search"
    frequency = float(decision_freq) if decision_freq is not None else float(trace_freq)

    return {
        "method": method,
        "frequency": frequency,
        "decision_based": {
            "searches": int(decision_search),
            "total": int(decision_total),
            "frequency": float(decision_freq) if decision_freq is not None else None,
        },
        "trace_based": {
            "traces_with_search": int(traces_with_search),
            "total_traces": int(total_traces),
            "frequency": float(trace_freq),
        },
        "search_events_total": int(search_events),
    }


def _format_console_summary(
    *,
    total: int,
    correct: int,
    wrong: int,
    no_answer: int,
    acc: float,
    step_dist: dict,
    search_stats: dict,
    has_beams: bool,
    beam_search_details: dict | None,
    beam_method_comparison: dict | None,
) -> str:
    lines: List[str] = []
    hr = "============================================"

    lines += [hr, "MERGED RESULTS SUMMARY", hr]
    lines.append(f"  Total:      {total}")
    if total:
        lines.append(f"  Correct:    {correct} ({correct/total:.1%})")
        lines.append(f"  Wrong:      {wrong} ({wrong/total:.1%})")
        lines.append(f"  No answer:  {no_answer} ({no_answer/total:.1%})")
    else:
        lines += ["  Correct:    0", "  Wrong:      0", "  No answer:  0"]
    lines.append(f"  Accuracy:   {acc:.2%}")

    lines += [hr, "SEARCH FREQUENCY", hr]
    if search_stats["method"] == "beam_reward_decision":
        d = search_stats["decision_based"]
        freq = d["frequency"] or 0.0
        lines.append(f"  Per-step (decision):  {d['searches']}/{d['total']} ({freq:.2%})")
    t = search_stats["trace_based"]
    lines.append(
        f"  Per-trace (any):      {t['traces_with_search']}/{t['total_traces']} ({t['frequency']:.2%})"
    )
    lines.append(f"  Search events total:  {search_stats['search_events_total']}")

    if has_beams:
        lines += [hr, "STEP DISTRIBUTION (K)", hr]
        lines.append(f"  Avg steps:  {step_dist['avg_steps']:.2f}")
        for k_str, count in step_dist["counts"].items():
            k = int(k_str)
            pct = count / total * 100 if total else 0.0
            bar = "█" * int(pct / 2)
            lines.append(f"  K={k:2d}: {count:4d} ({pct:5.1f}%) {bar}")

        if beam_search_details:
            lines += [hr, "BEAM SEARCH DETAILS", hr]
            lines.append(f"  Beam width:      {beam_search_details['beam_width']}")
            lines.append(f"  Candidates/beam: {beam_search_details['candidates_per_beam']}")
            lines.append(f"  Avg final beams: {beam_search_details['avg_final_beams']:.2f}")
            if beam_search_details["avg_best_score"] is not None:
                lines.append(f"  Avg best score:  {beam_search_details['avg_best_score']:.3f}")

        if beam_method_comparison:
            lines += [hr, "BEAM SELECTION METHOD COMPARISON", hr]
            lines.append(f"{'Method':<25} {'Accuracy':>10} {'Correct':>10} {'Wrong':>10} {'NoAns':>10}")
            lines.append("-" * 70)
            for method, stats in sorted(
                beam_method_comparison.items(), key=lambda x: x[1]["accuracy"], reverse=True
            ):
                lines.append(
                    f"{method:<25} {stats['accuracy']:>10.2%} {stats['correct']:>10} "
                    f"{stats['wrong']:>10} {stats['no_answer']:>10}"
                )
            lines.append(hr)

    lines.append(hr)
    return "\n".join(lines)


def _collect_shards(args) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Return (shard_dirs, shard_json_paths) as parallel sorted lists of (shard_id, path)."""
    shard_dirs: List[Tuple[int, str]] = []
    shard_json_paths: List[Tuple[int, str]] = []

    if args.shard_dirs or args.shard_jsons:
        if args.shard_dirs and args.shard_jsons:
            raise ValueError("Provide only one of --shard_dirs or --shard_jsons (not both).")

        if args.shard_dirs:
            for d in args.shard_dirs:
                d = os.path.abspath(d)
                shard_id = infer_shard_id_from_path(d)
                if any(sid == shard_id for sid, _ in shard_dirs):
                    raise ValueError(f"Duplicate shard id inferred from --shard_dirs: sh{shard_id}")
                shard_json = os.path.join(d, "all_medqa_input_outputs.json")
                if not os.path.isfile(shard_json):
                    raise FileNotFoundError(f"Missing {shard_json}")
                shard_dirs.append((shard_id, d))
                shard_json_paths.append((shard_id, shard_json))
        else:
            for p in args.shard_jsons:
                p = os.path.abspath(p)
                shard_id = infer_shard_id_from_path(p)
                if any(sid == shard_id for sid, _ in shard_json_paths):
                    raise ValueError(f"Duplicate shard id inferred from --shard_jsons: sh{shard_id}")
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Missing {p}")
                shard_json_paths.append((shard_id, p))
                shard_dirs.append((shard_id, os.path.dirname(p)))
    else:
        if not args.run_tag:
            raise ValueError("Must provide --run_tag unless using --shard_dirs/--shard_jsons.")
        if args.num_shards < 1:
            raise ValueError("--num_shards must be >= 1")

        trace_dir = os.path.abspath(args.trace_dir)
        for shard_id in range(args.num_shards):
            prefix = f"{args.run_tag}_sh{shard_id}"
            result_dir = find_latest_result_dir(trace_dir, prefix)
            shard_json = os.path.join(result_dir, "all_medqa_input_outputs.json")
            if not os.path.isfile(shard_json):
                raise FileNotFoundError(f"Missing {shard_json}")
            shard_dirs.append((shard_id, result_dir))
            shard_json_paths.append((shard_id, shard_json))

    shard_dirs.sort(key=lambda x: x[0])
    shard_json_paths.sort(key=lambda x: x[0])
    return shard_dirs, shard_json_paths


def _merge_records(shard_json_paths: List[Tuple[int, str]]) -> list:
    merged_by_j: Dict[int, dict] = {}
    for shard_id, shard_json in shard_json_paths:
        for rec in load_json(shard_json):
            j = rec["j"]
            if j is None:
                raise ValueError(f"Record missing 'j' in shard {shard_id}: {shard_json}")
            if j in merged_by_j:
                raise ValueError(f"Duplicate j={j} across shards (seen again in shard {shard_id})")
            merged_by_j[int(j)] = rec
    return [merged_by_j[j] for j in sorted(merged_by_j)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True, help="Base results directory (e.g. ./test/results)")
    ap.add_argument("--run_tag", default=None,
                    help="Run tag used in config_name for auto-discovery (e.g. beam_qwen3_4b_job12345).")
    ap.add_argument("--num_shards", type=int, default=3,
                    help="Number of shards for auto-discovery via --run_tag.")
    ap.add_argument("--primary_method", default="majority_voting",
                    help="Primary method used for the top-level RESULTS SUMMARY when beam data is present.")
    ap.add_argument("--shard_dirs", nargs="+", default=None,
                    help="Explicit shard result directories. Shard id is inferred from `_sh<id>`.")
    ap.add_argument("--shard_jsons", nargs="+", default=None,
                    help="Explicit shard JSON paths. Shard id is inferred from `_sh<id>`.")
    ap.add_argument("--out_prefix", default=None,
                    help="Optional output prefix (default: <run_tag>_merged)")
    args = ap.parse_args()

    trace_dir = os.path.abspath(args.trace_dir)
    out_prefix = args.out_prefix or (f"{args.run_tag}_merged" if args.run_tag else "merged")

    shard_dirs, shard_json_paths = _collect_shards(args)
    merged = _merge_records(shard_json_paths)
    total = len(merged)

    has_beams = any(isinstance(rec["all_beams"], list) for rec in merged)
    step_dist = _compute_step_distribution_from_steps(merged)
    search_stats = _compute_search_frequency(merged)

    beam_method_comparison = None
    beam_search_details = None

    if has_beams:
        methods = ["oracle", "majority_voting", "cumulative"]
        beam_method_comparison = {m: _beam_method_stats(merged, m) for m in methods}
        if args.primary_method not in beam_method_comparison:
            raise ValueError(
                f"Unknown --primary_method {args.primary_method!r}. "
                f"Available: {sorted(beam_method_comparison)}"
            )
        primary_stats = beam_method_comparison[args.primary_method]

        widths, final_beams, best_scores = [], [], []
        for rec in merged:
            all_beams = rec["all_beams"]
            final_beams.append(len(all_beams))
            widths.append(int(rec["num_beams"] or len(all_beams) or 0))
            if all_beams:
                best_scores.append(max(float(b["cumulative_score"]) for b in all_beams))
        beam_search_details = {
            "beam_width": int(round(sum(widths) / len(widths))) if widths else None,
            "avg_final_beams": (sum(final_beams) / len(final_beams)) if final_beams else 0.0,
            "avg_best_score": (sum(best_scores) / len(best_scores)) if best_scores else None,
            "candidates_per_beam": None,
        }
    else:
        c = w = na = 0
        for rec in merged:
            predicted = extract_predicted_answer(rec["steps"])
            correct_letter = get_correct_answer_letter(rec["q"]["answer_idx"])
            if predicted is None:
                na += 1
            elif predicted == correct_letter:
                c += 1
            else:
                w += 1
        primary_stats = {
            "total": total, "correct": c, "wrong": w, "no_answer": na,
            "accuracy": c / total if total else 0.0,
        }

    correct = primary_stats["correct"]
    wrong = primary_stats["wrong"]
    no_answer = primary_stats["no_answer"]
    acc = primary_stats["accuracy"]

    os.makedirs(trace_dir, exist_ok=True)
    out_dir = os.path.join(trace_dir, out_prefix + "_merged")
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, "all_medqa_input_outputs.json")
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    console_summary = _format_console_summary(
        total=total, correct=correct, wrong=wrong, no_answer=no_answer, acc=acc,
        step_dist=step_dist, search_stats=search_stats,
        has_beams=has_beams,
        beam_search_details=beam_search_details,
        beam_method_comparison=beam_method_comparison,
    )

    out_report = os.path.join(out_dir, "summary.txt")
    with open(out_report, "w") as f:
        f.write(console_summary + "\n")

    out_summary = os.path.join(out_dir, "summary.json")
    new_summary = {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "no_answer": no_answer,
        "accuracy": acc,
        "avg_steps": step_dist["avg_steps"],
        "step_distribution": step_dist["counts"],
        "search_frequency": search_stats,
        "beam_search_details": beam_search_details,
        "beam_method_comparison": beam_method_comparison,
        "primary_method": args.primary_method if has_beams else "single_trace_steps",
        "trace_dir": trace_dir,
        "run_tag": args.run_tag,
        "num_shards": len(shard_dirs),
        "shard_result_dirs": {str(sid): sdir for sid, sdir in shard_dirs},
        "console_summary": console_summary,
        "console_summary_path": out_report,
    }
    with open(out_summary, "w") as f:
        json.dump(new_summary, f, indent=2)

    print(console_summary)
    print(f"[DONE] Wrote merged JSON: {out_json}")
    print(f"[DONE] Wrote summary:    {out_summary}")


def main_cli() -> None:
    main()


if __name__ == "__main__":
    main_cli()
