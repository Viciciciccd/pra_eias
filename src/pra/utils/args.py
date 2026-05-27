import argparse
import os
import sys
from typing import Any, Optional, Sequence


def get_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML file whose top-level keys are used as argparse defaults.",
    )

    parser.add_argument("--input_jsonl_path", type=str)
    parser.add_argument("--policy_model_path", type=str)
    parser.add_argument("--trace_dir", type=str)

    parser.add_argument("--reward_model_path", type=str)
    parser.add_argument("--reward_model_url", type=str)

    parser.add_argument("--visible_gpus", type=str)
    parser.add_argument("--tensor_parallel_size", type=int)

    parser.add_argument("--retriever_device", type=str)
    parser.add_argument("--retriever_index_path", type=str)
    parser.add_argument("--retriever_topk", type=int)
    parser.add_argument("--retriever_rerank_topk", type=int)
    parser.add_argument("--retriever_sources", type=str)
    parser.add_argument("--use_reranker", action="store_true")
    parser.add_argument("--rerank_workers", type=int)
    parser.add_argument("--max_docs", type=int)
    parser.add_argument(
        "--retriever_use_all_steps",
        action="store_true",
        help="Use all reasoning steps for retrieval query (default: last 2 steps only)",
    )

    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--batch_size_pi_theta", type=int)
    parser.add_argument("--batch_size_mu_phi", type=int)
    parser.add_argument("--batch_size_rho", type=int)
    parser.add_argument("--max_workers", type=int)

    parser.add_argument("--K_max", type=int)
    parser.add_argument("--theta_qual", type=float)
    parser.add_argument("--theta_dep", type=float)
    parser.add_argument("--aggregate_steps_for_reward", action="store_true")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    parser.add_argument("--repetition_penalty", type=float)
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--kmax_no_stop", action="store_true")
    parser.add_argument("--pi_theta_max_tokens", type=int)
    parser.add_argument("--pi_theta_temperature", type=float)
    parser.add_argument("--pi_theta_top_p", type=float)
    parser.add_argument("--pi_theta_top_k", type=int)

    parser.add_argument("--config_name", type=str)

    return parser


def add_beam_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments specific to beam search mode."""
    parser.add_argument("--beam_width", type=int,
                        help="Number of top beams to keep at each step")
    parser.add_argument("--beam_candidates", type=int,
                        help="Number of candidates to generate per beam expansion")
    parser.add_argument("--beam_final_selection", type=str,
                        choices=["cumulative"],
                        help="How to select the final beam (affects saved steps + headline accuracy)")
    parser.add_argument(
        "--beam_require_parsable_answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For cumulative selection, only consider beams with a parsable final answer (default: true)",
    )
    parser.add_argument("--adaptive_beam", action="store_true",
                        help="Enable adaptive beam settings (vary beam_width / beam_candidates by step)")
    parser.add_argument("--beam_candidates_start", type=int,
                        help="If set with --beam_candidates_end, interpolate candidates from start->end over steps")
    parser.add_argument("--beam_candidates_end", type=int,
                        help="If set with --beam_candidates_start, interpolate candidates from start->end over steps")
    parser.add_argument("--beam_width_start", type=int,
                        help="If set with --beam_width_end, interpolate beam width from start->end over steps")
    parser.add_argument("--beam_width_end", type=int,
                        help="If set with --beam_width_start, interpolate beam width from start->end over steps")
    parser.add_argument("--beam_schedule_power", type=float, default=1.0,
                        help="Interpolation curvature for adaptive beam (>1 keeps closer to start longer; 1.0=linear)")
    return parser


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` placeholders in strings using the environment."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(loaded).__name__}")
    return _expand_env(loaded)


def parse_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
    required: Sequence[str] = ("input_jsonl_path", "policy_model_path", "trace_dir", "retriever_index_path"),
) -> argparse.Namespace:
    """Parse args, layering YAML config file values on top of argparse defaults.

    Precedence (highest first): explicit CLI flag > YAML file > parser default.
    Unknown keys in the YAML are ignored with a warning so old configs don't break.
    """
    argv = list(sys.argv[1:]) if argv is None else list(argv)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str)
    known, _ = pre.parse_known_args(argv)

    if known.config:
        cfg = _load_yaml(known.config)
        valid_dests = {a.dest for a in parser._actions}
        unknown = sorted(set(cfg) - valid_dests)
        if unknown:
            print(f"[args] Ignoring unknown config keys: {unknown}", file=sys.stderr)
        parser.set_defaults(**{k: v for k, v in cfg.items() if k in valid_dests})

    args = parser.parse_args(argv)

    missing = [name for name in required if getattr(args, name, None) in (None, "")]
    if missing:
        parser.error(
            "missing required values (pass via --config YAML or CLI flag): "
            + ", ".join(f"--{m}" for m in missing)
        )
    return args
