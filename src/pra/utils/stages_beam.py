import time
from tqdm import tqdm

from pra.utils.types import Action, Config
from pra.utils.reward_model import call_mu_phi, parse_mu_phi_output
from pra.utils.retriever import batch_retrieve


def run_reward_beam(traces: list, mu_phi_tokenizer, config: Config, args) -> tuple:
    """Handle REWARD action for beam search (two-pass: before and after SEARCH)."""
    if not traces:
        return [], []

    start_time = time.time()
    total_scores = 0
    mu_phi_params = {"max_tokens": 5, "temperature": 0.0, "logprobs": 20}

    total_candidates = sum(len(t.pending_candidates) for t in traces)
    pass_name = "with_E" if any(getattr(t, "pending_evidence", False) for t in traces) else "no_E"
    print(f"  [Beam Reward:{pass_name}] Scoring {total_candidates} candidates from {len(traces)} traces")

    pbar = tqdm(traces, desc=f"Beam reward scoring ({pass_name})", unit="trace")
    for trace in pbar:
        candidates_with_scores = []

        # Fast-path: if theta_dep <= 0, SEARCH is always desired, so skip pass-1 reward
        # (avoid scoring candidates without evidence).
        if (not trace.pending_evidence) and float(getattr(trace.config, "theta_dep", 0.0)) <= 0.0:
            trace._candidates_for_query = list(trace.pending_candidates)
            trace.history.append(
                {
                    "action": "beam_reward_decision",
                    "step": trace.i,
                    "decision": "skip_reward_force_search",
                    "theta_dep": trace.config.theta_dep,
                }
            )
            trace.action = Action.SEARCH
            continue

        # Score all pending candidates
        for beam_idx, text, pending_header, reached_eos in trace.pending_candidates:
            messages = trace.build_mu_phi_prompt(beam_idx, text)
            output = call_mu_phi(messages, mu_phi_tokenizer, args.reward_model_path, args.reward_model_url, mu_phi_params)
            scores = parse_mu_phi_output(output)

            candidates_with_scores.append(
                (beam_idx, text, scores["hat_r_qual"], scores["hat_r_dep"], reached_eos, pending_header)
            )
            total_scores += 1

        if not trace.pending_evidence:
            # Pass 1 (no evidence): decide whether to search
            if trace.should_search(candidates_with_scores):
                # Keep pending candidates for rescore; store scores for query-building
                trace._candidates_with_scores_for_query = candidates_with_scores
                trace.history.append(
                    {
                        "action": "beam_reward_decision",
                        "step": trace.i,
                        "decision": "search",
                        "theta_dep": trace.config.theta_dep,
                    }
                )
                trace.action = Action.SEARCH
            else:
                trace.history.append(
                    {
                        "action": "beam_reward_decision",
                        "step": trace.i,
                        "decision": "no_search",
                        "theta_dep": trace.config.theta_dep,
                    }
                )
                trace.pending_candidates = []
                trace.finalize_expansion(candidates_with_scores)
        else:
            # Pass 2 (with evidence): finalize expansion
            trace.history.append(
                {
                    "action": "beam_reward_decision",
                    "step": trace.i,
                    "decision": "finalize_after_search",
                    "theta_dep": trace.config.theta_dep,
                }
            )
            trace.pending_candidates = []
            trace.finalize_expansion(candidates_with_scores)

        elapsed = time.time() - start_time
        speed = total_scores / elapsed if elapsed > 0 else 0
        pbar.set_postfix({"speed": f"{speed:.2f}it/s", "beams": len(getattr(trace, "beams", []))})

    elapsed = time.time() - start_time
    speed = total_scores / elapsed if elapsed > 0 else 0
    print(f"  [Beam Reward:{pass_name}] {total_scores} scores in {elapsed:.2f}s ({speed:.2f}it/s)")

    return [t for t in traces if t.action != Action.DONE], [t for t in traces if t.action == Action.DONE]


def run_search_beam(traces: list, retriever_sources: list, args) -> tuple:
    """Handle SEARCH action for beam search: retrieve docs and re-enter REWARD."""
    if not traces:
        return [], []

    queries = [trace.build_rho_query() for trace in traces]

    all_docs = batch_retrieve(
        queries=queries,
        allowed_sources=retriever_sources,
        topk=args.retriever_topk,
        rerank_topk=args.retriever_rerank_topk,
        device=args.retriever_device,
        rerank_workers=args.rerank_workers,
        use_reranker=args.use_reranker,
        show_progress=True,
    )

    for trace, query, docs in zip(traces, queries, all_docs):
        trace.after_search(query, docs)

    return [t for t in traces if t.action != Action.DONE], [t for t in traces if t.action == Action.DONE]

