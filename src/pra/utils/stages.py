from vllm import SamplingParams

from pra.utils.types import Action, Config


def run_reason_beam(traces: list, pi_theta, config: Config, args) -> tuple:
    """Handle REASON action for beam search mode."""
    if not traces:
        return [], []

    all_updated, all_done = [], []

    step_dist: dict[int, int] = {}
    for t in traces:
        step_dist[t.i] = step_dist.get(t.i, 0) + 1
    total_beams = sum(len(t.get_active_beams()) for t in traces)
    dist_str = " ".join(f"s{k}:{v}" for k, v in sorted(step_dist.items()))
    print(f"  [Beam Reason] {len(traces)} traces, {total_beams} active beams | {dist_str}")

    step_groups: dict[int, list] = {}
    for t in traces:
        step_groups.setdefault(t.i, []).append(t)

    for step_idx, group_traces in step_groups.items():
        next_step_num = step_idx + 2

        is_last_step = args.kmax_no_stop and config.K_max is not None and step_idx + 1 >= config.K_max
        if is_last_step:
            stop_seqs = []
        else:
            stop_seqs = [
                f"\nStep {next_step_num}:",
                f"\nStep_{next_step_num}:",
                f"\n\nStep {next_step_num}:",
                f"\n\nStep_{next_step_num}:",
            ]

        n_candidates = config.get_beam_candidates(step_idx)

        prompt_info = []  # (trace, beam_idx, prompt)
        for trace in group_traces:
            active_beams = trace.get_active_beams()
            if not active_beams:
                trace.action = Action.DONE
                all_done.append(trace)
                continue
            for beam_idx in active_beams:
                prompt = trace.build_pi_theta_prompt(beam_idx)
                prompt_info.append((trace, beam_idx, prompt))

        if not prompt_info:
            continue

        params = SamplingParams(
            max_tokens=args.pi_theta_max_tokens,
            n=n_candidates,
            temperature=args.pi_theta_temperature,
            top_p=args.pi_theta_top_p,
            top_k=args.pi_theta_top_k,
            stop=stop_seqs,
            repetition_penalty=args.repetition_penalty,
            include_stop_str_in_output=True,
        )

        prompts = [p for _, _, p in prompt_info]
        outputs = pi_theta.generate(prompts=prompts, sampling_params=params)

        trace_candidates: dict[int, list] = {id(t): [] for t in group_traces}

        for (trace, beam_idx, _prompt), out in zip(prompt_info, outputs):
            beam = trace.beams[beam_idx]
            for completion in out.outputs:
                raw_text = completion.text
                if beam["pending_step_header"]:
                    raw_text = beam["pending_step_header"] + raw_text

                cleaned, pending_header = trace.extract_step(raw_text, beam_idx)
                reached_eos = (completion.finish_reason == "stop" and completion.stop_reason is None)
                trace_candidates[id(trace)].append((beam_idx, cleaned, pending_header, reached_eos))

        for trace in group_traces:
            if trace.action == Action.DONE:
                continue
            candidates = trace_candidates[id(trace)]
            if candidates:
                trace.pending_candidates = candidates
                trace.action = Action.REWARD
                all_updated.append(trace)
            else:
                trace.action = Action.DONE
                all_done.append(trace)

    return all_updated, all_done
