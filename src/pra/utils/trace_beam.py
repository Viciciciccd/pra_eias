from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pra.utils.types import Action, Config
from pra.utils.prompts import PI_THETA_INSTRUCTION, MU_PHI_SYSTEM, STEP_PATTERN


def _extract_step_from_text(text: str, current_step: int) -> tuple:
    matches = list(STEP_PATTERN.finditer(text))
    if not matches:
        return text.strip(), None

    expected = str(current_step + 1)
    next_step = str(current_step + 2)

    for idx, m in enumerate(matches):
        if m.group(1) == expected:
            if idx + 1 < len(matches):
                next_match = matches[idx + 1]
                step_content = text[m.start() : next_match.start()].strip()
                if next_match.group(1) == next_step:
                    pending = text[next_match.start() :].strip()
                    return step_content, pending
                return step_content, None
            return text[m.start() :].strip(), None

    if len(matches) > 1:
        return text[: matches[1].start()].strip(), None
    return text[matches[0].start() :].strip(), None


class BeamTrace:
    """Tracks multiple reasoning beams for a single question."""

    def __init__(self, j: int, q_dict: dict, tokenizer, config: Config):
        self.j = j
        self.q_dict = q_dict
        self.config = config

        self.action = Action.REASON
        self.i = 0  # current beam step index (0-based)

        # Evidence shared across beams (reward-model only)
        self.E: Optional[List[str]] = None
        self.pending_evidence: bool = False

        self.history: List[Dict[str, Any]] = []

        opts = "\n".join(f"({k}): {v}" for k, v in q_dict["options"].items())
        user_content = f"Question: {q_dict['question']}\n{opts}\n\nAnswer:"
        self._base_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": PI_THETA_INSTRUCTION},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Each beam: {steps, cumulative_score, is_done, pending_step_header}
        self.beams: List[Dict[str, Any]] = [
            {"steps": [], "cumulative_score": 0.0, "is_done": False, "pending_step_header": None}
        ]

        # Candidates generated for the current beam step:
        # (beam_idx, candidate_text, pending_header, reached_eos)
        self.pending_candidates: List[Tuple[int, str, Optional[str], bool]] = []

        # When SEARCH is triggered after reward pass 1, we keep candidate scores to build the query.
        # (beam_idx, text, hat_r_qual, hat_r_dep, reached_eos, pending_header)
        self._candidates_with_scores_for_query: Optional[
            List[Tuple[int, str, float, float, bool, Optional[str]]]
        ] = None

        # When SEARCH is forced without running reward pass 1 (theta_dep <= 0),
        # we keep pending candidates to build a reasonable query.
        self._candidates_for_query: Optional[List[Tuple[int, str, Optional[str], bool]]] = None

    def get_active_beams(self) -> List[int]:
        if self.config.K_max is not None and self.i >= self.config.K_max:
            return []
        return [idx for idx, b in enumerate(self.beams) if not b["is_done"]]

    def extract_step(self, text: str, beam_idx: int = None) -> tuple:
        return _extract_step_from_text(text, self.i)

    def build_pi_theta_prompt(self, beam_idx: int) -> str:
        """Build prompt for pi_theta (NO evidence in beam mode)."""
        beam = self.beams[beam_idx]
        base = self._base_prompt if not beam["steps"] else f"{self._base_prompt}\n\n" + "\n\n".join(beam["steps"])
        if beam["pending_step_header"]:
            base += "\n\n" + beam["pending_step_header"]
        elif beam["steps"]:
            base += "\n\n"
        return base

    def build_mu_phi_prompt(self, beam_idx: int, candidate_text: str) -> list:
        """Build mu_phi prompt; includes evidence only when pending_evidence=True."""
        beam = self.beams[beam_idx]
        parts = []

        if self.E and self.pending_evidence:
            parts.append("=== DOCUMENTS ===\n" + "\n".join(f"Doc {i+1}: {doc}" for i, doc in enumerate(self.E)))

        opts = "\n".join(f"({k}): {v}" for k, v in self.q_dict["options"].items())
        parts.append(f"=== QUESTION ===\n{self.q_dict['question']}\n{opts}")

        if self.config.aggregate_steps_for_reward and beam["steps"]:
            reasoning = "\n\n".join(beam["steps"]) + "\n\n" + candidate_text
        else:
            reasoning = candidate_text
        parts.append(f"=== REASONING SOLUTION ===\n{reasoning}")

        return [
            {"role": "system", "content": MU_PHI_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def _best_proposed_beam_steps_for_query(
        self, candidates_with_scores: List[Tuple[int, str, float, float, bool, Optional[str]]]
    ) -> List[str]:
        """Steps for the highest-scoring beam after a hypothetical expansion."""
        proposed = []
        for beam_idx, text, hat_r_qual, _hat_r_dep, _reached_eos, _pending_header in candidates_with_scores:
            parent = self.beams[beam_idx]
            proposed_steps = parent["steps"] + [text]
            proposed_score = float(parent["cumulative_score"]) + float(hat_r_qual)
            proposed.append((proposed_score, proposed_steps))
        if not proposed:
            return []
        proposed.sort(key=lambda x: x[0], reverse=True)
        return proposed[0][1]

    def build_rho_query(self) -> str:
        """Build retrieval query for SEARCH from the best *proposed* beam after pass-1 reward."""
        q = self.q_dict["question"]

        if self._candidates_with_scores_for_query:
            steps = self._best_proposed_beam_steps_for_query(self._candidates_with_scores_for_query)
            context = "\n".join(steps if self.config.retriever_use_all_steps else steps[-2:])
            return f"{q}\n\nContext:\n{context}" if context else q

        if self._candidates_for_query:
            # No pass-1 reward scores available; build query from one candidate per active beam.
            first_by_beam: Dict[int, str] = {}
            for beam_idx, text, _pending_header, _reached_eos in self._candidates_for_query:
                first_by_beam.setdefault(beam_idx, text)

            contexts = []
            for beam_idx in sorted(first_by_beam):
                parent = self.beams[beam_idx]
                proposed_steps = parent["steps"] + [first_by_beam[beam_idx]]
                ctx = "\n".join(
                    proposed_steps if self.config.retriever_use_all_steps else proposed_steps[-2:]
                )
                if ctx:
                    contexts.append(ctx)

            # Cap context blocks at beam_width to keep the query compact.
            contexts = contexts[: max(1, int(self.config.get_beam_width(self.i)))]
            if contexts:
                return f"{q}\n\nContext:\n" + "\n\n---\n\n".join(contexts)
            return q

        if not self.beams:
            return q
        best_beam = max(self.beams, key=lambda b: b["cumulative_score"])
        steps = best_beam["steps"]
        context = "\n".join(steps if self.config.retriever_use_all_steps else steps[-2:])
        return f"{q}\n\nContext:\n{context}" if context else q

    def should_search(self, candidates_with_scores: List[Tuple[int, str, float, float, bool, Optional[str]]]) -> bool:
        """Decide whether to search based on max dep among top kept beams."""
        if self.pending_evidence or not candidates_with_scores:
            return False

        new_beams = []
        for beam_idx, text, hat_r_qual, hat_r_dep, reached_eos, _pending_header in candidates_with_scores:
            parent = self.beams[beam_idx]
            cumulative = parent["cumulative_score"] + hat_r_qual
            new_beams.append((cumulative, hat_r_dep))

        new_beams.sort(key=lambda x: x[0], reverse=True)
        top = new_beams[: self.config.get_beam_width(self.i)]
        max_dep = max(d for _c, d in top) if top else 0.0
        return max_dep >= self.config.theta_dep

    def after_search(self, query: str, docs: list):
        """SEARCH: retrieve docs, then REWARD again (do NOT advance i)."""
        self.E = list(docs)[: self.config.max_docs]
        self.pending_evidence = True
        self.history.append(
            {
                "action": "search",
                "step": self.i,
                "input": query,
                "output": {"count": len(self.E), "preview": [d[:200] for d in self.E[:5]]},
            }
        )
        self._candidates_with_scores_for_query = None
        self._candidates_for_query = None
        self.action = Action.REWARD

    def finalize_expansion(self, candidates_with_scores: List[Tuple[int, str, float, float, bool, Optional[str]]]):
        """Finalize beam expansion/pruning using provided scores, then advance to next step."""
        if not candidates_with_scores:
            self.action = Action.DONE
            return

        new_beams = []
        for beam_idx, text, hat_r_qual, hat_r_dep, reached_eos, pending_header in candidates_with_scores:
            parent = self.beams[beam_idx]
            new_beam = {
                "steps": parent["steps"] + [text],
                "cumulative_score": parent["cumulative_score"] + hat_r_qual,
                "is_done": reached_eos,
                "pending_step_header": pending_header if not reached_eos else None,
            }
            new_beams.append((new_beam, hat_r_qual, hat_r_dep, beam_idx))

        new_beams.sort(key=lambda x: x[0]["cumulative_score"], reverse=True)
        top = new_beams[: self.config.get_beam_width(self.i)]

        self.history.append(
            {
                "action": "beam_expansion",
                "step": self.i,
                "num_candidates": len(candidates_with_scores),
                "num_beams_kept": len(top),
                "beam_scores": [(b["cumulative_score"], q, d, p_idx) for b, q, d, p_idx in top],
                "all_candidate_scores": [(b_idx, q, d) for b_idx, _t, q, d, _eos, _hdr in candidates_with_scores],
                "reward_with_evidence": bool(self.pending_evidence),
            }
        )

        self.beams = [b for b, _q, _d, _p in top]

        # Force-stop all beams at K_max.
        if self.config.K_max is not None and self.i + 1 >= self.config.K_max:
            for b in self.beams:
                b["is_done"] = True

        if not self.beams or all(b["is_done"] for b in self.beams):
            self.action = Action.DONE
            return

        self.i += 1
        self.E = None
        self.pending_evidence = False
        self.action = Action.REASON

    def get_best_beam(self) -> Optional[Dict[str, Any]]:
        """Select best beam using ``config.beam_final_selection``."""
        from pra.utils.scoring import select_final_beam

        return select_final_beam(
            self.beams,
            self.config.beam_final_selection,
            require_parsable=self.config.beam_require_parsable_answer,
        )

    def to_dict(self) -> dict:
        best_beam = self.get_best_beam()
        d = {
            "j": self.j,
            "i": self.i,
            "q": self.q_dict,
            "steps": best_beam["steps"] if best_beam else [],
            "all_beams": [
                {"steps": b["steps"], "cumulative_score": b["cumulative_score"], "is_done": b["is_done"]}
                for b in self.beams
            ],
            "num_beams": len(self.beams),
            "history": self.history,
            "E": self.E,
        }
        if hasattr(self, "result"):
            d["predicted_answer"] = self.predicted_answer
            d["correct_answer"] = self.correct_answer
            d["result"] = self.result
        return d
