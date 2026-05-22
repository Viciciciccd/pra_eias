from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Action(str, Enum):
    REASON = "reason"   # pi_theta generates next reasoning step
    REWARD = "reward"   # mu_phi observes step + assigns reward scores
    SEARCH = "search"   # rho retrieves evidence
    DONE = "done"


@dataclass
class Config:
    theta_qual: float
    theta_dep: float
    K_max: Optional[int]
    max_docs: int
    aggregate_steps_for_reward: bool
    beam_width: int = 3
    beam_candidates: int = 5
    beam_final_selection: str = "cumulative"
    # Adaptive beam search: when adaptive_beam is True and both *_start and *_end
    # are set, values are interpolated over steps with curvature beam_schedule_power
    # (1.0 = linear; >1 keeps closer to start for longer).
    adaptive_beam: bool = False
    beam_candidates_start: Optional[int] = None
    beam_candidates_end: Optional[int] = None
    beam_width_start: Optional[int] = None
    beam_width_end: Optional[int] = None
    beam_schedule_power: float = 1.0
    retriever_use_all_steps: bool = False

    def _step_frac(self, step_idx: int) -> float:
        """Map step index to [0,1] using K_max when available."""
        if self.K_max is None or self.K_max <= 1:
            return 0.0
        denom = max(1, self.K_max - 1)
        return min(1.0, max(0.0, step_idx / denom))

    def _interp_int(self, start: int, end: int, step_idx: int) -> int:
        """Interpolate int value from start->end over steps with power curve."""
        power = self.beam_schedule_power if self.beam_schedule_power > 0 else 1.0
        frac = self._step_frac(step_idx) ** power
        return max(1, int(round(start + (end - start) * frac)))

    def get_beam_candidates(self, step_idx: int) -> int:
        """Beam candidates to sample at a given step."""
        if (
            self.adaptive_beam
            and self.beam_candidates_start is not None
            and self.beam_candidates_end is not None
        ):
            return self._interp_int(self.beam_candidates_start, self.beam_candidates_end, step_idx)
        return max(1, int(self.beam_candidates))

    def get_beam_width(self, step_idx: int) -> int:
        """Beam width (top beams kept) at a given step."""
        if (
            self.adaptive_beam
            and self.beam_width_start is not None
            and self.beam_width_end is not None
        ):
            return self._interp_int(self.beam_width_start, self.beam_width_end, step_idx)
        return max(1, int(self.beam_width))
