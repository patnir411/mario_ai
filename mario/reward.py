"""Death-aware reward / state scoring for the search teacher.

This is the most important module to get right (DESIGN.md §6). The two canonical
failure modes it must avoid, both from Tom7's playfun:
  1. Death must be a HARD, dominant negative — otherwise search/learner jumps into
     pits because respawning a screen back still "scores OK".
  2. Only TRUE rightward level progress (x_pos) counts. Counter-style RAM (score,
     coins, timer) is ignored — it creates fake monotone progress that traps search
     on e.g. the 1-2 coin ledge.

Framerule note (DESIGN.md §6): SMB rounds level time up to a 21-frame boundary, so
shaving <21 frames mid-level is worthless. Hence `time` weight defaults to 0 during
the "beat it" phase; only the speed phase / 8-4 raises it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 1.0      # per pixel of true rightward level progress
    flag: float = 10000.0      # reaching the flagpole — dominates everything good
    death: float = 10000.0     # dying — dominates everything (HARD negative)
    time: float = 0.0          # ~0 mid-level (framerules); raise only for speed phase
    stuck: float = 5.0         # per stuck-step (no forward progress); search supplies count


DEFAULT = RewardWeights()


def is_success(info: dict) -> bool:
    """True iff Mario reached the flagpole / completed the level."""
    return bool(info.get("flag_get", False))


def is_death(info: dict, done: bool) -> bool:
    """A finished episode that did NOT reach the flag is a death (incl. timeout)."""
    return bool(done) and not is_success(info)


def death_cause(info: dict) -> str:
    """Best-effort death cause for diagnostics/histograms.

    Refined empirically in V3 once we have real death trajectories. `timeout` is
    reliable from the clock; pit vs enemy is approximate and flagged as such.
    """
    if int(info.get("time", 1)) <= 0:
        return "timeout"
    # y_pos near the bottom of the play-field indicates a fall into a pit. The exact
    # threshold is calibrated against real deaths in V3; until then report best-effort.
    y = int(info.get("y_pos", 0))
    if y < 80:
        return "pit"
    return "enemy"


def state_score(info: dict, x_start: int, frames: int = 0, died: bool = False,
                stuck: int = 0, w: RewardWeights = DEFAULT) -> float:
    """Scalar score of a (possibly partial) trajectory ending in `info`.

    Higher is better. Beam search keeps the top-k by this score, so it must rank
    "further right and alive" above "stuck" above "dead", with the flag dominating.
    Deliberately reads ONLY x_pos / flag / death — never score/coins/timer.
    """
    score = w.progress * (int(info.get("x_pos", 0)) - x_start)
    if is_success(info):
        score += w.flag
    if died:
        score -= w.death
    score -= w.time * frames
    score -= w.stuck * stuck
    return score
