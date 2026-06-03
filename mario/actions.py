"""Action-chunk vocabulary for search and (later) the policy.

A "chunk" = hold one SIMPLE_MOVEMENT action for `chunk_frames` frames (frame-skip).
This is the single biggest lever on search tractability (DESIGN.md §5): branching stays
at 7 and a depth-d beam covers d*chunk_frames frames. Larger chunks = shallower search
but coarser control; 1-1 is forgiving enough for chunk_frames=8.

Index meaning (mario.env.ACTIONS == SIMPLE_MOVEMENT):
  0 NOOP, 1 right, 2 right+A, 3 right+B, 4 right+A+B, 5 A, 6 left
"""
from __future__ import annotations

from mario.env import ACTIONS, N_ACTIONS  # noqa: F401  (re-exported)

DEFAULT_CHUNK_FRAMES = 8

ACTION_NAMES = ["NOOP", "right", "right+A", "right+B", "right+A+B", "A", "left"]


def action_name(idx: int) -> str:
    return ACTION_NAMES[idx] if 0 <= idx < len(ACTION_NAMES) else f"action{idx}"
