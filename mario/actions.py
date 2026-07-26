"""Emulator-free action vocabularies shared across Mario backends.

Action indices are part of every saved trajectory's schema.  Keeping the SMB1
vocabulary in a dependency-free module lets search/adapters inspect that schema
without importing the optional NES emulator stack, while ``mario.env`` passes
the exact same list to ``JoypadSpace``.
"""
from __future__ import annotations


# gym-super-mario-bros SIMPLE_MOVEMENT plus DOWN/UP.  Never reorder these
# entries: committed SMB1 solution manifests store their integer indices.
SMB1_ACTIONS: list[list[str]] = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
    ["down"],
    ["up"],
]
SMB1_N_ACTIONS = len(SMB1_ACTIONS)

# Backward-compatible public names used by older experiment scripts.
ACTIONS = SMB1_ACTIONS
N_ACTIONS = SMB1_N_ACTIONS
DEFAULT_CHUNK_FRAMES = 8
ACTION_NAMES = [
    "NOOP",
    "right",
    "right+A",
    "right+B",
    "right+A+B",
    "A",
    "left",
    "down",
    "up",
]


def action_name(idx: int) -> str:
    return ACTION_NAMES[idx] if 0 <= idx < len(ACTION_NAMES) else f"action{idx}"
