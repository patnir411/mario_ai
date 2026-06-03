"""Thin, deterministic wrapper around gym-super-mario-bros for forward-model search.

The whole project depends on three capabilities this module exposes cleanly:
  1. RAM access (`ram`)             -> compact observations + reward
  2. exact snapshot/restore        -> beam-search node cloning (`snapshot`/`restore`)
  3. chunked stepping              -> action-chunk search + frame-skip (`run_chunk`)

Everything is deterministic given (seed, action sequence). See tests/test_determinism.py
and tests/test_snapshot.py for the invariants this relies on.

NOTE on ROM version: we use the `-v0` ("standard") variant on purpose — it returns the
full-color screen as the observation, which we need for human-readable contact sheets.
We read RAM directly, so the pixel-preprocessing variants (v1/v2/v3) give us nothing.
"""
from __future__ import annotations

import warnings

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Canonical action set for the whole project (7 discrete actions).
# Index meaning (gym_super_mario_bros SIMPLE_MOVEMENT):
#   0 NOOP, 1 right, 2 right+A, 3 right+B, 4 right+A+B, 5 A, 6 left
ACTIONS = SIMPLE_MOVEMENT
N_ACTIONS = len(ACTIONS)


def _silence(fn):
    """gym-super-mario-bros emits a benign 'out of date / upgrade to v3' DeprecationWarning."""
    def wrapped(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a, **k)
    return wrapped


class MarioSim:
    """Deterministic, snapshot-able Mario environment for search.

    Usage:
        sim = MarioSim(world=1, stage=1)
        info = sim.reset(seed=0)
        snap = sim.snapshot()
        info, done = sim.run_chunk(action_idx=2, frames=4)
        sim.restore(snap)          # back to exactly where snapshot() was taken
    """

    def __init__(self, world: int = 1, stage: int = 1, version: str = "v0",
                 actions=ACTIONS):
        self.world = world
        self.stage = stage
        self.version = version
        self.actions = actions
        self.env_id = f"SuperMarioBros-{world}-{stage}-{version}"
        make = _silence(gym_super_mario_bros.make)
        self.env = JoypadSpace(make(self.env_id), actions)
        self.u = self.env.unwrapped
        self._last_info: dict = {}
        self._last_obs = None

    # --- lifecycle ---------------------------------------------------------
    def reset(self, seed: int = 0) -> dict:
        reset = _silence(self.env.reset)
        obs, info = reset(seed=seed)
        self._last_obs = obs
        self._last_info = dict(info)
        return self._last_info

    def close(self) -> None:
        self.env.close()

    # --- state access ------------------------------------------------------
    @property
    def ram(self):
        """Full 2048-byte NES RAM as a uint8 numpy array (live view)."""
        return self.u.ram

    @property
    def last_info(self) -> dict:
        return self._last_info

    @property
    def last_obs(self):
        """Most recent RGB screen (256x240x3 uint8) — used for contact sheets."""
        return self._last_obs

    # --- snapshots (the beam-search primitive) -----------------------------
    def snapshot(self):
        """Opaque in-memory clone of full emulator state. Reusable for many restores."""
        return self.u.dump_state()

    def restore(self, snap) -> None:
        self.u.load_state(snap)

    # --- stepping ----------------------------------------------------------
    def step(self, action_idx: int):
        """One env frame. Returns (obs, info, done)."""
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        self._last_obs = obs
        self._last_info = dict(info)
        return obs, self._last_info, (terminated or truncated)

    def run_chunk(self, action_idx: int, frames: int):
        """Hold `action_idx` for `frames` frames (frame-skip / action-chunk unit).

        Stops early if the episode ends. Returns (info, done).
        """
        done = False
        info = self._last_info
        for _ in range(frames):
            _obs, info, done = self.step(action_idx)
            if done:
                break
        return info, done


def beat_flag(info: dict) -> bool:
    """True iff Mario reached the flagpole / level end."""
    return bool(info.get("flag_get", False))
