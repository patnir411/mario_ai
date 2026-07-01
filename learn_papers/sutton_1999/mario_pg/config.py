"""Experiment configuration loader.

Usage:
    from mario_pg.config import load_config
    cfg = load_config("learn_papers/sutton_1999/configs/exp01_reinforce.yaml")
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ExperimentConfig:
    exp_id: str
    trainer: Literal["reinforce", "actor_critic", "oracle_ac", "sanity"] = "reinforce"
    world: int = 1
    stage: int = 1
    chunk_frames: int = 8
    episodes: int = 2000
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    entropy_coef: float = 0.01
    baseline: Literal["none", "constant", "learned", "oracle"] = "none"
    policy: Literal["mlp", "linear_compatible", "shared_trunk"] = "mlp"
    K: int = 4
    seed: int = 0
    eval_every: int = 100
    eval_seeds: list[int] = field(default_factory=lambda: list(range(30)))
    max_chunks: int = 400
    log_steps: bool = False
    label_every: int = 4          # oracle AC: call label_state every N chunks
    n_states_align: int = 200     # grad alignment sample count
    policy_iteration_rounds: int = 5

    def level_key(self) -> str:
        return f"{self.world}-{self.stage}"


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    return ExperimentConfig(**raw)


def save_config_json(cfg: ExperimentConfig, path: Path, *, git_rev: str = "") -> None:
    d = asdict(cfg)
    d["git_rev"] = git_rev
    path.write_text(json.dumps(d, indent=2))
