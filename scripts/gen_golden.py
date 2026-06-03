"""Generate the committed golden RAM hashes for the determinism regression test.

Run once (and re-run intentionally if the emulator/ROM/canonical sequence changes):
    ./venv/bin/python scripts/gen_golden.py
Commit the resulting tests/golden/ram_hashes.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import canonical_actions, run_sequence  # noqa: E402
from mario.env import MarioSim  # noqa: E402

CHECKPOINTS = [0, 50, 100, 150, 200]


def main() -> None:
    sim = MarioSim(world=1, stage=1)
    sim.reset(seed=0)
    final, marks, done_at, done = run_sequence(sim, canonical_actions(200), CHECKPOINTS)
    sim.close()

    out = {
        "level": "1-1",
        "seed": 0,
        "n_actions": 200,
        "checkpoints": {str(k): v for k, v in marks.items()},
        "final": final,
        "done_at": done_at,
        "done": done,
    }
    dest = ROOT / "tests" / "golden" / "ram_hashes.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", dest)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
