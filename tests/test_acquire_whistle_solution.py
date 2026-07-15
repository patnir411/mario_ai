"""Non-ROM checks for the verified AcquireWhistle_1_3 solution artifact."""
from __future__ import annotations

import json
from pathlib import Path

SOLUTION = Path("data/solutions/sma4/acquire_whistle_1_3.json")


def test_acquire_whistle_1_3_solution_schema():
    assert SOLUTION.exists(), "missing acquire_whistle_1_3.json"
    sol = json.loads(SOLUTION.read_text())
    assert sol["game_id"] == "sma4"
    assert sol["level_id"] == "1-3"
    assert sol["option_id"] == "acquire_whistle_1_3"
    assert sol["knowledge_tier"] == 2
    assert sol["solved"] is True
    assert sol["replay_verified"] is True
    assert sol["inventory_item"] == 0x0C
    assert sol["inventory_after"][0] == 0x0C
    assert sol["n_frames"] == len(sol["path_buttons"])
    assert sol["n_frames"] > 2000
    assert Path(sol["entry_snapshot"]).name == "1-3_pwing_entry.pkl"
    # Button traces are lists of button-name strings.
    assert all(isinstance(frame, list) for frame in sol["path_buttons"][:5])
    assert any("DOWN" in frame for frame in sol["path_buttons"])
    assert any("B" in frame for frame in sol["path_buttons"])
