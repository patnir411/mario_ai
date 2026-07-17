"""Non-ROM checks for the verified AcquireWhistle_fortress solution artifact."""
from __future__ import annotations

import json
from pathlib import Path

SOLUTION = Path("data/solutions/sma4/acquire_whistle_fortress.json")


def test_acquire_whistle_fortress_solution_schema():
    assert SOLUTION.exists(), "missing acquire_whistle_fortress.json"
    sol = json.loads(SOLUTION.read_text())
    assert sol["game_id"] == "sma4"
    assert sol["level_id"] == "1-fortress"
    assert sol["option_id"] == "acquire_whistle_fortress"
    assert sol["knowledge_tier"] == 2
    assert sol["solved"] is True
    assert sol["replay_verified"] is True
    assert sol["inventory_item"] == 0x0C
    assert sol["inventory_after"][0] == 0x0C
    assert sol["n_frames"] == len(sol["path_buttons"])
    assert sol["n_frames"] > 800
    assert Path(sol["entry_snapshot"]).name == "1-fortress_pwing_leaf_entry.pkl"
    assert all(isinstance(frame, list) for frame in sol["path_buttons"][:5])
    # Roof route uses UP into the chest room; chest opens with A/B.
    assert any("UP" in frame for frame in sol["path_buttons"])
    assert any("A" in frame or "B" in frame for frame in sol["path_buttons"])
    facts = sol.get("injected_facts") or []
    assert "pwing_fortress_entry_snapshot" in facts
    assert "fortress_door_entry_snapshot" not in facts
    assert "fortress_inventory_rehosted_to_pre_door_map" not in facts
