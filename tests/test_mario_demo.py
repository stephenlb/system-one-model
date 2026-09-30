"""World 1-1 is winnable by the rule the Mario demo's questions describe."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mario"))
sys.path.insert(0, str(ROOT / "demo"))

import mario_demo as m


def test_ground_state_verdicts():
    assert "clear" in m.ground_state(None)
    assert "too far" in m.ground_state(m.Hazard("pit", 90))
    assert "Jump now" in m.ground_state(m.Hazard("enemy", 20))


def test_baseline_rule():
    near = m.View(0, 208, False, False, m.Hazard("pit", 5))
    far = m.View(0, 208, False, False, m.Hazard("pit", 80))
    assert m.baseline_action(near) == "jump"
    assert m.baseline_action(far) == "run"
    rising = m.View(0, 190, True, True, m.Hazard("enemy", 20))
    assert m.baseline_action(rising) == "hold"
    over_wall = m.View(0, 80, True, True, m.Hazard("wall", 10, top=150, tiles=2))
    assert m.baseline_action(over_wall) == "release"
    assert m.baseline_action(m.View(0, 190, True, False, None)) == "release"


def test_baseline_clears_world_1_1():
    pytest.importorskip("gym_super_mario_bros")
    result = m.run_episode(None, max_decisions=600, window=None, quiet=True)
    assert result["result"] == "FLAG"
    assert result["x"] >= 3100
