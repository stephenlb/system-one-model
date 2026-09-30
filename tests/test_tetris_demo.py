"""The Tetris demo's engine, planner, question and synthesised sound."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tetris"))
sys.path.insert(0, str(ROOT / "mario"))
sys.path.insert(0, str(ROOT / "demo"))

import tetris_demo as t


def test_line_clear_and_score():
    game = t.Game(seed=1)
    for c in range(t.WIDTH):
        game.board[t.HEIGHT - 1][c] = "I"
    game.board[t.HEIGHT - 1][0] = ""
    game.piece = t.Piece("I", rot=1, x=-2, y=0)  # vertical I into column 0
    assert not game.collides(game.piece)
    game.hard_drop()
    assert game.lines == 1 and game.score == 100
    assert "clear" in game.events
    for _ in range(t.CLEAR_FRAMES):
        game.frame()
    assert not any(game.board[t.HEIGHT - 1][c] == "I" and c > 0 for c in range(1, t.WIDTH))
    assert game.piece is not None


def test_walls_block_moves():
    game = t.Game(seed=2)
    game.piece = t.Piece("O", 0, 0, 0)
    game.move(-1)
    assert game.piece.x == 0


def test_planner_completes_a_line():
    game = t.Game(seed=3)
    for c in range(t.WIDTH - 2):
        game.board[t.HEIGHT - 1][c] = "J"
    game.piece = t.Piece("O", 0, 4, 0)
    target = t.plan(game)
    assert target.x == t.WIDTH - 2  # the O fills the two-wide gap


def test_question_verdicts_match_the_rule():
    game = t.Game(seed=4)
    for rot, x, tr, tx in [(0, 3, 0, 3), (0, 6, 0, 2), (0, 1, 0, 6), (0, 3, 2, 3)]:
        game.piece = t.Piece("T", rot, x, 0)
        target = t.Target(tr, tx, 0, 0.0)
        action = t.baseline_action(game, target)
        assert action in t.CRITERIA
        assert (action == "drop") == (t.place_state(game, target).endswith("in the target column."))


def test_baseline_clears_ten_lines():
    result = t.run_episode(None, lines=10, max_decisions=600, window=None, quiet=True, seed=2)
    assert result["result"] == "CLEARED"
    assert result["lines"] >= 10


def test_synth_frames_are_one_sixtieth_of_a_second():
    synth = t.Synth()
    quiet = synth.frame()
    assert quiet.shape == (t.AUDIO_FRAME, 2) and quiet.dtype.name == "int16"
    assert abs(int(quiet.max())) > 0  # the music is playing
    synth.trigger("tetris")
    loud = synth.frame()
    assert int(abs(loud).max()) > int(abs(quiet).max())
    synth.trigger("gameover")
    assert not synth.music_on
