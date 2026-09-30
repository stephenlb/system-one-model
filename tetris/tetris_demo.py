"""Gemma 4 plays Tetris in real time, with music and sound effects.

Tetris here is a small self-contained engine (10x20 board, 7-bag pieces, gravity,
line clears). It runs on the same clock as the Mario demo: 60 frames per second,
one decision every ``FRAMES_PER_DECISION`` (8) frames = 133ms of game time, video
paced at 60fps and the next decision computed while the current block is on screen
and in the speakers.

The decision
------------
The piece falls whether or not anyone has decided anything, so the model steers it
one step per decision. The demo does the search: it tries every rotation and column
for the falling piece, scores the resulting boards (height, holes, bumpiness, lines
cleared) and marks the best one as the target. The model reads how the piece sits
against that target and picks the move:

  ==========  ==================================================================
  ``rotate``  the piece is not turned the way the target needs
  ``left``    turned correctly but too far right of the target column
  ``right``   turned correctly but too far left of the target column
  ``drop``    turned correctly and in the target column: hard drop
  ==========  ==================================================================

That is the split used by the other demos: the demo does the geometry, the model
reads a verdict in words. The verdict names the mistake and carries no distances,
because numbers in the state get read instead of the verdict (see the Flappy Bird
demo).

Sound
-----
All audio is synthesised (numpy square waves), so there is nothing to install:
a Korobeiniki loop with a bass line, and effects for moving, turning, locking, line
clears, a four-line Tetris, winning and topping out. ``--sound`` streams it through
the same ``Speaker`` as the Mario demo.

Run (from the repo root):
    python tetris/tetris_demo.py                  # 10 lines, model decides
    python tetris/tetris_demo.py --sound --watch  # with music, in a window
    python tetris/tetris_demo.py --baseline       # reference player, no model
    python tetris/tetris_demo.py --self-test      # check the question, no game
    python tetris/tetris_demo.py --lines 40 --max-decisions 2000 --episodes 3
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "mario"))

from common import LatencyLog, print_header
from mario_demo import Block, Pacer, Speaker, present  # same clock, same speaker

WIDTH, HEIGHT = 10, 20
FRAME_HZ = 60
FRAMES_PER_DECISION = 8  # 133ms of game time, longer than a decision takes
GRAVITY_FRAMES = 16  # one row every 16 frames: ~7 rows in a 2s piece journey
CLEAR_FRAMES = 18  # rows flash this long before they collapse
KICKS = (0, -1, 1, -2, 2)  # sideways nudges tried when a turn is blocked

SHAPES = {
    "I": ["....", "XXXX", "....", "...."],
    "O": ["XX", "XX"],
    "T": [".X.", "XXX", "..."],
    "S": [".XX", "XX.", "..."],
    "Z": ["XX.", ".XX", "..."],
    "J": ["X..", "XXX", "..."],
    "L": ["..X", "XXX", "..."],
}
COLORS = {
    "I": (0, 200, 220), "O": (230, 200, 40), "T": (170, 70, 200), "S": (70, 200, 80),
    "Z": (220, 60, 60), "J": (60, 90, 220), "L": (230, 140, 40),
}
KINDS = list(SHAPES)


def _rotate(matrix: list[str]) -> list[str]:
    n = len(matrix)
    return ["".join(matrix[n - 1 - c][r] for c in range(n)) for r in range(n)]


ROTATIONS: dict[str, list[list[tuple[int, int]]]] = {}
for _kind, _base in SHAPES.items():
    _states, _m = [], _base
    for _ in range(4):
        _states.append([(r, c) for r, row in enumerate(_m) for c, ch in enumerate(row) if ch == "X"])
        _m = _rotate(_m)
    ROTATIONS[_kind] = _states


# --------------------------------------------------------------------------- game
@dataclass
class Piece:
    kind: str
    rot: int = 0
    x: int = 0
    y: int = 0

    def cells(self, rot=None, x=None, y=None):
        rot = self.rot if rot is None else rot
        x = self.x if x is None else x
        y = self.y if y is None else y
        return [(y + r, x + c) for r, c in ROTATIONS[self.kind][rot % 4]]


@dataclass
class Game:
    seed: int | None = None
    board: list = field(default_factory=lambda: [[""] * WIDTH for _ in range(HEIGHT)])
    piece: Piece | None = None
    next_kind: str = ""
    score: int = 0
    lines: int = 0
    pieces: int = 0
    over: bool = False
    clearing: int = 0  # frames left in the line-clear flash
    full_rows: list = field(default_factory=list)
    events: list = field(default_factory=list)
    gravity: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.bag: list[str] = []
        self.next_kind = self._draw()
        self.spawn()

    def _draw(self) -> str:
        if not self.bag:
            self.bag = KINDS[:]
            self.rng.shuffle(self.bag)
        return self.bag.pop()

    def spawn(self) -> None:
        kind, self.next_kind = self.next_kind, self._draw()
        size = len(SHAPES[kind])
        self.piece = Piece(kind, 0, (WIDTH - size) // 2, 0)
        self.gravity = 0
        self.pieces += 1
        if self.collides(self.piece):
            self.over = True
            self.events.append("gameover")

    def collides(self, piece: Piece, **at) -> bool:
        for r, c in piece.cells(**at):
            if c < 0 or c >= WIDTH or r >= HEIGHT or (r >= 0 and self.board[r][c]):
                return True
        return False

    # -- moves
    def move(self, dx: int) -> None:
        p = self.piece
        if not self.collides(p, x=p.x + dx):
            p.x += dx
            self.events.append("move")

    def rotate(self) -> None:
        p = self.piece
        for kick in KICKS:
            if not self.collides(p, rot=p.rot + 1, x=p.x + kick):
                p.rot, p.x = (p.rot + 1) % 4, p.x + kick
                self.events.append("rotate")
                return

    def drop_y(self, piece: Piece, **at) -> int:
        y = piece.y if at.get("y") is None else at["y"]
        rot, x = at.get("rot", piece.rot), at.get("x", piece.x)
        while not self.collides(piece, rot=rot, x=x, y=y + 1):
            y += 1
        return y

    def hard_drop(self) -> None:
        self.piece.y = self.drop_y(self.piece)
        self.events.append("drop")
        self.lock()

    def lock(self) -> None:
        p = self.piece
        for r, c in p.cells():
            if r >= 0:
                self.board[r][c] = p.kind
        self.events.append("lock")
        self.piece = None
        rows = [r for r in range(HEIGHT) if all(self.board[r])]
        if rows:
            self.full_rows, self.clearing = rows, CLEAR_FRAMES
            self.lines += len(rows)
            self.score += {1: 100, 2: 300, 3: 500, 4: 800}[len(rows)]
            self.events.append("tetris" if len(rows) == 4 else "clear")
        else:
            self.spawn()

    # -- clock
    def frame(self, action: str | None = None) -> None:
        """Advance one frame; ``action`` is applied at the start of it."""
        if self.over:
            return
        if self.clearing:
            self.clearing -= 1
            if not self.clearing:
                for r in self.full_rows:
                    del self.board[r]
                    self.board.insert(0, [""] * WIDTH)
                self.full_rows = []
                self.spawn()
            return
        if action == "left":
            self.move(-1)
        elif action == "right":
            self.move(1)
        elif action == "rotate":
            self.rotate()
        elif action == "drop":
            self.hard_drop()
            return
        self.gravity += 1
        if self.gravity >= GRAVITY_FRAMES:
            self.gravity = 0
            if self.collides(self.piece, y=self.piece.y + 1):
                self.lock()
            else:
                self.piece.y += 1


# ------------------------------------------------------------------------ planner
@dataclass
class Target:
    rot: int
    x: int
    y: int
    score: float


def board_features(board) -> tuple[int, int, int]:
    """Aggregate height, holes and bumpiness of a board of kind-or-empty cells."""
    heights = []
    holes = 0
    for c in range(WIDTH):
        top = next((r for r in range(HEIGHT) if board[r][c]), HEIGHT)
        heights.append(HEIGHT - top)
        holes += sum(1 for r in range(top, HEIGHT) if not board[r][c])
    bumps = sum(abs(heights[i] - heights[i + 1]) for i in range(WIDTH - 1))
    return sum(heights), holes, bumps


def plan(game: Game) -> Target:
    """The best (rotation, column) for the falling piece, found by trying them all."""
    piece = game.piece
    best: Target | None = None
    for rot in range(4):
        for x in range(-3, WIDTH):
            if game.collides(piece, rot=rot, x=x, y=piece.y):
                continue
            y = game.drop_y(piece, rot=rot, x=x)
            board = [row[:] for row in game.board]
            for r, c in piece.cells(rot=rot, x=x, y=y):
                if r >= 0:
                    board[r][c] = piece.kind
            kept = [row for row in board if not all(row)]
            cleared = HEIGHT - len(kept)
            board = [[""] * WIDTH for _ in range(cleared)] + kept
            height, holes, bumps = board_features(board)
            score = -0.51 * height + 0.76 * cleared - 0.36 * holes - 0.18 * bumps
            # Prefer fewer turns and less sideways travel when boards are equal.
            score -= 1e-3 * rot + 1e-4 * abs(x - piece.x)
            if best is None or score > best.score:
                best = Target(rot, x, y, score)
    return best or Target(piece.rot, piece.x, piece.y, 0.0)


# ------------------------------------------------------------------------ prompts
# Every verdict names what is wrong with the piece, and the state carries no
# distances or piece names (see the Flappy Bird demo on why).
CRITERIA = {
    "drop": "Drop - the piece is turned the right way and is in the target column",
    "left": "Move left - the piece is too far right of the target column",
    "right": "Move right - the piece is too far left of the target column",
    "rotate": "Turn the piece - it is not turned the way the target spot needs",
}

CONTRACT = {
    "place": {
        "type": "choice",
        "instructions": (
            "A Tetris piece is falling and must land in the target spot. "
            "Decide the next move."
        ),
        "criteria": CRITERIA,
    },
}


def place_state(game: Game, target: Target) -> str:
    p = game.piece
    if p.rot % 4 != target.rot:
        return "The piece is not turned the way the target spot needs."
    if p.x < target.x:
        return "The piece is too far left of the target column."
    if p.x > target.x:
        return "The piece is too far right of the target column."
    return "The piece is turned correctly and is in the target column."


def baseline_action(game: Game, target: Target) -> str:
    """The rule the criteria describe, used with ``--baseline`` and the tests."""
    p = game.piece
    if p.rot % 4 != target.rot:
        return "rotate"
    if p.x < target.x:
        return "right"
    if p.x > target.x:
        return "left"
    return "drop"


class _Fake:
    def __init__(self, rot, x):
        self.rot, self.x = rot, x


def _case(piece_rot: int, piece_x: int, target_rot: int, target_x: int) -> str:
    game = Game.__new__(Game)
    game.piece = _Fake(piece_rot, piece_x)
    return place_state(game, Target(target_rot, target_x, 0, 0.0))


SELF_TEST_CASES: list[tuple[str, str]] = [
    (_case(0, 3, 0, 3), "drop"),
    (_case(2, 5, 2, 5), "drop"),
    (_case(0, 6, 0, 2), "left"),
    (_case(1, 8, 1, 4), "left"),
    (_case(0, 1, 0, 6), "right"),
    (_case(3, 0, 3, 5), "right"),
    (_case(0, 3, 2, 3), "rotate"),
    (_case(0, 6, 1, 2), "rotate"),
    (_case(1, 1, 3, 6), "rotate"),
]


def self_test(service) -> bool:
    print("Self-test: checking the question against synthetic states")
    service.warm(CONTRACT)
    passed = 0
    for state, expected in SELF_TEST_CASES:
        answer = service.system_one(state=state, questions={"place": CONTRACT["place"]}).answers["place"]
        hit = answer["choice"] == expected
        passed += hit
        print(
            f"  {'ok  ' if hit else 'FAIL'} expected={expected:<7} got={answer['choice']:<7} "
            f"conf={answer['confidence']:.2f}  | {state[:56]}"
        )
    print(f"  {passed}/{len(SELF_TEST_CASES)}")
    return passed == len(SELF_TEST_CASES)


# -------------------------------------------------------------------------- sound
AUDIO_RATE = 44100
AUDIO_FRAME = AUDIO_RATE // FRAME_HZ  # 735 samples per game frame

_NOTE = {"E5": 659.26, "B4": 493.88, "C5": 523.25, "D5": 587.33, "A4": 440.0,
         "F5": 698.46, "A5": 880.0, "G5": 783.99}
_BASS = {"E": 164.81, "A": 110.0, "D": 146.83, "C": 130.81}

# Korobeiniki (the "Type A" theme, a Russian folk song): (note, beats) per bar.
_MELODY = [
    [("E5", 1), ("B4", .5), ("C5", .5), ("D5", 1), ("C5", .5), ("B4", .5)],
    [("A4", 1), ("A4", .5), ("C5", .5), ("E5", 1), ("D5", .5), ("C5", .5)],
    [("B4", 1.5), ("C5", .5), ("D5", 1), ("E5", 1)],
    [("C5", 1), ("A4", 1), ("A4", 2)],
    [("D5", 1.5), ("F5", .5), ("A5", 1), ("G5", .5), ("F5", .5)],
    [("E5", 1.5), ("C5", .5), ("E5", 1), ("D5", .5), ("C5", .5)],
    [("B4", 1), ("B4", .5), ("C5", .5), ("D5", 1), ("E5", 1)],
    [("C5", 1), ("A4", 1), ("A4", 2)],
]
_BASS_ROOTS = ["E", "A", "E", "A", "D", "C", "E", "A"]
BEAT = 0.42  # seconds


def _tone(freq: float, dur: float, *, duty: float = 0.5, decay: float = 0.0, gate: float = 1.0):
    import numpy as np

    n = int(AUDIO_RATE * dur)
    t = np.arange(n) / AUDIO_RATE
    wave = np.where((t * freq) % 1.0 < duty, 1.0, -1.0)
    env = np.minimum(1.0, t / 0.004) * np.minimum(1.0, np.maximum(0.0, dur * gate - t) / 0.02 + 0.0)
    if decay:
        env = env * np.exp(-t * decay)
    return wave * env


def _sequence(notes, *, duty=0.5, decay=0.0):
    import numpy as np

    return np.concatenate([_tone(f, d, duty=duty, decay=decay, gate=0.9) for f, d in notes])


def build_music():
    """One loop of the theme: melody (25% pulse) over a plucked bass."""
    import numpy as np

    melody = _sequence(
        [(_NOTE[n], b * BEAT) for bar in _MELODY for n, b in bar], duty=0.25
    )
    bass = np.concatenate([
        np.concatenate([
            _tone(_BASS[root] * (2 if i % 2 else 1), BEAT / 2, decay=6.0, gate=1.0)
            for i in range(8)
        ])
        for root in _BASS_ROOTS
    ])
    n = min(len(melody), len(bass))
    return melody[:n] * 1800 + bass[:n] * 1400


def build_effects() -> dict:
    import numpy as np

    def arp(freqs, each, vol):
        return _sequence([(f, each) for f in freqs], duty=0.5, decay=4.0) * vol

    rng = np.random.default_rng(1)
    thud = (_tone(110, 0.09, decay=30) * 5000 + rng.uniform(-1, 1, int(AUDIO_RATE * 0.09)) *
            np.exp(-np.arange(int(AUDIO_RATE * 0.09)) / AUDIO_RATE * 60) * 2500)
    return {
        "move": _tone(880, 0.025, duty=0.25, decay=40) * 2600,
        "rotate": _sequence([(660, 0.02), (990, 0.03)], duty=0.25, decay=30) * 3000,
        "drop": thud,
        "lock": _tone(196, 0.06, duty=0.5, decay=25) * 3200,
        "clear": arp([523, 659, 784, 1047], 0.07, 4500),
        "tetris": arp([523, 659, 784, 1047, 1319, 1568, 2093], 0.07, 5000),
        "win": arp([523, 523, 523, 659, 784, 659, 784, 1047, 1047], 0.13, 5000),
        "gameover": arp([392, 349, 330, 262, 196, 131], 0.22, 5000),
    }


class Synth:
    """Mixes the looping music and triggered effects, one game frame at a time."""

    def __init__(self) -> None:
        import numpy as np

        self.np = np
        self.music = build_music()
        self.effects = build_effects()
        self.pos = 0
        self.active: list[list] = []  # [samples, position]
        self.music_on = True

    def trigger(self, name: str) -> None:
        if name in self.effects:
            self.active.append([self.effects[name], 0])
        if name in ("win", "gameover"):
            self.music_on = False

    def frame(self):
        """735 stereo int16 samples for the next 1/60s."""
        np = self.np
        out = np.zeros(AUDIO_FRAME)
        if self.music_on:
            idx = (self.pos + np.arange(AUDIO_FRAME)) % len(self.music)
            out += self.music[idx]
            self.pos = (self.pos + AUDIO_FRAME) % len(self.music)
        for item in self.active:
            samples, at = item
            chunk = samples[at:at + AUDIO_FRAME]
            out[:len(chunk)] += chunk
            item[1] += AUDIO_FRAME
        self.active = [item for item in self.active if item[1] < len(item[0])]
        mono = np.clip(out, -10000, 10000).astype(np.int16)
        return np.stack([mono, mono], axis=1)


# ---------------------------------------------------------------------------- view
@dataclass
class Snap:
    """Everything the window needs to draw one frame."""

    board: list
    piece: list
    piece_kind: str
    ghost: list
    flash: list
    next_kind: str
    score: int
    lines: int
    goal: int


def snapshot(game: Game, target: Target | None, goal: int) -> Snap:
    piece = game.piece
    ghost = []
    if piece is not None and target is not None:
        ghost = [(r, c) for r, c in piece.cells(rot=target.rot, x=target.x, y=target.y) if r >= 0]
    flash = game.full_rows if game.clearing and (game.clearing // 3) % 2 == 0 else []
    return Snap(
        [row[:] for row in game.board],
        [(r, c) for r, c in piece.cells() if r >= 0] if piece else [],
        piece.kind if piece else "",
        ghost,
        list(flash),
        game.next_kind,
        game.score,
        game.lines,
        goal,
    )


class Window:
    """Optional pygame window: the board, the target outline and the decision."""

    CELL = 32

    def __init__(self) -> None:
        import pygame

        self.pg = pygame
        pygame.init()
        self.board_w, self.board_h = WIDTH * self.CELL, HEIGHT * self.CELL
        self.size = (self.board_w + 220, self.board_h + 40)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Gemma 4 plays Tetris")
        self.font = pygame.font.SysFont("menlo", 16)
        self.big = pygame.font.SysFont("menlo", 26, bold=True)

    def _cell(self, r, c, color, ox=0, oy=0, size=None):
        size = size or self.CELL
        rect = self.pg.Rect(ox + c * size, oy + r * size, size, size)
        self.screen.fill(color, rect.inflate(-2, -2))
        self.screen.fill(tuple(min(255, v + 60) for v in color), (rect.x + 2, rect.y + 2, size - 6, 3))

    def draw(self, snap: Snap, phase: str, action: str, confidence: float, ms: float) -> None:
        pg = self.pg
        pg.event.pump()
        self.screen.fill((16, 16, 26))
        pg.draw.rect(self.screen, (60, 60, 90), (0, 0, self.board_w, self.board_h), 2)
        for i in range(1, WIDTH):
            pg.draw.line(self.screen, (28, 28, 44), (i * self.CELL, 0), (i * self.CELL, self.board_h))
        for j in range(1, HEIGHT):
            pg.draw.line(self.screen, (28, 28, 44), (0, j * self.CELL), (self.board_w, j * self.CELL))
        for r, row in enumerate(snap.board):
            for c, kind in enumerate(row):
                if kind:
                    self._cell(r, c, (235, 235, 245) if r in snap.flash else COLORS[kind])
        color = COLORS.get(snap.piece_kind, (255, 255, 255))
        for r, c in snap.ghost:
            pg.draw.rect(self.screen, color, (c * self.CELL + 1, r * self.CELL + 1, self.CELL - 2, self.CELL - 2), 2)
        for r, c in snap.piece:
            self._cell(r, c, color)

        x0 = self.board_w + 20
        self.screen.blit(self.big.render("TETRIS", True, (240, 240, 250)), (x0, 16))
        for i, (label, value) in enumerate((("score", snap.score), ("lines", f"{snap.lines}/{snap.goal}"))):
            self.screen.blit(self.font.render(f"{label:<6}{value}", True, (200, 200, 220)), (x0, 70 + i * 24))
        self.screen.blit(self.font.render("next", True, (200, 200, 220)), (x0, 150))
        if snap.next_kind:
            for r, c in ROTATIONS[snap.next_kind][0]:
                self._cell(r, c, COLORS[snap.next_kind], ox=x0, oy=176, size=24)
        bar = pg.Rect(x0, 300, 180, 10)
        pg.draw.rect(self.screen, (50, 50, 70), bar)
        pg.draw.rect(self.screen, (90, 200, 120), (bar.x, bar.y, int(bar.w * max(0.0, min(1.0, confidence))), bar.h))
        self.screen.blit(self.font.render(f"conf {confidence:.2f}", True, (200, 200, 220)), (x0, 316))
        self.screen.fill((20, 20, 30), (0, self.board_h, self.size[0], 40))
        label = f"{phase:<6} -> {action:<7} conf {confidence:.2f}   {ms:.0f}ms"
        self.screen.blit(self.font.render(label, True, (240, 240, 240)), (10, self.board_h + 10))
        pg.display.flip()

    def close(self) -> None:
        self.pg.quit()


# --------------------------------------------------------------------------- loop
def run_episode(service, *, lines: int, max_decisions: int, window: Window | None,
                quiet: bool, sound: bool = False, seed: int | None = None) -> dict:
    """Play one game of Tetris until ``lines`` are cleared or the stack tops out.

    Blocks of frames are emulated at once and shown at 60fps while the next decision
    runs on a worker thread, exactly as in the Mario demo. While rows flash there is
    no piece to steer, so those blocks are played without a decision.
    """
    from concurrent.futures import ThreadPoolExecutor

    game = Game(seed)
    synth = Synth() if sound else None
    speaker = Speaker(AUDIO_RATE) if sound else None
    pacer = Pacer() if (sound or window is not None) else None
    latency = LatencyLog("place")
    action_mix: dict[str, int] = {}
    decision_wall = 0.0
    step = 0
    frames_played = 0

    def run_frames(action: str | None, target: Target | None, n: int) -> tuple[list, list]:
        nonlocal frames_played
        frames, audio = [], []
        for i in range(n):
            game.frame(action if i == 0 else None)
            frames_played += 1
            events, game.events = game.events, []
            if synth is not None:
                for name in events:
                    synth.trigger(name)
                audio.append(synth.frame())
            if pacer is not None:
                frames.append(snapshot(game, target if game.piece else None, lines) if window else None)
            if game.over or game.lines >= lines:
                break
        return frames, audio

    def decide(state_text: str, target: Target):
        started = time.perf_counter()
        if service is None:
            action, confidence = baseline_action(game, target), 1.0
        else:
            answer = service.system_one(state=state_text, questions={"place": CONTRACT["place"]}).answers["place"]
            action, confidence = answer["choice"], answer["confidence"]
        return action, confidence, started, time.perf_counter() - started

    shown: Block | None = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        while step < max_decisions and not game.over and game.lines < lines:
            if game.piece is None:  # rows are flashing: nothing to steer
                if shown is not None:
                    present(shown, window, speaker, pacer)
                frames, audio = run_frames(None, None, FRAMES_PER_DECISION)
                shown = Block(frames, audio, "clear", "-", 1.0, 0.0)
                continue

            step += 1
            target = plan(game)
            state_text = place_state(game, target)
            future = pool.submit(decide, state_text, target)
            if shown is not None:
                present(shown, window, speaker, pacer)
                shown = None
            action, confidence, started, elapsed = future.result()
            decision_wall += elapsed
            ms = latency.record(started)
            action_mix[action] = action_mix.get(action, 0) + 1

            if not quiet:
                print(
                    f"{step:4d}  {action:<7} conf={confidence:.2f} {ms:6.1f}ms  "
                    f"piece={game.piece.kind} rot={game.piece.rot} x={game.piece.x}  "
                    f"lines={game.lines:2d}  | {state_text[:52]}"
                )
            frames, audio = run_frames(action, target, FRAMES_PER_DECISION)
            shown = Block(frames, audio, "place", action, confidence, ms)

        if shown is not None:
            present(shown, window, speaker, pacer)

        won = game.lines >= lines and not game.over
        if synth is not None or window is not None:
            # Let the closing jingle play over the final board.
            if synth is not None:
                synth.trigger("win" if won else "gameover")
            frames, audio = [], []
            for _ in range(FRAME_HZ * 2):
                if synth is not None:
                    audio.append(synth.frame())
                frames.append(snapshot(game, None, lines) if window else None)
            for i in range(0, len(frames), FRAMES_PER_DECISION):
                present(Block(frames[i:i + FRAMES_PER_DECISION], audio[i:i + FRAMES_PER_DECISION],
                              "end", "-", 1.0, 0.0), window, speaker, pacer)

    underruns = 0
    if speaker is not None:
        underruns = speaker.underruns
        speaker.close()
    return {
        "result": "CLEARED" if game.lines >= lines and not game.over else "topped out" if game.over else "out of decisions",
        "lines": game.lines,
        "score": game.score,
        "pieces": game.pieces,
        "underruns": underruns,
        "decisions": step,
        "game_seconds": frames_played / FRAME_HZ,
        "decision_wall": decision_wall,
        "action_mix": action_mix,
        "latency": latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 plays Tetris.")
    parser.add_argument("--lines", type=int, default=10, help="lines to clear (default: 10)")
    parser.add_argument("--max-decisions", type=int, default=800)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="random seed for the piece order")
    parser.add_argument("--watch", action="store_true", help="show the game in a pygame window")
    parser.add_argument("--sound", action="store_true", help="play the music and effects")
    parser.add_argument("--baseline", action="store_true", help="play with the criteria rule")
    parser.add_argument("--quiet", action="store_true", help="summary only")
    parser.add_argument("--self-test", action="store_true", help="check the question and exit")
    args = parser.parse_args()

    service = None
    if args.self_test or not args.baseline:
        from truetype.service import TypeSafeReplica

        print("Loading model (one-time cost)...")
        service = TypeSafeReplica()
        service.engine.load()
        print(f"Model loaded in {service.engine.load_seconds:.1f}s")
        if args.self_test:
            print_header("Question self-test")
            sys.exit(0 if self_test(service) else 1)
        print(f"Question prefixes warmed in {service.warm(CONTRACT):.0f}ms")

    who = "criteria rule" if args.baseline else "Gemma 4"
    print_header(f"Tetris - {who}, {args.lines} lines, {args.episodes} episode(s)")
    print(f"one decision every {FRAMES_PER_DECISION} frames = {FRAMES_PER_DECISION / FRAME_HZ * 1000:.0f}ms of game time\n")

    window = Window() if args.watch else None
    results = []
    try:
        for episode in range(1, args.episodes + 1):
            print(f"--- episode {episode} ---")
            r = run_episode(
                None if args.baseline else service,
                lines=args.lines,
                max_decisions=args.max_decisions,
                window=window,
                quiet=args.quiet,
                sound=args.sound,
                seed=None if args.seed is None else args.seed + episode,
            )
            results.append(r)
            print(
                f"    {r['result']}: lines={r['lines']} score={r['score']} pieces={r['pieces']} "
                f"decisions={r['decisions']} game_time={r['game_seconds']:.1f}s mix={r['action_mix']}"
                + (f" audio_underruns={r['underruns']}" if args.sound else "") + "\n"
            )
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if window is not None:
            window.close()

    if results:
        cleared = sum(r["result"] == "CLEARED" for r in results)
        print(f"Goal reached in {cleared}/{len(results)} episode(s)")
        merged = LatencyLog("place", [s for r in results for s in r["latency"].samples_ms])
        if merged.count:
            print(merged.summary())
        sys.exit(0 if cleared else 1)


if __name__ == "__main__":
    main()
