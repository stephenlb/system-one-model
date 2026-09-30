"""Gemma 4 plays the first level of Super Mario Bros. (NES).

The engine is a real NES emulator (``nes-py``, with the ROM bundled by
``gym-super-mario-bros``). Like ViZDoom it advances only when it is stepped, so a
slow decision costs wall-clock time but never game time. The goal is the one the
game sets: reach the flagpole of World 1-1.

The decision
------------
The demo reads the engine's RAM (Mario's position, the tile map, the enemy slots)
and turns it into a short report. One question is asked per decision, in one of
two phases chosen from the state:

  ==========  ================================================================
  ``ground``  Mario is on the floor: run / jump, judged against the next hazard
  ``air``     Mario is jumping: keep holding jump / let go
  ==========  ================================================================

Mario always holds RIGHT and B (run). The model only decides whether A (jump) is
pressed for the next block of ``FRAMES_PER_DECISION`` frames. The demo does the
geometry - which hazard is next (pit, wall or enemy) and whether it is inside the
takeoff window - and the model reads the verdict. That is the same split as the
Flappy Bird demo: the report says what is wrong with the position, in words, and
does not hand the model raw pixel distances to interpret.

Run (from the repo root):
    python mario/mario_demo.py                  # Gemma plays 1-1
    python mario/mario_demo.py --watch          # ...in a pygame window
    python mario/mario_demo.py --baseline       # reference player, no model
    python mario/mario_demo.py --self-test      # check the questions, no game
    python mario/mario_demo.py --episodes 3 --max-decisions 900
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from common import LatencyLog, print_header

# nes-py's raw joypad bitmask.
RIGHT, LEFT, DOWN, UP, START, SELECT, B, A = 128, 64, 32, 16, 8, 4, 2, 1

FRAMES_PER_DECISION = 4  # 60 fps NES: one decision is 67ms of game time
LOOKAHEAD = 112  # pixels of level scanned for the next hazard
TILE = 16

# Powerups that walk toward Mario are not hazards.
FRIENDLY_ENEMY_TYPES = {0x2E, 0x2F, 0x30, 0x31}

# Takeoff windows: a hazard this close (pixels between Mario's leading edge and
# the hazard) is inside the window where a jump clears it. Measured with
# ``--baseline`` on World 1-1.
PIT_WINDOW = 14
ENEMY_WINDOW = 60
WALL_WINDOW_BASE = 20
WALL_WINDOW_PER_TILE = 12


# -------------------------------------------------------------------------- state
@dataclass
class Hazard:
    kind: str  # "pit" | "wall" | "enemy"
    distance: int  # pixels from Mario's leading edge
    top: int = 0  # screen y of the top of a wall (feet must end above it)
    tiles: int = 0  # wall height in tiles

    @property
    def window(self) -> int:
        if self.kind == "pit":
            return PIT_WINDOW
        if self.kind == "enemy":
            return ENEMY_WINDOW
        return WALL_WINDOW_BASE + WALL_WINDOW_PER_TILE * self.tiles

    @property
    def in_window(self) -> bool:
        return self.distance <= self.window


@dataclass
class View:
    world_x: int
    feet_y: int
    airborne: bool
    rising: bool
    hazard: Hazard | None


def _ram(env):
    return env.unwrapped.ram


def tile_solid(ram, world_x: int, screen_y: int) -> bool:
    row = (screen_y - 32) // TILE
    if not 0 <= row < 13:
        return False
    page = (world_x // 256) % 2
    return ram[0x500 + page * 208 + row * 16 + (world_x % 256) // TILE] != 0


def read_view(env) -> View:
    ram = _ram(env)
    world_x = int(ram[0x6D]) * 256 + int(ram[0x86])
    height = 16 if ram[0x754] == 1 else 32
    feet_y = int(ram[0xCE]) + 16 + height
    airborne = ram[0x1D] != 0
    vy = int(ram[0x9F])
    rising = airborne and (vy > 127 and vy != 0)  # signed byte: negative is up
    return View(world_x, feet_y, airborne, rising, next_hazard(ram, world_x, feet_y))


def next_hazard(ram, world_x: int, feet_y: int) -> Hazard | None:
    lead = world_x + 12  # Mario's leading edge
    body_y = feet_y - 1
    found: list[Hazard] = []

    # Terrain, one tile column at a time.
    for col_x in range((lead // TILE + 1) * TILE, lead + LOOKAHEAD, TILE):
        dist = col_x - lead
        if tile_solid(ram, col_x, body_y):
            top_y = body_y
            while tile_solid(ram, col_x, top_y - TILE):
                top_y -= TILE
            top = 32 + ((top_y - 32) // TILE) * TILE
            found.append(Hazard("wall", dist, top=top, tiles=max(1, (feet_y - top + 8) // TILE)))
            break
        below = [tile_solid(ram, col_x, y) for y in range(feet_y, 32 + 13 * TILE, TILE)]
        if not any(below):
            found.append(Hazard("pit", dist))
            break

    # Enemies in the slots, on screen and at roughly Mario's height.
    for i in range(5):
        if ram[0x0F + i] != 1 or int(ram[0x16 + i]) in FRIENDLY_ENEMY_TYPES:
            continue
        ex = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i])
        ey = int(ram[0xCF + i]) + 24
        dist = ex - lead
        if 0 < dist < LOOKAHEAD and abs(ey - feet_y) <= 32:
            found.append(Hazard("enemy", dist))

    return min(found, key=lambda h: h.distance, default=None)


# ------------------------------------------------------------------------ prompts
# ``run`` is listed first so it takes letter A (see the Flappy Bird demo for why
# the no-op option goes first).
GROUND_CRITERIA = {
    "run": "Keep running - the path is clear or the hazard is still too far away to jump",
    "jump": "Jump - a hazard is right ahead and jumping now clears it",
}

AIR_CRITERIA = {
    "release": "Let go of jump - Mario is coming down or is already high enough for what is ahead",
    "hold": "Keep holding jump - Mario is still rising and has not cleared what is ahead",
}

CONTRACT = {
    "ground": {
        "type": "choice",
        "instructions": (
            "Mario is running right on the ground in Super Mario Bros. and must "
            "reach the flag. Decide whether to jump now."
        ),
        "criteria": GROUND_CRITERIA,
    },
    "air": {
        "type": "choice",
        "instructions": (
            "Mario is in the air after jumping in Super Mario Bros. Decide whether "
            "to keep holding the jump button."
        ),
        "criteria": AIR_CRITERIA,
    },
}

HAZARD_WORDS = {"pit": "A pit", "wall": "A wall", "enemy": "An enemy"}


def ground_state(hazard: Hazard | None) -> str:
    if hazard is None:
        return "The path ahead is clear."
    if hazard.in_window:
        return f"{HAZARD_WORDS[hazard.kind]} is right ahead. Jump now to clear it."
    return f"{HAZARD_WORDS[hazard.kind]} is ahead but still too far away to jump."


FAR_PIT = 40  # a pit this far off is a later jump, not this one


def hazard_cleared(view: View) -> bool:
    """True when nothing ahead needs the jump to go any higher or further."""
    h = view.hazard
    if h is None:
        return False
    if h.kind == "wall":
        return view.feet_y <= h.top - 8
    return h.kind == "pit" and h.distance > FAR_PIT


def air_state(view: View) -> str:
    # A jump cut short lands on the enemy or in the pit it was meant to clear, so
    # it is held all the way up. Over a wall a full jump overshoots the step it
    # should land on, and over a distant pit it overshoots the ground it should
    # land on before jumping again.
    if not view.rising:
        return "Mario is coming down."
    if hazard_cleared(view):
        return "Mario is rising and is already high enough for what is ahead."
    return "Mario is still rising and has not cleared what is ahead."


def phase_of(view: View) -> str:
    return "air" if view.airborne else "ground"


def baseline_action(view: View) -> str:
    """The rule the criteria describe, used with ``--baseline`` and the tests."""
    if view.airborne:
        return "hold" if view.rising and not hazard_cleared(view) else "release"
    return "jump" if view.hazard is not None and view.hazard.in_window else "run"


# --------------------------------------------------------------------- self test
SELF_TEST_CASES: list[tuple[str, str, str]] = [
    ("ground", ground_state(None), "run"),
    ("ground", ground_state(Hazard("pit", 90)), "run"),
    ("ground", ground_state(Hazard("enemy", 100)), "run"),
    ("ground", ground_state(Hazard("wall", 100, tiles=3)), "run"),
    ("ground", ground_state(Hazard("pit", 8)), "jump"),
    ("ground", ground_state(Hazard("enemy", 30)), "jump"),
    ("ground", ground_state(Hazard("wall", 30, tiles=2)), "jump"),
    ("air", air_state(View(0, 190, True, True, Hazard("wall", 10, top=150, tiles=2))), "hold"),
    ("air", air_state(View(0, 100, True, True, Hazard("pit", 20))), "hold"),
    ("air", air_state(View(0, 100, True, True, Hazard("pit", 70))), "release"),
    ("air", air_state(View(0, 100, True, True, None)), "hold"),
    ("air", air_state(View(0, 80, True, True, Hazard("wall", 10, top=100, tiles=2))), "release"),
    ("air", air_state(View(0, 100, True, False, Hazard("pit", 40))), "release"),
    ("air", air_state(View(0, 100, True, False, None)), "release"),
]


def self_test(service) -> bool:
    print("Self-test: checking each question against synthetic states")
    service.warm(CONTRACT)
    passed = 0
    for phase, state, expected in SELF_TEST_CASES:
        answer = service.system_one(state=state, questions={phase: CONTRACT[phase]}).answers[phase]
        hit = answer["choice"] == expected
        passed += hit
        print(
            f"  {'ok  ' if hit else 'FAIL'} {phase:<7} expected={expected:<8} "
            f"got={answer['choice']:<8} conf={answer['confidence']:.2f}  | {state[:50]}"
        )
    print(f"  {passed}/{len(SELF_TEST_CASES)}")
    return passed == len(SELF_TEST_CASES)


# --------------------------------------------------------------------------- loop
def make_env():
    import gym_super_mario_bros as smb

    env = smb.make("SuperMarioBros-1-1-v0")
    env.reset()
    return env


class Window:
    """Optional pygame window showing the emulator screen and the decision."""

    def __init__(self) -> None:
        import pygame

        self.pg = pygame
        pygame.init()
        self.size = (256 * 3, 240 * 3 + 40)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Gemma 4 plays Super Mario Bros. 1-1")
        self.font = pygame.font.SysFont("menlo", 16)

    def draw(self, frame, phase: str, action: str, confidence: float, ms: float) -> None:
        pg = self.pg
        pg.event.pump()
        surface = pg.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(pg.transform.scale(surface, (256 * 3, 240 * 3)), (0, 0))
        self.screen.fill((20, 20, 30), (0, 240 * 3, self.size[0], 40))
        label = f"{phase:<6} -> {action:<8} conf {confidence:.2f}   {ms:.0f}ms"
        self.screen.blit(self.font.render(label, True, (240, 240, 240)), (10, 240 * 3 + 10))
        pg.display.flip()

    def close(self) -> None:
        self.pg.quit()


def run_episode(service, *, max_decisions: int, window: Window | None, quiet: bool) -> dict:
    env = make_env()
    latency = {phase: LatencyLog(phase) for phase in CONTRACT}
    action_mix: dict[str, int] = {}
    decision_wall = 0.0
    a_held = False
    info: dict = {}
    step = 0
    result = "out of decisions"

    while step < max_decisions:
        step += 1
        view = read_view(env)
        phase = phase_of(view)
        state_text = ground_state(view.hazard) if phase == "ground" else air_state(view)

        started = time.perf_counter()
        if service is None:
            action, confidence = baseline_action(view), 1.0
        else:
            answer = service.system_one(
                state=state_text, questions={phase: CONTRACT[phase]}
            ).answers[phase]
            action, confidence = answer["choice"], answer["confidence"]
        decision_wall += time.perf_counter() - started
        ms = latency[phase].record(started)
        action_mix[action] = action_mix.get(action, 0) + 1

        press_a = action in ("jump", "hold")
        for frame in range(FRAMES_PER_DECISION):
            # A jump needs a fresh press: let go for one frame after a landing.
            a_now = press_a and not (frame == 0 and a_held and not view.airborne)
            _, _, terminated, truncated, info = env.step(RIGHT | B | (A if a_now else 0))
            a_held = a_now
            if window is not None:
                window.draw(env.unwrapped.screen, phase, action, confidence, ms)
            if terminated or truncated or info.get("flag_get"):
                break

        if not quiet:
            h = view.hazard
            note = f"{h.kind} {h.distance}px" if h else "clear"
            print(
                f"{step:4d}  {phase:<6} {action:<8} conf={confidence:.2f} {ms:6.1f}ms  "
                f"x={info['x_pos']:4d}  | {note}"
            )

        if info.get("flag_get"):
            result = "FLAG"
            break
        if info.get("is_dead") or info.get("is_dying") or info.get("death") or info.get("time", 1) == 0:
            result = "died"
            break

    env.close()
    return {
        "result": result,
        "decisions": step,
        "x": info.get("x_pos", 0),
        "game_seconds": step * FRAMES_PER_DECISION / 60,
        "decision_wall": decision_wall,
        "action_mix": action_mix,
        "latency": latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 plays Super Mario Bros. 1-1.")
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--watch", action="store_true", help="show the game in a pygame window")
    parser.add_argument("--baseline", action="store_true", help="play with the criteria rule")
    parser.add_argument("--quiet", action="store_true", help="summary only")
    parser.add_argument("--self-test", action="store_true", help="check the questions and exit")
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
    print_header(f"Super Mario Bros. 1-1 - {who}, {args.episodes} episode(s)")
    print(f"one decision every {FRAMES_PER_DECISION} frames = {FRAMES_PER_DECISION / 60 * 1000:.0f}ms of game time\n")

    window = Window() if args.watch else None
    results = []
    try:
        for episode in range(1, args.episodes + 1):
            print(f"--- episode {episode} ---")
            r = run_episode(
                None if args.baseline else service,
                max_decisions=args.max_decisions,
                window=window,
                quiet=args.quiet,
            )
            results.append(r)
            print(f"    {r['result']}: x={r['x']} decisions={r['decisions']} "
                  f"game_time={r['game_seconds']:.1f}s mix={r['action_mix']}\n")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if window is not None:
            window.close()

    if results:
        cleared = sum(r["result"] == "FLAG" for r in results)
        print(f"Level cleared in {cleared}/{len(results)} episode(s)")
        for phase in CONTRACT:
            merged = LatencyLog(phase, [s for r in results for s in r["latency"][phase].samples_ms])
            if merged.count:
                print(merged.summary())
        sys.exit(0 if cleared else 1)


if __name__ == "__main__":
    main()
