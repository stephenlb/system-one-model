"""Blocks System One plays the first level of Super Mario Bros. (NES).

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
    python mario/mario_demo.py --sound          # ...with the game's audio
    python mario/mario_demo.py --baseline       # reference player, no model
    python mario/mario_demo.py --self-test      # check the questions, no game
    python mario/mario_demo.py --episodes 3 --max-decisions 900
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from common import LatencyLog, print_header

# nes-py's raw joypad bitmask.
RIGHT, LEFT, DOWN, UP, START, SELECT, B, A = 128, 64, 32, 16, 8, 4, 2, 1

FRAMES_PER_DECISION = 8  # 60 fps NES: one decision is 133ms of game time, longer than a decision takes
LOOKAHEAD = 112  # pixels of level scanned for the next hazard
TILE = 16

# Powerups that walk toward Mario are not hazards.
FRIENDLY_ENEMY_TYPES = {0x2E, 0x2F, 0x30, 0x31}

# Takeoff windows: a hazard this close (pixels between Mario's leading edge and
# the hazard) is inside the window where a jump clears it. Measured with
# ``--baseline`` on World 1-1.
PIT_WINDOW = 20
ENEMY_WINDOW = 50
WALL_WINDOW_BASE = 30
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


def tile_solid(ram, world_x: int, screen_y: int) -> bool:
    row = (screen_y - 32) // TILE
    if not 0 <= row < 13:
        return False
    page = (world_x // 256) % 2
    return ram[0x500 + page * 208 + row * 16 + (world_x % 256) // TILE] != 0


def read_view(ram) -> View:
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
class Emulator:
    """One NES frame at a time, with RAM, the screen and (optionally) audio.

    The default backend is ``nes-py``: fast and headless, but it does not emulate
    the sound chip. ``sound=True`` uses ``stable-retro`` (FCEUmm core) instead,
    which does. Both run the same ROM and expose the same RAM map.
    """

    # nes-py bit -> stable-retro button index (B, -, SELECT, START, UP, DOWN, LEFT, RIGHT, A)
    _RETRO_BUTTONS = {B: 0, SELECT: 2, START: 3, UP: 4, DOWN: 5, LEFT: 6, RIGHT: 7, A: 8}

    def __init__(self, sound: bool = False) -> None:
        self.sound = sound
        if sound:
            import numpy as np
            import stable_retro as retro

            self._np = np
            install_retro_rom()
            self._env = retro.make("SuperMarioBros-Nes-v0", render_mode=None)
            self._env.reset()
            self.audio_rate = int(self._env.em.get_audio_rate())
        else:
            import gym_super_mario_bros as smb

            self._env = smb.make("SuperMarioBros-1-1-v0")
            self._env.reset()
            self.audio_rate = 0

    @property
    def ram(self):
        return self._env.get_ram() if self.sound else self._env.unwrapped.ram

    @property
    def screen(self):
        return self._env.em.get_screen() if self.sound else self._env.unwrapped.screen

    def frame(self, mask: int):
        """Advance one frame with a nes-py style button mask; return its audio."""
        if not self.sound:
            self._env.unwrapped._frame_advance(mask)
            return None
        pad = self._np.zeros(9, dtype=self._np.int8)
        for bit, index in self._RETRO_BUTTONS.items():
            pad[index] = 1 if mask & bit else 0
        self._env.step(pad)
        return self._env.em.get_audio()

    def close(self) -> None:
        self._env.close()


def install_retro_rom() -> None:
    """Give stable-retro the ROM that ``gym-super-mario-bros`` already ships.

    The bundled dump has a different checksum from the one stable-retro's importer
    accepts, so ``retro.import`` skips it; loading it directly works.
    """
    import shutil

    import gym_super_mario_bros
    import stable_retro

    target = Path(stable_retro.data.path()) / "stable" / "SuperMarioBros-Nes-v0" / "rom.nes"
    if not target.exists():
        source = Path(gym_super_mario_bros.__file__).parent / "_roms" / "super-mario-bros.nes"
        shutil.copy(source, target)


def make_env(sound: bool = False) -> Emulator:
    return Emulator(sound)


def mario_status(ram) -> dict:
    """What the environment wrapper used to report, read straight from RAM."""
    state = int(ram[0x0E])
    return {
        "x": int(ram[0x6D]) * 256 + int(ram[0x86]),
        "flag": state == 4,  # sliding down the flagpole
        "dead": state in (6, 0x0B) or int(ram[0xB5]) > 1,  # dying, or below the screen
        "time": int(ram[0x7F8]) * 100 + int(ram[0x7F9]) * 10 + int(ram[0x7FA]),
    }


GAIN = 3


class Speaker:
    """Streams the emulator's audio through pygame.mixer at its native speed.

    Sound is smooth only if the emulator stays ahead of the speaker. A decision
    costs ~110ms of wall time, so a block must hold at least that much game time:
    ``FRAMES_PER_DECISION`` is 8 (133ms). The mixer has one playing slot and one
    queued slot; ``flush`` waits for the queued slot to free up, which paces the
    game to real time when it would otherwise run ahead (``--baseline``, the flag
    walk) and does nothing when the model is the bottleneck.
    """

    def __init__(self, rate: int) -> None:
        import numpy as np
        import pygame

        self.np = np
        # pygame.init() (the --watch window) opens the mixer at 44100 Hz, and a
        # second init is then a no-op: the samples would play 37% too fast.
        pygame.mixer.quit()
        pygame.mixer.init(frequency=rate, size=-16, channels=2, buffer=512)
        got = pygame.mixer.get_init()
        if got is None or got[0] != rate:
            raise RuntimeError(f"audio mixer opened at {got}, needed {rate} Hz stereo")
        self.mixer = pygame.mixer
        self.channel = pygame.mixer.Channel(0)
        self.rate = rate
        self.pending: list = []
        self.underruns = 0
        self.blocks: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._feed, daemon=True)
        self.thread.start()

    def add(self, samples) -> None:
        self.pending.append(samples)

    def flush(self) -> None:
        """Hand the frames gathered since the last flush to the audio thread."""
        if not self.pending:
            return
        np = self.np
        block = np.concatenate(self.pending)
        self.pending = []
        # FCEUmm's output peaks around a fifth of full scale; bring it up.
        block = np.clip(block.astype(np.int32) * GAIN, -32768, 32767).astype(np.int16)
        self.blocks.put(block.tobytes())

    def _feed(self) -> None:
        """Keep the mixer's playing and queued slots full, off the video's clock."""
        while True:
            data = self.blocks.get()
            if data is None:
                return
            while self.channel.get_queue() is not None:
                time.sleep(0.002)
            sound = self.mixer.Sound(buffer=data)
            if self.channel.get_busy():
                self.channel.queue(sound)
            else:
                if self.started:
                    self.underruns += 1  # the speaker ran dry before this block arrived
                self.channel.play(sound)
                self.started = True

    started = False

    def close(self) -> None:
        self.blocks.put(None)
        self.thread.join()
        time.sleep(0.4)  # let the last block finish playing
        self.mixer.quit()


class Window:
    """Optional pygame window showing the emulator screen and the decision."""

    def __init__(self) -> None:
        import pygame

        self.pg = pygame
        pygame.init()
        self.size = (256 * 3, 240 * 3 + 40)
        self.view = (256 * 3, 240 * 3)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Blocks System One plays Super Mario Bros. 1-1")
        self.font = pygame.font.SysFont("menlo", 16)

    def draw(self, frame, phase: str, action: str, confidence: float, ms: float) -> None:
        pg = self.pg
        pg.event.pump()
        surface = pg.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(pg.transform.scale(surface, self.view), (0, 0))
        self.screen.fill((20, 20, 30), (0, 240 * 3, self.size[0], 40))
        label = f"{phase:<6} -> {action:<8} conf {confidence:.2f}   {ms:.0f}ms"
        self.screen.blit(self.font.render(label, True, (240, 240, 240)), (10, 240 * 3 + 10))
        pg.display.flip()

    def close(self) -> None:
        self.pg.quit()


CASTLE_X = 3260  # World 1-1: Mario stops at the castle door at x=3266


class Pacer:
    """Holds the game to 60 frames per second of wall time."""

    def __init__(self) -> None:
        self.next = time.perf_counter()

    def tick(self) -> None:
        self.next += 1 / 60
        delay = self.next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.1:  # stalled (a slow decision): resume from now, do not sprint
            self.next = time.perf_counter()


@dataclass
class Block:
    """FRAMES_PER_DECISION emulated frames waiting to be shown and heard."""

    frames: list
    audio: list
    phase: str
    action: str
    confidence: float
    ms: float


def present(block: Block, window: Window | None, speaker: Speaker | None, pacer: Pacer | None) -> None:
    """Show a block at 60fps and queue its sound. Runs while the next decision is made."""
    if speaker is not None:
        for samples in block.audio:
            speaker.add(samples)
        speaker.flush()
    for frame in block.frames:
        if window is not None:
            window.draw(frame, block.phase, block.action, block.confidence, block.ms)
        if pacer is not None:
            pacer.tick()


def walk_to_castle(env: Emulator, window: Window | None, speaker: Speaker | None = None,
                   pacer: Pacer | None = None, *, max_frames: int = 600) -> bool:
    """Play out the flagpole sequence until Mario is at the castle door.

    The episode is won when the flag is grabbed, but the game is not over: Mario
    slides down the pole and walks toward the castle if RIGHT is held. No
    decisions are made here, so nothing is timed.
    """
    still = 0
    last_x = -1
    for i in range(max_frames):
        audio = env.frame(RIGHT)
        if speaker is not None and audio is not None:
            speaker.add(audio)
            if i % 4 == 3:
                speaker.flush()
        if window is not None:
            window.draw(env.screen, "flag", "walk", 1.0, 0.0)
        if pacer is not None:
            pacer.tick()
        x = mario_status(env.ram)["x"]
        still = still + 1 if x == last_x else 0
        last_x = x
        if x >= CASTLE_X and still >= 20:
            if speaker is not None:
                speaker.flush()
            return True
    return False


def run_episode(service, *, max_decisions: int, window: Window | None, quiet: bool,
                sound: bool = False) -> dict:
    """Play one episode.

    Each block of frames is emulated at once (the emulator is far faster than the
    game clock), then shown at 60fps while the next decision is computed on a
    worker thread. The decision needs only the state at the end of the block, which
    already exists, so overlapping the two adds no delay to the game and hides the
    ~110ms decision inside the 133ms of playback. Without a window or sound the
    game runs flat out.
    """
    from concurrent.futures import ThreadPoolExecutor

    env = make_env(sound)
    speaker = Speaker(env.audio_rate) if sound else None
    pacer = Pacer() if (sound or window is not None) else None
    latency = {phase: LatencyLog(phase) for phase in CONTRACT}
    action_mix: dict[str, int] = {}
    decision_wall = 0.0
    a_held = False
    status = mario_status(env.ram)
    step = 0
    result = "out of decisions"
    castle = False

    def decide(view: View, phase: str, state_text: str):
        started = time.perf_counter()
        if service is None:
            action, confidence = baseline_action(view), 1.0
        else:
            answer = service.system_one(
                state=state_text, questions={phase: CONTRACT[phase]}
            ).answers[phase]
            action, confidence = answer["choice"], answer["confidence"]
        return action, confidence, started, time.perf_counter() - started

    if sound:
        # stable-retro starts on the title screen; press START to begin World 1-1.
        for i in range(240):
            audio = env.frame(START if 30 <= i < 40 else 0)
            speaker.add(audio)
            if i % 4 == 3:
                speaker.flush()
            if window is not None:
                window.draw(env.screen, "title", "start", 1.0, 0.0)
            pacer.tick()
            if int(env.ram[0x0770]) == 1 and int(env.ram[0x0E]) == 8 and i > 100:
                break

    shown: Block | None = None  # the block on screen while the next decision runs
    with ThreadPoolExecutor(max_workers=1) as pool:
        while step < max_decisions:
            step += 1
            view = read_view(env.ram)
            phase = phase_of(view)
            state_text = ground_state(view.hazard) if phase == "ground" else air_state(view)
            future = pool.submit(decide, view, phase, state_text)

            if shown is not None:
                present(shown, window, speaker, pacer)
                shown = None
            action, confidence, started, elapsed = future.result()
            decision_wall += elapsed
            ms = latency[phase].record(started)
            action_mix[action] = action_mix.get(action, 0) + 1

            press_a = action in ("jump", "hold")
            frames: list = []
            audio_blocks: list = []
            for frame in range(FRAMES_PER_DECISION):
                # A jump needs a fresh press: let go for one frame after a landing.
                a_now = press_a and not (frame == 0 and a_held and not view.airborne)
                audio = env.frame(RIGHT | B | (A if a_now else 0))
                a_held = a_now
                if audio is not None:
                    audio_blocks.append(audio)
                if window is not None:
                    frames.append(env.screen.copy())
                status = mario_status(env.ram)
                if status["flag"] or status["dead"]:
                    break
            shown = Block(frames, audio_blocks, phase, action, confidence, ms)

            if not quiet:
                h = view.hazard
                note = f"{h.kind} {h.distance}px" if h else "clear"
                print(
                    f"{step:4d}  {phase:<6} {action:<8} conf={confidence:.2f} {ms:6.1f}ms  "
                    f"x={status['x']:4d}  | {note}"
                )

            if status["flag"] or status["dead"] or status["time"] == 0:
                break

        if shown is not None:
            present(shown, window, speaker, pacer)
        if status["flag"]:
            result = "FLAG"
            castle = walk_to_castle(env, window, speaker, pacer)
        elif status["dead"] or status["time"] == 0:
            result = "died"

    underruns = 0
    if speaker is not None:
        underruns = speaker.underruns
        speaker.close()
    env.close()
    return {
        "result": result,
        "castle": castle,
        "underruns": underruns,
        "decisions": step,
        "x": status["x"],
        "game_seconds": step * FRAMES_PER_DECISION / 60,
        "decision_wall": decision_wall,
        "action_mix": action_mix,
        "latency": latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Blocks System One plays Super Mario Bros. 1-1.")
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--watch", action="store_true", help="show the game in a pygame window")
    parser.add_argument("--sound", action="store_true",
                        help="play the game audio (uses stable-retro instead of nes-py)")
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
                sound=args.sound,
            )
            results.append(r)
            print(f"    {r['result']}{' + castle' if r['castle'] else ''}: x={r['x']} decisions={r['decisions']} "
                  f"game_time={r['game_seconds']:.1f}s mix={r['action_mix']}"
                  + (f" audio_underruns={r['underruns']}" if args.sound else "") + "\n")
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
