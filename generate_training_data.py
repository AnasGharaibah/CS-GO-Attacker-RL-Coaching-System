"""
CS:GO Attacker RL — Synthetic Training Data Generator
======================================================
Generates realistic episode JSON files matching the per-frame schema
expected by rl/train.py.

Usage
-----
  python generate_training_data.py                     # 200 episodes → data/episodes/
  python generate_training_data.py --episodes 500 --out my_data/
  python generate_training_data.py --episodes 50  --seed 7 --verbose

Output
------
  data/episodes/
  ├── episode_0000.json
  ├── episode_0001.json
  └── ...

Each JSON file is a list of per-frame state dicts that can be fed directly
into CSGOAttackerEnv.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants matching the README schema / action space
# ---------------------------------------------------------------------------

MAPS = ["de_dust2"]   # single map — attacker-only dataset
WEAPONS = ["rifle", "pistol", "sniper", "knife"]
UTILITY = ["smoke", "flash", "he", "molotov", "none"]
BOMB_SITES = ["A", "B", "unknown"]

ROLE = "attacker"   # fixed role — no defender actions generated

# Attacker-only scenarios (no defensive/rotate scenarios)
SCENARIOS = [
    "rush_a",
    "rush_b",
    "slow_take_a",
    "slow_take_b",
    "mid_control",
    "eco_rush",
    "split_attack",
]

FPS = 30                    # frames per second
MAX_ROUND_SECS = 60         # max round length in seconds
MAX_FRAMES = FPS * MAX_ROUND_SECS   # 1800

# Attacker-valid actions (indices 0-13; 14-15 are always masked)
ATTACKER_ACTIONS = list(range(14))

# Reward values (mirrors reward.py from README)
R_WIN           = +15.0
R_LOSS          = -10.0
R_KILL          =  +3.0
R_DEATH         =  -4.0
R_ASSIST        =  +0.5
R_PLANT         =  +5.0
R_CROSSHAIR_OK  =  +0.1
R_CROSSHAIR_BAD = -0.05
R_GOOD_POS      =  +0.02
R_TIME_PENALTY  = -0.01


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def rand_pos() -> tuple[float, float]:
    """Random normalised map position."""
    return round(random.uniform(0.0, 1.0), 4), round(random.uniform(0.0, 1.0), 4)


def forward_pos(current_x: float, current_y: float, scenario: str) -> tuple[float, float]:
    """
    Nudge position toward bomb site depending on scenario.
    A-site scenarios push toward x > 0.5; B-site toward y > 0.5.
    """
    dx = dy = 0.0
    if "a" in scenario:
        dx = random.uniform(0.005, 0.02)
    elif "b" in scenario:
        dy = random.uniform(0.005, 0.02)
    else:
        dx = random.uniform(-0.01, 0.02)
        dy = random.uniform(-0.01, 0.02)
    # Add small noise
    dx += random.gauss(0, 0.003)
    dy += random.gauss(0, 0.003)
    return clamp(current_x + dx), clamp(current_y + dy)


def crosshair_near_enemy(enemy: dict | None) -> tuple[float, float]:
    """Return a crosshair position that is near (or misses) an enemy bbox."""
    if enemy is None or not enemy["is_visible"]:
        return round(random.uniform(0.3, 0.7), 4), round(random.uniform(0.3, 0.7), 4)
    # 60% chance of good placement
    if random.random() < 0.6:
        cx = clamp(enemy["bbox_x"] + random.gauss(0, 0.01))
        cy = clamp(enemy["bbox_y"] + random.gauss(0, 0.01))
    else:
        cx = clamp(enemy["bbox_x"] + random.uniform(-0.1, 0.1))
        cy = clamp(enemy["bbox_y"] + random.uniform(-0.1, 0.1))
    return round(cx, 4), round(cy, 4)


def random_enemy(alive: bool, confidence_floor: float = 0.6) -> dict:
    """Generate one enemy state."""
    bx = round(random.uniform(0.2, 0.8), 4)
    by = round(random.uniform(0.2, 0.8), 4)
    bw = round(random.uniform(0.02, 0.07), 4)
    bh = round(random.uniform(0.05, 0.14), 4)
    return {
        "enemy_id": random.randint(0, 4),
        "bbox_x": bx,
        "bbox_y": by,
        "bbox_w": bw,
        "bbox_h": bh,
        "distance_est": round(random.uniform(0.1, 1.0), 4),
        "is_visible": alive and random.random() > 0.35,
        "confidence": round(random.uniform(confidence_floor, 0.99), 4),
    }


# ---------------------------------------------------------------------------
# Round / episode generator
# ---------------------------------------------------------------------------

class RoundSimulator:
    """
    Simulates one CS:GO round as a sequence of per-frame JSON state dicts.
    """

    def __init__(self, round_number: int, map_name: str, scenario: str, rng: random.Random):
        self.round_number = round_number
        self.map_name = map_name
        self.scenario = scenario
        self.rng = rng

        # Player state
        self.pos_x, self.pos_y = rand_pos()
        self.health = 100
        self.armor = 100
        self.alive = True
        self.weapon = rng.choices(WEAPONS, weights=[0.55, 0.25, 0.15, 0.05])[0]
        self.utility = rng.choice(UTILITY)

        # Bomb state
        self.bomb_detected = rng.random() > 0.2
        self.bomb_site = rng.choices(BOMB_SITES, weights=[0.45, 0.45, 0.10])[0]
        self.bomb_planted = False
        self.plant_frame = None

        # Round outcome (determined upfront for story consistency)
        self.round_won = rng.random() < 0.52  # slight attacker edge

        # Enemy count (1-5)
        self.enemy_count = rng.randint(1, 5)
        self.enemies_alive = self.enemy_count

        # Utility active tracking
        self.smoke_count = 0
        self.flash_active = False
        self.fire_active = False

        # Event counters
        self.kills_done = 0
        self.death_done = False

        # Determine round length
        if "rush" in scenario:
            self.round_length_frames = rng.randint(int(FPS * 8), int(FPS * 25))
        elif "save" in scenario:
            self.round_length_frames = rng.randint(int(FPS * 5), int(FPS * 15))
        else:
            self.round_length_frames = rng.randint(int(FPS * 20), MAX_FRAMES)

    # ------------------------------------------------------------------
    def _choose_action(self, frame_id: int, visible_enemies: int) -> str:
        """
        Heuristic attacker action labelling — only uses attacker-valid actions 0-13.
        Actions 14 (DEFUSE_BOMB) and 15 (ROTATE_SITE) are never produced.
        """
        progress = frame_id / self.round_length_frames
        s = self.scenario

        if not self.alive:
            return "REPOSITION"

        # Early phase: push toward site
        if progress < 0.3:
            if "rush" in s:
                return self.rng.choices(["RUSH_SITE", "MOVE_FORWARD"], weights=[0.7, 0.3])[0]
            return self.rng.choices(
                ["MOVE_FORWARD", "RUSH_SITE", "STRAFE_LEFT", "STRAFE_RIGHT"],
                weights=[0.45, 0.20, 0.20, 0.15],
            )[0]

        # Mid phase: engage or use utility
        if progress < 0.7:
            if visible_enemies > 0:
                return self.rng.choices(
                    ["ENGAGE_ENEMY", "PEEK_CORNER", "THROW_SMOKE", "CROUCH_HOLD"],
                    weights=[0.50, 0.25, 0.15, 0.10],
                )[0]
            if self.utility in ("smoke", "flash") and not self.bomb_planted:
                return self.rng.choices(
                    ["THROW_SMOKE", "MOVE_FORWARD", "PEEK_CORNER"],
                    weights=[0.30, 0.40, 0.30],
                )[0]
            return self.rng.choices(
                ["MOVE_FORWARD", "REPOSITION", "HOLD_ANGLE"],
                weights=[0.50, 0.30, 0.20],
            )[0]

        # Late phase: plant or hold — attacker-only actions
        if not self.bomb_planted and self.bomb_detected:
            return "PLANT_BOMB"
        if visible_enemies > 0:
            return self.rng.choices(["ENGAGE_ENEMY", "HOLD_ANGLE"], weights=[0.6, 0.4])[0]
        return self.rng.choices(
            ["HOLD_ANGLE", "REPOSITION", "MOVE_FORWARD"],
            weights=[0.50, 0.30, 0.20],
        )[0]

    # ------------------------------------------------------------------
    def _build_enemies(self, frame_id: int) -> list[dict]:
        """Produce up to 5 enemy slot dicts."""
        slots = []
        for i in range(min(self.enemies_alive, 5)):
            e = random_enemy(alive=True)
            e["enemy_id"] = i
            slots.append(e)
        # Zero-pad up to 5 slots
        while len(slots) < 5:
            slots.append({
                "enemy_id": len(slots),
                "bbox_x": 0.0, "bbox_y": 0.0,
                "bbox_w": 0.0, "bbox_h": 0.0,
                "distance_est": 0.0,
                "is_visible": False,
                "confidence": 0.0,
            })
        return slots[:5]

    # ------------------------------------------------------------------
    def _tick_events(self, frame_id: int) -> dict:
        """
        Decide what events fire this frame.
        Returns game_events dict and mutates round state.
        """
        events = {
            "kill_this_frame": False,
            "death_this_frame": False,
            "assist_this_frame": False,
            "round_won": False,
            "round_lost": False,
        }
        progress = frame_id / self.round_length_frames
        is_last = frame_id == self.round_length_frames - 1

        if not self.alive:
            if is_last:
                events["round_won" if self.round_won else "round_lost"] = True
            return events

        # Kill chance increases with progress (more contact)
        kill_chance = 0.012 * (1 + progress) if self.enemies_alive > 0 else 0
        if self.rng.random() < kill_chance and self.enemies_alive > 0:
            events["kill_this_frame"] = True
            self.kills_done += 1
            self.enemies_alive = max(0, self.enemies_alive - 1)

        # Assist
        if self.rng.random() < 0.005:
            events["assist_this_frame"] = True

        # Death (only once; higher in aggressive scenarios)
        death_chance = 0.006 if "rush" in self.scenario else 0.003
        if not self.death_done and self.rng.random() < death_chance:
            if not self.round_won:  # more likely to die in a losing round
                events["death_this_frame"] = True
                self.death_done = True
                self.alive = False
                self.health = 0

        # Bomb plant
        if (
            not self.bomb_planted
            and self.alive
            and progress > 0.45
            and self.bomb_detected
            and self.enemies_alive == 0
            and self.rng.random() < 0.15
        ):
            self.bomb_planted = True
            self.plant_frame = frame_id

        # Health damage
        if self.alive and self.enemies_alive > 0 and self.rng.random() < 0.04:
            dmg = self.rng.randint(5, 30)
            if self.armor > 0:
                self.armor = max(0, self.armor - dmg // 2)
                dmg //= 2
            self.health = max(1, self.health - dmg)

        # Terminal frame
        if is_last:
            events["round_won" if self.round_won else "round_lost"] = True

        return events

    # ------------------------------------------------------------------
    def generate(self) -> list[dict[str, Any]]:
        """Return the full list of frame state dicts for this round."""
        frames = []
        for frame_id in range(self.round_length_frames):
            timestamp = round(frame_id / FPS, 4)

            # Move player
            self.pos_x, self.pos_y = forward_pos(self.pos_x, self.pos_y, self.scenario)

            # Enemy list
            enemies = self._build_enemies(frame_id)
            visible = [e for e in enemies if e["is_visible"]]
            visible_count = len(visible)

            # Crosshair placement
            best_enemy = max(visible, key=lambda e: e["confidence"]) if visible else None
            cx, cy = crosshair_near_enemy(best_enemy)

            # Action label
            action_label = self._choose_action(frame_id, visible_count)

            # Utility active (random smoke/fire duration events)
            if action_label == "THROW_SMOKE" and self.rng.random() < 0.4:
                self.smoke_count = min(5, self.smoke_count + 1)
            if self.smoke_count > 0 and self.rng.random() < 0.02:
                self.smoke_count -= 1
            self.flash_active = self.rng.random() < 0.04
            self.fire_active = self.rng.random() < 0.03

            # Utility in hand
            if action_label in ("THROW_SMOKE", "THROW_FLASH", "THROW_HE") and self.rng.random() < 0.5:
                self.utility = "none"

            # Events
            events = self._tick_events(frame_id)

            # Bomb timer (counts up from 0 after plant, then -1 if unplanted)
            timer = -1
            if self.bomb_planted and self.plant_frame is not None:
                timer = round((frame_id - self.plant_frame) / FPS, 2)

            frame = {
                "frame_id": frame_id,
                "timestamp_sec": timestamp,
                "round_number": self.round_number,
                "map_name": self.map_name,
                "player": {
                    "role": ROLE,
                    "position_x": round(self.pos_x, 4),
                    "position_y": round(self.pos_y, 4),
                    "crosshair_x": cx,
                    "crosshair_y": cy,
                    "health": self.health,
                    "armor": self.armor,
                    "alive": self.alive,
                    "weapon": self.weapon,
                    "utility_held": self.utility,
                },
                "enemies": enemies,
                "bomb": {
                    "detected": self.bomb_detected,
                    "site": self.bomb_site,
                    "planted": self.bomb_planted,
                    "timer_remaining": timer,
                },
                "utility_active": {
                    "smoke_count": self.smoke_count,
                    "flash_active": self.flash_active,
                    "fire_active": self.fire_active,
                },
                "game_events": events,
                # extra metadata (not consumed by env but useful for analysis)
                "_meta": {
                    "scenario": self.scenario,
                    "agent_action": action_label,
                },
            }
            frames.append(frame)

        return frames


# ---------------------------------------------------------------------------
# Episode generator (multi-round)
# ---------------------------------------------------------------------------

def generate_episode(
    episode_id: int,
    rounds_per_episode: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    One episode = one full match excerpt (multiple consecutive rounds).
    All rounds share the same map.
    """
    map_name = rng.choice(MAPS)
    all_frames: list[dict[str, Any]] = []

    for r in range(1, rounds_per_episode + 1):
        scenario = rng.choice(SCENARIOS)
        sim = RoundSimulator(
            round_number=r,
            map_name=map_name,
            scenario=scenario,
            rng=rng,
        )
        all_frames.extend(sim.generate())

    return all_frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic CS:GO attacker training episodes")
    p.add_argument("--episodes",  type=int, default=200,          help="Number of episode files to generate")
    p.add_argument("--rounds",    type=int, default=None,         help="Rounds per episode (default: random 5-15)")
    p.add_argument("--out",       type=str, default="data/episodes", help="Output directory")
    p.add_argument("--seed",      type=int, default=42,           help="Global RNG seed")
    p.add_argument("--verbose",   action="store_true",            help="Print progress")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for i in range(args.episodes):
        rounds = args.rounds if args.rounds else rng.randint(5, 15)
        episode = generate_episode(episode_id=i, rounds_per_episode=rounds, rng=rng)
        total_frames += len(episode)

        fname = out_dir / f"episode_{i:04d}.json"
        with open(fname, "w") as f:
            json.dump(episode, f, separators=(",", ":"))   # compact for speed

        if args.verbose or (i % 50 == 0):
            print(f"  [{i+1:>4}/{args.episodes}]  {fname.name}  —  {len(episode):>5} frames")

    print(f"\n✓  Generated {args.episodes} episodes  ({total_frames:,} total frames)  →  {out_dir}/")
    print(f"   Ready for:  python -m rl.train --episodes {out_dir}/ --steps 500000")


if __name__ == "__main__":
    main()