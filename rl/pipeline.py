from __future__ import annotations

import json
import math
from pathlib import Path


_WEAPON_MAP = {
    "ak47": "rifle", "m4a1": "rifle", "m4a1_silencer": "rifle",
    "sg556": "rifle", "aug": "rifle", "famas": "rifle", "galil": "rifle",
    "scar20": "rifle", "g3sg1": "rifle",
    "awp": "sniper", "ssg08": "sniper",
    "glock": "pistol", "usp_silencer": "pistol", "p250": "pistol",
    "deagle": "pistol", "fiveseven": "pistol", "tec9": "pistol",
    "cz75a": "pistol", "revolver": "pistol", "elite": "pistol",
    "p2000": "pistol", "hkp2000": "pistol",
    "knife": "knife", "knife_t": "knife", "bayonet": "knife",
}

_UTILITY_ITEMS = {
    "smokegrenade": "smoke",
    "flashbang":    "flash",
    "hegrenade":    "he",
    "molotov":      "molotov",
    "incgrenade":   "molotov",
    "decoy":        "none",
}

_MAP_BOUNDS = {
    "de_dust2":   {"x_min": -2476, "x_max": 1680,  "y_min": -3192, "y_max": 1632},
    "de_mirage":  {"x_min": -3230, "x_max": 1870,  "y_min": -3430, "y_max": 1850},
    "de_inferno": {"x_min": -2087, "x_max": 2200,  "y_min": -3050, "y_max": 3730},
    "de_nuke":    {"x_min": -3453, "x_max": 2923,  "y_min": -4000, "y_max": 4650},
    "de_overpass":{"x_min": -4831, "x_max": 1107,  "y_min": -942,  "y_max": 6436},
    "de_vertigo": {"x_min": -3137, "x_max": 2495,  "y_min": -2870, "y_max": 1713},
    "de_ancient": {"x_min": -2953, "x_max": 2164,  "y_min": -3462, "y_max": 1596},
    "de_anubis":  {"x_min": -2320, "x_max": 2348,  "y_min": -2786, "y_max": 2152},
    "__default__":{"x_min": -4000, "x_max": 4000,  "y_min": -4000, "y_max": 4000},
}


def _norm_pos(x, y, map_name):
    b = _MAP_BOUNDS.get(map_name, _MAP_BOUNDS["__default__"])
    nx = (x - b["x_min"]) / (b["x_max"] - b["x_min"])
    ny = (y - b["y_min"]) / (b["y_max"] - b["y_min"])
    return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))


def _parse_xyz(s):
    if not s:
        return 0.0, 0.0, 0.0
    try:
        parts = [float(v.strip()) for v in s.split(",")]
        return parts[0], parts[1], parts[2] if len(parts) >= 3 else 0.0
    except (ValueError, IndexError):
        return 0.0, 0.0, 0.0


def _active_weapon(weapons):
    weapon_cat = "knife"
    utility    = "none"
    for _slot, w in weapons.items():
        if w.get("state") != "active":
            continue
        name  = w.get("name", "").replace("weapon_", "").lower()
        wtype = w.get("type", "").lower()
        if "grenade" in wtype or name in _UTILITY_ITEMS:
            utility = _UTILITY_ITEMS.get(name, "none")
        else:
            weapon_cat = _WEAPON_MAP.get(name, "knife")
    return weapon_cat, utility


def _guess_site(bpos, map_name):
    if not bpos:
        return "unknown"
    x, y, _ = _parse_xyz(bpos)
    nx, _ = _norm_pos(x, y, map_name)
    if nx > 0.5:  return "A"
    if nx < 0.4:  return "B"
    return "unknown"


def _gsi_to_frame(tick, map_name, frame_id):
    rnd    = tick.get("round",  {}) or {}
    player = tick.get("player", {}) or {}
    state  = player.get("state", {}) or {}
    stats  = player.get("match_stats", {}) or {}
    bomb   = tick.get("bomb",   {}) or {}
    map_   = tick.get("map",    {}) or {}

    round_phase = rnd.get("phase", "")
    map_phase   = map_.get("phase", "live")
    if round_phase == "freezetime" or map_phase in ("warmup", "intermission", "gameover"):
        return None

    team = player.get("team", "")
    if team not in ("T", "CT"):
        return None

    role = "attacker" if team == "T" else "defender"
    px, py, _ = _parse_xyz(player.get("position"))
    nx, ny    = _norm_pos(px, py, map_name)
    weapon_cat, utility_held = _active_weapon(player.get("weapons", {}))

    b_state   = bomb.get("state", "")
    b_planted = b_state in ("planted", "defusing", "defused", "exploded")
    b_timer   = float(bomb.get("countdown") or -1)

    round_won  = rnd.get("win_team") == team and round_phase == "over"
    round_lost = rnd.get("win_team") not in (None, "", team) and round_phase == "over"

    prev       = tick.get("previously", {}) or {}
    prev_stats = prev.get("player", {}).get("match_stats", {})

    def _changed(key):
        prev_val = prev_stats.get(key)
        curr_val = stats.get(key, 0)
        return prev_val is not None and int(curr_val) > int(prev_val)

    return {
        "frame_id":      frame_id,
        "timestamp_sec": float(tick.get("provider", {}).get("timestamp", 0)),
        "round_number":  int(map_.get("round", 1)),
        "map_name":      map_name,
        "_round_phase":  round_phase,
        "player": {
            "role":         role,
            "position_x":   nx,
            "position_y":   ny,
            "crosshair_x":  0.5,
            "crosshair_y":  0.5,
            "health":       int(state.get("health", 0)),
            "armor":        int(state.get("armor",  0)),
            "alive":        int(state.get("health", 0)) > 0,
            "weapon":       weapon_cat,
            "utility_held": utility_held,
        },
        "enemies": [],
        "bomb": {
            "detected":        bool(bomb),
            "site":            _guess_site(bomb.get("position"), map_name),
            "planted":         b_planted,
            "timer_remaining": b_timer,
        },
        "utility_active": {"smoke_count": 0, "flash_active": False, "fire_active": False},
        "game_events": {
            "kill_this_frame":   _changed("kills"),
            "death_this_frame":  _changed("deaths"),
            "assist_this_frame": _changed("assists"),
            "round_won":         round_won,
            "round_lost":        round_lost,
        },
    }


def _yolo_to_enemies(yolo_frame):
    enemies      = []
    smoke_count  = 0
    flash_active = False
    fire_active  = False

    for det in yolo_frame.get("detections", []):
        cls  = str(det.get("class", "")).lower()
        conf = float(det.get("confidence", 0.0))

        if "smoke" in cls:
            smoke_count += 1
            continue
        if "flash" in cls or "flashbang" in cls:
            flash_active = True
            continue
        if "fire" in cls or "molotov" in cls or "inferno" in cls:
            fire_active = True
            continue

        if "enemy" in cls or "player" in cls:
            bh = float(det.get("bbox_h", 0.0))
            enemies.append({
                "enemy_id":    len(enemies),
                "bbox_x":      float(det.get("bbox_x", 0.0)),
                "bbox_y":      float(det.get("bbox_y", 0.0)),
                "bbox_w":      float(det.get("bbox_w", 0.0)),
                "bbox_h":      bh,
                "distance_est": max(0.0, 1.0 - bh * 10.0),
                "is_visible":  conf >= 0.5,
                "confidence":  conf,
            })
            if len(enemies) >= 5:
                break

    return enemies, {"smoke_count": smoke_count, "flash_active": flash_active, "fire_active": fire_active}


def _align(gsi_frames, yolo_frames, tol=0.1):
    pairs = []
    yi = 0
    for gf in gsi_frames:
        gts = gf["timestamp_sec"]
        while yi + 1 < len(yolo_frames):
            if abs(yolo_frames[yi + 1]["timestamp_sec"] - gts) < abs(yolo_frames[yi]["timestamp_sec"] - gts):
                yi += 1
            else:
                break
        yf = yolo_frames[yi] if yi < len(yolo_frames) else None
        if yf and abs(yf["timestamp_sec"] - gts) > tol:
            yf = None
        pairs.append((gf, yf))
    return pairs


def _group_episodes(frames):
    if not frames:
        return []
    episodes, current, prev_round = [], [], frames[0]["round_number"]
    for frame in frames:
        rnum = frame["round_number"]
        if rnum != prev_round and current:
            episodes.append(current)
            current = []
        current.append(frame)
        prev_round = rnum
    if current:
        episodes.append(current)
    return [ep for ep in episodes if len(ep) >= 2]


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def build_episodes(gsi_path, yolo_path, role="attacker", map_name=None):
    gsi_ticks   = load_jsonl(gsi_path)
    yolo_frames = load_jsonl(yolo_path)

    if map_name is None:
        for tick in gsi_ticks:
            mn = tick.get("map", {}).get("name")
            if mn:
                map_name = mn
                break
        map_name = map_name or "de_dust2"

    print(f"[pipeline] GSI: {len(gsi_ticks)} ticks | YOLO: {len(yolo_frames)} frames | Map: {map_name}")

    partial = []
    for i, tick in enumerate(gsi_ticks):
        pf = _gsi_to_frame(tick, map_name, i)
        if pf is None or pf["player"]["role"] != role:
            continue
        if pf["timestamp_sec"] == 0:
            pf["timestamp_sec"] = float(i) / 30.0
        partial.append(pf)

    if not partial:
        print(f"[pipeline] No live {role} frames found.")
        return []

    pairs  = _align(partial, yolo_frames)
    merged = []

    for gf, yf in pairs:
        if yf:
            enemies, util = _yolo_to_enemies(yf)
            gf["enemies"]        = enemies
            gf["utility_active"] = util
            if enemies:
                best = max(enemies, key=lambda e: e["confidence"])
                gf["player"]["crosshair_x"] = best["bbox_x"]
                gf["player"]["crosshair_y"] = best["bbox_y"]
        gf.pop("_round_phase", None)
        merged.append(gf)

    episodes = _group_episodes(merged)
    print(f"[pipeline] {len(episodes)} episodes, {len(merged)} frames total")
    return episodes


def build_episodes_from_folder(source_dir="source", role="attacker"):
    gsi_dir  = Path(source_dir) / "gsi"
    yolo_dir = Path(source_dir) / "yolo"

    if not gsi_dir.exists():
        raise FileNotFoundError(f"GSI folder not found: {gsi_dir}")
    if not yolo_dir.exists():
        raise FileNotFoundError(f"YOLO folder not found: {yolo_dir}")

    all_episodes = []
    for gsi_file in sorted(gsi_dir.glob("*.jsonl")):
        yolo_file = yolo_dir / gsi_file.name
        if not yolo_file.exists():
            print(f"[pipeline] Skipping {gsi_file.name} — no matching YOLO file.")
            continue
        print(f"[pipeline] {gsi_file.stem}")
        all_episodes.extend(build_episodes(str(gsi_file), str(yolo_file), role=role))

    print(f"[pipeline] Total episodes: {len(all_episodes)}")
    return all_episodes


if __name__ == "__main__":
    import argparse
    from .replay_buffer import ReplayBuffer

    p = argparse.ArgumentParser()
    p.add_argument("--source", default="source")
    p.add_argument("--out",    default="data/attacker_buffer.h5")
    p.add_argument("--role",   default="attacker", choices=["attacker", "defender"])
    args = p.parse_args()

    episodes = build_episodes_from_folder(args.source, role=args.role)
    if not episodes:
        raise SystemExit("[pipeline] No episodes built.")

    buf = ReplayBuffer.build_from_episodes(episodes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    buf.save(args.out)
    print(f"[pipeline] Saved → {args.out} ({len(buf)} transitions)")
