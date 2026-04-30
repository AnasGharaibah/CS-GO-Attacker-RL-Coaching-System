"""
CS2 GSI Server — Full Data Extraction
Receives every game tick from CS2 and parses all available fields.
No UI, no LLM, no audio.

Run:  python gsi_server.py
Port: 3000

CS2 cfg (place in CS2/cfg/gamestate_integration_gsi.cfg):
--------------------------------------------------------------
"CS2 GSI"
{
    "uri"           "http://127.0.0.1:3000/"
    "timeout"       "5.0"
    "heartbeat"     "10.0"
    "data"
    {
        "provider"            "1"
        "map"                 "1"
        "map_round_wins"      "1"
        "player_id"           "1"
        "player_state"        "1"
        "player_weapons"      "1"
        "player_match_stats"  "1"
        "round"               "1"
        "allplayers_id"       "1"
        "allplayers_state"    "1"
        "allplayers_match_stats" "1"
        "allplayers_weapons"  "1"
        "allplayers_position" "1"
        "bomb"                "1"
        "phase_ends_in"       "1"
    }
}
--------------------------------------------------------------
"""

import json
import os
import uvicorn
from datetime import datetime
from fastapi import FastAPI, Request
from pathlib import Path

# ── GSI recording (writes raw ticks to source/gsi/<match_id>.jsonl) ──────────
GSI_RECORD = True          # set False to disable recording
GSI_DIR    = Path("source/gsi")
GSI_DIR.mkdir(parents=True, exist_ok=True)
_gsi_file  = None          # current open file handle

app = FastAPI()

# ── Global state ──────────────────────────────────────────────
latest_payload:  dict | None = None
latest_gsi:      dict | None = None   # parsed, structured
current_match_id: str | None = None
match_history:   list        = []     # last 5 round summaries
round_win_history: dict      = {}     # round_number → win_team
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# PARSERS  — one function per GSI block
# ══════════════════════════════════════════════════════════════

def parse_provider(payload: dict) -> dict:
    """
    provider → who is sending the data.
    Fields: name, appid, version, steamid, timestamp
    """
    p = payload.get("provider", {})
    return {
        "name":      p.get("name"),        # "Counter-Strike: Global Offensive"
        "appid":     p.get("appid"),       # 730
        "version":   p.get("version"),
        "steamid":   p.get("steamid"),     # local player's SteamID64
        "timestamp": p.get("timestamp"),   # Unix epoch from game
    }


def parse_map(payload: dict) -> dict:
    """
    map → match-level info and team scores.
    """
    m = payload.get("map", {})

    def parse_team(t: dict) -> dict:
        return {
            "score":              t.get("score", 0),
            "consecutive_losses": t.get("consecutive_round_losses", 0),
            "timeouts_remaining": t.get("timeouts_remaining"),
            "matches_won_this_series": t.get("matches_won_this_series"),
        }

    return {
        "name":    m.get("name"),           # e.g. "de_dust2"
        "phase":   m.get("phase"),          # "warmup" | "live" | "intermission" | "gameover"
        "round":   m.get("round", 0),
        "mode":    m.get("mode"),           # "competitive" | "deathmatch" | etc.
        "num_matches_to_win_series": m.get("num_matches_to_win_series"),
        "current_spectators": m.get("current_spectators", 0),
        "souvenirs_total":    m.get("souvenirs_total", 0),
        "team_ct": parse_team(m.get("team_ct", {})),
        "team_t":  parse_team(m.get("team_t", {})),
        "round_wins": payload.get("map", {}).get("round_wins", {}),
        # round_wins is a dict like {"1": "ct_win_elimination", "2": "t_win_bomb", ...}
    }


def parse_round(payload: dict) -> dict:
    """
    round → current round phase and outcome data.
    """
    r = payload.get("round", {})
    return {
        "phase":       r.get("phase"),       # "freezetime" | "live" | "over" | "defuse" | "bomb"
        "win_team":    r.get("win_team"),     # "CT" | "T" (only when phase == "over")
        "bomb_state":  r.get("bomb"),         # "planted" | "exploded" | "defused"
        "phase_ends_in": payload.get("phase_ends_in"),  # seconds left in this phase
    }


def parse_player_state(state: dict) -> dict:
    """Parse the player.state block (stats that change each tick)."""
    return {
        "health":          state.get("health", 0),
        "armor":           state.get("armor", 0),
        "helmet":          state.get("helmet", False),
        "flashed":         state.get("flashed", 0),        # 0–255
        "smoked":          state.get("smoked", 0),         # 0–255
        "burning":         state.get("burning", 0),        # 0–255
        "money":           state.get("money", 0),
        "round_kills":     state.get("round_kills", 0),
        "round_killhs":    state.get("round_killhs", 0),   # headshot kills this round
        "round_totaldmg":  state.get("round_totaldmg", 0),
        "equip_value":     state.get("equip_value", 0),    # $ value of current loadout
        "defusekit":       state.get("defusekit", False),
        "in_buyzone":      state.get("in_buyzone", False),
        "in_bombzone":     state.get("in_bombzone", False),
    }


def parse_match_stats(stats: dict) -> dict:
    return {
        "kills":   stats.get("kills", 0),
        "assists": stats.get("assists", 0),
        "deaths":  stats.get("deaths", 0),
        "mvps":    stats.get("mvps", 0),
        "score":   stats.get("score", 0),
    }


def parse_weapons(weapons: dict) -> dict:
    """
    Parse player.weapons into a clean structure.
    Returns active weapon separately for quick access.
    """
    parsed = {}
    active = None

    for slot, w in weapons.items():
        entry = {
            "name":          w.get("name", "").replace("weapon_", ""),
            "paintkit":      w.get("paintkit"),
            "type":          w.get("type"),           # "Rifle" | "Pistol" | "Knife" | "Grenade" | etc.
            "state":         w.get("state"),          # "active" | "holstered" | "reloading"
            "ammo_clip":     w.get("ammo_clip"),
            "ammo_clip_max": w.get("ammo_clip_max"),
            "ammo_reserve":  w.get("ammo_reserve"),
        }
        parsed[slot] = entry
        if entry["state"] == "active":
            active = entry

    return {"slots": parsed, "active": active}


def parse_local_player(payload: dict) -> dict:
    """
    player → the local player (the one running the game).
    Includes position and forward vector (requires 'observer' token in cfg).
    """
    p = payload.get("player", {})
    return {
        "steamid":       p.get("steamid"),
        "name":          p.get("name"),
        "team":          p.get("team"),          # "CT" | "T" | "Spectator"
        "observer_slot": p.get("observer_slot"),
        "activity":      p.get("activity"),      # "playing" | "menu" | "textinput"
        "position":      p.get("position"),      # "x, y, z" string
        "forward":       p.get("forward"),       # "x, y, z" facing direction string
        "state":         parse_player_state(p.get("state", {})),
        "match_stats":   parse_match_stats(p.get("match_stats", {})),
        "weapons":       parse_weapons(p.get("weapons", {})),
    }


def parse_all_players(payload: dict) -> dict:
    """
    allplayers → every player in the server (only populated when spectating
    or when the cfg uses an observer auth token with sufficient permission).
    Returns a dict keyed by SteamID64.
    """
    all_players = payload.get("allplayers", {})
    result = {}

    for steamid, p in all_players.items():
        result[steamid] = {
            "name":          p.get("name"),
            "team":          p.get("team"),
            "observer_slot": p.get("observer_slot"),
            "position":      p.get("position"),
            "forward":       p.get("forward"),
            "state":         parse_player_state(p.get("state", {})),
            "match_stats":   parse_match_stats(p.get("match_stats", {})),
            "weapons":       parse_weapons(p.get("weapons", {})),
        }

    return result


def parse_bomb(payload: dict) -> dict:
    """
    bomb → C4 state (only present when bomb is planted/dropped/picked up).
    """
    b = payload.get("bomb", {})
    if not b:
        return {}
    return {
        "state":    b.get("state"),      # "carried" | "planted" | "dropped" | "defusing" | "defused" | "exploded"
        "position": b.get("position"),   # "x, y, z"
        "player":   b.get("player"),     # SteamID of carrier
        "countdown": b.get("countdown"), # seconds until explosion (when planted)
    }


def parse_grenades(payload: dict) -> dict:
    """
    grenades → all active grenades in flight / exploding (when requested in cfg).
    Returns a dict keyed by grenade entity id.
    """
    grenades = payload.get("grenades", {})
    result = {}
    for gid, g in grenades.items():
        result[gid] = {
            "owner":        g.get("owner"),       # SteamID
            "type":         g.get("type"),         # "smoke" | "flashbang" | "frag" | "molotov" | "inferno"
            "position":     g.get("position"),
            "velocity":     g.get("velocity"),
            "lifetime":     g.get("lifetime"),
            "effecttime":   g.get("effecttime"),
        }
    return result


def build_gsi(payload: dict) -> dict:
    """Assembles all parsed blocks into one structured object."""
    return {
        "provider":    parse_provider(payload),
        "map":         parse_map(payload),
        "round":       parse_round(payload),
        "player":      parse_local_player(payload),
        "all_players": parse_all_players(payload),
        "bomb":        parse_bomb(payload),
        "grenades":    parse_grenades(payload),
        "previously":  payload.get("previously", {}),  # what changed vs last tick
        "added":       payload.get("added", {}),        # fields added this tick
        "received_at": datetime.utcnow().isoformat(),
    }


# ══════════════════════════════════════════════════════════════
# MATCH / ROUND TRACKING
# ══════════════════════════════════════════════════════════════

def on_round_end(gsi: dict):
    """Stores a round summary when round phase == 'over'."""
    global match_history

    rnd    = gsi["round"]
    player = gsi["player"]
    map_   = gsi["map"]

    round_num = map_["round"]
    summary = {
        "round":       round_num,
        "win_team":    rnd["win_team"],
        "bomb_state":  rnd["bomb_state"],
        "team":        player["team"],
        "won":         rnd["win_team"] == player["team"],
        "kills":       player["state"]["round_kills"],
        "hs_kills":    player["state"]["round_killhs"],
        "damage":      player["state"]["round_totaldmg"],
        "died":        player["state"]["health"] == 0,
        "equip_value": player["state"]["equip_value"],
        "score_ct":    map_["team_ct"]["score"],
        "score_t":     map_["team_t"]["score"],
    }

    if not any(r["round"] == round_num for r in match_history):
        match_history.append(summary)
        print(f"[ROUND END] Round {round_num} | {'WIN' if summary['won'] else 'LOSS'} "
              f"| {summary['kills']}K {summary['damage']}DMG")

    if len(match_history) > 5:
        match_history.pop(0)


# ══════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.post("/")
async def gsi_endpoint(request: Request):
    """Main GSI receiver — CS2 POSTs here every tick."""
    global latest_payload, latest_gsi, current_match_id, _gsi_file

    try:
        payload = await request.json()
    except Exception:
        return {"status": "error", "reason": "invalid json"}

    if not payload.get("map"):
        return {"status": "ignored"}

    latest_payload = payload
    latest_gsi     = build_gsi(payload)

    map_phase   = latest_gsi["map"]["phase"]
    round_phase = latest_gsi["round"]["phase"]

    # ── New match detection ───────────────────────────────────
    if map_phase == "live" and current_match_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_match_id = f"match_{ts}"
        print(f"\n[NEW MATCH] {current_match_id} | Map: {latest_gsi['map']['name']}")
        if GSI_RECORD:
            _gsi_file = open(GSI_DIR / f"{current_match_id}.jsonl", "a", encoding="utf-8")
            print(f"[GSI RECORD] Writing ticks → source/gsi/{current_match_id}.jsonl")

    # ── Record raw tick ───────────────────────────────────────
    if GSI_RECORD and _gsi_file and not _gsi_file.closed:
        _gsi_file.write(json.dumps(payload) + "\n")
        _gsi_file.flush()

    # ── Match over ────────────────────────────────────────────
    if map_phase == "gameover":
        print(f"[MATCH OVER] {current_match_id}")
        if _gsi_file and not _gsi_file.closed:
            _gsi_file.close()
            _gsi_file = None
        current_match_id = None
        match_history.clear()

    # ── Round over ────────────────────────────────────────────
    if round_phase == "over":
        on_round_end(latest_gsi)

    # ── Per-tick console log (active gameplay only) ───────────
    player = latest_gsi["player"]
    if player["activity"] == "playing":
        s      = player["state"]
        active = player["weapons"]["active"]
        weapon = active["name"] if active else "none"
        ammo   = f"{active['ammo_clip']}/{active['ammo_reserve']}" if active else "-"
        print(
            f"[TICK] Rd {latest_gsi['map']['round']:>2} | "
            f"Phase: {round_phase:<10} | "
            f"HP:{s['health']:>3} Armor:{s['armor']:>3} | "
            f"${s['money']:>5} | "
            f"Weapon: {weapon:<20} Ammo: {ammo}"
        )

    # ══════════════════════════════════════════════════════════
    # YOUR LOGIC GOES HERE
    # ══════════════════════════════════════════════════════════
    # You have access to:
    #
    #   latest_gsi        — fully parsed game state (see build_gsi)
    #   latest_payload    — raw JSON from CS2
    #   current_match_id  — e.g. "match_20240429_143201"
    #   match_history     — list of last 5 round summaries
    #
    # Useful fields:
    #   latest_gsi["player"]["state"]["health"]
    #   latest_gsi["player"]["state"]["money"]
    #   latest_gsi["player"]["state"]["flashed"]      # 0–255
    #   latest_gsi["player"]["state"]["burning"]      # 0–255
    #   latest_gsi["player"]["state"]["equip_value"]
    #   latest_gsi["player"]["weapons"]["active"]     # active weapon dict
    #   latest_gsi["player"]["match_stats"]           # kills/deaths/assists/mvps
    #   latest_gsi["map"]["team_ct"]["score"]
    #   latest_gsi["map"]["team_t"]["score"]
    #   latest_gsi["map"]["round_wins"]               # {"1": "ct_win_elimination", ...}
    #   latest_gsi["round"]["phase"]                  # "freezetime"|"live"|"over"
    #   latest_gsi["round"]["bomb_state"]             # "planted"|"exploded"|"defused"
    #   latest_gsi["bomb"]["countdown"]               # seconds until explosion
    #   latest_gsi["all_players"]                     # dict of all players (spectator/auth only)
    #   latest_gsi["grenades"]                        # active grenades in the air
    # ══════════════════════════════════════════════════════════

    return {"status": "ok"}


@app.get("/status")
async def status():
    """Human-readable match status."""
    if not latest_gsi:
        return {"status": "waiting", "message": "No data yet. Launch CS2."}

    m = latest_gsi["map"]
    p = latest_gsi["player"]
    return {
        "match_id":   current_match_id,
        "map":        m["name"],
        "map_phase":  m["phase"],
        "round":      m["round"],
        "round_phase": latest_gsi["round"]["phase"],
        "score":      {"ct": m["team_ct"]["score"], "t": m["team_t"]["score"]},
        "player": {
            "name":   p["name"],
            "team":   p["team"],
            "health": p["state"]["health"],
            "money":  p["state"]["money"],
            "stats":  p["match_stats"],
        },
        "bomb":       latest_gsi["bomb"],
        "history":    match_history,
    }


@app.get("/player")
async def player():
    """Full local player data."""
    if not latest_gsi:
        return {"error": "no data"}
    return latest_gsi["player"]


@app.get("/map")
async def map_data():
    """Map + team scores + round win history."""
    if not latest_gsi:
        return {"error": "no data"}
    return latest_gsi["map"]


@app.get("/round")
async def round_data():
    """Current round state."""
    if not latest_gsi:
        return {"error": "no data"}
    return latest_gsi["round"]


@app.get("/bomb")
async def bomb():
    """C4 state (position, countdown, carrier)."""
    if not latest_gsi:
        return {"error": "no data"}
    return latest_gsi["bomb"]


@app.get("/allplayers")
async def all_players():
    """All players in the server (requires observer auth token in cfg)."""
    if not latest_gsi:
        return {"error": "no data"}
    return latest_gsi["all_players"]


@app.get("/history")
async def history():
    """Last 5 round summaries."""
    return match_history


@app.get("/raw")
async def raw():
    """Raw unprocessed GSI payload (last tick)."""
    if not latest_payload:
        return {"error": "no data"}
    return latest_payload


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("CS2 GSI Server  →  http://0.0.0.0:3000")
    print("Endpoints: /status  /player  /map  /round  /bomb  /allplayers  /history  /raw")
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="warning")
