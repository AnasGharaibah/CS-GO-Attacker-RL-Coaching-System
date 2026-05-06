import json
import argparse
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request

GSI_DIR = Path("source/gsi")
GSI_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

_match_id    = None
_round_phase = "unknown"
_map_name    = None
_gsi_file    = None
_tick_count  = 0
_player_team = None
_last_team_warn = None
_last_player = {}
_last_map    = {}
_last_bomb   = {}


def _open_file(mid):
    global _gsi_file
    if _gsi_file and not _gsi_file.closed:
        _gsi_file.close()
    _gsi_file = open(GSI_DIR / f"{mid}.jsonl", "a", encoding="utf-8")
    print(f"[gsi] Recording → source/gsi/{mid}.jsonl")


def _close_file():
    global _gsi_file
    if _gsi_file and not _gsi_file.closed:
        _gsi_file.close()
        _gsi_file = None


def _write(payload):
    if _gsi_file and not _gsi_file.closed:
        _gsi_file.write(json.dumps(payload) + "\n")
        _gsi_file.flush()


@app.post("/")
async def receive_tick(request: Request):
    global _match_id, _round_phase, _map_name, _tick_count, _player_team, _last_team_warn
    global _last_player, _last_map, _last_bomb

    try:
        payload = await request.json()
    except Exception:
        return {"status": "error"}

    if not payload.get("map"):
        return {"status": "ignored"}

    map_data    = payload.get("map",   {})
    round_data  = payload.get("round", {})
    map_phase   = map_data.get("phase",  "")
    round_phase = round_data.get("phase", "unknown")
    map_name    = map_data.get("name",   "unknown")

    _round_phase = round_phase
    _map_name    = map_name
    _last_player = payload.get("player", {})
    _last_map    = payload.get("map",    {})
    _last_bomb   = payload.get("bomb",   {})

    if map_phase == "live" and _match_id is None:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        _match_id = f"match_{ts}"
        _tick_count = 0
        _open_file(_match_id)
        print(f"[gsi] New match: {_match_id} | {map_name}")

    if map_phase == "gameover" and _match_id:
        print(f"[gsi] Match over: {_match_id} | {_tick_count} ticks")
        _close_file()
        _match_id = None

    if _match_id and map_phase == "live":
        player = payload.get("player", {})
        state  = player.get("state", {})
        team   = player.get("team", "")

        if team and team != _player_team:
            _player_team = team
            if team == "CT":
                print(f"[gsi] WARNING: you are CT — this data will be ignored by the model.")
                print(f"[gsi]          Switch to T side for useful training data.")
            elif team == "T":
                print(f"[gsi] You are T (attacker) — recording useful data.")

        _write(payload)
        _tick_count += 1
        if _tick_count % 300 == 0:
            side = "T  ✓" if team == "T" else "CT  ← not useful"
            print(f"[gsi] Rd {map_data.get('round', '?'):>2} | "
                  f"{side} | "
                  f"HP {state.get('health', '?'):>3} | "
                  f"{round_phase} | ticks={_tick_count}")

    return {"status": "ok"}


@app.get("/player")
async def get_player():
    return _last_player


@app.get("/map")
async def get_map():
    return _last_map


@app.get("/bomb")
async def get_bomb():
    return _last_bomb


@app.get("/match_id")
async def get_match_id():
    return {"match_id": _match_id}


@app.get("/round_phase")
async def get_round_phase():
    return {"phase": _round_phase}


@app.get("/status")
async def get_status():
    return {
        "match_id":    _match_id,
        "map":         _map_name,
        "round_phase": _round_phase,
        "ticks_saved": _tick_count,
        "recording":   _match_id is not None,
        "team":        _player_team,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=3000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    print(f"[gsi] Listening on http://{args.host}:{args.port}/")
    print(f"[gsi] Saving to source/gsi/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
