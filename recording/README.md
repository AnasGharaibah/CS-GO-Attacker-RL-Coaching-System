# recording/

Data collection module. Records CS2 game state and screen detections simultaneously while you play.

Both files write to `source/` using the **same match_id** so `prepare.py` can pair them automatically.

---

## Files

### `gsi_recorder.py`

Receives every CS2 game-state tick via the built-in Game State Integration (GSI) system and saves raw payloads to disk.

**What it records:**
- Player position, health, armor, team, active weapon
- Round phase, round number, bomb state and countdown
- Per-tick kill / death / assist deltas (via the `previously` diff block)
- Match start and end events

**Output:** `source/gsi/<match_id>.jsonl` — one raw JSON payload per line

**Sync endpoints (used by `capture.py`):**

| Endpoint | Returns |
|---|---|
| `GET /match_id` | `{"match_id": "match_20240429_143201"}` or `null` |
| `GET /round_phase` | `{"phase": "live"}` |
| `GET /status` | full recording status |

**Run standalone:**
```bash
python recording/gsi_recorder.py
python recording/gsi_recorder.py --port 3000 --host 0.0.0.0
```

---

### `capture.py`

Captures your screen at up to 30fps and runs your trained YOLO model on every frame to detect visible enemies, smoke, flashbangs, and fire.

**What it records:**
- Bounding box of each detected object (centre x/y, width/height — all normalised 0-1)
- Object class (player_enemy, smoke, flash, fire, bomb, weapon)
- YOLO confidence score per detection
- Frame timestamp relative to match start

**Output:** `source/yolo/<match_id>.jsonl` — one detection frame per line

**Sync behaviour:**
- Polls `gsi_recorder.py /match_id` every frame to get the active match id
- Polls `gsi_recorder.py /round_phase` to skip `freezetime` frames

**Class name config** — edit `CLASS_NAMES` at the top of the file to match your model:
```python
CLASS_NAMES = {
    0: "player_enemy",
    1: "player_ct",
    2: "player_t",
    3: "smoke",
    4: "flash",
    5: "fire",
    6: "bomb",
    7: "weapon",
}
```

**Run standalone:**
```bash
python recording/capture.py --model models/csgo_yolo.pt
python recording/capture.py --model models/csgo_yolo.pt --fps 30 --conf 0.4 --monitor 1
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `models/csgo_yolo.pt` | Path to your YOLO `.pt` file |
| `--fps` | `30` | Capture rate |
| `--conf` | `0.40` | Minimum detection confidence |
| `--monitor` | `1` | Monitor index: 1 = primary, 2 = secondary |

---

### `gsi_server.py`

The full-featured CS2 GSI server with a complete REST API. Use this instead of `gsi_recorder.py` if you also need to query live game state from other apps or scripts.

**Extra endpoints (over gsi_recorder.py):**

| Endpoint | Returns |
|---|---|
| `GET /status` | overview: map, score, round, player HP/money |
| `GET /player` | full local player data |
| `GET /map` | map name, phase, team scores |
| `GET /round` | round phase, win team, bomb state |
| `GET /bomb` | C4 state, position, countdown |
| `GET /allplayers` | all players (spectator / auth token only) |
| `GET /history` | last 5 round summaries |
| `GET /raw` | last unprocessed CS2 payload |

Also auto-records ticks to `source/gsi/` (same as `gsi_recorder.py`).

**Run standalone:**
```bash
python recording/gsi_server.py
```

> Only run ONE of `gsi_recorder.py` or `gsi_server.py` at a time — both listen on port 3000.

---

## Output format

### GSI line (`source/gsi/<match_id>.jsonl`)

```json
{
  "provider": {"timestamp": 1714400000},
  "map": {"name": "de_dust2", "round": 3, "phase": "live"},
  "player": {
    "team": "T",
    "position": "120.5, -450.2, 64.0",
    "state": {"health": 100, "armor": 100},
    "weapons": {"weapon_0": {"name": "weapon_ak47", "type": "Rifle", "state": "active"}},
    "match_stats": {"kills": 2, "deaths": 1, "assists": 0}
  },
  "round": {"phase": "live"},
  "bomb": {},
  "previously": {"player": {"match_stats": {"kills": 1}}}
}
```

### YOLO line (`source/yolo/<match_id>.jsonl`)

```json
{
  "frame_id": 42,
  "timestamp_sec": 1.4,
  "detections": [
    {
      "class": "player_enemy",
      "bbox_x": 0.55, "bbox_y": 0.47,
      "bbox_w": 0.04, "bbox_h": 0.09,
      "confidence": 0.91
    },
    {
      "class": "smoke",
      "bbox_x": 0.30, "bbox_y": 0.60,
      "bbox_w": 0.12, "bbox_h": 0.08,
      "confidence": 0.88
    }
  ]
}
```
