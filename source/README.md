# Source data folder

Place your raw match data here before running the pipeline.

```
source/
├── gsi/
│   ├── match_20240429_143201.jsonl   ← one GSI tick per line
│   └── match_20240501_112033.jsonl
└── yolo/
    ├── match_20240429_143201.jsonl   ← one YOLO detection frame per line
    └── match_20240501_112033.jsonl
```

Both files for the same match **must have the same filename stem** so the pipeline can pair them automatically.

---

## GSI file format

Each line is a single raw JSON payload exactly as CS2 POSTs to `gsi_server.py`.
Save every tick by appending to a `.jsonl` file inside `gsi_endpoint()`:

```python
# Inside gsi_server.py — gsi_endpoint()
import json
with open(f"source/gsi/{current_match_id}.jsonl", "a") as f:
    f.write(json.dumps(payload) + "\n")
```

Each line looks like:

```json
{"provider": {"timestamp": 1714400000}, "map": {"name": "de_dust2", "round": 3, "phase": "live", ...}, "player": {"team": "T", "position": "120.5, -450.2, 64.0", "state": {"health": 100, "armor": 100}, "weapons": {"weapon_0": {"name": "weapon_ak47", "type": "Rifle", "state": "active", "ammo_clip": 28}}, "match_stats": {"kills": 2, "deaths": 1, "assists": 0}}, "round": {"phase": "live", "bomb": null}, "bomb": {}, "previously": {"player": {"match_stats": {"kills": 1}}}}
```

---

## YOLO file format

Each line is one JSON object corresponding to one video frame.
Your YOLO model should output this format:

```json
{"frame_id": 42, "timestamp_sec": 1.40, "detections": [{"class": "player_enemy", "bbox_x": 0.55, "bbox_y": 0.47, "bbox_w": 0.04, "bbox_h": 0.09, "confidence": 0.91}, {"class": "smoke", "bbox_x": 0.30, "bbox_y": 0.60, "bbox_w": 0.12, "bbox_h": 0.08, "confidence": 0.88}]}
```

### Supported class names

| Class | Meaning |
|---|---|
| `player_enemy` | Visible enemy player |
| `player_t` / `player_ct` | Team-labelled player (enemy inferred from role) |
| `smoke` | Smoke grenade cloud |
| `flashbang` / `flash` | Active flashbang |
| `fire` / `molotov` / `inferno` | Fire / molotov |

### YOLO output adapter (example)

If your YOLO model uses Ultralytics, convert its output like this:

```python
import json
from ultralytics import YOLO

model   = YOLO("your_model.pt")
classes = {0: "player_enemy", 1: "smoke", 2: "flash", 3: "fire"}

cap = cv2.VideoCapture("match.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
fid = 0

with open("source/yolo/match_abc.jsonl", "w") as out:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        detections = []
        h, w = frame.shape[:2]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class":      classes.get(int(box.cls), "player_enemy"),
                "bbox_x":     ((x1 + x2) / 2) / w,
                "bbox_y":     ((y1 + y2) / 2) / h,
                "bbox_w":     (x2 - x1) / w,
                "bbox_h":     (y2 - y1) / h,
                "confidence": float(box.conf),
            })
        out.write(json.dumps({
            "frame_id":     fid,
            "timestamp_sec": fid / fps,
            "detections":   detections,
        }) + "\n")
        fid += 1
```

---

## Running the pipeline

```bash
# Build replay buffer from all matched pairs in source/
python -m rl.pipeline --source source --out data/attacker_buffer.h5

# Then train
python -m rl.train --episodes source --steps 500000
```
