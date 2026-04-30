# run.py

Single-command launcher that starts `gsi_recorder` and `capture` together and shuts both down cleanly when you press Ctrl+C.

---

## Usage

```bash
python run.py
python run.py --model models/csgo_yolo.pt
python run.py --model models/csgo_yolo.pt --fps 30 --conf 0.4 --monitor 1
```

## What it does

1. Starts `recording/gsi_recorder.py` in a background process (cyan output prefix `[GSI]`)
2. Waits 1.5 s for the GSI server to bind its port
3. Starts `recording/capture.py` in a background process (yellow output prefix `[CAP]`)
4. Streams both outputs to your terminal with colour coding
5. On Ctrl+C — terminates both processes and prints a shutdown message

## Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `models/csgo_yolo.pt` | Path to your trained YOLO `.pt` file |
| `--fps` | `30` | Screen capture frame rate |
| `--conf` | `0.40` | YOLO detection confidence threshold |
| `--monitor` | `1` | Monitor to capture: 1 = primary, 2 = secondary |
| `--gsi-port` | `3000` | Port for the GSI recorder (must match CS2 cfg) |

## Example output

```
========================================================
  CS2 Data Recorder — GSI + Screen Capture
========================================================
  YOLO model : models/csgo_yolo.pt
  Capture FPS: 30    Confidence: 0.4
  Monitor    : 1   GSI port: 3000
  Output     : source/gsi/   source/yolo/
========================================================
  Press Ctrl+C to stop recording.

[GSI] CS2 GSI Recorder
[GSI] Listening  →  http://0.0.0.0:3000/
[CAP] YOLO model loaded: models/csgo_yolo.pt
[CAP] Screen: 2560x1440 (monitor 1)
[CAP] Waiting for CS2 match via http://127.0.0.1:3000 ...
[GSI] NEW MATCH  match_20240429_143201  |  Map: de_dust2
[GSI] Recording  →  source/gsi/match_20240429_143201.jsonl
[CAP] Recording → source/yolo/match_20240429_143201.jsonl
[CAP] Frame      0 | t=  0.03s | enemies=0 | phase=live
[GSI] Rd  1 | T | HP 100 | phase=live       | ticks=300
[CAP] Frame    150 | t=  5.01s | enemies=1 | phase=live
```

## Notes

- Both processes save files with the **same match_id** (coordinated via the GSI `/match_id` endpoint), so `prepare.py` can pair them automatically.
- If either process exits unexpectedly, `run.py` detects it and shuts down the other one too.
- YOLO model must exist before starting. If not found, run exits with an error message.
