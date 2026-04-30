# CS:GO Attacker RL Coaching System

A complete pipeline that watches you play CS:GO, learns from your gameplay, and coaches you to play better as an Attacker (T-side).

---

## How it works

```
┌─────────────────────────────────────────────────────┐
│  STEP 1 — RECORD  (while playing CS2)               │
│                                                     │
│  CS2 game state ──► recording/gsi_recorder.py       │
│                         │                           │
│                         ▼  source/gsi/<match>.jsonl │
│                                                     │
│  Your screen   ──► recording/capture.py             │
│  + YOLO model               │                       │
│                         ▼  source/yolo/<match>.jsonl│
└─────────────────────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2 — PREPARE  (after playing)                  │
│                                                     │
│  prepare.py  merges GSI + YOLO into RL frames       │
│              builds replay buffer                   │
│              saves →  data/attacker_buffer.h5       │
└─────────────────────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3 — TRAIN                                     │
│                                                     │
│  python -m rl.train --steps 500000                  │
│              saves →  data/models/                  │
└─────────────────────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────┐
│  OUTPUT — coaching report per match                 │
│                                                     │
│  - Action decision log (what you did vs agent)      │
│  - Kill opportunity report (missed engagements)     │
│  - Win rate, K/D, crosshair score, utility score    │
│  - Role coaching score  0-100                       │
└─────────────────────────────────────────────────────┘
```

---

## Project structure

```
pt/
├── README.md              ← you are here
├── run.py                 ← START HERE: launches recording
├── prepare.py             ← Step 2: build training dataset
│
├── recording/             ← data collection during gameplay
│   ├── README.md
│   ├── capture.py         ← screen capture + YOLO inference
│   ├── gsi_recorder.py    ← CS2 game state recorder
│   └── gsi_server.py      ← full GSI API server (optional)
│
├── rl/                    ← reinforcement learning model
│   ├── README.md
│   ├── pipeline.py        ← GSI + YOLO → RL frame schema
│   ├── observation.py     ← 62-dim observation encoder
│   ├── actions.py         ← 16-action space + attacker mask
│   ├── reward.py          ← attacker reward function
│   ├── action_inference.py← CV heuristics → action labels
│   ├── replay_buffer.py   ← offline experience store
│   ├── env.py             ← Gymnasium environment
│   ├── policy.py          ← MLP neural network
│   └── train.py           ← PPO training entry point
│
├── source/                ← raw recorded data (auto-created)
│   ├── README.md
│   ├── gsi/               ← GSI tick logs (.jsonl per match)
│   └── yolo/              ← YOLO detection logs (.jsonl per match)
│
└── data/                  ← processed buffers + trained models
    └── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install fastapi uvicorn mss ultralytics opencv-python-headless \
            requests numpy gymnasium stable-baselines3 sb3-contrib \
            torch h5py wandb
```

### 2. Set up CS2 GSI config

Create `gamestate_integration_gsi.cfg` in your CS2 `cfg/` folder:

```
"CS2 GSI"
{
    "uri"       "http://127.0.0.1:3000/"
    "timeout"   "5.0"
    "heartbeat" "10.0"
    "data"
    {
        "provider" "1"  "map" "1"  "map_round_wins" "1"
        "player_id" "1"  "player_state" "1"  "player_weapons" "1"
        "player_match_stats" "1"  "round" "1"  "bomb" "1"
        "phase_ends_in" "1"
    }
}
```

Add `-gamestateintegration` to CS2 Steam launch options.

### 3. Record gameplay

```bash
python run.py --model models/csgo_yolo.pt
```

Play CS2 normally. Both recorders start and stop automatically with each match.

### 4. Prepare the dataset

```bash
python prepare.py
```

### 5. Train

```bash
python -m rl.train --steps 500000
```

---

## File reference

| File | What it does |
|---|---|
| `run.py` | Launches `gsi_recorder` + `capture` together with one command |
| `prepare.py` | Merges raw data, builds replay buffer, prints dataset stats |
| `recording/capture.py` | Captures screen at 30fps, runs YOLO, saves detections |
| `recording/gsi_recorder.py` | Records CS2 GSI ticks to JSONL files |
| `recording/gsi_server.py` | Full GSI server with REST API (optional, for live queries) |
| `rl/pipeline.py` | Merges GSI + YOLO into the 62-dim RL frame schema |
| `rl/observation.py` | Encodes frame JSON to float32 vector |
| `rl/actions.py` | 16-action enum + per-frame attacker mask |
| `rl/reward.py` | Composite attacker reward function |
| `rl/action_inference.py` | Labels frame transitions with action IDs |
| `rl/replay_buffer.py` | Stores and serialises offline experience tuples |
| `rl/env.py` | CSGOAttackerEnv — Gymnasium environment |
| `rl/policy.py` | MLP [256,256] network with action masking |
| `rl/train.py` | PPO training loop with checkpoints and logging |

---

## Requirements

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | GSI server |
| `mss` | Screen capture |
| `ultralytics` | YOLO model inference |
| `opencv-python-headless` | Frame processing |
| `requests` | capture.py to gsi_recorder.py sync |
| `numpy` | Array operations |
| `gymnasium` | RL environment interface |
| `stable-baselines3` + `sb3-contrib` | MaskablePPO training |
| `torch` | Neural network |
| `h5py` | Replay buffer storage |
| `wandb` | Training metrics (optional) |
