# prepare.py

Merges raw GSI + YOLO recordings into a ready-to-train RL replay buffer.

Run this after one or more recording sessions with `run.py`.

---

## Usage

```bash
# Process all matches in source/
python prepare.py

# Process one specific match
python prepare.py --match match_20240429_143201

# Custom output path
python prepare.py --out data/my_session.h5

# Print stats without saving
python prepare.py --stats-only
```

## What it does

1. Scans `source/gsi/` and `source/yolo/` for matched `.jsonl` file pairs
2. For each pair, runs `rl/pipeline.py` to merge GSI ticks + YOLO frames into per-frame state dicts
3. Groups frames into episodes (one episode = one CS:GO round)
4. Builds the RL replay buffer: `(obs, action, reward, next_obs, done)` tuples
5. Prints a full dataset report
6. Saves the buffer to `data/attacker_buffer.h5` and a 70/15/15 split index file

## Flags

| Flag | Default | Description |
|---|---|---|
| `--source` | `source` | Root folder with `gsi/` and `yolo/` subfolders |
| `--out` | `data/attacker_buffer.h5` | Output replay buffer path |
| `--match` | all | Process only files with this name prefix |
| `--role` | `attacker` | Player role to extract (`attacker` or `defender`) |
| `--stats-only` | off | Print report but do not save the buffer |

## Example output

```
[prepare] Processing 3 match(es)...

[prepare] ── match_20240429_143201
          GSI ticks: 54820  |  YOLO frames: 49140
          Episodes : 18

[prepare] ── match_20240501_112033
          GSI ticks: 61200  |  YOLO frames: 58900
          Episodes : 23

[prepare] Building replay buffer from 41 episodes...
[prepare] Buffer size: 138,400 transitions

────────────────────────────────────────────────────────
  DATASET REPORT
────────────────────────────────────────────────────────
  Matches processed : 3
    • match_20240429_143201
    • match_20240501_112033
    • match_20240503_091845
────────────────────────────────────────────────────────
  EPISODES
    Total rounds     : 41
    Total frames     : 138,400
    Avg round length : 3375.6 frames
    Win rate         : 56.1%
    K/D ratio        : 1.31
    Total kills      : 87
    Total deaths     : 66
    Bomb plants      : 24
────────────────────────────────────────────────────────
  REPLAY BUFFER
    Transitions      : 138,400
    Episodes (done)  : 41
    Reward  mean     : +0.1842
    Reward  std      : 1.2140
    Reward  range    : [-10.05, +23.10]
────────────────────────────────────────────────────────
  ACTION DISTRIBUTION
    ENGAGE_ENEMY       31.2%  ███████████████
    MOVE_FORWARD       18.7%  █████████
    RUSH_SITE          14.3%  ███████
    PLANT_BOMB          9.8%  ████
    HOLD_ANGLE          8.1%  ████
    REPOSITION          6.4%  ███
    PEEK_CORNER         5.9%  ██
    STRAFE_LEFT         2.4%  █
    STRAFE_RIGHT        1.8%  
    MOVE_BACK           1.4%  
────────────────────────────────────────────────────────
  Saved → data/attacker_buffer.h5
  Next  →  python -m rl.train --steps 500000
────────────────────────────────────────────────────────

[prepare] Split indices → data/attacker_buffer.splits.json
[prepare] Train: 96,880   Val: 20,760   Test: 20,760 transitions
```

## Outputs

| File | Description |
|---|---|
| `data/attacker_buffer.h5` | HDF5 replay buffer with all transitions |
| `data/attacker_buffer.splits.json` | Train/val/test index split (70/15/15) |
