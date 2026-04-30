# CS:GO Attacker RL Agent

Reinforcement Learning coaching model for CS:GO **Attacker (T-side)** players.  
Trained offline on recorded gameplay footage processed through a YOLOv8 CV pipeline.  
Uses PPO (Proximal Policy Optimization) via `sb3-contrib`'s `MaskablePPO`.

---

## How it works

The pipeline has two stages:

```
Raw .mp4 / .dem footage
        │
   YOLOv8 Detector
        │
   Per-frame JSON state (§3.1 schema)
        │
   ┌────┴──────────────────────────────────────────┐
   │              rl/ (this module)                │
   │                                               │
   │  observation.py → 62-dim float32 vector       │
   │  action_inference.py → action label per frame │
   │  reward.py → scalar reward per frame          │
   │  replay_buffer.py → (s, a, r, s', done) store │
   │  env.py → Gymnasium environment               │
   │  policy.py → MLP [256, 256] + action masking  │
   │  train.py → MaskablePPO training loop         │
   └───────────────────────────────────────────────┘
        │
   Trained policy → coaching feedback report
```

---

## Module reference

| File | Purpose |
|---|---|
| `actions.py` | 16-action `Action` enum and per-frame boolean mask for the attacker role |
| `observation.py` | Converts a JSON frame state into the fixed 62-dim observation vector |
| `reward.py` | Composite attacker reward function (win/loss, kills, bomb plant, crosshair, positioning, time) |
| `action_inference.py` | Labels each frame transition with a discrete action using CV heuristics |
| `replay_buffer.py` | Stores offline `(obs, action, reward, next_obs, done)` tuples; HDF5 / npz serialisation |
| `env.py` | `CSGOAttackerEnv` — Gymnasium environment wrapping a replay buffer |
| `policy.py` | `AttackerPolicyNetwork` — shared MLP trunk + separate policy/value heads with action masking |
| `train.py` | Training entry point: PPO hyperparameters, dataset split, checkpointing, W&B logging |

---

## Installation

```bash
pip install gymnasium stable-baselines3 sb3-contrib torch numpy h5py wandb
```

---

## Quick start

### 1. Smoke test (no real data needed)

```bash
python -m rl.train --smoke-test --steps 2048
```

This generates 10 tiny synthetic episodes and runs one PPO update to verify the full pipeline works end-to-end.

### 2. Train on real data

Your episode data must be a folder of JSON files.  Each file is an ordered list of per-frame state dicts matching the schema below.

```bash
python -m rl.train --episodes data/episodes/ --steps 500000
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--steps` | 500000 | Total environment steps |
| `--checkpoint-freq` | 10000 | Save checkpoint every N steps |
| `--wandb` | off | Enable Weights & Biases logging |
| `--seed` | 42 | Global RNG seed |

### 3. Load a trained model

```python
from sb3_contrib import MaskablePPO
from rl.env import CSGOAttackerEnv

episodes = [...]   # list of frame sequences
env      = CSGOAttackerEnv(episodes)
model    = MaskablePPO.load("rl/models/attacker_ppo_final", env=env)

obs, _  = env.reset()
action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
```

---

## Input schema (per-frame JSON)

Each frame state dict must follow this structure (produced by the YOLOv8 pipeline):

```json
{
  "frame_id": 0,
  "timestamp_sec": 0.0,
  "round_number": 1,
  "map_name": "de_dust2",
  "player": {
    "role": "attacker",
    "position_x": 0.45,
    "position_y": 0.30,
    "crosshair_x": 0.52,
    "crosshair_y": 0.48,
    "health": 100,
    "armor": 100,
    "alive": true,
    "weapon": "rifle",
    "utility_held": "smoke"
  },
  "enemies": [
    {
      "enemy_id": 0,
      "bbox_x": 0.55, "bbox_y": 0.47,
      "bbox_w": 0.04, "bbox_h": 0.09,
      "distance_est": 0.6,
      "is_visible": true,
      "confidence": 0.91
    }
  ],
  "bomb": {
    "detected": true,
    "site": "A",
    "planted": false,
    "timer_remaining": -1
  },
  "utility_active": {
    "smoke_count": 0,
    "flash_active": false,
    "fire_active": false
  },
  "game_events": {
    "kill_this_frame": false,
    "death_this_frame": false,
    "assist_this_frame": false,
    "round_won": false,
    "round_lost": false
  }
}
```

All position and bounding-box fields are normalized to `[0.0, 1.0]`.  
Up to 5 enemies are tracked per frame; extras are ignored and missing slots are zero-padded.

---

## Observation vector (62 dimensions)

| Slice | Fields | Size |
|---|---|---|
| `[0:7]` | position_x, position_y, crosshair_x, crosshair_y, health/100, armor/100, alive | 7 |
| `[7:11]` | weapon one-hot: rifle / pistol / sniper / knife | 4 |
| `[11:16]` | utility one-hot: smoke / flash / he / molotov / none | 5 |
| `[16:51]` | 5 enemy slots × 7 fields (bbox_x, bbox_y, bbox_w, bbox_h, dist_est, is_visible, confidence) — sorted by confidence desc, zero-padded | 35 |
| `[51:57]` | bomb: detected, site one-hot (A/B/unknown), planted, timer/40 | 6 |
| `[57:60]` | utility active: smoke_count/5, flash_active, fire_active | 3 |
| `[60:62]` | timestamp_sec/60, round_number/30 | 2 |

---

## Action space (16 actions)

| ID | Action | Attacker? |
|---|---|---|
| 0 | `MOVE_FORWARD` | Yes |
| 1 | `MOVE_BACK` | Yes |
| 2 | `STRAFE_LEFT` | Yes |
| 3 | `STRAFE_RIGHT` | Yes |
| 4 | `CROUCH_HOLD` | Yes |
| 5 | `PEEK_CORNER` | Yes |
| 6 | `ENGAGE_ENEMY` | Yes |
| 7 | `REPOSITION` | Yes |
| 8 | `THROW_SMOKE` | Yes (masked if no utility) |
| 9 | `THROW_FLASH` | Yes (masked if no utility) |
| 10 | `THROW_HE` | Yes (masked if no utility) |
| 11 | `PLANT_BOMB` | Yes |
| 12 | `RUSH_SITE` | Yes |
| 13 | `HOLD_ANGLE` | Yes |
| 14 | `DEFUSE_BOMB` | **Always masked** |
| 15 | `ROTATE_SITE` | **Always masked** |

Actions 14-15 are defender-only and are permanently masked to `-inf` logits before softmax.

---

## Reward function

| Component | Trigger | Value |
|---|---|---|
| Round win | Terminal frame, team wins | +15.0 |
| Round loss | Terminal frame, team loses | −10.0 |
| Kill | `kill_this_frame = true` | +3.0 |
| Death | `death_this_frame = true` | −4.0 |
| Assist | `assist_this_frame = true` | +0.5 |
| Bomb plant | `bomb.planted` flips to `true` | +5.0 |
| Good crosshair | Crosshair inside visible enemy bbox | +0.1 |
| Poor crosshair | Enemy visible, crosshair misses | −0.05 |
| Good position | Player in forward half of map (alive) | +0.02 |
| Time penalty | Every frame (encourages aggression) | −0.01 |

---

## Neural network architecture

```
Input (62)
    │
Linear(256) → ReLU → LayerNorm(256)
    │
Linear(256) → ReLU → LayerNorm(256)
    │                     │
Policy Head           Value Head
Linear(128) → ReLU    Linear(128) → ReLU
Linear(16)             Linear(1)
[masked logits]        [scalar V(s)]
```

---

## PPO hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 (linear decay) |
| Discount factor γ | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Mini-batch size | 64 |
| Rollout buffer | 2048 frames |
| Epochs per update | 10 |
| Max episode length | 1800 frames (60 s × 30 fps) |

---

## Dataset split

| Split | Fraction | Purpose |
|---|---|---|
| Train | 70% | Policy learning |
| Validation | 15% | Hyperparameter tuning, overfitting check |
| Test | 15% | Generalisation evaluation (unseen maps) |

---

## Saved files

```
rl/models/
├── checkpoints/          # Periodic checkpoints (every 10k steps)
│   └── attacker_ppo_*.zip
├── best/                 # Best model by eval reward
│   └── best_model.zip
├── eval_logs/            # EvalCallback result logs
└── attacker_ppo_final.zip
```

---

## Evaluation metrics

| Metric | Target |
|---|---|
| Win rate | > 55% on test set |
| K/D ratio lift vs. baseline | > 10% |
| Crosshair alignment score | > 40% of frames |
| Utility efficiency | > 60% tactically correct uses |
| Mean episode reward | Increasing over epochs |
| Policy entropy | Remains > 0.5 |

---

## What the model output looks like

### 1. Per-frame prediction (raw model call)

```python
obs, _   = env.reset()
action, _states = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
probs    = model.policy.get_distribution(obs_tensor).distribution.probs
```

```
Observation  (62,): [0.45 0.30 0.52 0.48 1.00 1.00 1.00  1 0 0 0  0 0 0 0 1  ...]
Action mask  (16,): [True True True True True True True True False False False True True True False False]
                     ^^^^                                    ^^^^^ no utility held    ^^^^^^ defender-only

Predicted action : 6  →  ENGAGE_ENEMY
Action probs     :
  0  MOVE_FORWARD   0.031
  1  MOVE_BACK      0.008
  2  STRAFE_LEFT    0.041
  3  STRAFE_RIGHT   0.055
  4  CROUCH_HOLD    0.019
  5  PEEK_CORNER    0.072
  6  ENGAGE_ENEMY   0.581  ◄ selected
  7  REPOSITION     0.044
  8  THROW_SMOKE    0.000  (masked)
  9  THROW_FLASH    0.000  (masked)
 10  THROW_HE       0.000  (masked)
 11  PLANT_BOMB     0.109
 12  RUSH_SITE      0.040
 13  HOLD_ANGLE     0.000
 14  DEFUSE_BOMB    0.000  (masked)
 15  ROTATE_SITE    0.000  (masked)

State value V(s) : +2.74
```

---

### 2. Action decision log (per round CSV)

Written to `rl/models/coaching/action_log_round_<N>.csv` after each evaluated episode.

```
frame_id , timestamp , player_action   , agent_action   , reward  , gap
00042    , 1.40 s    , MOVE_FORWARD    , RUSH_SITE      , -0.010  , -0.571
00043    , 1.43 s    , MOVE_FORWARD    , RUSH_SITE      , -0.010  , -0.571
00044    , 1.47 s    , PEEK_CORNER     , ENGAGE_ENEMY   , +0.050  , -0.531
00045    , 1.50 s    , ENGAGE_ENEMY    , ENGAGE_ENEMY   , +0.100  ,  0.000  ✓
00046    , 1.53 s    , ENGAGE_ENEMY    , ENGAGE_ENEMY   , +3.100  ,  0.000  ✓ KILL
00047    , 1.57 s    , REPOSITION      , PLANT_BOMB     , -0.010  , -0.619
00048    , 1.60 s    , PLANT_BOMB      , PLANT_BOMB     , +4.990  ,  0.000  ✓ PLANT
```

`gap` = agent's recommended action value − value of action player actually took (negative = player underperformed the agent's suggestion).

---

### 3. Per-match coaching report (console summary)

Printed at the end of each evaluated match:

```
╔══════════════════════════════════════════════════════╗
║         ATTACKER COACHING REPORT — de_dust2         ║
╠══════════════════════════════════════════════════════╣
║  Rounds analysed       : 15                         ║
║  Win rate              : 9 / 15  (60.0%)            ║
║  K/D ratio             : 1.42   (baseline 1.08)  ↑  ║
║  Crosshair alignment   : 47.3%  of frames        ↑  ║
║  Utility efficiency    : 68.2%  correct use       ↑  ║
║  Mean episode reward   : +34.81                  ↑  ║
╠══════════════════════════════════════════════════════╣
║  TOP DECISION GAPS (frames where player diverged)   ║
║  1. T-spawn rush → agent preferred RUSH_SITE x 38  ║
║  2. Post-kill → agent preferred PLANT_BOMB x 12    ║
║  3. Smoke thrown too early → HOLD_ANGLE x 9        ║
╠══════════════════════════════════════════════════════╣
║  ATTACKER ROLE SCORE   :  73 / 100                  ║
╚══════════════════════════════════════════════════════╝
```

---

### 4. Action distribution across a full match

Shows which actions the agent recommended most often versus what the player actually did:

```
Action          Agent %   Player %   Delta
─────────────────────────────────────────
ENGAGE_ENEMY     31.2%     28.4%     -2.8%
MOVE_FORWARD     18.7%     24.1%     +5.4%  ← player over-rotates
RUSH_SITE        14.3%      7.9%     -6.4%  ← player too passive
PLANT_BOMB        9.8%      6.2%     -3.6%  ← player delays plant
HOLD_ANGLE        8.1%      9.3%     +1.2%
REPOSITION        6.4%      8.8%     +2.4%
PEEK_CORNER       5.9%      8.1%     +2.2%
STRAFE_LEFT       2.4%      4.3%     +1.9%
STRAFE_RIGHT      1.8%      2.7%     +0.9%
MOVE_BACK         1.4%      0.2%     -1.2%
```

---

### 5. Kill opportunity report

Frames where an enemy was visible but the player did NOT take the `ENGAGE_ENEMY` action:

```
Round  Frame   Enemy visible   Player action    Agent said      Reward gap
  3    00182   confidence 0.94  MOVE_FORWARD    ENGAGE_ENEMY    -0.63
  3    00183   confidence 0.91  MOVE_FORWARD    ENGAGE_ENEMY    -0.63
  5    00341   confidence 0.87  REPOSITION      ENGAGE_ENEMY    -0.55
  7    00521   confidence 0.95  THROW_SMOKE     ENGAGE_ENEMY    -0.60
  9    00712   confidence 0.89  MOVE_BACK       ENGAGE_ENEMY    -0.58
─────────────────────────────────────────────────────────────────────────
Total missed engagements: 23  across 15 rounds
```

---

### 6. Training progress (logged to TensorBoard / W&B)

```
Step      ep_rew_mean   policy_entropy   value_loss   approx_kl
 2048        -8.41          2.771           0.412       0.0021
 4096        -5.23          2.694           0.387       0.0034
 8192        +1.87          2.531           0.301       0.0041
16384       +12.44          2.318           0.245       0.0039
32768       +24.91          2.107           0.198       0.0028
65536       +31.22          1.983           0.171       0.0025
```

`ep_rew_mean` rising from negative toward ~+30 over training is the primary health signal.  
`policy_entropy` staying above `0.5` confirms the agent keeps exploring diverse actions.
