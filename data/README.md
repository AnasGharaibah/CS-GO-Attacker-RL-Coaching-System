# data/

Output folder for processed datasets, replay buffers, and trained models.

All files here are **generated** — do not edit them by hand. Delete and regenerate with `prepare.py` and `rl/train.py` if needed.

---

## What gets saved here

```
data/
├── attacker_buffer.h5         ← replay buffer (default output of prepare.py)
├── attacker_buffer.splits.json← train/val/test index split
└── models/
    ├── checkpoints/           ← periodic PPO checkpoints
    │   └── attacker_ppo_<N>_steps.zip
    ├── best/
    │   └── best_model.zip     ← best model by eval reward
    ├── eval_logs/             ← EvalCallback result CSVs
    └── attacker_ppo_final.zip ← final trained model
```

---

## Replay buffer (`attacker_buffer.h5`)

HDF5 file containing every `(obs, action, reward, next_obs, done)` transition built from your recorded matches.

| Dataset | Shape | dtype | Description |
|---|---|---|---|
| `obs` | `(N, 62)` | float32 | Observation vectors |
| `actions` | `(N,)` | int32 | Action IDs (0-15) |
| `rewards` | `(N,)` | float32 | Per-frame rewards |
| `next_obs` | `(N, 62)` | float32 | Next observation vectors |
| `dones` | `(N,)` | bool | Episode terminal flags |

Load it in Python:
```python
from rl.replay_buffer import ReplayBuffer
buf = ReplayBuffer.load("data/attacker_buffer.h5")
obs, actions, rewards, next_obs, dones = buf.get_all()
print(f"Transitions: {len(buf)}")
```

---

## Split file (`attacker_buffer.splits.json`)

```json
{
  "total": 150000,
  "train": [0, 3, 7, ...],
  "val":   [1, 5, 9, ...],
  "test":  [2, 4, 6, ...],
  "train_frac": 0.7,
  "val_frac":   0.15,
  "test_frac":  0.15
}
```

---

## Trained model (`models/attacker_ppo_final.zip`)

SB3 `MaskablePPO` checkpoint. Load and run predictions:

```python
from sb3_contrib import MaskablePPO
from rl.env import CSGOAttackerEnv

episodes = [...]   # list of episode frame sequences
env      = CSGOAttackerEnv(episodes)
model    = MaskablePPO.load("data/models/attacker_ppo_final", env=env)

obs, _  = env.reset()
action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
```

---

## Regenerating

```bash
# Rebuild buffer from recordings
python prepare.py

# Retrain from scratch
python -m rl.train --steps 500000
```
