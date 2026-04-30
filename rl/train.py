from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import CSGOAttackerEnv
from .policy import make_policy_kwargs
from .replay_buffer import ReplayBuffer


DEFAULTS = dict(
    learning_rate = 3e-4,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    vf_coef       = 0.5,
    batch_size    = 64,
    n_steps       = 2048,
    n_epochs      = 10,
    max_grad_norm = 0.5,
    verbose       = 1,
)

MODEL_DIR = Path("rl/models")


def split_episodes(episodes, train_frac=0.70, val_frac=0.15, seed=42):
    rng = random.Random(seed)
    eps = episodes.copy()
    rng.shuffle(eps)
    n     = len(eps)
    t_end = int(n * train_frac)
    v_end = t_end + int(n * val_frac)
    return eps[:t_end], eps[t_end:v_end], eps[v_end:]


def _try_wandb(config: dict) -> bool:
    try:
        import wandb
        wandb.init(project="csgo-attacker-rl", config=config, name="attacker-ppo")
        return True
    except ImportError:
        return False


def _linear_schedule(lr: float):
    return lambda progress: lr * progress


def _load_episodes_from_dir(path: str) -> list[list[dict]]:
    episodes = []
    for fpath in sorted(Path(path).glob("*.json")):
        with open(fpath) as f:
            ep = json.load(f)
        if isinstance(ep, list) and ep:
            episodes.append(ep)
    if not episodes:
        print(f"[train] No episode JSON files found in {path!r}.")
        sys.exit(1)
    return episodes


def _make_dummy_episodes(n: int = 10, length: int = 30) -> list[list[dict]]:
    def frame(i):
        return {
            "frame_id": i,
            "timestamp_sec": i / 30.0,
            "round_number": 1,
            "map_name": "de_dust2",
            "player": {
                "role": "attacker",
                "position_x": float(np.random.rand()),
                "position_y": float(np.random.rand()),
                "crosshair_x": float(np.random.rand()),
                "crosshair_y": float(np.random.rand()),
                "health": 100,
                "armor": 100,
                "alive": True,
                "weapon": "rifle",
                "utility_held": "none",
            },
            "enemies": [],
            "bomb": {
                "detected": False,
                "site": "unknown",
                "planted": i == length - 1,
                "timer_remaining": -1,
            },
            "utility_active": {"smoke_count": 0, "flash_active": False, "fire_active": False},
            "game_events": {
                "kill_this_frame": False,
                "death_this_frame": False,
                "assist_this_frame": False,
                "round_won": i == length - 1,
                "round_lost": False,
            },
        }
    return [[frame(i) for i in range(length)] for _ in range(n)]


def _tb_available() -> bool:
    try:
        import tensorboard
        return True
    except Exception:
        return False


def train(
    episodes_train,
    episodes_val=None,
    total_steps=500_000,
    use_wandb=False,
    checkpoint_freq=10_000,
    seed=42,
) -> MaskablePPO:
    np.random.seed(seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = {**DEFAULTS, "total_steps": total_steps, "seed": seed}
    if use_wandb:
        _try_wandb(config)

    train_env = DummyVecEnv([lambda: Monitor(CSGOAttackerEnv(episodes_train, seed=seed))])

    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(MODEL_DIR / "checkpoints"),
            name_prefix="attacker_ppo",
        )
    ]

    if episodes_val:
        eval_env = DummyVecEnv([lambda: Monitor(CSGOAttackerEnv(episodes_val, seed=seed + 1))])
        callbacks.append(
            EvalCallback(
                eval_env,
                eval_freq=checkpoint_freq,
                best_model_save_path=str(MODEL_DIR / "best"),
                log_path=str(MODEL_DIR / "eval_logs"),
                deterministic=True,
            )
        )

    model = MaskablePPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = _linear_schedule(DEFAULTS["learning_rate"]),
        gamma         = DEFAULTS["gamma"],
        gae_lambda    = DEFAULTS["gae_lambda"],
        clip_range    = DEFAULTS["clip_range"],
        ent_coef      = DEFAULTS["ent_coef"],
        vf_coef       = DEFAULTS["vf_coef"],
        batch_size    = DEFAULTS["batch_size"],
        n_steps       = DEFAULTS["n_steps"],
        n_epochs      = DEFAULTS["n_epochs"],
        max_grad_norm = DEFAULTS["max_grad_norm"],
        policy_kwargs = make_policy_kwargs(),
        verbose       = DEFAULTS["verbose"],
        seed          = seed,
        tensorboard_log = str(MODEL_DIR / "tb_logs") if _tb_available() else None,
    )

    print(f"[train] Starting — {total_steps:,} steps")
    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)

    final_path = str(MODEL_DIR / "attacker_ppo_final")
    model.save(final_path)
    print(f"[train] Saved → {final_path}.zip")
    return model


def _parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group()
    src.add_argument("--episodes",   type=str)
    src.add_argument("--smoke-test", action="store_true")
    p.add_argument("--steps",           type=int,   default=500_000)
    p.add_argument("--wandb",           action="store_true")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--checkpoint-freq", type=int,   default=10_000)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.smoke_test:
        print("[train] Smoke test — 10 dummy episodes")
        episodes = _make_dummy_episodes(n=10, length=30)
    elif args.episodes:
        episodes = _load_episodes_from_dir(args.episodes)
        print(f"[train] Loaded {len(episodes)} episodes from {args.episodes!r}")
    else:
        print("[train] Pass --episodes <dir> or --smoke-test")
        sys.exit(1)

    train_eps, val_eps, test_eps = split_episodes(episodes, seed=args.seed)
    print(f"[train] Split: {len(train_eps)} train / {len(val_eps)} val / {len(test_eps)} test")

    train(
        episodes_train  = train_eps,
        episodes_val    = val_eps,
        total_steps     = args.steps,
        use_wandb       = args.wandb,
        checkpoint_freq = args.checkpoint_freq,
        seed            = args.seed,
    )
    print("[train] Done.")
