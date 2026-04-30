from __future__ import annotations

import random
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .observation import OBS_DIM, encode_state
from .reward import compute_attacker_reward
from .actions import get_action_mask, NUM_ACTIONS


class CSGOAttackerEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        episodes: list[list[dict]],
        max_episode_steps: int = 1800,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if not episodes:
            raise ValueError("Need at least one episode.")

        self._episodes          = episodes
        self._max_episode_steps = max_episode_steps
        self._rng               = random.Random(seed)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(NUM_ACTIONS)

        self._frames:     list[dict]  = []
        self._frame_idx:  int         = 0
        self._prev_frame: dict | None = None
        self._obs:        np.ndarray  = np.zeros(OBS_DIM, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        self._frames     = self._rng.choice(self._episodes)
        self._frame_idx  = 0
        self._prev_frame = None
        self._obs        = encode_state(self._frames[0])
        return self._obs.copy(), self._info()

    def step(self, action: int):
        frame      = self._frames[self._frame_idx]
        terminated = self._frame_idx >= len(self._frames) - 1
        truncated  = self._frame_idx >= self._max_episode_steps - 1

        reward = compute_attacker_reward(
            state=frame,
            prev_state=self._prev_frame,
            done=terminated or truncated,
        )

        self._prev_frame = frame

        if not (terminated or truncated):
            self._frame_idx += 1
            self._obs = encode_state(self._frames[self._frame_idx])

        return self._obs.copy(), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        util = str(self._frames[self._frame_idx].get("player", {}).get("utility_held", "none"))
        return get_action_mask(util)

    def render(self) -> dict:
        frame = self._frames[self._frame_idx]
        return {
            "frame_id":    frame.get("frame_id"),
            "frame_index": self._frame_idx,
            "obs":         self._obs.tolist(),
            "action_mask": self.action_masks().tolist(),
        }

    def close(self) -> None:
        pass

    def _info(self) -> dict[str, Any]:
        frame = self._frames[self._frame_idx]
        return {
            "frame_id":    frame.get("frame_id"),
            "round":       frame.get("round_number"),
            "frame_index": self._frame_idx,
            "episode_len": len(self._frames),
        }

    @property
    def current_frame(self) -> dict:
        return self._frames[self._frame_idx]
