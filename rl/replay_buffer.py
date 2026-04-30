from __future__ import annotations

import numpy as np

from .observation import OBS_DIM, encode_state
from .action_inference import label_episode
from .reward import compute_attacker_reward

try:
    import h5py
    _HAS_H5 = True
except Exception:
    _HAS_H5 = False

try:
    import pyarrow.parquet
    _HAS_PA = True
except Exception:
    _HAS_PA = False


class ReplayBuffer:

    def __init__(self, capacity: int = 500_000) -> None:
        self.capacity = capacity
        self._ptr  = 0
        self.size  = 0

        self._obs      = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self._actions  = np.zeros(capacity,            dtype=np.int32)
        self._rewards  = np.zeros(capacity,            dtype=np.float32)
        self._next_obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self._dones    = np.zeros(capacity,            dtype=bool)

    def add(self, obs, action, reward, next_obs, done):
        idx = self._ptr % self.capacity
        self._obs[idx]      = obs
        self._actions[idx]  = action
        self._rewards[idx]  = reward
        self._next_obs[idx] = next_obs
        self._dones[idx]    = done
        self._ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size < batch_size:
            raise ValueError(f"Buffer has {self.size} transitions, requested {batch_size}.")
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self._obs[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_obs[idx],
            self._dones[idx],
        )

    def get_all(self):
        return (
            self._obs[:self.size],
            self._actions[:self.size],
            self._rewards[:self.size],
            self._next_obs[:self.size],
            self._dones[:self.size],
        )

    def save(self, path: str):
        if _HAS_H5 and path.endswith(".h5"):
            self._save_h5(path)
        else:
            self._save_npz(path.replace(".h5", ".npz"))

    def _save_h5(self, path: str):
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("obs",      data=self._obs[:self.size],      compression="gzip")
            f.create_dataset("actions",  data=self._actions[:self.size],  compression="gzip")
            f.create_dataset("rewards",  data=self._rewards[:self.size],  compression="gzip")
            f.create_dataset("next_obs", data=self._next_obs[:self.size], compression="gzip")
            f.create_dataset("dones",    data=self._dones[:self.size],    compression="gzip")
            f.attrs["size"] = self.size

    def _save_npz(self, path: str):
        np.savez_compressed(
            path,
            obs=self._obs[:self.size],
            actions=self._actions[:self.size],
            rewards=self._rewards[:self.size],
            next_obs=self._next_obs[:self.size],
            dones=self._dones[:self.size],
        )

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        if _HAS_H5 and path.endswith(".h5"):
            return cls._load_h5(path)
        return cls._load_npz(path)

    @classmethod
    def _load_h5(cls, path: str) -> "ReplayBuffer":
        import h5py
        with h5py.File(path, "r") as f:
            obs      = f["obs"][:]
            actions  = f["actions"][:]
            rewards  = f["rewards"][:]
            next_obs = f["next_obs"][:]
            dones    = f["dones"][:]
        buf = cls(capacity=len(obs))
        buf._obs[:len(obs)]      = obs
        buf._actions[:len(obs)]  = actions
        buf._rewards[:len(obs)]  = rewards
        buf._next_obs[:len(obs)] = next_obs
        buf._dones[:len(obs)]    = dones
        buf.size = len(obs)
        buf._ptr = len(obs)
        return buf

    @classmethod
    def _load_npz(cls, path: str) -> "ReplayBuffer":
        data = np.load(path)
        n = len(data["obs"])
        buf = cls(capacity=n)
        buf._obs[:n]      = data["obs"]
        buf._actions[:n]  = data["actions"]
        buf._rewards[:n]  = data["rewards"]
        buf._next_obs[:n] = data["next_obs"]
        buf._dones[:n]    = data["dones"]
        buf.size = n
        buf._ptr = n
        return buf

    @classmethod
    def build_from_episodes(cls, episodes: list[list[dict]], capacity: int | None = None) -> "ReplayBuffer":
        total = sum(max(len(ep) - 1, 0) for ep in episodes)
        buf   = cls(capacity=capacity or max(total, 1))

        for ep in episodes:
            if len(ep) < 2:
                continue
            actions = label_episode(ep)
            for i in range(len(ep) - 1):
                obs      = encode_state(ep[i])
                next_obs = encode_state(ep[i + 1])
                reward   = compute_attacker_reward(
                    state=ep[i],
                    prev_state=ep[i - 1] if i > 0 else None,
                    done=i == len(ep) - 2,
                )
                buf.add(obs, actions[i], reward, next_obs, i == len(ep) - 2)

        return buf

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"ReplayBuffer(size={self.size}, capacity={self.capacity})"
