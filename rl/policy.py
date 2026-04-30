from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .observation import OBS_DIM
from .actions import NUM_ACTIONS


class AttackerPolicyNetwork(nn.Module):

    def __init__(
        self,
        obs_dim:     int = OBS_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden_dim:  int = 256,
        head_dim:    int = 128,
    ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

        self._num_actions = num_actions
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        features = self.trunk(obs)
        logits   = self.policy_head(features)
        value    = self.value_head(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits, value

    def get_action_probs(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        logits, _ = self.forward(obs, action_mask)
        return F.softmax(logits, dim=-1)

    def get_value(self, obs: torch.Tensor):
        return self.value_head(self.trunk(obs))

    def act(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None, deterministic: bool = False):
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor | None = None):
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy().mean()


def make_policy_kwargs(hidden_dim: int = 256, head_dim: int = 128) -> dict:
    return {
        "net_arch": {
            "pi": [hidden_dim, hidden_dim, head_dim],
            "vf": [hidden_dim, hidden_dim, head_dim],
        },
        "activation_fn": nn.ReLU,
    }
