import numpy as np

OBS_DIM = 62

# obs layout:
# [0:7]   player state
# [7:16]  weapon + utility one-hots
# [16:51] up to 5 enemies, 7 values each
# [51:57] bomb state
# [57:60] utility active
# [60:62] time

_WEAPON_IDX  = {"rifle": 0, "pistol": 1, "sniper": 2, "knife": 3}
_UTILITY_IDX = {"smoke": 0, "flash": 1, "he": 2, "molotov": 3, "none": 4}
_SITE_IDX    = {"A": 0, "B": 1, "unknown": 2}

_MAX_ENEMIES    = 5
_BOMB_TIMER_MAX = 40.0
_ROUND_MAX      = 30
_TIMESTAMP_MAX  = 60.0


def _one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def encode_state(state: dict) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    player  = state.get("player", {}) or {}
    enemies = state.get("enemies", []) or []
    bomb    = state.get("bomb", {}) or {}
    utility = state.get("utility_active", {}) or {}

    obs[0] = float(player.get("position_x", 0.0))
    obs[1] = float(player.get("position_y", 0.0))
    obs[2] = float(player.get("crosshair_x", 0.0))
    obs[3] = float(player.get("crosshair_y", 0.0))
    obs[4] = float(player.get("health", 0)) / 100.0
    obs[5] = float(player.get("armor",  0)) / 100.0
    obs[6] = 1.0 if player.get("alive", False) else 0.0

    weapon_idx  = _WEAPON_IDX.get(str(player.get("weapon", "knife")), 3)
    utility_idx = _UTILITY_IDX.get(str(player.get("utility_held", "none")), 4)
    obs[7:11]  = _one_hot(weapon_idx, 4)
    obs[11:16] = _one_hot(utility_idx, 5)

    sorted_enemies = sorted(
        enemies,
        key=lambda e: float(e.get("confidence", 0.0)),
        reverse=True,
    )[:_MAX_ENEMIES]

    for i, enemy in enumerate(sorted_enemies):
        base = 16 + i * 7
        obs[base + 0] = float(enemy.get("bbox_x", 0.0))
        obs[base + 1] = float(enemy.get("bbox_y", 0.0))
        obs[base + 2] = float(enemy.get("bbox_w", 0.0))
        obs[base + 3] = float(enemy.get("bbox_h", 0.0))
        obs[base + 4] = float(enemy.get("distance_est", 0.0))
        obs[base + 5] = 1.0 if enemy.get("is_visible", False) else 0.0
        obs[base + 6] = float(enemy.get("confidence", 0.0))

    site_idx = _SITE_IDX.get(str(bomb.get("site", "unknown")), 2)
    obs[51] = 1.0 if bomb.get("detected", False) else 0.0
    obs[52:55] = _one_hot(site_idx, 3)
    obs[55] = 1.0 if bomb.get("planted", False) else 0.0
    timer = float(bomb.get("timer_remaining", -1))
    obs[56] = max(timer, 0.0) / _BOMB_TIMER_MAX

    obs[57] = min(float(utility.get("smoke_count", 0)), 5.0) / 5.0
    obs[58] = 1.0 if utility.get("flash_active", False) else 0.0
    obs[59] = 1.0 if utility.get("fire_active",  False) else 0.0

    obs[60] = min(float(state.get("timestamp_sec", 0.0)), _TIMESTAMP_MAX) / _TIMESTAMP_MAX
    obs[61] = min(float(state.get("round_number",  1)),   _ROUND_MAX)     / _ROUND_MAX

    return obs
