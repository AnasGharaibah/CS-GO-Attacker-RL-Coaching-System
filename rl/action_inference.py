import math
from .actions import Action

_STILL        = 0.005
_RUSH_THRESH  = 0.04
_LAT_RATIO    = 1.5
_AIM_STABLE   = 0.03

_UTIL_MAP = [
    ("smoke",   Action.THROW_SMOKE),
    ("flash",   Action.THROW_FLASH),
    ("he",      Action.THROW_HE),
    ("molotov", Action.THROW_HE),
]


def _delta(state, prev):
    px = float(state.get("player", {}).get("position_x", 0))
    py = float(state.get("player", {}).get("position_y", 0))
    ppx = float(prev.get("player", {}).get("position_x", px))
    ppy = float(prev.get("player", {}).get("position_y", py))
    return px - ppx, py - ppy


def _aim_delta(state, prev):
    cx  = float(state.get("player", {}).get("crosshair_x", 0))
    cy  = float(state.get("player", {}).get("crosshair_y", 0))
    pcx = float(prev.get("player", {}).get("crosshair_x", cx))
    pcy = float(prev.get("player", {}).get("crosshair_y", cy))
    return math.hypot(cx - pcx, cy - pcy)


def _aim_on_enemy(state):
    cx = float(state.get("player", {}).get("crosshair_x", -1))
    cy = float(state.get("player", {}).get("crosshair_y", -1))
    for e in state.get("enemies", []):
        if not e.get("is_visible", False):
            continue
        bx, by = float(e.get("bbox_x", 0)), float(e.get("bbox_y", 0))
        hw, hh = float(e.get("bbox_w", 0)) / 2, float(e.get("bbox_h", 0)) / 2
        if (bx - hw <= cx <= bx + hw) and (by - hh <= cy <= by + hh):
            return True
    return False


def _enemy_visible(state):
    return any(e.get("is_visible", False) for e in state.get("enemies", []))


def _util_thrown(state, prev):
    prev_u = str(prev.get("player", {}).get("utility_held", "none"))
    curr_u = str(state.get("player", {}).get("utility_held", "none"))
    if prev_u != "none" and curr_u == "none":
        for name, action in _UTIL_MAP:
            if name in prev_u:
                return action
        return Action.THROW_HE
    return None


def infer_action(state: dict, prev_state: dict | None = None) -> int:
    if prev_state is None:
        return int(Action.MOVE_FORWARD)

    events = state.get("game_events", {}) or {}

    prev_planted = prev_state.get("bomb", {}).get("planted", False)
    curr_planted = state.get("bomb", {}).get("planted", False)
    if curr_planted and not prev_planted:
        return int(Action.PLANT_BOMB)

    if (events.get("kill_this_frame") or _aim_on_enemy(state)) and _enemy_visible(state):
        return int(Action.ENGAGE_ENEMY)

    util_action = _util_thrown(state, prev_state)
    if util_action is not None:
        return int(util_action)

    dx, dy = _delta(state, prev_state)
    dist = math.hypot(dx, dy)

    if dist >= _RUSH_THRESH and dy < 0:
        return int(Action.RUSH_SITE)

    if dist < _STILL and not _enemy_visible(state):
        return int(Action.CROUCH_HOLD)

    if dist < _RUSH_THRESH and _enemy_visible(state) and abs(dx) > abs(dy):
        return int(Action.PEEK_CORNER)

    if dist < _STILL and _aim_delta(state, prev_state) < _AIM_STABLE:
        return int(Action.HOLD_ANGLE)

    if dist >= _STILL and not _enemy_visible(state):
        return int(Action.REPOSITION)

    if dist >= _STILL and abs(dx) >= abs(dy) * _LAT_RATIO:
        return int(Action.STRAFE_LEFT if dx < 0 else Action.STRAFE_RIGHT)

    if dy > 0:
        return int(Action.MOVE_BACK)

    return int(Action.MOVE_FORWARD)


def label_episode(frames: list[dict]) -> list[int]:
    return [infer_action(f, frames[i - 1] if i > 0 else None) for i, f in enumerate(frames)]
