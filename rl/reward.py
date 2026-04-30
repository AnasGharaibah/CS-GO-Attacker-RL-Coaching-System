R_WIN            =  15.0
R_LOSS           = -10.0
R_KILL           =   3.0
R_DEATH          =  -4.0
R_ASSIST         =   0.5
R_BOMB_PLANT     =   5.0
R_CROSSHAIR_GOOD =   0.1
R_CROSSHAIR_POOR = -0.05
R_POSITION_GOOD  =   0.02
R_TIME_PENALTY   =  -0.01


def _crosshair_on_enemy(state: dict):
    cx = float(state.get("player", {}).get("crosshair_x", -1))
    cy = float(state.get("player", {}).get("crosshair_y", -1))
    visible = False
    for e in state.get("enemies", []):
        if not e.get("is_visible", False):
            continue
        visible = True
        bx, by = float(e.get("bbox_x", 0)), float(e.get("bbox_y", 0))
        hw, hh = float(e.get("bbox_w", 0)) / 2, float(e.get("bbox_h", 0)) / 2
        if (bx - hw <= cx <= bx + hw) and (by - hh <= cy <= by + hh):
            return True
    return False if visible else None


def _bomb_just_planted(state: dict, prev: dict) -> bool:
    return state.get("bomb", {}).get("planted", False) and \
           not (prev or {}).get("bomb", {}).get("planted", False)


def compute_attacker_reward(state: dict, prev_state: dict | None = None, done: bool = False) -> float:
    prev_state = prev_state or {}
    events = state.get("game_events", {}) or {}
    r = 0.0

    if done:
        if events.get("round_won"):  r += R_WIN
        if events.get("round_lost"): r += R_LOSS

    if events.get("kill_this_frame"):   r += R_KILL
    if events.get("death_this_frame"):  r += R_DEATH
    if events.get("assist_this_frame"): r += R_ASSIST

    if _bomb_just_planted(state, prev_state):
        r += R_BOMB_PLANT

    overlap = _crosshair_on_enemy(state)
    if overlap is True:
        r += R_CROSSHAIR_GOOD
    elif overlap is False:
        r += R_CROSSHAIR_POOR

    player = state.get("player", {}) or {}
    if player.get("alive", False) and float(player.get("position_y", 1.0)) < 0.5:
        r += R_POSITION_GOOD

    r += R_TIME_PENALTY
    return r
