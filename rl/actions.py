from enum import IntEnum
import numpy as np


class Action(IntEnum):
    MOVE_FORWARD = 0
    MOVE_BACK    = 1
    STRAFE_LEFT  = 2
    STRAFE_RIGHT = 3
    CROUCH_HOLD  = 4
    PEEK_CORNER  = 5
    ENGAGE_ENEMY = 6
    REPOSITION   = 7
    THROW_SMOKE  = 8
    THROW_FLASH  = 9
    THROW_HE     = 10
    PLANT_BOMB   = 11
    RUSH_SITE    = 12
    HOLD_ANGLE   = 13
    DEFUSE_BOMB  = 14  # defender only
    ROTATE_SITE  = 15  # defender only


NUM_ACTIONS = 16

ATTACKER_VALID_ACTIONS = list(range(14))

_UTILITY_ACTIONS = (Action.THROW_SMOKE, Action.THROW_FLASH, Action.THROW_HE)

_WEAPON_TO_IDX  = {"rifle": 0, "pistol": 1, "sniper": 2, "knife": 3}
_UTILITY_TO_IDX = {"smoke": 0, "flash": 1, "he": 2, "molotov": 3, "none": 4}


def get_action_mask(utility_held: str = "none") -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[ATTACKER_VALID_ACTIONS] = True
    mask[Action.DEFUSE_BOMB] = False
    mask[Action.ROTATE_SITE] = False
    if utility_held == "none":
        for a in _UTILITY_ACTIONS:
            mask[a] = False
    return mask
