"""Gym registration for the AeroStrike navigation task."""

from __future__ import annotations

import gymnasium as gym

from . import agents
from aerostrike_lab.tasks.navigation.nav_env_cfg import AEROSTRIKE_NAVIGATION_SETTINGS


TASK_ID = AEROSTRIKE_NAVIGATION_SETTINGS.task_id


gym.register(
    id=TASK_ID,
    entry_point="aerostrike_lab.tasks.navigation.nav_env:AeroStrikeNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "aerostrike_lab.tasks.navigation.nav_env_cfg:AeroStrikeNavigationEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


__all__ = ["TASK_ID"]
