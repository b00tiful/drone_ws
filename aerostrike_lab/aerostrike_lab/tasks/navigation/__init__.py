"""Gym registration for the AeroStrike navigation task."""

from __future__ import annotations

import gymnasium as gym

from . import agents
from aerostrike_lab.tasks.navigation.nav_env_cfg import (
    AEROSTRIKE_NAVIGATION_SETTINGS,
    AEROSTRIKE_NAVIGATION_V2_SAFE_CAPTURE_SETTINGS,
    AEROSTRIKE_NAVIGATION_V2_SETTINGS,
)


TASK_ID = AEROSTRIKE_NAVIGATION_SETTINGS.task_id
V2_TASK_ID = AEROSTRIKE_NAVIGATION_V2_SETTINGS.task_id
V2_SAFE_CAPTURE_TASK_ID = AEROSTRIKE_NAVIGATION_V2_SAFE_CAPTURE_SETTINGS.task_id


gym.register(
    id=TASK_ID,
    entry_point="aerostrike_lab.tasks.navigation.nav_env:AeroStrikeNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "aerostrike_lab.tasks.navigation.nav_env_cfg:AeroStrikeNavigationEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id=V2_TASK_ID,
    entry_point="aerostrike_lab.tasks.navigation.nav_env:AeroStrikeNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "aerostrike_lab.tasks.navigation.nav_env_cfg:AeroStrikeNavigationV2EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_v2_cfg.yaml",
    },
)

gym.register(
    id=V2_SAFE_CAPTURE_TASK_ID,
    entry_point="aerostrike_lab.tasks.navigation.nav_env:AeroStrikeNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aerostrike_lab.tasks.navigation.nav_env_cfg:AeroStrikeNavigationV2SafeCaptureEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_v2_safe_capture_cfg.yaml",
    },
)

__all__ = ["TASK_ID", "V2_TASK_ID", "V2_SAFE_CAPTURE_TASK_ID"]
