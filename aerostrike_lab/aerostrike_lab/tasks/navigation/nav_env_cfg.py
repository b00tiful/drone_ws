"""Isaac Lab configuration for the AeroStrike navigation DirectRLEnv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from aerostrike_lab.assets.quadrotor import make_quadrotor_cfg

try:
    import yaml
except ImportError:  # pragma: no cover - Isaac Lab environments normally include PyYAML.
    yaml = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "navigation.yaml"

DEFAULT_TASK_ID = "AeroStrike-Navigation-Direct-v0"
DEFAULT_NUM_ENVS = 16
DEFAULT_ENV_SPACING_M = 25.0
DEFAULT_PHYSICS_DT_S = 0.005
DEFAULT_DECIMATION = 4
DEFAULT_EPISODE_LENGTH_S = 10.0
DEFAULT_ACTION_SPACE = 3
DEFAULT_OBSERVATION_SPACE = 41
DEFAULT_STATE_SPACE = 0
DEFAULT_ROBOT_PRIM_PATH = "/World/envs/env_.*/Robot"
DEFAULT_WAREHOUSE_ROOT_PRIM_PATH = "/World/envs/env_0/Warehouse"
DEFAULT_WAREHOUSE_MESH_PRIM_EXPR = "/World/envs/env_.*/Warehouse"
DEFAULT_LAYOUT_SEED = 7
DEFAULT_ACTION_VELOCITY_LIMIT_MPS = 5.0
DEFAULT_START_HEIGHT_M = 0.5
DEFAULT_GOAL_HEIGHT_M = 1.5
DEFAULT_MIN_HEIGHT_M = 0.1
DEFAULT_MAX_HEIGHT_M = 4.0
DEFAULT_GOAL_DISTANCE_NORMALIZER_M = 20.0
DEFAULT_HOVER_THRUST_SCALE = 1.0
DEFAULT_GOAL_RADIUS_M = 0.75
DEFAULT_COLLISION_DISTANCE_M = 0.35
DEFAULT_PROXIMITY_DISTANCE_M = 1.5
DEFAULT_PROGRESS_WEIGHT = 8.0
DEFAULT_FORWARD_VELOCITY_WEIGHT = 0.5
DEFAULT_PROXIMITY_PENALTY_WEIGHT = 2.0
DEFAULT_COLLISION_PENALTY = 10.0
DEFAULT_SUCCESS_BONUS = 20.0
DEFAULT_INSTABILITY_PENALTY_WEIGHT = 0.05
DEFAULT_ACTION_SMOOTHNESS_PENALTY_WEIGHT = 0.02
DEFAULT_ALIVE_PENALTY = 0.01
DEFAULT_TARGET_SPEED_MPS = 3.0
DEFAULT_DEBUG_VIS = False


@dataclass(frozen=True)
class NavigationSettings:
    """YAML-backed navigation environment settings."""

    task_id: str = DEFAULT_TASK_ID
    num_envs: int = DEFAULT_NUM_ENVS
    env_spacing_m: float = DEFAULT_ENV_SPACING_M
    physics_dt_s: float = DEFAULT_PHYSICS_DT_S
    decimation: int = DEFAULT_DECIMATION
    episode_length_s: float = DEFAULT_EPISODE_LENGTH_S
    action_space: int = DEFAULT_ACTION_SPACE
    observation_space: int = DEFAULT_OBSERVATION_SPACE
    state_space: int = DEFAULT_STATE_SPACE
    robot_prim_path: str = DEFAULT_ROBOT_PRIM_PATH
    warehouse_root_prim_path: str = DEFAULT_WAREHOUSE_ROOT_PRIM_PATH
    warehouse_mesh_prim_expr: str = DEFAULT_WAREHOUSE_MESH_PRIM_EXPR
    layout_seed: int = DEFAULT_LAYOUT_SEED
    action_velocity_limit_mps: float = DEFAULT_ACTION_VELOCITY_LIMIT_MPS
    start_height_m: float = DEFAULT_START_HEIGHT_M
    goal_height_m: float = DEFAULT_GOAL_HEIGHT_M
    min_height_m: float = DEFAULT_MIN_HEIGHT_M
    max_height_m: float = DEFAULT_MAX_HEIGHT_M
    goal_distance_normalizer_m: float = DEFAULT_GOAL_DISTANCE_NORMALIZER_M
    hover_thrust_scale: float = DEFAULT_HOVER_THRUST_SCALE
    goal_radius_m: float = DEFAULT_GOAL_RADIUS_M
    collision_distance_m: float = DEFAULT_COLLISION_DISTANCE_M
    proximity_distance_m: float = DEFAULT_PROXIMITY_DISTANCE_M
    progress_weight: float = DEFAULT_PROGRESS_WEIGHT
    forward_velocity_weight: float = DEFAULT_FORWARD_VELOCITY_WEIGHT
    proximity_penalty_weight: float = DEFAULT_PROXIMITY_PENALTY_WEIGHT
    collision_penalty: float = DEFAULT_COLLISION_PENALTY
    success_bonus: float = DEFAULT_SUCCESS_BONUS
    instability_penalty_weight: float = DEFAULT_INSTABILITY_PENALTY_WEIGHT
    action_smoothness_penalty_weight: float = DEFAULT_ACTION_SMOOTHNESS_PENALTY_WEIGHT
    alive_penalty: float = DEFAULT_ALIVE_PENALTY
    target_speed_mps: float = DEFAULT_TARGET_SPEED_MPS
    debug_vis: bool = DEFAULT_DEBUG_VIS


def _as_section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def load_navigation_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> NavigationSettings:
    """Load navigation settings from YAML, falling back to safe code defaults."""
    path = Path(config_path)
    if yaml is None or not path.exists():
        return NavigationSettings()

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    env = _as_section(data, "env")
    sim = _as_section(data, "sim")
    scene = _as_section(data, "scene")
    robot = _as_section(data, "robot")
    warehouse = _as_section(data, "warehouse")
    action = _as_section(data, "action")
    reset = _as_section(data, "reset")
    observation = _as_section(data, "observation")
    control = _as_section(data, "control")
    termination = _as_section(data, "termination")
    reward = _as_section(data, "reward")

    return NavigationSettings(
        task_id=str(env.get("task_id", DEFAULT_TASK_ID)),
        num_envs=int(scene.get("num_envs", DEFAULT_NUM_ENVS)),
        env_spacing_m=float(scene.get("env_spacing_m", DEFAULT_ENV_SPACING_M)),
        physics_dt_s=float(sim.get("physics_dt_s", DEFAULT_PHYSICS_DT_S)),
        decimation=int(sim.get("decimation", DEFAULT_DECIMATION)),
        episode_length_s=float(env.get("episode_length_s", DEFAULT_EPISODE_LENGTH_S)),
        action_space=int(env.get("action_space", DEFAULT_ACTION_SPACE)),
        observation_space=int(env.get("observation_space", DEFAULT_OBSERVATION_SPACE)),
        state_space=int(env.get("state_space", DEFAULT_STATE_SPACE)),
        robot_prim_path=str(robot.get("prim_path", DEFAULT_ROBOT_PRIM_PATH)),
        warehouse_root_prim_path=str(warehouse.get("root_prim_path", DEFAULT_WAREHOUSE_ROOT_PRIM_PATH)),
        warehouse_mesh_prim_expr=str(warehouse.get("mesh_prim_expr", DEFAULT_WAREHOUSE_MESH_PRIM_EXPR)),
        layout_seed=int(warehouse.get("layout_seed", DEFAULT_LAYOUT_SEED)),
        action_velocity_limit_mps=float(action.get("velocity_limit_mps", DEFAULT_ACTION_VELOCITY_LIMIT_MPS)),
        start_height_m=float(reset.get("start_height_m", DEFAULT_START_HEIGHT_M)),
        goal_height_m=float(reset.get("goal_height_m", DEFAULT_GOAL_HEIGHT_M)),
        min_height_m=float(reset.get("min_height_m", DEFAULT_MIN_HEIGHT_M)),
        max_height_m=float(reset.get("max_height_m", DEFAULT_MAX_HEIGHT_M)),
        goal_distance_normalizer_m=float(
            observation.get("goal_distance_normalizer_m", DEFAULT_GOAL_DISTANCE_NORMALIZER_M)
        ),
        hover_thrust_scale=float(control.get("hover_thrust_scale", DEFAULT_HOVER_THRUST_SCALE)),
        goal_radius_m=float(termination.get("goal_radius_m", DEFAULT_GOAL_RADIUS_M)),
        collision_distance_m=float(termination.get("collision_distance_m", DEFAULT_COLLISION_DISTANCE_M)),
        proximity_distance_m=float(termination.get("proximity_distance_m", DEFAULT_PROXIMITY_DISTANCE_M)),
        progress_weight=float(reward.get("progress_weight", DEFAULT_PROGRESS_WEIGHT)),
        forward_velocity_weight=float(reward.get("forward_velocity_weight", DEFAULT_FORWARD_VELOCITY_WEIGHT)),
        proximity_penalty_weight=float(reward.get("proximity_penalty_weight", DEFAULT_PROXIMITY_PENALTY_WEIGHT)),
        collision_penalty=float(reward.get("collision_penalty", DEFAULT_COLLISION_PENALTY)),
        success_bonus=float(reward.get("success_bonus", DEFAULT_SUCCESS_BONUS)),
        instability_penalty_weight=float(reward.get("instability_penalty_weight", DEFAULT_INSTABILITY_PENALTY_WEIGHT)),
        action_smoothness_penalty_weight=float(
            reward.get("action_smoothness_penalty_weight", DEFAULT_ACTION_SMOOTHNESS_PENALTY_WEIGHT)
        ),
        alive_penalty=float(reward.get("alive_penalty", DEFAULT_ALIVE_PENALTY)),
        target_speed_mps=float(reward.get("target_speed_mps", DEFAULT_TARGET_SPEED_MPS)),
        debug_vis=bool(env.get("debug_vis", DEFAULT_DEBUG_VIS)),
    )


AEROSTRIKE_NAVIGATION_SETTINGS = load_navigation_settings()
_NAV_SETTINGS = AEROSTRIKE_NAVIGATION_SETTINGS


@configclass
class AeroStrikeNavigationEnvCfg(DirectRLEnvCfg):
    """Configuration for the AeroStrike DirectRLEnv skeleton."""

    episode_length_s = _NAV_SETTINGS.episode_length_s
    decimation = _NAV_SETTINGS.decimation
    action_space = _NAV_SETTINGS.action_space
    observation_space = _NAV_SETTINGS.observation_space
    state_space = _NAV_SETTINGS.state_space
    debug_vis = _NAV_SETTINGS.debug_vis
    num_rerenders_on_reset = 1

    sim: SimulationCfg = SimulationCfg(
        dt=_NAV_SETTINGS.physics_dt_s,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=_NAV_SETTINGS.num_envs,
        env_spacing=_NAV_SETTINGS.env_spacing_m,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot: ArticulationCfg = make_quadrotor_cfg(prim_path=_NAV_SETTINGS.robot_prim_path)
    warehouse_root_prim_path = _NAV_SETTINGS.warehouse_root_prim_path
    warehouse_mesh_prim_expr = _NAV_SETTINGS.warehouse_mesh_prim_expr
    warehouse_layout_seed = _NAV_SETTINGS.layout_seed
    action_velocity_limit_mps = _NAV_SETTINGS.action_velocity_limit_mps
    start_height_m = _NAV_SETTINGS.start_height_m
    goal_height_m = _NAV_SETTINGS.goal_height_m
    min_height_m = _NAV_SETTINGS.min_height_m
    max_height_m = _NAV_SETTINGS.max_height_m
    goal_distance_normalizer_m = _NAV_SETTINGS.goal_distance_normalizer_m
    hover_thrust_scale = _NAV_SETTINGS.hover_thrust_scale
    goal_radius_m = _NAV_SETTINGS.goal_radius_m
    collision_distance_m = _NAV_SETTINGS.collision_distance_m
    proximity_distance_m = _NAV_SETTINGS.proximity_distance_m
    progress_weight = _NAV_SETTINGS.progress_weight
    forward_velocity_weight = _NAV_SETTINGS.forward_velocity_weight
    proximity_penalty_weight = _NAV_SETTINGS.proximity_penalty_weight
    collision_penalty = _NAV_SETTINGS.collision_penalty
    success_bonus = _NAV_SETTINGS.success_bonus
    instability_penalty_weight = _NAV_SETTINGS.instability_penalty_weight
    action_smoothness_penalty_weight = _NAV_SETTINGS.action_smoothness_penalty_weight
    alive_penalty = _NAV_SETTINGS.alive_penalty
    target_speed_mps = _NAV_SETTINGS.target_speed_mps


__all__ = [
    "AEROSTRIKE_NAVIGATION_SETTINGS",
    "AeroStrikeNavigationEnvCfg",
    "NavigationSettings",
    "load_navigation_settings",
]
