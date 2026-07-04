"""Quadrotor and onboard range-sensor configuration for AeroStrike."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.utils import configclass

try:
    import yaml
except ImportError:  # pragma: no cover - Isaac Lab environments normally include PyYAML.
    yaml = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "quadrotor.yaml"

DEFAULT_ROBOT_PRIM_PATH = "{ENV_REGEX_NS}/Robot"
DEFAULT_SENSOR_PRIM_SUFFIX = "body"
DEFAULT_SENSOR_MESH_PRIM_PATHS = ("/World/defaultGroundPlane",)
DEFAULT_RAY_COUNT = 24
DEFAULT_MIN_RANGE_M = 0.2
DEFAULT_MAX_RANGE_M = 10.0
DEFAULT_HORIZONTAL_FOV_DEG = (-150.0, 150.0)
DEFAULT_VERTICAL_FOV_DEG = (-15.0, 15.0)
DEFAULT_VERTICAL_CHANNELS = 3


@dataclass
class RaySensorSettings:
    """YAML-backed settings used to build the AeroStrike RayCaster."""

    prim_suffix: str = DEFAULT_SENSOR_PRIM_SUFFIX
    mesh_prim_paths: tuple[str, ...] = DEFAULT_SENSOR_MESH_PRIM_PATHS
    ray_count: int = DEFAULT_RAY_COUNT
    min_range_m: float = DEFAULT_MIN_RANGE_M
    max_range_m: float = DEFAULT_MAX_RANGE_M
    horizontal_fov_degrees: tuple[float, float] = DEFAULT_HORIZONTAL_FOV_DEG
    vertical_fov_degrees: tuple[float, float] = DEFAULT_VERTICAL_FOV_DEG
    vertical_channels: int = DEFAULT_VERTICAL_CHANNELS
    debug_vis: bool = False

    @property
    def horizontal_samples(self) -> int:
        """Number of horizontal samples required to meet ``ray_count``."""
        if self.ray_count % self.vertical_channels != 0:
            raise ValueError(
                f"ray_count ({self.ray_count}) must be divisible by vertical_channels "
                f"({self.vertical_channels})"
            )
        samples = self.ray_count // self.vertical_channels
        if samples < 2:
            raise ValueError("RayCaster horizontal sample count must be at least 2")
        return samples

    @property
    def horizontal_resolution_degrees(self) -> float:
        """Horizontal angular resolution for Isaac Lab's inclusive FOV range."""
        start, end = self.horizontal_fov_degrees
        return (end - start) / float(self.horizontal_samples - 1)


def fixed_count_lidar_pattern(
    cfg: "FixedCountLidarPatternCfg",
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate an exact-count LiDAR pattern from channel and sample counts."""
    vertical_angles = torch.linspace(
        cfg.vertical_fov_range[0],
        cfg.vertical_fov_range[1],
        cfg.channels,
        device=device,
    )
    horizontal_angles = torch.linspace(
        cfg.horizontal_fov_range[0],
        cfg.horizontal_fov_range[1],
        cfg.horizontal_samples,
        device=device,
    )

    vertical_angles_rad = torch.deg2rad(vertical_angles)
    horizontal_angles_rad = torch.deg2rad(horizontal_angles)
    v_angles, h_angles = torch.meshgrid(vertical_angles_rad, horizontal_angles_rad, indexing="ij")

    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions


@configclass
class FixedCountLidarPatternCfg(patterns.PatternBaseCfg):
    """LiDAR pattern cfg that honors an exact horizontal sample count."""

    func: Callable = fixed_count_lidar_pattern
    channels: int = DEFAULT_VERTICAL_CHANNELS
    vertical_fov_range: tuple[float, float] = DEFAULT_VERTICAL_FOV_DEG
    horizontal_fov_range: tuple[float, float] = DEFAULT_HORIZONTAL_FOV_DEG
    horizontal_samples: int = DEFAULT_RAY_COUNT // DEFAULT_VERTICAL_CHANNELS


def _as_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return default
    return float(value[0]), float(value[1])


def _as_str_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not value:
        return default
    return tuple(str(item) for item in value)


def load_ray_sensor_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> RaySensorSettings:
    """Load RayCaster settings from YAML, falling back to safe code defaults."""
    path = Path(config_path)
    if yaml is None or not path.exists():
        return RaySensorSettings()

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    sensor = data.get("sensor", {})
    if not isinstance(sensor, dict):
        sensor = {}

    return RaySensorSettings(
        prim_suffix=str(sensor.get("prim_suffix", DEFAULT_SENSOR_PRIM_SUFFIX)),
        mesh_prim_paths=_as_str_tuple(sensor.get("mesh_prim_paths"), DEFAULT_SENSOR_MESH_PRIM_PATHS),
        ray_count=int(sensor.get("ray_count", DEFAULT_RAY_COUNT)),
        min_range_m=float(sensor.get("min_range_m", DEFAULT_MIN_RANGE_M)),
        max_range_m=float(sensor.get("max_range_m", DEFAULT_MAX_RANGE_M)),
        horizontal_fov_degrees=_as_float_pair(
            sensor.get("horizontal_fov_degrees"), DEFAULT_HORIZONTAL_FOV_DEG
        ),
        vertical_fov_degrees=_as_float_pair(sensor.get("vertical_fov_degrees"), DEFAULT_VERTICAL_FOV_DEG),
        vertical_channels=int(sensor.get("vertical_channels", DEFAULT_VERTICAL_CHANNELS)),
        debug_vis=bool(sensor.get("debug_vis", False)),
    )


def make_quadrotor_cfg(prim_path: str = DEFAULT_ROBOT_PRIM_PATH) -> ArticulationCfg:
    """Return the Crazyflie articulation config at the requested prim path."""
    return CRAZYFLIE_CFG.replace(prim_path=prim_path)


def make_lidar_pattern_cfg(settings: RaySensorSettings) -> FixedCountLidarPatternCfg:
    """Return a fixed-count LiDAR pattern cfg from YAML-backed sensor settings."""
    return FixedCountLidarPatternCfg(
        channels=settings.vertical_channels,
        vertical_fov_range=settings.vertical_fov_degrees,
        horizontal_fov_range=settings.horizontal_fov_degrees,
        horizontal_samples=settings.horizontal_samples,
    )


def make_raycaster_cfg(
    robot_prim_path: str = DEFAULT_ROBOT_PRIM_PATH,
    settings: RaySensorSettings | None = None,
    *,
    debug_vis: bool | None = None,
    mesh_prim_paths: tuple[str, ...] | None = None,
) -> RayCasterCfg:
    """Return the AeroStrike RayCaster config attached to the Crazyflie body."""
    ray_settings = settings or load_ray_sensor_settings()
    sensor_prim_path = f"{robot_prim_path.rstrip('/')}/{ray_settings.prim_suffix.lstrip('/')}"
    visual_debug = ray_settings.debug_vis if debug_vis is None else debug_vis
    targets = list(mesh_prim_paths or ray_settings.mesh_prim_paths)

    return RayCasterCfg(
        prim_path=sensor_prim_path,
        ray_alignment="base",
        pattern_cfg=make_lidar_pattern_cfg(ray_settings),
        debug_vis=visual_debug,
        max_distance=ray_settings.max_range_m,
        mesh_prim_paths=targets,
    )


CRAZYFLIE_ASSET_NAME = "crazyflie"
AEROSTRIKE_RAY_SENSOR_SETTINGS = load_ray_sensor_settings()
AEROSTRIKE_CRAZYFLIE_CFG = make_quadrotor_cfg()
AEROSTRIKE_RAYCASTER_CFG = make_raycaster_cfg(settings=AEROSTRIKE_RAY_SENSOR_SETTINGS)

__all__ = [
    "AEROSTRIKE_CRAZYFLIE_CFG",
    "AEROSTRIKE_RAYCASTER_CFG",
    "AEROSTRIKE_RAY_SENSOR_SETTINGS",
    "CRAZYFLIE_ASSET_NAME",
    "FixedCountLidarPatternCfg",
    "RaySensorSettings",
    "fixed_count_lidar_pattern",
    "load_ray_sensor_settings",
    "make_lidar_pattern_cfg",
    "make_quadrotor_cfg",
    "make_raycaster_cfg",
]
