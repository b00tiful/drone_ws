"""Procedural industrial warehouse scene generation for AeroStrike."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable, Literal

try:
    import yaml
except ImportError:  # pragma: no cover - Isaac Lab environments normally include PyYAML.
    yaml = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "scene_variants.yaml"
WAREHOUSE_ENV_NAME = "warehouse_env"
DEFAULT_ROOT_PRIM_PATH = "/World/Warehouse"

DEFAULT_ARENA_SIZE_M = (20.0, 20.0)
DEFAULT_WALL_HEIGHT_M = 4.0
DEFAULT_WALL_THICKNESS_M = 0.25
DEFAULT_FLOOR_THICKNESS_M = 0.1
DEFAULT_OBSTACLE_COUNT_RANGE = (8, 15)
DEFAULT_PILLAR_COUNT_RANGE = (2, 4)
DEFAULT_BOX_SIZE_RANGE_M = (0.5, 3.0)
DEFAULT_OBSTACLE_HEIGHT_RANGE_M = (1.0, 4.0)
DEFAULT_PILLAR_RADIUS_RANGE_M = (0.2, 0.5)
DEFAULT_MIN_START_GOAL_DISTANCE_M = 10.0
DEFAULT_TARGET_ALTITUDE_M = 1.5
DEFAULT_START_GOAL_CLEARANCE_M = 1.5
DEFAULT_OBSTACLE_CLEARANCE_M = 0.6
DEFAULT_BOUNDARY_MARGIN_M = 1.0
DEFAULT_LAYOUT_SEED = 7
DEFAULT_BOX_YAW_RANGE_DEGREES = (0.0, 180.0)
DEFAULT_SAMPLING_ATTEMPTS = 100
DEFAULT_FLOOR_COLOR = (0.18, 0.18, 0.17)
DEFAULT_WALL_COLOR = (0.46, 0.48, 0.50)
DEFAULT_BOX_COLOR = (0.55, 0.42, 0.25)
DEFAULT_PILLAR_COLOR = (0.35, 0.38, 0.40)
DEFAULT_LIGHT_INTENSITY = 3500.0
DEFAULT_LIGHT_COLOR = (0.80, 0.82, 0.85)
DEFAULT_ROUTE_MARKING_COLOR = (0.95, 0.72, 0.18)
DEFAULT_SAFETY_STRIPE_COLOR = (0.06, 0.07, 0.08)
DEFAULT_LIGHT_FIXTURE_COLOR = (0.88, 0.92, 0.86)
DEFAULT_SIDE_CLUTTER_COLOR = (0.30, 0.25, 0.18)
DEFAULT_CEILING_BEAM_COLOR = (0.18, 0.20, 0.21)
DEFAULT_TEXTURES_ENABLED = True
DEFAULT_TEXTURE_ROOT = WORKSPACE_ROOT / "assets" / "textures" / "polyhaven"
DEFAULT_FLOOR_TEXTURE_ID = "concrete_floor_worn_001"
DEFAULT_WALL_TEXTURE_ID = "factory_wall"
DEFAULT_BOX_TEXTURE_ID = "metal_plate"
DEFAULT_PILLAR_TEXTURE_ID = "metal_plate"
DEFAULT_SCENE_VARIANT = "warehouse"
DEFAULT_HALLWAY_ARENA_SIZE_M = (12.0, 36.0)
DEFAULT_HALLWAY_OBSTACLE_COUNT = 14
DEFAULT_HALLWAY_START_Y_M = -15.0
DEFAULT_HALLWAY_GOAL_Y_M = 15.0
DEFAULT_HALLWAY_LANE_X_RANGE_M = (-3.0, 3.0)
DEFAULT_HALLWAY_OBSTACLE_X_RANGE_M = (-2.8, 2.8)
DEFAULT_HALLWAY_CENTER_CLEARANCE_M = 0.0
DEFAULT_HALLWAY_OBSTACLE_Y_MARGIN_M = 4.0
DEFAULT_HALLWAY_OBSTACLE_SPACING_M = 2.2
DEFAULT_VISUAL_SIDE_CLUTTER_COUNT = 0
DEFAULT_VISUAL_SIDE_CLUTTER_SIZE_M = (0.72, 0.48, 0.42)
DEFAULT_VISUAL_SIDE_CLUTTER_MARGIN_M = 0.85
DEFAULT_VISUAL_CEILING_BEAM_COUNT = 0
DEFAULT_VISUAL_CEILING_BEAM_DEPTH_M = 0.08
DEFAULT_VISUAL_CEILING_BEAM_HEIGHT_M = 0.06
DEFAULT_VISUAL_CEILING_BEAM_TOP_OFFSET_M = 0.06
DEFAULT_VISUAL_MODE = "cinematic"
DEFAULT_VISUAL_MODEL_ASSET_ROOT = WORKSPACE_ROOT / "assets" / "models" / "kenney_factory_kit"
DEFAULT_VISUAL_MODEL_USD_CACHE_DIR = DEFAULT_VISUAL_MODEL_ASSET_ROOT / "usd_cache"
DEFAULT_VISUAL_MODEL_COUNT = 0
DEFAULT_VISUAL_MODEL_SCALE = 0.42
DEFAULT_VISUAL_MODEL_SIDE_MARGIN_M = 0.62
DEFAULT_VISUAL_MODEL_ROUTE_CLEARANCE_M = 1.65


@dataclass(frozen=True)
class VisualModelSpec:
    """YAML-backed visual-only model placement spec."""

    asset: str
    count: int
    scale: float


@dataclass(frozen=True)
class WarehouseSceneSettings:
    """YAML-backed settings for the procedural warehouse."""

    scene_variant: Literal["warehouse", "hallway"] = DEFAULT_SCENE_VARIANT
    arena_size_m: tuple[float, float] = DEFAULT_ARENA_SIZE_M
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M
    floor_thickness_m: float = DEFAULT_FLOOR_THICKNESS_M
    obstacle_count_range: tuple[int, int] = DEFAULT_OBSTACLE_COUNT_RANGE
    pillar_count_range: tuple[int, int] = DEFAULT_PILLAR_COUNT_RANGE
    box_size_range_m: tuple[float, float] = DEFAULT_BOX_SIZE_RANGE_M
    obstacle_height_range_m: tuple[float, float] = DEFAULT_OBSTACLE_HEIGHT_RANGE_M
    pillar_radius_range_m: tuple[float, float] = DEFAULT_PILLAR_RADIUS_RANGE_M
    min_start_goal_distance_m: float = DEFAULT_MIN_START_GOAL_DISTANCE_M
    target_altitude_m: float = DEFAULT_TARGET_ALTITUDE_M
    start_goal_clearance_m: float = DEFAULT_START_GOAL_CLEARANCE_M
    obstacle_clearance_m: float = DEFAULT_OBSTACLE_CLEARANCE_M
    boundary_margin_m: float = DEFAULT_BOUNDARY_MARGIN_M
    layout_seed: int = DEFAULT_LAYOUT_SEED
    box_yaw_range_degrees: tuple[float, float] = DEFAULT_BOX_YAW_RANGE_DEGREES
    sampling_attempts: int = DEFAULT_SAMPLING_ATTEMPTS
    floor_color: tuple[float, float, float] = DEFAULT_FLOOR_COLOR
    wall_color: tuple[float, float, float] = DEFAULT_WALL_COLOR
    box_color: tuple[float, float, float] = DEFAULT_BOX_COLOR
    pillar_color: tuple[float, float, float] = DEFAULT_PILLAR_COLOR
    light_intensity: float = DEFAULT_LIGHT_INTENSITY
    light_color: tuple[float, float, float] = DEFAULT_LIGHT_COLOR
    route_marking_color: tuple[float, float, float] = DEFAULT_ROUTE_MARKING_COLOR
    safety_stripe_color: tuple[float, float, float] = DEFAULT_SAFETY_STRIPE_COLOR
    light_fixture_color: tuple[float, float, float] = DEFAULT_LIGHT_FIXTURE_COLOR
    side_clutter_color: tuple[float, float, float] = DEFAULT_SIDE_CLUTTER_COLOR
    ceiling_beam_color: tuple[float, float, float] = DEFAULT_CEILING_BEAM_COLOR
    textures_enabled: bool = DEFAULT_TEXTURES_ENABLED
    texture_root: Path = DEFAULT_TEXTURE_ROOT
    floor_texture_id: str = DEFAULT_FLOOR_TEXTURE_ID
    wall_texture_id: str = DEFAULT_WALL_TEXTURE_ID
    box_texture_id: str = DEFAULT_BOX_TEXTURE_ID
    pillar_texture_id: str = DEFAULT_PILLAR_TEXTURE_ID
    hallway_arena_size_m: tuple[float, float] = DEFAULT_HALLWAY_ARENA_SIZE_M
    hallway_obstacle_count: int = DEFAULT_HALLWAY_OBSTACLE_COUNT
    hallway_start_y_m: float = DEFAULT_HALLWAY_START_Y_M
    hallway_goal_y_m: float = DEFAULT_HALLWAY_GOAL_Y_M
    hallway_lane_x_range_m: tuple[float, float] = DEFAULT_HALLWAY_LANE_X_RANGE_M
    hallway_obstacle_x_range_m: tuple[float, float] = DEFAULT_HALLWAY_OBSTACLE_X_RANGE_M
    hallway_center_clearance_m: float = DEFAULT_HALLWAY_CENTER_CLEARANCE_M
    hallway_obstacle_y_margin_m: float = DEFAULT_HALLWAY_OBSTACLE_Y_MARGIN_M
    hallway_obstacle_spacing_m: float = DEFAULT_HALLWAY_OBSTACLE_SPACING_M
    visual_side_clutter_count: int = DEFAULT_VISUAL_SIDE_CLUTTER_COUNT
    visual_side_clutter_size_m: tuple[float, float, float] = DEFAULT_VISUAL_SIDE_CLUTTER_SIZE_M
    visual_side_clutter_margin_m: float = DEFAULT_VISUAL_SIDE_CLUTTER_MARGIN_M
    visual_ceiling_beam_count: int = DEFAULT_VISUAL_CEILING_BEAM_COUNT
    visual_ceiling_beam_depth_m: float = DEFAULT_VISUAL_CEILING_BEAM_DEPTH_M
    visual_ceiling_beam_height_m: float = DEFAULT_VISUAL_CEILING_BEAM_HEIGHT_M
    visual_ceiling_beam_top_offset_m: float = DEFAULT_VISUAL_CEILING_BEAM_TOP_OFFSET_M
    visual_mode: Literal["cinematic", "bright"] = DEFAULT_VISUAL_MODE
    visual_model_asset_root: Path = DEFAULT_VISUAL_MODEL_ASSET_ROOT
    visual_model_usd_cache_dir: Path = DEFAULT_VISUAL_MODEL_USD_CACHE_DIR
    visual_model_count: int = DEFAULT_VISUAL_MODEL_COUNT
    visual_model_scale: float = DEFAULT_VISUAL_MODEL_SCALE
    visual_model_side_margin_m: float = DEFAULT_VISUAL_MODEL_SIDE_MARGIN_M
    visual_model_route_clearance_m: float = DEFAULT_VISUAL_MODEL_ROUTE_CLEARANCE_M
    visual_model_specs: tuple[VisualModelSpec, ...] = ()


@dataclass(frozen=True)
class WarehousePrimitive:
    """A sampled static scene primitive."""

    name: str
    kind: Literal["box", "pillar"]
    translation: tuple[float, float, float]
    size: tuple[float, float, float] | None = None
    radius: float | None = None
    height: float | None = None
    yaw_degrees: float = 0.0


@dataclass(frozen=True)
class WarehouseLayout:
    """Deterministic layout sampled from ``WarehouseSceneSettings``."""

    root_prim_path: str
    settings: WarehouseSceneSettings
    seed: int
    obstacles: tuple[WarehousePrimitive, ...]
    start_position: tuple[float, float, float]
    goal_position: tuple[float, float, float]

    @property
    def mesh_prim_paths(self) -> tuple[str, ...]:
        """Static mesh roots that RayCaster should target."""
        paths = [
            f"{self.root_prim_path}/Floor",
            f"{self.root_prim_path}/WallNorth",
            f"{self.root_prim_path}/WallSouth",
            f"{self.root_prim_path}/WallEast",
            f"{self.root_prim_path}/WallWest",
        ]
        paths.extend(f"{self.root_prim_path}/{primitive.name}" for primitive in self.obstacles)
        return tuple(paths)


def _as_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return default
    return float(value[0]), float(value[1])


def _as_float_triple(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return default
    return float(value[0]), float(value[1]), float(value[2])


def _as_int_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return default
    low, high = int(value[0]), int(value[1])
    return min(low, high), max(low, high)


def _as_color(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return default
    return float(value[0]), float(value[1]), float(value[2])


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def _as_texture_root(value: Any, default: Path) -> Path:
    if not isinstance(value, str) or not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def _as_workspace_path(value: Any, default: Path) -> Path:
    if not isinstance(value, str) or not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def _scene_variant(value: Any, default: Literal["warehouse", "hallway"]) -> Literal["warehouse", "hallway"]:
    variant = str(value or default)
    if variant not in ("warehouse", "hallway"):
        return default
    return variant


def _visual_mode(value: Any, default: Literal["cinematic", "bright"]) -> Literal["cinematic", "bright"]:
    mode = str(value or default)
    if mode not in ("cinematic", "bright"):
        return default
    return mode


def _visual_model_specs(value: Any, default_scale: float) -> tuple[VisualModelSpec, ...]:
    if not isinstance(value, list):
        return ()

    specs: list[VisualModelSpec] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        asset = str(item.get("asset", "")).strip()
        if not asset:
            continue
        count = max(0, int(item.get("count", 0)))
        if count <= 0:
            continue
        scale = float(item.get("scale", default_scale))
        if scale <= 0.0:
            continue
        specs.append(VisualModelSpec(asset=asset, count=count, scale=scale))
    return tuple(specs)


def load_warehouse_scene_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> WarehouseSceneSettings:
    """Load warehouse settings from YAML, falling back to safe code defaults."""
    path = Path(config_path)
    if yaml is None or not path.exists():
        return WarehouseSceneSettings()

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    warehouse = data.get("warehouse", {})
    if not isinstance(warehouse, dict):
        warehouse = {}
    materials = warehouse.get("materials", {})
    if not isinstance(materials, dict):
        materials = {}
    lighting = warehouse.get("lighting", {})
    if not isinstance(lighting, dict):
        lighting = {}
    hallway = warehouse.get("hallway", {})
    if not isinstance(hallway, dict):
        hallway = {}
    visuals = warehouse.get("visuals", {})
    if not isinstance(visuals, dict):
        visuals = {}
    models = visuals.get("models", {})
    if not isinstance(models, dict):
        models = {}
    model_scale = float(models.get("scale", DEFAULT_VISUAL_MODEL_SCALE))

    return WarehouseSceneSettings(
        scene_variant=_scene_variant(warehouse.get("scene_variant"), DEFAULT_SCENE_VARIANT),
        arena_size_m=_as_float_pair(warehouse.get("arena_size_m"), DEFAULT_ARENA_SIZE_M),
        wall_height_m=float(warehouse.get("wall_height_m", DEFAULT_WALL_HEIGHT_M)),
        wall_thickness_m=float(warehouse.get("wall_thickness_m", DEFAULT_WALL_THICKNESS_M)),
        floor_thickness_m=float(warehouse.get("floor_thickness_m", DEFAULT_FLOOR_THICKNESS_M)),
        obstacle_count_range=_as_int_pair(
            warehouse.get("obstacle_count_range"), DEFAULT_OBSTACLE_COUNT_RANGE
        ),
        pillar_count_range=_as_int_pair(warehouse.get("pillar_count_range"), DEFAULT_PILLAR_COUNT_RANGE),
        box_size_range_m=_as_float_pair(warehouse.get("box_size_range_m"), DEFAULT_BOX_SIZE_RANGE_M),
        obstacle_height_range_m=_as_float_pair(
            warehouse.get("obstacle_height_range_m"), DEFAULT_OBSTACLE_HEIGHT_RANGE_M
        ),
        pillar_radius_range_m=_as_float_pair(
            warehouse.get("pillar_radius_range_m"), DEFAULT_PILLAR_RADIUS_RANGE_M
        ),
        min_start_goal_distance_m=float(
            warehouse.get("min_start_goal_distance_m", DEFAULT_MIN_START_GOAL_DISTANCE_M)
        ),
        target_altitude_m=float(warehouse.get("target_altitude_m", DEFAULT_TARGET_ALTITUDE_M)),
        start_goal_clearance_m=float(
            warehouse.get("start_goal_clearance_m", DEFAULT_START_GOAL_CLEARANCE_M)
        ),
        obstacle_clearance_m=float(warehouse.get("obstacle_clearance_m", DEFAULT_OBSTACLE_CLEARANCE_M)),
        boundary_margin_m=float(warehouse.get("boundary_margin_m", DEFAULT_BOUNDARY_MARGIN_M)),
        layout_seed=int(warehouse.get("layout_seed", DEFAULT_LAYOUT_SEED)),
        box_yaw_range_degrees=_as_float_pair(
            warehouse.get("box_yaw_range_degrees"), DEFAULT_BOX_YAW_RANGE_DEGREES
        ),
        sampling_attempts=int(warehouse.get("sampling_attempts", DEFAULT_SAMPLING_ATTEMPTS)),
        floor_color=_as_color(materials.get("floor_color"), DEFAULT_FLOOR_COLOR),
        wall_color=_as_color(materials.get("wall_color"), DEFAULT_WALL_COLOR),
        box_color=_as_color(materials.get("box_color"), DEFAULT_BOX_COLOR),
        pillar_color=_as_color(materials.get("pillar_color"), DEFAULT_PILLAR_COLOR),
        light_intensity=float(lighting.get("intensity", DEFAULT_LIGHT_INTENSITY)),
        light_color=_as_color(lighting.get("color"), DEFAULT_LIGHT_COLOR),
        route_marking_color=_as_color(materials.get("route_marking_color"), DEFAULT_ROUTE_MARKING_COLOR),
        safety_stripe_color=_as_color(materials.get("safety_stripe_color"), DEFAULT_SAFETY_STRIPE_COLOR),
        light_fixture_color=_as_color(materials.get("light_fixture_color"), DEFAULT_LIGHT_FIXTURE_COLOR),
        side_clutter_color=_as_color(materials.get("side_clutter_color"), DEFAULT_SIDE_CLUTTER_COLOR),
        ceiling_beam_color=_as_color(materials.get("ceiling_beam_color"), DEFAULT_CEILING_BEAM_COLOR),
        textures_enabled=_as_bool(materials.get("textures_enabled"), DEFAULT_TEXTURES_ENABLED),
        texture_root=_as_texture_root(materials.get("texture_root"), DEFAULT_TEXTURE_ROOT),
        floor_texture_id=str(materials.get("floor_texture_id", DEFAULT_FLOOR_TEXTURE_ID)),
        wall_texture_id=str(materials.get("wall_texture_id", DEFAULT_WALL_TEXTURE_ID)),
        box_texture_id=str(materials.get("box_texture_id", DEFAULT_BOX_TEXTURE_ID)),
        pillar_texture_id=str(materials.get("pillar_texture_id", DEFAULT_PILLAR_TEXTURE_ID)),
        hallway_arena_size_m=_as_float_pair(hallway.get("arena_size_m"), DEFAULT_HALLWAY_ARENA_SIZE_M),
        hallway_obstacle_count=int(hallway.get("obstacle_count", DEFAULT_HALLWAY_OBSTACLE_COUNT)),
        hallway_start_y_m=float(hallway.get("start_y_m", DEFAULT_HALLWAY_START_Y_M)),
        hallway_goal_y_m=float(hallway.get("goal_y_m", DEFAULT_HALLWAY_GOAL_Y_M)),
        hallway_lane_x_range_m=_as_float_pair(
            hallway.get("lane_x_range_m"), DEFAULT_HALLWAY_LANE_X_RANGE_M
        ),
        hallway_obstacle_x_range_m=_as_float_pair(
            hallway.get("obstacle_x_range_m"), DEFAULT_HALLWAY_OBSTACLE_X_RANGE_M
        ),
        hallway_center_clearance_m=float(
            hallway.get("center_clearance_m", DEFAULT_HALLWAY_CENTER_CLEARANCE_M)
        ),
        hallway_obstacle_y_margin_m=float(
            hallway.get("obstacle_y_margin_m", DEFAULT_HALLWAY_OBSTACLE_Y_MARGIN_M)
        ),
        hallway_obstacle_spacing_m=float(
            hallway.get("obstacle_spacing_m", DEFAULT_HALLWAY_OBSTACLE_SPACING_M)
        ),
        visual_side_clutter_count=max(
            0, int(visuals.get("side_clutter_count", DEFAULT_VISUAL_SIDE_CLUTTER_COUNT))
        ),
        visual_side_clutter_size_m=_as_float_triple(
            visuals.get("side_clutter_size_m"), DEFAULT_VISUAL_SIDE_CLUTTER_SIZE_M
        ),
        visual_side_clutter_margin_m=float(
            visuals.get("side_clutter_margin_m", DEFAULT_VISUAL_SIDE_CLUTTER_MARGIN_M)
        ),
        visual_ceiling_beam_count=max(
            0, int(visuals.get("ceiling_beam_count", DEFAULT_VISUAL_CEILING_BEAM_COUNT))
        ),
        visual_ceiling_beam_depth_m=float(
            visuals.get("ceiling_beam_depth_m", DEFAULT_VISUAL_CEILING_BEAM_DEPTH_M)
        ),
        visual_ceiling_beam_height_m=float(
            visuals.get("ceiling_beam_height_m", DEFAULT_VISUAL_CEILING_BEAM_HEIGHT_M)
        ),
        visual_ceiling_beam_top_offset_m=float(
            visuals.get("ceiling_beam_top_offset_m", DEFAULT_VISUAL_CEILING_BEAM_TOP_OFFSET_M)
        ),
        visual_mode=_visual_mode(visuals.get("mode"), DEFAULT_VISUAL_MODE),
        visual_model_asset_root=_as_workspace_path(models.get("asset_root"), DEFAULT_VISUAL_MODEL_ASSET_ROOT),
        visual_model_usd_cache_dir=_as_workspace_path(
            models.get("usd_cache_dir"), DEFAULT_VISUAL_MODEL_USD_CACHE_DIR
        ),
        visual_model_count=max(0, int(models.get("count", DEFAULT_VISUAL_MODEL_COUNT))),
        visual_model_scale=model_scale,
        visual_model_side_margin_m=float(
            models.get("side_margin_m", DEFAULT_VISUAL_MODEL_SIDE_MARGIN_M)
        ),
        visual_model_route_clearance_m=float(
            models.get("route_clearance_m", DEFAULT_VISUAL_MODEL_ROUTE_CLEARANCE_M)
        ),
        visual_model_specs=_visual_model_specs(models.get("assets"), model_scale),
    )


def sample_warehouse_layout(
    settings: WarehouseSceneSettings | None = None,
    *,
    seed: int | None = None,
    root_prim_path: str = DEFAULT_ROOT_PRIM_PATH,
    scene_variant: Literal["warehouse", "hallway"] | None = None,
) -> WarehouseLayout:
    """Sample a deterministic obstacle layout and start/goal pair."""
    scene_settings = settings or load_warehouse_scene_settings()
    layout_seed = scene_settings.layout_seed if seed is None else seed
    variant = scene_settings.scene_variant if scene_variant is None else scene_variant
    if variant == "hallway":
        return sample_hallway_layout(scene_settings, seed=layout_seed, root_prim_path=root_prim_path)

    rng = Random(layout_seed)
    start = _sample_free_position(rng, scene_settings, ())
    if start is None:
        raise ValueError("Failed to sample a start position with the configured arena settings")
    goal = _sample_goal_position(rng, scene_settings, start)

    blocked: list[tuple[float, float, float]] = [start, goal]
    obstacles: list[WarehousePrimitive] = []
    box_count = rng.randint(*scene_settings.obstacle_count_range)
    pillar_count = rng.randint(*scene_settings.pillar_count_range)

    for index in range(box_count):
        obstacle = _sample_box(rng, scene_settings, tuple(blocked), index)
        if obstacle is not None:
            obstacles.append(obstacle)
            blocked.append((obstacle.translation[0], obstacle.translation[1], scene_settings.target_altitude_m))

    for index in range(pillar_count):
        obstacle = _sample_pillar(rng, scene_settings, tuple(blocked), index)
        if obstacle is not None:
            obstacles.append(obstacle)
            blocked.append((obstacle.translation[0], obstacle.translation[1], scene_settings.target_altitude_m))

    return WarehouseLayout(
        root_prim_path=root_prim_path.rstrip("/"),
        settings=scene_settings,
        seed=layout_seed,
        obstacles=tuple(obstacles),
        start_position=start,
        goal_position=goal,
    )


def sample_hallway_layout(
    settings: WarehouseSceneSettings | None = None,
    *,
    seed: int | None = None,
    root_prim_path: str = DEFAULT_ROOT_PRIM_PATH,
) -> WarehouseLayout:
    """Sample a deterministic long-hallway layout for demo recording."""
    base_settings = settings or load_warehouse_scene_settings()
    layout_seed = base_settings.layout_seed if seed is None else seed
    rng = Random(layout_seed)
    hallway_settings = _with_hallway_arena(base_settings)

    lane_min_x, lane_max_x = hallway_settings.hallway_lane_x_range_m
    start = (
        rng.uniform(lane_min_x, lane_max_x),
        hallway_settings.hallway_start_y_m,
        hallway_settings.target_altitude_m,
    )
    goal = (
        rng.uniform(lane_min_x, lane_max_x),
        hallway_settings.hallway_goal_y_m,
        hallway_settings.target_altitude_m,
    )

    y_min = min(start[1], goal[1]) + hallway_settings.hallway_obstacle_y_margin_m
    y_max = max(start[1], goal[1]) - hallway_settings.hallway_obstacle_y_margin_m
    if y_min >= y_max:
        raise ValueError("Hallway start/goal and obstacle margins leave no obstacle corridor")

    obstacles: list[WarehousePrimitive] = []
    blocked: list[tuple[float, float, float]] = [start, goal]
    for index in range(max(0, hallway_settings.hallway_obstacle_count)):
        obstacle = _sample_hallway_box(rng, hallway_settings, tuple(blocked), index, y_min, y_max)
        if obstacle is not None:
            obstacles.append(obstacle)
            blocked.append((obstacle.translation[0], obstacle.translation[1], hallway_settings.target_altitude_m))

    return WarehouseLayout(
        root_prim_path=root_prim_path.rstrip("/"),
        settings=hallway_settings,
        seed=layout_seed,
        obstacles=tuple(obstacles),
        start_position=start,
        goal_position=goal,
    )


def _with_hallway_arena(settings: WarehouseSceneSettings) -> WarehouseSceneSettings:
    return WarehouseSceneSettings(
        scene_variant="hallway",
        arena_size_m=settings.hallway_arena_size_m,
        wall_height_m=settings.wall_height_m,
        wall_thickness_m=settings.wall_thickness_m,
        floor_thickness_m=settings.floor_thickness_m,
        obstacle_count_range=settings.obstacle_count_range,
        pillar_count_range=(0, 0),
        box_size_range_m=settings.box_size_range_m,
        obstacle_height_range_m=settings.obstacle_height_range_m,
        pillar_radius_range_m=settings.pillar_radius_range_m,
        min_start_goal_distance_m=settings.min_start_goal_distance_m,
        target_altitude_m=settings.target_altitude_m,
        start_goal_clearance_m=settings.start_goal_clearance_m,
        obstacle_clearance_m=settings.obstacle_clearance_m,
        boundary_margin_m=settings.boundary_margin_m,
        layout_seed=settings.layout_seed,
        box_yaw_range_degrees=settings.box_yaw_range_degrees,
        sampling_attempts=settings.sampling_attempts,
        floor_color=settings.floor_color,
        wall_color=settings.wall_color,
        box_color=settings.box_color,
        pillar_color=settings.pillar_color,
        light_intensity=settings.light_intensity,
        light_color=settings.light_color,
        route_marking_color=settings.route_marking_color,
        safety_stripe_color=settings.safety_stripe_color,
        light_fixture_color=settings.light_fixture_color,
        side_clutter_color=settings.side_clutter_color,
        ceiling_beam_color=settings.ceiling_beam_color,
        textures_enabled=settings.textures_enabled,
        texture_root=settings.texture_root,
        floor_texture_id=settings.floor_texture_id,
        wall_texture_id=settings.wall_texture_id,
        box_texture_id=settings.box_texture_id,
        pillar_texture_id=settings.pillar_texture_id,
        hallway_arena_size_m=settings.hallway_arena_size_m,
        hallway_obstacle_count=settings.hallway_obstacle_count,
        hallway_start_y_m=settings.hallway_start_y_m,
        hallway_goal_y_m=settings.hallway_goal_y_m,
        hallway_lane_x_range_m=settings.hallway_lane_x_range_m,
        hallway_obstacle_x_range_m=settings.hallway_obstacle_x_range_m,
        hallway_center_clearance_m=settings.hallway_center_clearance_m,
        hallway_obstacle_y_margin_m=settings.hallway_obstacle_y_margin_m,
        hallway_obstacle_spacing_m=settings.hallway_obstacle_spacing_m,
        visual_side_clutter_count=settings.visual_side_clutter_count,
        visual_side_clutter_size_m=settings.visual_side_clutter_size_m,
        visual_side_clutter_margin_m=settings.visual_side_clutter_margin_m,
        visual_ceiling_beam_count=settings.visual_ceiling_beam_count,
        visual_ceiling_beam_depth_m=settings.visual_ceiling_beam_depth_m,
        visual_ceiling_beam_height_m=settings.visual_ceiling_beam_height_m,
        visual_ceiling_beam_top_offset_m=settings.visual_ceiling_beam_top_offset_m,
        visual_mode=settings.visual_mode,
        visual_model_asset_root=settings.visual_model_asset_root,
        visual_model_usd_cache_dir=settings.visual_model_usd_cache_dir,
        visual_model_count=settings.visual_model_count,
        visual_model_scale=settings.visual_model_scale,
        visual_model_side_margin_m=settings.visual_model_side_margin_m,
        visual_model_route_clearance_m=settings.visual_model_route_clearance_m,
        visual_model_specs=settings.visual_model_specs,
    )


def spawn_warehouse_scene(layout: WarehouseLayout | None = None) -> WarehouseLayout:
    """Spawn the warehouse as static collidable primitives and return its layout."""
    scene_layout = layout or sample_warehouse_layout()
    settings = scene_layout.settings
    root = scene_layout.root_prim_path

    import isaaclab.sim as sim_utils

    light_cfg = sim_utils.DistantLightCfg(
        intensity=settings.light_intensity,
        color=settings.light_color,
        angle=2.0,
    )
    if not sim_utils.get_current_stage().GetPrimAtPath("/World/Light").IsValid():
        light_cfg.func("/World/Light", light_cfg)

    floor_cfg = _make_cuboid_cfg(
        settings.arena_size_m + (settings.floor_thickness_m,),
        settings.floor_color,
        roughness=0.82,
        texture_id=settings.floor_texture_id,
        texture_root=settings.texture_root,
        textures_enabled=settings.textures_enabled,
        texture_scale=(settings.arena_size_m[0] / 5.0, settings.arena_size_m[1] / 5.0),
    )
    floor_cfg.func(
        f"{root}/Floor",
        floor_cfg,
        translation=(0.0, 0.0, -settings.floor_thickness_m / 2.0),
    )

    for name, size, translation in _wall_specs(settings):
        wall_cfg = _make_cuboid_cfg(
            size,
            settings.wall_color,
            roughness=0.68,
            texture_id=settings.wall_texture_id,
            texture_root=settings.texture_root,
            textures_enabled=settings.textures_enabled,
            texture_scale=_texture_scale_for_size(size, meters_per_tile=4.0),
        )
        wall_cfg.func(f"{root}/{name}", wall_cfg, translation=translation)

    for primitive in scene_layout.obstacles:
        if primitive.kind == "box":
            if primitive.size is None:
                raise ValueError(f"Box primitive {primitive.name} is missing size")
            cfg = _make_cuboid_cfg(
                primitive.size,
                settings.box_color,
                roughness=0.74,
                metallic=0.28,
                texture_id=settings.box_texture_id,
                texture_root=settings.texture_root,
                textures_enabled=settings.textures_enabled,
                texture_scale=_texture_scale_for_size(primitive.size, meters_per_tile=2.0),
            )
            cfg.func(
                f"{root}/{primitive.name}",
                cfg,
                translation=primitive.translation,
                orientation=_yaw_quat(primitive.yaw_degrees),
            )
        else:
            if primitive.radius is None or primitive.height is None:
                raise ValueError(f"Pillar primitive {primitive.name} is missing radius/height")
            cfg = sim_utils.CylinderCfg(
                radius=primitive.radius,
                height=primitive.height,
                axis="Z",
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=_make_material_cfg(
                    settings.pillar_color,
                    roughness=0.46,
                    metallic=0.35,
                    texture_id=settings.pillar_texture_id,
                    texture_root=settings.texture_root,
                    textures_enabled=settings.textures_enabled,
                    texture_scale=(1.0, max(1.0, primitive.height / 2.0)),
                ),
            )
            cfg.func(f"{root}/{primitive.name}", cfg, translation=primitive.translation)

    _spawn_visual_polish(
        _visual_root_for(root),
        settings,
        scene_layout.seed,
        scene_layout.start_position,
        scene_layout.goal_position,
    )
    return scene_layout


def _make_cuboid_cfg(
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    roughness: float = 0.55,
    metallic: float = 0.0,
    opacity: float = 1.0,
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    texture_id: str | None = None,
    texture_root: Path = DEFAULT_TEXTURE_ROOT,
    textures_enabled: bool = DEFAULT_TEXTURES_ENABLED,
    texture_scale: tuple[float, float] = (1.0, 1.0),
) -> Any:
    import isaaclab.sim as sim_utils

    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=_make_material_cfg(
            color,
            roughness=roughness,
            metallic=metallic,
            opacity=opacity,
            emissive_color=emissive_color,
            texture_id=texture_id,
            texture_root=texture_root,
            textures_enabled=textures_enabled,
            texture_scale=texture_scale,
        ),
    )


def _make_material_cfg(
    color: tuple[float, float, float],
    *,
    roughness: float,
    metallic: float = 0.0,
    opacity: float = 1.0,
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    texture_id: str | None = None,
    texture_root: Path = DEFAULT_TEXTURE_ROOT,
    textures_enabled: bool = DEFAULT_TEXTURES_ENABLED,
    texture_scale: tuple[float, float] = (1.0, 1.0),
) -> Any:
    import isaaclab.sim as sim_utils

    texture_set = _texture_set(texture_root, texture_id) if textures_enabled else None
    if texture_set is not None:
        return _OmniPbrTextureCfg(
            diffuse_color=color,
            diffuse_texture=texture_set.diffuse,
            roughness=roughness,
            roughness_texture=texture_set.roughness,
            metallic=metallic,
            normal_texture=texture_set.normal,
            texture_scale=texture_scale,
        )

    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        emissive_color=emissive_color,
        roughness=roughness,
        metallic=metallic,
        opacity=opacity,
    )


@dataclass(frozen=True)
class _TextureSet:
    diffuse: str
    roughness: str | None = None
    normal: str | None = None


@dataclass(frozen=True)
class _OmniPbrTextureCfg:
    func: Callable[[str, "_OmniPbrTextureCfg"], Any] = (
        lambda prim_path, cfg: _spawn_omni_pbr_material(prim_path, cfg)
    )
    diffuse_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    diffuse_texture: str = ""
    roughness: float = 0.65
    roughness_texture: str | None = None
    metallic: float = 0.0
    normal_texture: str | None = None
    texture_scale: tuple[float, float] = (1.0, 1.0)


def _texture_set(texture_root: Path, texture_id: str | None) -> _TextureSet | None:
    if texture_id is None or not texture_id:
        return None

    root = texture_root / texture_id
    diffuse = root / f"{texture_id}_diff_1k.jpg"
    if not diffuse.exists():
        return None

    roughness = root / f"{texture_id}_rough_1k.jpg"
    normal = root / f"{texture_id}_nor_gl_1k.jpg"
    return _TextureSet(
        diffuse=str(diffuse),
        roughness=str(roughness) if roughness.exists() else None,
        normal=str(normal) if normal.exists() else None,
    )


def _spawn_omni_pbr_material(prim_path: str, cfg: _OmniPbrTextureCfg) -> Any:
    import omni.kit.commands
    import omni.usd
    from pxr import Gf, Sdf, UsdShade

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    if not stage.GetPrimAtPath(prim_path).IsValid():
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_path=prim_path,
            select_new_prim=False,
        )

    material_prim = stage.GetPrimAtPath(prim_path)
    shader_prim = stage.GetPrimAtPath(f"{prim_path}/Shader")
    if not shader_prim.IsValid():
        shader_prim = UsdShade.Shader(
            omni.usd.get_shader_from_material(material_prim, get_prim=True)
        ).GetPrim()
    shader = UsdShade.Shader(shader_prim)

    _set_shader_input(
        shader,
        "diffuse_color_constant",
        Sdf.ValueTypeNames.Color3f,
        Gf.Vec3f(*cfg.diffuse_color),
    )
    _set_shader_input(shader, "diffuse_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(cfg.diffuse_texture))
    _set_shader_input(shader, "reflection_roughness_constant", Sdf.ValueTypeNames.Float, cfg.roughness)
    _set_shader_input(shader, "metallic_constant", Sdf.ValueTypeNames.Float, cfg.metallic)
    _set_shader_input(shader, "project_uvw", Sdf.ValueTypeNames.Bool, True)
    _set_shader_input(shader, "texture_scale", Sdf.ValueTypeNames.Float2, Gf.Vec2f(*cfg.texture_scale))
    if cfg.roughness_texture is not None:
        _set_shader_input(
            shader,
            "reflectionroughness_texture",
            Sdf.ValueTypeNames.Asset,
            Sdf.AssetPath(cfg.roughness_texture),
        )
        _set_shader_input(shader, "reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, 0.75)
    if cfg.normal_texture is not None:
        _set_shader_input(shader, "normalmap_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(cfg.normal_texture))
        _set_shader_input(shader, "bump_factor", Sdf.ValueTypeNames.Float, 0.18)

    return shader_prim


def _set_shader_input(shader: Any, name: str, value_type: Any, value: Any) -> None:
    shader_input = shader.GetInput(name)
    if not shader_input:
        shader_input = shader.CreateInput(name, value_type)
    shader_input.Set(value)


def _texture_scale_for_size(size: tuple[float, float, float], meters_per_tile: float) -> tuple[float, float]:
    ordered = sorted((abs(size[0]), abs(size[1]), abs(size[2])), reverse=True)
    return max(1.0, ordered[0] / meters_per_tile), max(1.0, ordered[1] / meters_per_tile)


def _spawn_visual_polish(
    visual_root: str,
    settings: WarehouseSceneSettings,
    seed: int,
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
) -> None:
    _spawn_route_markings(visual_root, settings, start, goal)
    _spawn_wall_safety_bands(visual_root, settings)
    _spawn_overhead_lights(visual_root, settings)
    if settings.scene_variant == "hallway":
        _spawn_ceiling_beams(visual_root, settings)
        _spawn_side_clutter(visual_root, settings, seed)
        _spawn_cinematic_models(visual_root, settings, seed)


def _spawn_route_markings(
    root: str,
    settings: WarehouseSceneSettings,
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
) -> None:
    import math

    route_length = _xy_distance(start, goal)
    if route_length <= 1.0e-3:
        return

    center = ((start[0] + goal[0]) / 2.0, (start[1] + goal[1]) / 2.0, 0.012)
    yaw = math.degrees(math.atan2(goal[1] - start[1], goal[0] - start[0]))
    lateral_offsets = (-0.45, 0.45)
    for index, offset in enumerate(lateral_offsets):
        stripe_center = _offset_xy(center, yaw + 90.0, offset)
        stripe_cfg = _make_visual_cuboid_cfg(
            size=(route_length, 0.045, 0.012),
            color=settings.route_marking_color,
            roughness=0.88,
        )
        stripe_cfg.func(
            f"{root}/VisualRouteStripe_{index:02d}",
            stripe_cfg,
            translation=stripe_center,
            orientation=_yaw_quat(yaw),
        )

    for index, position in enumerate((start, goal)):
        marker_cfg = _make_visual_cuboid_cfg(
            size=(1.2, 0.08, 0.014),
            color=settings.route_marking_color,
            roughness=0.88,
        )
        for cross_index, cross_yaw in enumerate((0.0, 90.0)):
            marker_cfg.func(
                f"{root}/VisualEndpoint_{index:02d}_{cross_index:02d}",
                marker_cfg,
                translation=(position[0], position[1], 0.016),
                orientation=_yaw_quat(cross_yaw),
            )


def _spawn_wall_safety_bands(root: str, settings: WarehouseSceneSettings) -> None:
    arena_x, arena_y = settings.arena_size_m
    z = 0.62
    band_thickness = 0.026
    band_height = 0.16
    specs = (
        ("North", (arena_x, band_thickness, band_height), (0.0, arena_y / 2.0 - 0.018, z), 0.0),
        ("South", (arena_x, band_thickness, band_height), (0.0, -arena_y / 2.0 + 0.018, z), 0.0),
        ("East", (arena_y, band_thickness, band_height), (arena_x / 2.0 - 0.018, 0.0, z), 90.0),
        ("West", (arena_y, band_thickness, band_height), (-arena_x / 2.0 + 0.018, 0.0, z), 90.0),
    )
    for name, size, translation, yaw in specs:
        cfg = _make_visual_cuboid_cfg(size, settings.safety_stripe_color, roughness=0.7)
        cfg.func(f"{root}/VisualWallBand{name}", cfg, translation=translation, orientation=_yaw_quat(yaw))


def _spawn_overhead_lights(root: str, settings: WarehouseSceneSettings) -> None:
    arena_y = settings.arena_size_m[1]
    fixture_y_positions = _fixture_y_positions(arena_y)
    z = settings.wall_height_m - 0.18
    emissive_scale = 1.1 if settings.visual_mode == "cinematic" else 0.65
    for index, y in enumerate(fixture_y_positions):
        flicker = 0.72 if settings.visual_mode == "cinematic" and index % 2 == 1 else 1.0
        fixture_cfg = _make_visual_cuboid_cfg(
            size=(2.2, 0.16, 0.045),
            color=_scale_color(settings.light_fixture_color, flicker),
            roughness=0.2,
            emissive_color=_scale_color(settings.light_fixture_color, emissive_scale * flicker),
        )
        fixture_cfg.func(
            f"{root}/VisualLightFixture_{index:02d}",
            fixture_cfg,
            translation=(0.0, y, z),
        )


def _spawn_ceiling_beams(root: str, settings: WarehouseSceneSettings) -> None:
    if settings.visual_ceiling_beam_count <= 0:
        return

    arena_x, arena_y = settings.arena_size_m
    z = settings.wall_height_m - settings.visual_ceiling_beam_top_offset_m
    y_positions = _even_positions(
        count=settings.visual_ceiling_beam_count,
        low=-arena_y / 2.0 + settings.wall_thickness_m,
        high=arena_y / 2.0 - settings.wall_thickness_m,
    )
    for index, y in enumerate(y_positions):
        beam_cfg = _make_visual_cuboid_cfg(
            size=(
                arena_x + 2.0 * settings.wall_thickness_m,
                settings.visual_ceiling_beam_depth_m,
                settings.visual_ceiling_beam_height_m,
            ),
            color=settings.ceiling_beam_color,
            roughness=0.58,
            metallic=0.18,
        )
        beam_cfg.func(
            f"{root}/VisualCeilingBeam_{index:02d}",
            beam_cfg,
            translation=(0.0, y, z),
        )


def _spawn_side_clutter(root: str, settings: WarehouseSceneSettings, seed: int) -> None:
    if settings.visual_side_clutter_count <= 0:
        return

    import math

    rng = Random(seed + 9173)
    arena_x, arena_y = settings.arena_size_m
    base_size = settings.visual_side_clutter_size_m
    usable_y = arena_y - 2.0 * settings.boundary_margin_m
    if usable_y <= 0.0:
        return

    for index in range(settings.visual_side_clutter_count):
        side = -1.0 if index % 2 == 0 else 1.0
        row = index // 2
        row_fraction = (row + 0.5) / max(1, math.ceil(settings.visual_side_clutter_count / 2.0))
        y = -usable_y / 2.0 + row_fraction * usable_y + rng.uniform(-0.35, 0.35)
        x = side * (arena_x / 2.0 - settings.visual_side_clutter_margin_m)
        size = (
            base_size[0] * rng.uniform(0.85, 1.25),
            base_size[1] * rng.uniform(0.85, 1.35),
            base_size[2] * rng.uniform(0.75, 1.45),
        )
        yaw = 90.0 if side < 0.0 else -90.0
        yaw += rng.uniform(-8.0, 8.0)
        cfg = _make_visual_cuboid_cfg(
            size=size,
            color=_scale_color(settings.side_clutter_color, rng.uniform(0.86, 1.12)),
            roughness=0.76,
            metallic=0.05,
        )
        cfg.func(
            f"{root}/VisualSideClutter_{index:02d}",
            cfg,
            translation=(x, y, size[2] / 2.0),
            orientation=_yaw_quat(yaw),
        )


def _spawn_cinematic_models(root: str, settings: WarehouseSceneSettings, seed: int) -> None:
    if settings.visual_model_count <= 0 or not settings.visual_model_specs:
        return

    import isaaclab.sim as sim_utils
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

    placements = _cinematic_model_placements(settings, seed)
    if not placements:
        return

    usd_paths: dict[str, str] = {}
    for spec in settings.visual_model_specs:
        source = settings.visual_model_asset_root / "fbx" / f"{spec.asset}.fbx"
        if not source.exists():
            continue
        converter = MeshConverter(
            MeshConverterCfg(
                asset_path=str(source),
                usd_dir=str(settings.visual_model_usd_cache_dir / spec.asset),
                usd_file_name=f"{spec.asset}.usd",
                scale=(spec.scale, spec.scale, spec.scale),
                make_instanceable=True,
            )
        )
        usd_paths[spec.asset] = converter.usd_path

    if not usd_paths:
        return

    weighted_assets = [
        spec.asset
        for spec in settings.visual_model_specs
        for _ in range(spec.count)
        if spec.asset in usd_paths
    ]
    if not weighted_assets:
        return

    rng = Random(seed + 29011)
    for index, placement in enumerate(placements):
        asset = weighted_assets[index % len(weighted_assets)]
        cfg = sim_utils.UsdFileCfg(usd_path=usd_paths[asset])
        yaw = placement.yaw_degrees + rng.uniform(-5.0, 5.0)
        cfg.func(
            f"{root}/CinematicModel_{index:02d}_{asset.replace('-', '_')}",
            cfg,
            translation=placement.translation,
            orientation=_yaw_quat(yaw),
        )


@dataclass(frozen=True)
class _VisualModelPlacement:
    translation: tuple[float, float, float]
    yaw_degrees: float


def _cinematic_model_placements(
    settings: WarehouseSceneSettings,
    seed: int,
) -> tuple[_VisualModelPlacement, ...]:
    import math

    rng = Random(seed + 25117)
    arena_x, arena_y = settings.arena_size_m
    usable_y = arena_y - 2.0 * settings.boundary_margin_m
    if usable_y <= 0.0:
        return ()

    overhead_count = min(3, max(0, settings.visual_model_count // 6))
    side_count = max(0, settings.visual_model_count - overhead_count)
    placements: list[_VisualModelPlacement] = []
    for index in range(side_count):
        side = -1.0 if index % 2 == 0 else 1.0
        row = index // 2
        row_fraction = (row + 0.5) / max(1, math.ceil(side_count / 2.0))
        y = -usable_y / 2.0 + row_fraction * usable_y + rng.uniform(-0.55, 0.55)
        x = side * (
            arena_x / 2.0
            - settings.visual_model_side_margin_m
            - rng.uniform(0.0, 0.55)
        )
        if abs(x) < settings.visual_model_route_clearance_m:
            x = side * settings.visual_model_route_clearance_m
        yaw = -90.0 if side > 0.0 else 90.0
        placements.append(_VisualModelPlacement((x, y, 0.02), yaw))

    overhead_y = _even_positions(
        count=overhead_count,
        low=-arena_y * 0.32,
        high=arena_y * 0.32,
    )
    for y in overhead_y:
        placements.append(
            _VisualModelPlacement(
                (0.0, y, max(2.15, settings.wall_height_m - 0.95)),
                0.0,
            )
        )

    return tuple(placements[: settings.visual_model_count])


def _make_visual_cuboid_cfg(
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    roughness: float,
    metallic: float = 0.0,
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Any:
    import isaaclab.sim as sim_utils

    return sim_utils.CuboidCfg(
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            emissive_color=emissive_color,
            roughness=roughness,
            metallic=metallic,
        ),
    )


def _fixture_y_positions(arena_y: float) -> tuple[float, ...]:
    if arena_y <= 18.0:
        return (-arena_y * 0.25, arena_y * 0.25)
    return (-arena_y * 0.33, 0.0, arena_y * 0.33)


def _even_positions(count: int, low: float, high: float) -> tuple[float, ...]:
    if count <= 0 or low >= high:
        return ()
    if count == 1:
        return ((low + high) / 2.0,)

    step = (high - low) / float(count - 1)
    return tuple(low + index * step for index in range(count))


def _offset_xy(
    center: tuple[float, float, float],
    yaw_degrees: float,
    distance_m: float,
) -> tuple[float, float, float]:
    import math

    yaw = math.radians(yaw_degrees)
    return (
        center[0] + math.cos(yaw) * distance_m,
        center[1] + math.sin(yaw) * distance_m,
        center[2],
    )


def _scale_color(
    color: tuple[float, float, float],
    scale: float,
) -> tuple[float, float, float]:
    return tuple(min(1.0, max(0.0, channel * scale)) for channel in color)


def _visual_root_for(root: str) -> str:
    parts = [part for part in root.rstrip("/").split("/") if part]
    if "envs" in parts:
        env_index = parts.index("envs")
        if env_index + 1 < len(parts):
            return f"/World/Visuals/{parts[env_index + 1]}/Warehouse"
    return "/World/Visuals/Warehouse"


def _sample_box(
    rng: Random,
    settings: WarehouseSceneSettings,
    blocked_positions: tuple[tuple[float, float, float], ...],
    index: int,
) -> WarehousePrimitive | None:
    width = rng.uniform(*settings.box_size_range_m)
    depth = rng.uniform(*settings.box_size_range_m)
    height = rng.uniform(*settings.obstacle_height_range_m)
    footprint_radius = max(width, depth) / 2.0 + settings.obstacle_clearance_m
    position = _sample_free_position(rng, settings, blocked_positions, footprint_radius=footprint_radius)
    if position is None:
        return None
    x, y, _ = position
    return WarehousePrimitive(
        name=f"Box_{index:02d}",
        kind="box",
        size=(width, depth, height),
        translation=(x, y, height / 2.0),
        yaw_degrees=rng.uniform(*settings.box_yaw_range_degrees),
    )


def _sample_pillar(
    rng: Random,
    settings: WarehouseSceneSettings,
    blocked_positions: tuple[tuple[float, float, float], ...],
    index: int,
) -> WarehousePrimitive | None:
    radius = rng.uniform(*settings.pillar_radius_range_m)
    height = rng.uniform(*settings.obstacle_height_range_m)
    position = _sample_free_position(
        rng,
        settings,
        blocked_positions,
        footprint_radius=radius + settings.obstacle_clearance_m,
    )
    if position is None:
        return None
    x, y, _ = position
    return WarehousePrimitive(
        name=f"Pillar_{index:02d}",
        kind="pillar",
        radius=radius,
        height=height,
        translation=(x, y, height / 2.0),
    )


def _sample_hallway_box(
    rng: Random,
    settings: WarehouseSceneSettings,
    blocked_positions: tuple[tuple[float, float, float], ...],
    index: int,
    y_min: float,
    y_max: float,
) -> WarehousePrimitive | None:
    width = rng.uniform(*settings.box_size_range_m)
    depth = rng.uniform(*settings.box_size_range_m)
    height = rng.uniform(*settings.obstacle_height_range_m)
    footprint_radius = max(width, depth) / 2.0 + settings.obstacle_clearance_m
    x_min, x_max = settings.hallway_obstacle_x_range_m
    min_spacing = max(settings.hallway_obstacle_spacing_m, settings.start_goal_clearance_m + footprint_radius)

    for _ in range(settings.sampling_attempts):
        x = _sample_hallway_obstacle_x(rng, x_min, x_max, settings.hallway_center_clearance_m)
        y = rng.uniform(y_min, y_max)
        candidate = (x, y, settings.target_altitude_m)
        if all(_xy_distance(candidate, blocked) >= min_spacing for blocked in blocked_positions):
            return WarehousePrimitive(
                name=f"HallwayBox_{index:02d}",
                kind="box",
                size=(width, depth, height),
                translation=(x, y, height / 2.0),
                yaw_degrees=rng.uniform(*settings.box_yaw_range_degrees),
            )
    return None


def _sample_hallway_obstacle_x(rng: Random, x_min: float, x_max: float, center_clearance_m: float) -> float:
    clearance = max(0.0, center_clearance_m)
    left = (x_min, min(x_max, -clearance))
    right = (max(x_min, clearance), x_max)
    bands = [band for band in (left, right) if band[0] < band[1]]
    if not bands:
        return rng.uniform(x_min, x_max)
    low, high = rng.choice(bands)
    return rng.uniform(low, high)


def _sample_free_position(
    rng: Random,
    settings: WarehouseSceneSettings,
    blocked_positions: tuple[tuple[float, float, float], ...],
    *,
    footprint_radius: float = 0.0,
    attempts: int | None = None,
) -> tuple[float, float, float] | None:
    half_x = settings.arena_size_m[0] / 2.0 - settings.boundary_margin_m - footprint_radius
    half_y = settings.arena_size_m[1] / 2.0 - settings.boundary_margin_m - footprint_radius
    if half_x <= 0.0 or half_y <= 0.0:
        return None

    for _ in range(attempts or settings.sampling_attempts):
        x = rng.uniform(-half_x, half_x)
        y = rng.uniform(-half_y, half_y)
        candidate = (x, y, settings.target_altitude_m)
        min_clearance = settings.start_goal_clearance_m + footprint_radius
        if all(_xy_distance(candidate, blocked) >= min_clearance for blocked in blocked_positions):
            return candidate
    return None


def _sample_goal_position(
    rng: Random,
    settings: WarehouseSceneSettings,
    start: tuple[float, float, float] | None,
    attempts: int | None = None,
) -> tuple[float, float, float]:
    if start is None:
        raise ValueError("Cannot sample a goal without a valid start position")

    for _ in range(attempts or settings.sampling_attempts):
        candidate = _sample_free_position(rng, settings, (start,))
        if candidate is not None and _xy_distance(start, candidate) >= settings.min_start_goal_distance_m:
            return candidate
    raise ValueError("Failed to sample a start/goal pair with the configured minimum distance")


def _xy_distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _wall_specs(
    settings: WarehouseSceneSettings,
) -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    arena_x, arena_y = settings.arena_size_m
    thickness = settings.wall_thickness_m
    height = settings.wall_height_m
    z = height / 2.0
    return (
        ("WallNorth", (arena_x + 2.0 * thickness, thickness, height), (0.0, arena_y / 2.0, z)),
        ("WallSouth", (arena_x + 2.0 * thickness, thickness, height), (0.0, -arena_y / 2.0, z)),
        ("WallEast", (thickness, arena_y, height), (arena_x / 2.0, 0.0, z)),
        ("WallWest", (thickness, arena_y, height), (-arena_x / 2.0, 0.0, z)),
    )


def _yaw_quat(yaw_degrees: float) -> tuple[float, float, float, float]:
    import math

    half_yaw = math.radians(yaw_degrees) / 2.0
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


AEROSTRIKE_WAREHOUSE_SETTINGS = load_warehouse_scene_settings()

__all__ = [
    "AEROSTRIKE_WAREHOUSE_SETTINGS",
    "DEFAULT_ROOT_PRIM_PATH",
    "WAREHOUSE_ENV_NAME",
    "WarehouseLayout",
    "WarehousePrimitive",
    "WarehouseSceneSettings",
    "load_warehouse_scene_settings",
    "sample_hallway_layout",
    "sample_warehouse_layout",
    "spawn_warehouse_scene",
]
