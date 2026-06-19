#!/usr/bin/env python3
"""Smoke-check the AeroStrike procedural warehouse scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
AEROSTRIKE_LAB_PATH = WORKSPACE_ROOT / "aerostrike_lab"
if str(AEROSTRIKE_LAB_PATH) not in sys.path:
    sys.path.insert(0, str(AEROSTRIKE_LAB_PATH))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments before launching Isaac Sim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Override the YAML layout seed.")
    parser.add_argument(
        "--scene_variant",
        choices=("warehouse", "hallway"),
        default="warehouse",
        help="Procedural scene variant to spawn.",
    )
    parser.add_argument("--steps", type=int, default=30, help="Number of simulation steps to run.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from aerostrike_lab.scenes.warehouse import (
    load_warehouse_scene_settings,
    sample_warehouse_layout,
    spawn_warehouse_scene,
)
from isaaclab.sim import SimulationContext


def main() -> None:
    """Run a short simulation to verify the warehouse scene spawns."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[13.0, 13.0, 9.0], target=[0.0, 0.0, 1.0])

    settings = load_warehouse_scene_settings()
    layout = sample_warehouse_layout(
        settings=settings,
        seed=args_cli.seed,
        scene_variant=args_cli.scene_variant,
    )
    spawned_layout = spawn_warehouse_scene(layout)

    sim.reset()
    print("[INFO]: AeroStrike warehouse smoke-check setup complete.")
    print(f"[INFO]: Scene variant: {spawned_layout.settings.scene_variant}")
    print(f"[INFO]: Layout seed: {spawned_layout.seed}")
    print(f"[INFO]: Arena size: {spawned_layout.settings.arena_size_m}")
    print(f"[INFO]: Static mesh targets: {len(spawned_layout.mesh_prim_paths)}")
    print(f"[INFO]: Obstacles: {len(spawned_layout.obstacles)}")
    print(f"[INFO]: Start position: {spawned_layout.start_position}")
    print(f"[INFO]: Goal position: {spawned_layout.goal_position}")

    sim_dt = sim.get_physics_dt()
    for _ in range(args_cli.steps):
        sim.step()
        sim.render()

    print("[INFO]: Warehouse smoke check complete.")
    _ = sim_dt


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
