#!/usr/bin/env python3
"""Smoke-check the AeroStrike Crazyflie asset and RayCaster config."""

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
    parser.add_argument("--steps", type=int, default=120, help="Number of simulation steps to run.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from aerostrike_lab.assets.quadrotor import (
    AEROSTRIKE_RAY_SENSOR_SETTINGS,
    make_quadrotor_cfg,
    make_raycaster_cfg,
)
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster
from isaaclab.sim import SimulationContext


def main() -> None:
    """Run a short simulation to verify the Crazyflie spawns."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.2], target=[0.0, 0.0, 0.5])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = make_quadrotor_cfg(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    ray_cfg = make_raycaster_cfg(
        robot_prim_path="/World/Crazyflie",
        settings=AEROSTRIKE_RAY_SENSOR_SETTINGS,
        debug_vis=not args_cli.headless,
        mesh_prim_paths=("/World/defaultGroundPlane",),
    )
    ray_caster = RayCaster(ray_cfg)

    sim.reset()
    print("[INFO]: AeroStrike quadrotor smoke-check setup complete.")
    print(f"[INFO]: Robot prim: {robot_cfg.prim_path}")
    print(f"[INFO]: RayCaster prim: {ray_cfg.prim_path}")
    print(f"[INFO]: Ray count: {AEROSTRIKE_RAY_SENSOR_SETTINGS.ray_count}")

    sim_dt = sim.get_physics_dt()
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    for _ in range(args_cli.steps):
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = robot_mass * gravity / 4.0
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=prop_body_ids,
        )
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        ray_caster.update(sim_dt)

    root_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
    print(f"[INFO]: Smoke check complete. root_pos={root_pos}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
