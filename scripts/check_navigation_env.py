#!/usr/bin/env python3
"""Smoke-check the AeroStrike DirectRLEnv skeleton."""

from __future__ import annotations

import argparse
import atexit
import os
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
    parser.add_argument(
        "--profile",
        choices=("v1", "v2", "v2-safe-capture"),
        default="v1",
        help="Environment profile to smoke-check.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments for the smoke check.")
    parser.add_argument("--steps", type=int, default=10, help="Number of zero-action environment steps to run.")
    parser.add_argument(
        "--action",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("VX", "VY", "VZ"),
        help="Normalized body-frame velocity action to apply for every step.",
    )
    parser.add_argument(
        "--min_displacement_m",
        type=float,
        default=0.0,
        help="Minimum root displacement required after stepping.",
    )
    parser.add_argument(
        "--check_goal_vector",
        action="store_true",
        default=False,
        help="Verify the goal-direction observation matches the robot-to-goal vector.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

_VALIDATION_COMPLETE = False


def _fail_if_validation_incomplete() -> None:
    """Make an early Isaac/Python shutdown fail the smoke gate."""
    if not _VALIDATION_COMPLETE:
        print(
            "[ERROR]: Navigation env smoke check exited before validation completed.",
            file=sys.stderr,
            flush=True,
        )
        os._exit(1)


atexit.register(_fail_if_validation_incomplete)

import torch

from aerostrike_lab.tasks.navigation.nav_env import AeroStrikeNavigationEnv
from aerostrike_lab.tasks.navigation.nav_env_cfg import (
    AeroStrikeNavigationEnvCfg,
    AeroStrikeNavigationV2SafeCaptureEnvCfg,
    AeroStrikeNavigationV2EnvCfg,
)
from isaaclab.utils.math import subtract_frame_transforms


def main() -> None:
    """Create, reset, and step the navigation environment."""
    global _VALIDATION_COMPLETE
    if args_cli.profile == "v2":
        cfg = AeroStrikeNavigationV2EnvCfg()
    elif args_cli.profile == "v2-safe-capture":
        cfg = AeroStrikeNavigationV2SafeCaptureEnvCfg()
    else:
        cfg = AeroStrikeNavigationEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = AeroStrikeNavigationEnv(cfg)

    try:
        observations, _ = env.reset()
        initial_root_pos = env._robot.data.root_pos_w.detach().clone()
        action = torch.tensor(args_cli.action, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
        total_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        for _ in range(args_cli.steps):
            observations, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        policy_obs = observations["policy"]
        final_root_pos = env._robot.data.root_pos_w.detach().clone()
        displacement = torch.linalg.norm(final_root_pos - initial_root_pos, dim=1)
        root_pos = final_root_pos[0].detach().cpu().tolist()
        root_velocity = env._robot.data.root_lin_vel_w[0].detach().cpu().tolist()
        goal_pos = env._desired_pos_w[0].detach().cpu().tolist()
        ray_count = cfg.ray_sensor_settings.ray_count
        goal_direction_obs = policy_obs[:, ray_count + 9 : ray_count + 12]
        expected_goal_pos_b, _ = subtract_frame_transforms(
            final_root_pos,
            env._robot.data.root_quat_w,
            env._desired_pos_w,
        )
        expected_goal_distance = torch.linalg.norm(expected_goal_pos_b, dim=1)
        expected_goal_direction_b = expected_goal_pos_b / expected_goal_distance.unsqueeze(-1).clamp_min(1.0e-6)
        goal_direction_error = torch.linalg.norm(goal_direction_obs - expected_goal_direction_b, dim=1)
        print("[INFO]: AeroStrike navigation env smoke-check setup complete.", flush=True)
        print(f"[INFO]: Num envs: {env.num_envs}", flush=True)
        print(f"[INFO]: Observation shape: {tuple(policy_obs.shape)}", flush=True)
        print(f"[INFO]: Action shape: {tuple(action.shape)}", flush=True)
        print(f"[INFO]: Applied action: {list(args_cli.action)}", flush=True)
        print(f"[INFO]: Ray count: {env._ray_caster.num_rays}", flush=True)
        print(f"[INFO]: Robot root position: {root_pos}", flush=True)
        print(f"[INFO]: Robot root velocity: {root_velocity}", flush=True)
        print(f"[INFO]: Robot displacement: {float(displacement[0].detach().cpu())}", flush=True)
        print(f"[INFO]: Total reward: {float(total_reward[0].detach().cpu())}", flush=True)
        print(f"[INFO]: Goal distance: {float(env._goal_distance[0].detach().cpu())}", flush=True)
        print(f"[INFO]: Min ray distance: {float(env._min_ray_distance_m[0].detach().cpu())}", flush=True)
        print(f"[INFO]: Goal position: {goal_pos}", flush=True)
        print(f"[INFO]: Goal direction observation: {goal_direction_obs[0].detach().cpu().tolist()}", flush=True)
        print(f"[INFO]: Goal direction error: {float(goal_direction_error.max().detach().cpu())}", flush=True)
        print(f"[INFO]: Any terminated: {bool(terminated.any())}", flush=True)
        print(f"[INFO]: Any truncated: {bool(truncated.any())}", flush=True)
        if env._ray_caster.num_rays != ray_count:
            raise RuntimeError(
                f"Ray count mismatch: expected {ray_count}, "
                f"got {env._ray_caster.num_rays}"
            )
        if policy_obs.shape[-1] != cfg.observation_space:
            raise RuntimeError(
                f"Observation shape mismatch: expected last dim {cfg.observation_space}, "
                f"got {policy_obs.shape[-1]}"
            )
        if bool((displacement < args_cli.min_displacement_m).any()):
            raise RuntimeError(
                f"Movement check failed: minimum displacement {float(displacement.min().detach().cpu())}m "
                f"is below required {args_cli.min_displacement_m}m"
            )
        if args_cli.check_goal_vector:
            if not torch.isfinite(goal_direction_obs).all():
                raise RuntimeError("Goal direction observation contains non-finite values")
            goal_direction_norm = torch.linalg.norm(goal_direction_obs, dim=1)
            if bool(torch.abs(goal_direction_norm - 1.0).max() > 1.0e-4):
                raise RuntimeError(
                    f"Goal direction norm mismatch: max norm error "
                    f"{float(torch.abs(goal_direction_norm - 1.0).max().detach().cpu())}"
                )
            if bool(goal_direction_error.max() > 1.0e-4):
                raise RuntimeError(
                    f"Goal direction mismatch: max error {float(goal_direction_error.max().detach().cpu())}"
                )
        _VALIDATION_COMPLETE = True
        print("[INFO]: Navigation env smoke check complete.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
