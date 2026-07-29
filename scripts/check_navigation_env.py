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
    parser.add_argument(
        "--check-speed-aware-proximity",
        action="store_true",
        default=False,
        help="Verify the configured proximity reward scales with forward speed.",
    )
    parser.add_argument(
        "--check-clearance-margin-reward",
        action="store_true",
        default=False,
        help="Verify the configured clearance-margin reward at safe and violating distances.",
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


def check_speed_aware_proximity(env: AeroStrikeNavigationEnv) -> None:
    """Compare shaped and baseline proximity rewards at the same simulated state."""
    if env.cfg.proximity_speed_scale <= 0.0:
        raise RuntimeError("Speed-aware proximity check requires a positive proximity_speed_scale")

    goal_delta_w = env._desired_pos_w - env._robot.data.root_pos_w
    goal_distance = torch.linalg.norm(goal_delta_w, dim=1)
    goal_direction_w = goal_delta_w / goal_distance.unsqueeze(-1).clamp_min(1.0e-6)
    forward_velocity = torch.sum(
        env._robot.data.root_lin_vel_w * goal_direction_w,
        dim=1,
    ).clamp_min(0.0)
    if bool((forward_velocity <= 0.1).any()):
        raise RuntimeError("Speed-aware proximity check requires positive goal-aligned velocity")

    original_proximity_distance_m = env.cfg.proximity_distance_m
    original_proximity_speed_scale = env.cfg.proximity_speed_scale
    env.cfg.proximity_distance_m = env.cfg.ray_sensor_settings.max_range_m
    try:
        min_ray_distance_m = env._get_ray_distances_m().min(dim=1).values
        proximity_ratio = (
            (env.cfg.proximity_distance_m - min_ray_distance_m) / env.cfg.proximity_distance_m
        ).clamp(0.0, 1.0)
        if bool((proximity_ratio <= 0.0).any()):
            raise RuntimeError("Speed-aware proximity check requires a ray hit inside the test band")

        env.cfg.proximity_speed_scale = 0.0
        env._previous_goal_distance[:] = goal_distance
        baseline_reward = env._get_rewards().clone()
        env.cfg.proximity_speed_scale = original_proximity_speed_scale
        env._previous_goal_distance[:] = goal_distance
        shaped_reward = env._get_rewards().clone()

        expected_delta = (
            -env.cfg.proximity_penalty_weight
            * proximity_ratio.square()
            * original_proximity_speed_scale
            * forward_velocity
            / env.cfg.target_speed_mps
            * env.step_dt
        )
        torch.testing.assert_close(shaped_reward - baseline_reward, expected_delta)
        print(
            "[INFO]: Speed-aware proximity reward delta: "
            f"{float((shaped_reward - baseline_reward)[0].detach().cpu())}",
            flush=True,
        )
    finally:
        env.cfg.proximity_distance_m = original_proximity_distance_m
        env.cfg.proximity_speed_scale = original_proximity_speed_scale
        env._previous_goal_distance[:] = goal_distance


def check_clearance_margin_reward(env: AeroStrikeNavigationEnv) -> None:
    """Compare enabled and disabled clearance-margin rewards at the same state."""
    if env.cfg.clearance_margin_penalty_weight <= 0.0:
        raise RuntimeError("Clearance-margin reward check requires a positive penalty weight")

    goal_distance = torch.linalg.norm(
        env._desired_pos_w - env._robot.data.root_pos_w,
        dim=1,
    )
    min_ray_distance_m = env._get_ray_distances_m().min(dim=1).values
    original_clearance_margin_m = env.cfg.clearance_margin_m
    original_penalty_weight = env.cfg.clearance_margin_penalty_weight
    try:
        env.cfg.clearance_margin_m = float(min_ray_distance_m.min().detach().cpu())
        env.cfg.clearance_margin_penalty_weight = 0.0
        env._previous_goal_distance[:] = goal_distance
        safe_baseline_reward = env._get_rewards().clone()
        env.cfg.clearance_margin_penalty_weight = original_penalty_weight
        env._previous_goal_distance[:] = goal_distance
        safe_shaped_reward = env._get_rewards().clone()
        torch.testing.assert_close(
            safe_shaped_reward - safe_baseline_reward,
            torch.zeros_like(safe_baseline_reward),
        )

        env.cfg.clearance_margin_m = float(min_ray_distance_m.max().detach().cpu()) + 0.25
        clearance_deficit_m = (env.cfg.clearance_margin_m - min_ray_distance_m).clamp_min(0.0)
        env.cfg.clearance_margin_penalty_weight = 0.0
        env._previous_goal_distance[:] = goal_distance
        violating_baseline_reward = env._get_rewards().clone()
        env.cfg.clearance_margin_penalty_weight = original_penalty_weight
        env._previous_goal_distance[:] = goal_distance
        violating_shaped_reward = env._get_rewards().clone()
        expected_delta = -original_penalty_weight * clearance_deficit_m.square() * env.step_dt
        torch.testing.assert_close(violating_shaped_reward - violating_baseline_reward, expected_delta)
        print(
            "[INFO]: Clearance-margin reward delta: "
            f"{float((violating_shaped_reward - violating_baseline_reward)[0].detach().cpu())}",
            flush=True,
        )
    finally:
        env.cfg.clearance_margin_m = original_clearance_margin_m
        env.cfg.clearance_margin_penalty_weight = original_penalty_weight
        env._previous_goal_distance[:] = goal_distance


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
        if args_cli.check_speed_aware_proximity:
            check_speed_aware_proximity(env)
        if args_cli.check_clearance_margin_reward:
            check_clearance_margin_reward(env)
        _VALIDATION_COMPLETE = True
        print("[INFO]: Navigation env smoke check complete.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
