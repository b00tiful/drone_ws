#!/usr/bin/env python3
"""Evaluate an AeroStrike skrl checkpoint and print navigation metrics."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
AEROSTRIKE_LAB_PATH = WORKSPACE_ROOT / "aerostrike_lab"
if str(AEROSTRIKE_LAB_PATH) not in sys.path:
    sys.path.insert(0, str(AEROSTRIKE_LAB_PATH))

from isaaclab.app import AppLauncher

DEFAULT_TASK_ID = "AeroStrike-Navigation-Direct-v0"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher arguments and leave Hydra overrides for Isaac Lab."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a skrl checkpoint file, or a directory containing checkpoint files.",
    )
    parser.add_argument("--episodes", type=int, default=32, help="Completed episodes to collect.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum vectorized env steps before stopping.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK_ID, help="Gym task id to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed. Use -1 to randomize.")
    parser.add_argument(
        "--layout_seed",
        type=int,
        default=None,
        help="Warehouse layout seed override for robustness checks.",
    )
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch", "jax"],
        help="skrl backend.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_known_args()


def resolve_checkpoint_path(path_text: str) -> Path:
    """Resolve and validate a local skrl checkpoint path before launching Isaac Sim."""
    checkpoint_path = Path(path_text).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (WORKSPACE_ROOT / checkpoint_path).resolve()
    if checkpoint_path.is_file():
        return checkpoint_path
    if checkpoint_path.is_dir():
        best_checkpoints = sorted(
            checkpoint_path.glob("**/best_agent.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if best_checkpoints:
            return best_checkpoints[0]
        checkpoints = sorted(
            checkpoint_path.glob("**/*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if checkpoints:
            return checkpoints[0]
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")


args_cli, hydra_args_cli = parse_args()
checkpoint_path_cli = resolve_checkpoint_path(args_cli.checkpoint)
sys.argv = [sys.argv[0]] + hydra_args_cli

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import skrl
import torch
from packaging import version

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import aerostrike_lab.tasks.navigation  # noqa: F401

SKRL_VERSION = "2.0.0"


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict) -> None:
    """Load a checkpoint, run evaluation episodes, and print aggregate metrics."""
    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        raise RuntimeError(f"Unsupported skrl version {skrl.__version__}; expected >= {SKRL_VERSION}")
    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args_cli.max_steps <= 0:
        raise ValueError("--max_steps must be positive")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.layout_seed is not None:
        env_cfg.warehouse_layout_seed = args_cli.layout_seed
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    if args_cli.ml_framework == "jax":
        skrl.config.jax.backend = "jax"

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(raw_env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, agent_cfg)
    runner.agent.load(str(checkpoint_path_cli))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    completed = 0
    successes = 0
    collisions = 0
    timeouts = 0
    other_terminations = 0
    reward_sum = 0.0
    speed_sum = 0.0
    forward_speed_sum = 0.0
    vertical_speed_sum = 0.0
    final_goal_distance_sum = 0.0
    min_ray_distance_sum = 0.0
    speed_samples = 0
    steps = 0

    try:
        obs, _ = env.reset()
        states = env.state()
        while completed < args_cli.episodes and steps < args_cli.max_steps:
            with torch.inference_mode():
                outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                obs, reward, terminated, truncated, _ = env.step(actions)
                states = env.state()

            steps += 1
            reward_sum += float(reward.sum().detach().cpu())
            root_velocity_w = raw_env.unwrapped._robot.data.root_lin_vel_w
            goal_delta_w = raw_env.unwrapped._desired_pos_w - raw_env.unwrapped._robot.data.root_pos_w
            goal_distance = torch.linalg.norm(goal_delta_w, dim=1, keepdim=True)
            goal_direction_w = goal_delta_w / goal_distance.clamp_min(1.0e-6)
            root_speed = torch.linalg.norm(root_velocity_w, dim=1)
            forward_speed = torch.sum(root_velocity_w * goal_direction_w, dim=1).clamp_min(0.0)
            speed_sum += float(root_speed.sum().detach().cpu())
            forward_speed_sum += float(forward_speed.sum().detach().cpu())
            vertical_speed_sum += float(torch.abs(root_velocity_w[:, 2]).sum().detach().cpu())
            speed_samples += int(root_speed.numel())

            terminated_count = int(torch.count_nonzero(terminated).detach().cpu())
            truncated_count = int(torch.count_nonzero(truncated).detach().cpu())
            if terminated_count or truncated_count:
                log = raw_env.unwrapped.extras.get("log", {})
                step_successes = int(log.get("Episode_Termination/success", 0))
                step_collisions = int(log.get("Episode_Termination/collision", 0))
                step_completed = terminated_count + truncated_count

                successes += step_successes
                collisions += step_collisions
                timeouts += truncated_count
                other_terminations += max(0, terminated_count - step_successes - step_collisions)
                final_goal_distance_sum += float(log.get("Metrics/final_goal_distance", 0.0)) * step_completed
                min_ray_distance_sum += float(log.get("Metrics/min_ray_distance_m", 0.0)) * step_completed
                completed += step_completed

        if completed == 0:
            raise RuntimeError("No episodes completed before max_steps; increase --max_steps or lower --episodes")

        print("[INFO]: AeroStrike benchmark complete.")
        print(f"[INFO]: Checkpoint: {checkpoint_path_cli}")
        print(f"[INFO]: Seed: {args_cli.seed}")
        print(f"[INFO]: Layout seed: {raw_env.unwrapped.cfg.warehouse_layout_seed}")
        print(f"[INFO]: Num envs: {raw_env.unwrapped.num_envs}")
        print(f"[INFO]: Vectorized steps: {steps}")
        print(f"[INFO]: Episodes completed: {completed}")
        print(f"[INFO]: Successes: {successes}")
        print(f"[INFO]: Collisions: {collisions}")
        print(f"[INFO]: Timeouts: {timeouts}")
        print(f"[INFO]: Other terminations: {other_terminations}")
        print(f"[INFO]: Success rate: {successes / completed:.3f}")
        print(f"[INFO]: Collision rate: {collisions / completed:.3f}")
        print(f"[INFO]: Timeout rate: {timeouts / completed:.3f}")
        print(f"[INFO]: Mean speed m/s: {speed_sum / max(speed_samples, 1):.3f}")
        print(f"[INFO]: Mean forward speed m/s: {forward_speed_sum / max(speed_samples, 1):.3f}")
        print(f"[INFO]: Mean vertical speed m/s: {vertical_speed_sum / max(speed_samples, 1):.3f}")
        print(f"[INFO]: Mean final goal distance m: {final_goal_distance_sum / completed:.3f}")
        print(f"[INFO]: Mean min ray distance m: {min_ray_distance_sum / completed:.3f}")
        print(f"[INFO]: Mean reward per env-step: {reward_sum / max(steps * raw_env.unwrapped.num_envs, 1):.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
