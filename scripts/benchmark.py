#!/usr/bin/env python3
"""Evaluate an AeroStrike skrl checkpoint and print navigation metrics."""

from __future__ import annotations

import argparse
import atexit
import os
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
    parser.add_argument("--min_success_rate", type=float, default=None, help="Fail if success rate is below this.")
    parser.add_argument("--max_collision_rate", type=float, default=None, help="Fail if collision rate is above this.")
    parser.add_argument("--max_timeout_rate", type=float, default=None, help="Fail if timeout rate is above this.")
    parser.add_argument("--min_mean_speed", type=float, default=None, help="Fail if mean speed is below this m/s.")
    parser.add_argument(
        "--min_mean_forward_speed",
        type=float,
        default=None,
        help="Fail if mean forward speed is below this m/s.",
    )
    parser.add_argument(
        "--max_mean_final_goal_distance",
        type=float,
        default=None,
        help="Fail if mean final goal distance is above this meters.",
    )
    parser.add_argument(
        "--min_mean_min_ray_distance",
        type=float,
        default=None,
        help="Fail if mean minimum ray distance is below this meters.",
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


def check_minimum(name: str, value: float, threshold: float | None, failures: list[str]) -> None:
    """Append a gate failure if a metric is below its minimum threshold."""
    if threshold is not None and value < threshold:
        failures.append(f"{name} {value:.3f} < {threshold:.3f}")


def check_maximum(name: str, value: float, threshold: float | None, failures: list[str]) -> None:
    """Append a gate failure if a metric is above its maximum threshold."""
    if threshold is not None and value > threshold:
        failures.append(f"{name} {value:.3f} > {threshold:.3f}")


args_cli, hydra_args_cli = parse_args()
checkpoint_path_cli = resolve_checkpoint_path(args_cli.checkpoint)
sys.argv = [sys.argv[0]] + hydra_args_cli

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

_BENCHMARK_COMPLETE = False


def _enforce_benchmark_exit_status() -> None:
    """Make early Isaac/Python shutdown fail the benchmark gate."""
    if not _BENCHMARK_COMPLETE:
        print(
            "[ERROR]: AeroStrike benchmark exited before metrics completed.",
            file=sys.stderr,
            flush=True,
        )
        os._exit(1)


atexit.register(_enforce_benchmark_exit_status)

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
    global _BENCHMARK_COMPLETE
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
        env_cfg.warehouse_layout_seeds = (args_cli.layout_seed,)
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

        success_rate = successes / completed
        collision_rate = collisions / completed
        timeout_rate = timeouts / completed
        mean_speed = speed_sum / max(speed_samples, 1)
        mean_forward_speed = forward_speed_sum / max(speed_samples, 1)
        mean_vertical_speed = vertical_speed_sum / max(speed_samples, 1)
        mean_final_goal_distance = final_goal_distance_sum / completed
        mean_min_ray_distance = min_ray_distance_sum / completed
        mean_reward_per_env_step = reward_sum / max(steps * raw_env.unwrapped.num_envs, 1)

        print("[INFO]: AeroStrike benchmark complete.", flush=True)
        print(f"[INFO]: Checkpoint: {checkpoint_path_cli}", flush=True)
        print(f"[INFO]: Seed: {args_cli.seed}", flush=True)
        print(f"[INFO]: Layout seed: {raw_env.unwrapped.cfg.warehouse_layout_seed}", flush=True)
        print(f"[INFO]: Num envs: {raw_env.unwrapped.num_envs}", flush=True)
        print(f"[INFO]: Vectorized steps: {steps}", flush=True)
        print(f"[INFO]: Episodes completed: {completed}", flush=True)
        print(f"[INFO]: Successes: {successes}", flush=True)
        print(f"[INFO]: Collisions: {collisions}", flush=True)
        print(f"[INFO]: Timeouts: {timeouts}", flush=True)
        print(f"[INFO]: Other terminations: {other_terminations}", flush=True)
        print(f"[INFO]: Success rate: {success_rate:.3f}", flush=True)
        print(f"[INFO]: Collision rate: {collision_rate:.3f}", flush=True)
        print(f"[INFO]: Timeout rate: {timeout_rate:.3f}", flush=True)
        print(f"[INFO]: Mean speed m/s: {mean_speed:.3f}", flush=True)
        print(f"[INFO]: Mean forward speed m/s: {mean_forward_speed:.3f}", flush=True)
        print(f"[INFO]: Mean vertical speed m/s: {mean_vertical_speed:.3f}", flush=True)
        print(f"[INFO]: Mean final goal distance m: {mean_final_goal_distance:.3f}", flush=True)
        print(f"[INFO]: Mean min ray distance m: {mean_min_ray_distance:.3f}", flush=True)
        print(f"[INFO]: Mean reward per env-step: {mean_reward_per_env_step:.3f}", flush=True)

        gate_failures: list[str] = []
        check_minimum("success rate", success_rate, args_cli.min_success_rate, gate_failures)
        check_maximum("collision rate", collision_rate, args_cli.max_collision_rate, gate_failures)
        check_maximum("timeout rate", timeout_rate, args_cli.max_timeout_rate, gate_failures)
        check_minimum("mean speed", mean_speed, args_cli.min_mean_speed, gate_failures)
        check_minimum(
            "mean forward speed",
            mean_forward_speed,
            args_cli.min_mean_forward_speed,
            gate_failures,
        )
        check_maximum(
            "mean final goal distance",
            mean_final_goal_distance,
            args_cli.max_mean_final_goal_distance,
            gate_failures,
        )
        check_minimum(
            "mean min ray distance",
            mean_min_ray_distance,
            args_cli.min_mean_min_ray_distance,
            gate_failures,
        )
        if gate_failures:
            _BENCHMARK_COMPLETE = True
            print("[ERROR]: Benchmark gate failed:", flush=True)
            for failure in gate_failures:
                print(f"[ERROR]: - {failure}", flush=True)
            raise RuntimeError(f"Benchmark gate failed: {'; '.join(gate_failures)}")
        if any(
            threshold is not None
            for threshold in (
                args_cli.min_success_rate,
                args_cli.max_collision_rate,
                args_cli.max_timeout_rate,
                args_cli.min_mean_speed,
                args_cli.min_mean_forward_speed,
                args_cli.max_mean_final_goal_distance,
                args_cli.min_mean_min_ray_distance,
            )
        ):
            print("[INFO]: Benchmark gate passed.", flush=True)
        _BENCHMARK_COMPLETE = True
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
        if not _BENCHMARK_COMPLETE:
            raise RuntimeError("AeroStrike benchmark exited before metrics completed")
    finally:
        simulation_app.close()
