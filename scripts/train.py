#!/usr/bin/env python3
"""Train the AeroStrike navigation policy with Isaac Lab and skrl PPO."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime
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
    parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
    parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval in steps.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK_ID, help="Gym task id to train.")
    parser.add_argument("--seed", type=int, default=None, help="Training seed. Use -1 to randomize.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file or directory to resume from.",
    )
    parser.add_argument("--max_iterations", type=int, default=None, help="PPO iterations to train.")
    parser.add_argument(
        "--layout_seed",
        type=int,
        default=None,
        help="Warehouse layout seed override for curriculum training.",
    )
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch", "jax"],
        help="skrl backend.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    if args.video:
        args.enable_cameras = True
    return args, hydra_args


def resolve_checkpoint_path(path_text: str) -> Path:
    """Resolve a local skrl checkpoint file or directory before launching Isaac Sim."""
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
checkpoint_path_cli = resolve_checkpoint_path(args_cli.checkpoint) if args_cli.checkpoint else None
sys.argv = [sys.argv[0]] + hydra_args_cli

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import skrl
from packaging import version

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import aerostrike_lab.tasks.navigation  # noqa: F401

SKRL_VERSION = "2.0.0"


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict) -> None:
    """Create the env, wrap it for skrl, and run PPO training."""
    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        raise RuntimeError(f"Unsupported skrl version {skrl.__version__}; expected >= {SKRL_VERSION}")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.layout_seed is not None:
        env_cfg.warehouse_layout_seed = args_cli.layout_seed
        env_cfg.warehouse_layout_seeds = (args_cli.layout_seed,)
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    if args_cli.ml_framework == "jax":
        skrl.config.jax.backend = "jax"

    log_root_path = os.path.abspath(
        os.path.join(str(WORKSPACE_ROOT), "logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    )
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_ppo_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    log_dir = os.path.join(log_root_path, log_dir_name)
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir_name
    env_cfg.log_dir = log_dir

    print(f"[INFO] Logging experiment in directory: {log_dir}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, agent_cfg)
    if checkpoint_path_cli:
        print(f"[INFO] Loading model checkpoint from: {checkpoint_path_cli}")
        runner.agent.load(str(checkpoint_path_cli))

    start_time = time.time()
    runner.run()
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
