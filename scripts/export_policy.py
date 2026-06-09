#!/usr/bin/env python3
"""Export an AeroStrike skrl PPO policy to ONNX for ROS 2 inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = WORKSPACE_ROOT / "checkpoints" / "aerostrike_policy.onnx"
NAVIGATION_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "navigation.yaml"
QUADROTOR_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "quadrotor.yaml"
DEFAULT_METADATA_SUFFIX = ".yaml"
OBSERVATION_DIM = 41
ACTION_DIM = 3
SCALER_EPSILON = 1.0e-8
SCALER_CLIP_THRESHOLD = 5.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a skrl checkpoint file, for example logs/.../checkpoints/best_agent.pt.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Output metadata YAML path. Defaults to the ONNX path with .yaml suffix.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    """Resolve a workspace-relative file path."""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (WORKSPACE_ROOT / path).resolve()
    return path


def make_metadata_path(output_path: Path, path_text: str | None) -> Path:
    """Resolve the metadata output path."""
    if path_text:
        return resolve_path(path_text)
    return output_path.with_suffix(DEFAULT_METADATA_SUFFIX)


def require_dependency(name: str) -> Any:
    """Import a runtime dependency with a clear setup hint."""
    try:
        return __import__(name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing Python dependency '{name}'. Run this exporter with the Isaac Lab Python, "
            "for example: /home/user/IsaacLab/isaaclab.sh -p scripts/export_policy.py ..."
        ) from exc


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Check that the checkpoint has the tensors needed for deterministic policy export."""
    required_sections = {"policy", "observation_preprocessor"}
    missing_sections = sorted(required_sections.difference(checkpoint))
    if missing_sections:
        raise ValueError(f"Checkpoint is missing sections: {missing_sections}")

    required_policy_tensors = {
        "net_container.0.weight",
        "net_container.0.bias",
        "net_container.2.weight",
        "net_container.2.bias",
        "net_container.4.weight",
        "net_container.4.bias",
        "policy_layer.weight",
        "policy_layer.bias",
    }
    missing_tensors = sorted(required_policy_tensors.difference(checkpoint["policy"]))
    if missing_tensors:
        raise ValueError(f"Checkpoint policy is missing tensors: {missing_tensors}")

    required_scaler_tensors = {"running_mean", "running_variance"}
    missing_scaler = sorted(required_scaler_tensors.difference(checkpoint["observation_preprocessor"]))
    if missing_scaler:
        raise ValueError(f"Checkpoint observation preprocessor is missing tensors: {missing_scaler}")


def build_policy_module(
    torch: Any,
    checkpoint: dict[str, Any],
    observation_dim: int,
    action_dim: int,
) -> Any:
    """Create a standalone policy module from a skrl checkpoint."""

    class ExportedPolicy(torch.nn.Module):
        """Deterministic policy mean with embedded skrl observation standardization."""

        def __init__(self) -> None:
            super().__init__()
            policy = checkpoint["policy"]
            scaler = checkpoint["observation_preprocessor"]

            self.register_buffer("running_mean", scaler["running_mean"].float())
            self.register_buffer("running_variance", scaler["running_variance"].float())
            self.net = torch.nn.Sequential(
                torch.nn.Linear(observation_dim, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, 64),
                torch.nn.ELU(),
            )
            self.policy_layer = torch.nn.Linear(64, action_dim)

            self.net[0].weight.data.copy_(policy["net_container.0.weight"])
            self.net[0].bias.data.copy_(policy["net_container.0.bias"])
            self.net[2].weight.data.copy_(policy["net_container.2.weight"])
            self.net[2].bias.data.copy_(policy["net_container.2.bias"])
            self.net[4].weight.data.copy_(policy["net_container.4.weight"])
            self.net[4].bias.data.copy_(policy["net_container.4.bias"])
            self.policy_layer.weight.data.copy_(policy["policy_layer.weight"])
            self.policy_layer.bias.data.copy_(policy["policy_layer.bias"])

        def forward(self, observations: Any) -> Any:
            normalized = (observations - self.running_mean) / (
                torch.sqrt(self.running_variance) + SCALER_EPSILON
            )
            normalized = torch.clamp(normalized, -SCALER_CLIP_THRESHOLD, SCALER_CLIP_THRESHOLD)
            mean_actions = self.policy_layer(self.net(normalized))
            return torch.clamp(mean_actions, -1.0, 1.0)

    return ExportedPolicy().eval()


def load_yaml_file(yaml: Any, path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def section(data: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a YAML section as a dictionary."""
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping for section '{name}'")
    return value


def export_settings(yaml: Any) -> dict[str, Any]:
    """Load export-relevant settings from project YAML configs."""
    navigation = load_yaml_file(yaml, NAVIGATION_CONFIG_PATH)
    quadrotor = load_yaml_file(yaml, QUADROTOR_CONFIG_PATH)
    env = section(navigation, "env")
    action = section(navigation, "action")
    sensor = section(quadrotor, "sensor")
    observation_dim = int(env.get("observation_space", OBSERVATION_DIM))
    action_dim = int(env.get("action_space", ACTION_DIM))
    ray_count = int(sensor["ray_count"])
    return {
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "ray_count": ray_count,
        "velocity_limit_mps": float(action["velocity_limit_mps"]),
        "vertical_velocity_limit_mps": float(action["vertical_velocity_limit_mps"]),
    }


def validate_settings_against_checkpoint(settings: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    """Verify YAML dimensions against checkpoint tensor shapes."""
    observation_dim = settings["observation_dim"]
    action_dim = settings["action_dim"]
    policy = checkpoint["policy"]
    scaler = checkpoint["observation_preprocessor"]
    if tuple(policy["net_container.0.weight"].shape) != (256, observation_dim):
        raise ValueError(
            "navigation.yaml observation_space does not match checkpoint first layer: "
            f"{observation_dim} vs {tuple(policy['net_container.0.weight'].shape)}"
        )
    if tuple(policy["policy_layer.weight"].shape) != (action_dim, 64):
        raise ValueError(
            "navigation.yaml action_space does not match checkpoint policy layer: "
            f"{action_dim} vs {tuple(policy['policy_layer.weight'].shape)}"
        )
    if tuple(scaler["running_mean"].shape) != (observation_dim,):
        raise ValueError(
            "navigation.yaml observation_space does not match checkpoint observation scaler: "
            f"{observation_dim} vs {tuple(scaler['running_mean'].shape)}"
        )


def make_metadata(
    checkpoint_path: Path,
    output_path: Path,
    checkpoint: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Build ROS-facing metadata for the exported policy artifact."""
    current_count = checkpoint["observation_preprocessor"].get("current_count")
    current_count_value = float(current_count.item()) if current_count is not None else None
    ray_count = settings["ray_count"]
    return {
        "policy": {
            "format": "onnx",
            "source_checkpoint": str(checkpoint_path),
            "output_path": str(output_path),
            "type": "skrl_ppo_gaussian_mean",
            "exported_output": "clipped_mean_actions",
            "opset": None,
        },
        "observation": {
            "dimension": settings["observation_dim"],
            "layout": [
                f"ray_distances_normalized[{ray_count}]",
                "root_linear_velocity_body[3]",
                "root_angular_velocity_body[3]",
                "projected_gravity_body[3]",
                "goal_direction_body[3]",
                "goal_distance_normalized[1]",
                "previous_actions[3]",
                "height_normalized[1]",
            ],
            "normalization": {
                "type": "skrl_running_standard_scaler",
                "epsilon": SCALER_EPSILON,
                "clip_threshold": SCALER_CLIP_THRESHOLD,
                "current_count": current_count_value,
                "embedded_in_onnx": True,
            },
        },
        "action": {
            "dimension": settings["action_dim"],
            "layout": ["desired_vx_body", "desired_vy_body", "desired_vz_body"],
            "range": [-1.0, 1.0],
            "velocity_limit_mps": settings["velocity_limit_mps"],
            "vertical_velocity_limit_mps": settings["vertical_velocity_limit_mps"],
        },
        "ros2": {
            "target_node": "policy_node",
            "runtime": "onnxruntime",
        },
    }


def write_metadata(yaml: Any, metadata_path: Path, metadata: dict[str, Any]) -> None:
    """Write export metadata as YAML."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)


def main() -> None:
    """Entrypoint for policy export."""
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    output_path = resolve_path(args.output)
    metadata_path = make_metadata_path(output_path, args.metadata)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    torch = require_dependency("torch")
    yaml = require_dependency("yaml")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_checkpoint(checkpoint)
    settings = export_settings(yaml)
    validate_settings_against_checkpoint(settings, checkpoint)
    policy = build_policy_module(
        torch,
        checkpoint,
        observation_dim=settings["observation_dim"],
        action_dim=settings["action_dim"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_observation = torch.zeros(1, settings["observation_dim"], dtype=torch.float32)
    torch.onnx.export(
        policy,
        dummy_observation,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
    )

    metadata = make_metadata(checkpoint_path, output_path, checkpoint, settings)
    metadata["policy"]["opset"] = args.opset
    write_metadata(yaml, metadata_path, metadata)

    print(f"[INFO] Exported ONNX policy: {output_path}")
    print(f"[INFO] Wrote policy metadata: {metadata_path}")


if __name__ == "__main__":
    main()
