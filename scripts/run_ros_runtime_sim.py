#!/usr/bin/env python3
"""Run the Isaac Lab warehouse sim as fake ROS 2 sensor/command I/O."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Literal

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
AEROSTRIKE_LAB_PATH = WORKSPACE_ROOT / "aerostrike_lab"
if str(AEROSTRIKE_LAB_PATH) not in sys.path:
    sys.path.insert(0, str(AEROSTRIKE_LAB_PATH))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments before launching Isaac Sim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2500, help="Maximum Isaac env steps to run.")
    parser.add_argument(
        "--stop_on_termination",
        action="store_true",
        help="Exit after publishing the first terminal episode state instead of resetting.",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Launch Isaac Sim with a viewport instead of the default headless mode.",
    )
    parser.add_argument(
        "--camera_mode",
        choices=("free", "third_person", "first_person"),
        default="third_person",
        help="Visible-mode camera behavior. Use free to leave the viewport user-controlled.",
    )
    parser.add_argument(
        "--camera_distance_m",
        type=float,
        default=0.40,
        help="Third-person follow distance behind the drone.",
    )
    parser.add_argument(
        "--camera_height_m",
        type=float,
        default=0.25,
        help="Third-person camera height above the drone.",
    )
    parser.add_argument(
        "--camera_target_height_m",
        type=float,
        default=0.20,
        help="Camera target height above the drone root.",
    )
    parser.add_argument(
        "--camera_smoothing",
        type=float,
        default=0.50,
        help="Camera smoothing alpha in [0, 1]; 1 snaps to the target pose.",
    )
    parser.add_argument(
        "--camera_max_yaw_rate_dps",
        type=float,
        default=90.0,
        help="Maximum visible camera heading turn rate in degrees per second.",
    )
    parser.add_argument(
        "--scene_variant",
        choices=("warehouse", "hallway"),
        default="warehouse",
        help="Procedural scene variant to run without changing the policy observation/action contract.",
    )
    parser.add_argument(
        "--demo_robot_marker",
        action="store_true",
        help="Show a non-physics visual marker on the drone for video readability.",
    )
    parser.add_argument(
        "--demo_robot_marker_radius_m",
        type=float,
        default=0.16,
        help="Half-length of the optional drone readability marker axes.",
    )
    parser.add_argument("--publish_rate_hz", type=float, default=50.0, help="ROS state/ray publish rate.")
    parser.add_argument(
        "--real_time_factor",
        type=float,
        default=1.0,
        help="Wall-clock pacing for Isaac env steps; use 0 to run as fast as possible.",
    )
    parser.add_argument(
        "--command_timeout_s",
        type=float,
        default=0.25,
        help="Zero the simulated velocity if no command arrives within this many seconds.",
    )
    parser.add_argument("--layout_seed", type=int, default=7, help="Single warehouse layout seed to run.")
    parser.add_argument(
        "--odom_topic",
        type=str,
        default="/aerostrike/odom",
        help="Odometry topic consumed by observation_builder.",
    )
    parser.add_argument(
        "--ray_distances_topic",
        type=str,
        default="/aerostrike/ray_distances",
        help="Raw ray-distance topic consumed by observation_builder.",
    )
    parser.add_argument(
        "--command_topic",
        type=str,
        default="/aerostrike/body_velocity_cmd",
        help="Body-frame command topic produced by command_adapter.",
    )
    parser.add_argument(
        "--terminal_metrics_topic",
        type=str,
        default="/aerostrike/terminal_metrics",
        help="Episode terminal metrics topic consumed by metrics_logger.",
    )
    parser.add_argument("--world_frame_id", type=str, default="world", help="Odometry frame id.")
    parser.add_argument("--body_frame_id", type=str, default="base_link", help="Odometry child frame id.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    if args.visible:
        args.headless = False
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import rclpy
import torch
import isaaclab.sim as sim_utils
from geometry_msgs.msg import TwistStamped
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

from aerostrike_lab.assets.quadrotor import AEROSTRIKE_RAY_SENSOR_SETTINGS
from aerostrike_lab.tasks.navigation.nav_env import AeroStrikeNavigationEnv
from aerostrike_lab.tasks.navigation.nav_env_cfg import AeroStrikeNavigationEnvCfg


CameraMode = Literal["free", "third_person", "first_person"]


def sensor_qos() -> QoSProfile:
    """Return a sensor-style QoS profile matching the C++ observation builder."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )


def command_qos() -> QoSProfile:
    """Return a reliable command QoS profile matching command_adapter."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


class IsaacRuntimeBridge(Node):
    """Publish Isaac state/rays and store the latest body-frame velocity command."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("isaac_runtime_bridge")
        self._args = args
        self._latest_command_b = torch.zeros(3, dtype=torch.float)
        self._last_command_wall_time = 0.0

        self._odom_pub = self.create_publisher(Odometry, args.odom_topic, sensor_qos())
        self._rays_pub = self.create_publisher(Float32MultiArray, args.ray_distances_topic, sensor_qos())
        self._terminal_metrics_pub = self.create_publisher(
            Float32MultiArray,
            args.terminal_metrics_topic,
            command_qos(),
        )
        self._command_sub = self.create_subscription(
            TwistStamped,
            args.command_topic,
            self._handle_command,
            command_qos(),
        )

    def _handle_command(self, msg: TwistStamped) -> None:
        values = (
            float(msg.twist.linear.x),
            float(msg.twist.linear.y),
            float(msg.twist.linear.z),
        )
        if not all(math.isfinite(value) for value in values):
            self.get_logger().warn("Ignoring non-finite body velocity command")
            return

        self._latest_command_b = torch.tensor(values, dtype=torch.float)
        self._last_command_wall_time = time.monotonic()

    def normalized_action(self, env: AeroStrikeNavigationEnv) -> torch.Tensor:
        """Convert latest body-frame m/s command into the env's normalized action."""
        if time.monotonic() - self._last_command_wall_time > self._args.command_timeout_s:
            command_b = torch.zeros(3, dtype=torch.float, device=env.device)
        else:
            command_b = self._latest_command_b.to(device=env.device)

        action = torch.empty(1, 3, dtype=torch.float, device=env.device)
        action[0, 0] = command_b[0] / env.cfg.action_velocity_limit_mps
        action[0, 1] = command_b[1] / env.cfg.action_velocity_limit_mps
        action[0, 2] = command_b[2] / env.cfg.action_vertical_velocity_limit_mps
        return action.clamp(-1.0, 1.0)

    def publish_state(self, env: AeroStrikeNavigationEnv) -> None:
        """Publish odometry and raw ray distances for env 0."""
        now = self.get_clock().now().to_msg()

        root_pos_w = env._robot.data.root_pos_w[0].detach().cpu()
        root_quat_w = env._robot.data.root_quat_w[0].detach().cpu()
        root_lin_vel_b = env._robot.data.root_lin_vel_b[0].detach().cpu()
        root_ang_vel_b = env._robot.data.root_ang_vel_b[0].detach().cpu()
        ray_distances_m = env._get_ray_distances_m()[0].detach().cpu()

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self._args.world_frame_id
        odom.child_frame_id = self._args.body_frame_id
        odom.pose.pose.position.x = float(root_pos_w[0])
        odom.pose.pose.position.y = float(root_pos_w[1])
        odom.pose.pose.position.z = float(root_pos_w[2])
        odom.pose.pose.orientation.w = float(root_quat_w[0])
        odom.pose.pose.orientation.x = float(root_quat_w[1])
        odom.pose.pose.orientation.y = float(root_quat_w[2])
        odom.pose.pose.orientation.z = float(root_quat_w[3])
        odom.twist.twist.linear.x = float(root_lin_vel_b[0])
        odom.twist.twist.linear.y = float(root_lin_vel_b[1])
        odom.twist.twist.linear.z = float(root_lin_vel_b[2])
        odom.twist.twist.angular.x = float(root_ang_vel_b[0])
        odom.twist.twist.angular.y = float(root_ang_vel_b[1])
        odom.twist.twist.angular.z = float(root_ang_vel_b[2])
        self._odom_pub.publish(odom)

        rays = Float32MultiArray()
        rays.data = [float(value) for value in ray_distances_m.tolist()]
        self._rays_pub.publish(rays)

    def publish_terminal_metrics(
        self,
        env: AeroStrikeNavigationEnv,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Publish episode-level metrics captured by Isaac Lab before auto-reset."""
        log = env.extras.get("log", {})
        terminal = Float32MultiArray()
        terminal.data = [
            float(log.get("Episode_Termination/success", 0.0)),
            float(log.get("Episode_Termination/collision", 0.0)),
            float(torch.count_nonzero(truncated).detach().cpu().item()),
            float(log.get("Metrics/final_goal_distance", math.nan)),
            float(log.get("Metrics/min_ray_distance_m", math.nan)),
        ]
        self._terminal_metrics_pub.publish(terminal)
        self.get_logger().info(
            "Terminal metrics: success=%.0f collision=%.0f timeout=%.0f "
            "final_goal_distance=%.3f min_ray=%.3f terminated=%s truncated=%s"
            % (
                terminal.data[0],
                terminal.data[1],
                terminal.data[2],
                terminal.data[3],
                terminal.data[4],
                bool(torch.count_nonzero(terminated).detach().cpu().item()),
                bool(torch.count_nonzero(truncated).detach().cpu().item()),
            )
        )


class DemoCamera:
    """Visible-mode viewport camera tracking for demo recording."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._mode: CameraMode = args.camera_mode
        self._distance_m = args.camera_distance_m
        self._height_m = args.camera_height_m
        self._target_height_m = args.camera_target_height_m
        self._smoothing = args.camera_smoothing
        self._max_yaw_rate_rad_s = math.radians(args.camera_max_yaw_rate_dps)
        self._eye: torch.Tensor | None = None
        self._target: torch.Tensor | None = None
        self._forward: torch.Tensor | None = None
        self._last_update_wall_time = time.monotonic()

    def update(self, env: AeroStrikeNavigationEnv) -> None:
        """Update the active Isaac viewport camera."""
        if self._mode == "free":
            return

        now = time.monotonic()
        dt = max(1.0e-3, now - self._last_update_wall_time)
        self._last_update_wall_time = now
        root_pos = env._robot.data.root_pos_w[0].detach().cpu()
        goal_pos = env._desired_pos_w[0].detach().cpu()
        desired_forward = goal_pos - root_pos
        desired_forward[2] = 0.0
        norm = torch.linalg.norm(desired_forward).item()
        if norm < 1.0e-4:
            desired_forward = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float)
        else:
            desired_forward = desired_forward / norm

        forward = self._limit_forward_yaw(desired_forward, dt)

        if self._mode == "third_person":
            desired_eye = root_pos - forward * self._distance_m
            desired_eye[2] = root_pos[2] + self._height_m
            desired_target = root_pos + forward * 2.0
            desired_target[2] = root_pos[2] + self._target_height_m
        else:
            desired_eye = root_pos + forward * 0.35
            desired_eye[2] = root_pos[2] + self._target_height_m
            desired_target = root_pos + forward * max(3.0, self._distance_m)
            desired_target[2] = root_pos[2] + self._target_height_m

        alpha = max(0.0, min(1.0, self._smoothing))
        if self._eye is None or self._target is None:
            self._eye = desired_eye
            self._target = desired_target
        else:
            self._eye = self._eye * (1.0 - alpha) + desired_eye * alpha
            self._target = self._target * (1.0 - alpha) + desired_target * alpha

        env.sim.set_camera_view(eye=self._eye.tolist(), target=self._target.tolist())

    def _limit_forward_yaw(self, desired_forward: torch.Tensor, dt: float) -> torch.Tensor:
        if self._forward is None:
            self._forward = desired_forward
            return desired_forward

        current_yaw = math.atan2(float(self._forward[1]), float(self._forward[0]))
        desired_yaw = math.atan2(float(desired_forward[1]), float(desired_forward[0]))
        yaw_error = math.atan2(math.sin(desired_yaw - current_yaw), math.cos(desired_yaw - current_yaw))
        max_delta = max(0.0, self._max_yaw_rate_rad_s) * dt
        yaw_delta = max(-max_delta, min(max_delta, yaw_error))
        yaw = current_yaw + yaw_delta
        self._forward = torch.tensor((math.cos(yaw), math.sin(yaw), 0.0), dtype=torch.float)
        return self._forward


class DemoRobotMarker:
    """Non-physics visual marker that makes the small Crazyflie readable on video."""

    def __init__(self, radius_m: float) -> None:
        axis_thickness_m = max(0.018, radius_m * 0.18)
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/AeroStrikeDroneMarker",
            markers={
                "marker_x": sim_utils.CuboidCfg(
                    size=(radius_m * 2.0, axis_thickness_m, axis_thickness_m),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.85, 1.0),
                        emissive_color=(0.0, 0.18, 0.25),
                        roughness=0.35,
                    ),
                ),
                "marker_y": sim_utils.CuboidCfg(
                    size=(axis_thickness_m, radius_m * 2.0, axis_thickness_m),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.52, 0.12),
                        emissive_color=(0.22, 0.08, 0.0),
                        roughness=0.35,
                    ),
                ),
                "marker_z": sim_utils.CuboidCfg(
                    size=(axis_thickness_m, axis_thickness_m, radius_m * 1.6),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.95, 0.92),
                        emissive_color=(0.16, 0.16, 0.14),
                        roughness=0.35,
                    ),
                ),
            },
        )
        self._visualizer = VisualizationMarkers(marker_cfg)

    def update(self, env: AeroStrikeNavigationEnv) -> None:
        position = env._robot.data.root_pos_w[0].detach().cpu()
        positions = position.repeat((3, 1))
        orientations = env._robot.data.root_quat_w[0].detach().cpu().repeat((3, 1))
        self._visualizer.visualize(
            positions,
            orientations,
            marker_indices=torch.tensor((0, 1, 2), dtype=torch.int32),
        )


def make_env(args: argparse.Namespace) -> AeroStrikeNavigationEnv:
    """Create the single-env warehouse simulation used by the ROS bridge."""
    cfg = AeroStrikeNavigationEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device if args.device is not None else cfg.sim.device
    cfg.warehouse_scene_variant = args.scene_variant
    cfg.warehouse_layout_seed = args.layout_seed
    cfg.warehouse_layout_seeds = (args.layout_seed,)
    return AeroStrikeNavigationEnv(cfg)


def main() -> None:
    """Run Isaac Sim and bridge one environment to ROS 2 topics."""
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive")
    if args_cli.publish_rate_hz <= 0.0:
        raise ValueError("--publish_rate_hz must be positive")
    if args_cli.real_time_factor < 0.0:
        raise ValueError("--real_time_factor must be non-negative")
    if args_cli.command_timeout_s <= 0.0:
        raise ValueError("--command_timeout_s must be positive")
    if args_cli.camera_distance_m <= 0.0:
        raise ValueError("--camera_distance_m must be positive")
    if args_cli.camera_height_m <= 0.0:
        raise ValueError("--camera_height_m must be positive")
    if args_cli.camera_smoothing < 0.0 or args_cli.camera_smoothing > 1.0:
        raise ValueError("--camera_smoothing must be in [0, 1]")
    if args_cli.camera_max_yaw_rate_dps <= 0.0:
        raise ValueError("--camera_max_yaw_rate_dps must be positive")
    if args_cli.demo_robot_marker_radius_m <= 0.0:
        raise ValueError("--demo_robot_marker_radius_m must be positive")

    rclpy.init()
    node: IsaacRuntimeBridge | None = None
    env: AeroStrikeNavigationEnv | None = None
    camera: DemoCamera | None = None
    marker: DemoRobotMarker | None = None
    env_started = False
    publish_period_s = 1.0 / args_cli.publish_rate_hz
    next_publish_wall_time = time.monotonic()

    try:
        node = IsaacRuntimeBridge(args_cli)
        env = make_env(args_cli)
        env.reset()
        env_started = True
        if args_cli.visible:
            camera = DemoCamera(args_cli)
            camera.update(env)
            if args_cli.demo_robot_marker:
                marker = DemoRobotMarker(args_cli.demo_robot_marker_radius_m)
                marker.update(env)
        if env._ray_caster.num_rays != AEROSTRIKE_RAY_SENSOR_SETTINGS.ray_count:
            raise RuntimeError(
                f"Ray count mismatch: expected {AEROSTRIKE_RAY_SENSOR_SETTINGS.ray_count}, "
                f"got {env._ray_caster.num_rays}"
            )

        start_pos = env._robot.data.root_pos_w[0].detach().cpu().tolist()
        goal_pos = env._desired_pos_w[0].detach().cpu().tolist()
        node.get_logger().info(
            "Isaac runtime bridge ready: "
            f"{args_cli.odom_topic} + {args_cli.ray_distances_topic} -> "
            f"{args_cli.command_topic}, layout_seed={args_cli.layout_seed}, "
            f"scene_variant={args_cli.scene_variant}"
        )
        node.get_logger().info(
            "Simulation start="
            f"({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}) "
            "goal="
            f"({goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}); "
            "keep goal_publisher.yaml aligned with this goal"
        )

        for _ in range(args_cli.steps):
            step_started_at = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.0)
            action = node.normalized_action(env)
            _, _, terminated, truncated, _ = env.step(action)
            if camera is not None:
                camera.update(env)
            if marker is not None:
                marker.update(env)

            if bool(torch.logical_or(terminated, truncated).any()):
                node.publish_terminal_metrics(env, terminated, truncated)
                node.publish_state(env)
                if args_cli.stop_on_termination:
                    rclpy.spin_once(node, timeout_sec=0.05)
                    time.sleep(0.1)
                    break
                env.reset()
                if camera is not None:
                    camera.update(env)
                if marker is not None:
                    marker.update(env)
                next_publish_wall_time = time.monotonic() + publish_period_s
                continue

            now = time.monotonic()
            if now >= next_publish_wall_time:
                node.publish_state(env)
                next_publish_wall_time = now + publish_period_s

            if args_cli.real_time_factor > 0.0:
                target_step_wall_s = env.step_dt / args_cli.real_time_factor
                elapsed_s = time.monotonic() - step_started_at
                if elapsed_s < target_step_wall_s:
                    time.sleep(target_step_wall_s - elapsed_s)
    finally:
        if env is not None:
            if env_started:
                zero_action = torch.zeros(1, 3, dtype=torch.float, device=env.device)
                env.step(zero_action)
            env.close()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
