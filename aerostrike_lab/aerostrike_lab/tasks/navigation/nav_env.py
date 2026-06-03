"""AeroStrike Isaac Lab DirectRLEnv skeleton for goal-directed quadrotor navigation."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from aerostrike_lab.assets.quadrotor import AEROSTRIKE_RAY_SENSOR_SETTINGS, make_lidar_pattern_cfg
from aerostrike_lab.scenes.warehouse import sample_warehouse_layout, spawn_warehouse_scene
from aerostrike_lab.tasks.navigation.nav_env_cfg import AeroStrikeNavigationEnvCfg


class AeroStrikeNavigationEnv(DirectRLEnv):
    """Minimal DirectRLEnv lifecycle for AeroStrike navigation."""

    cfg: AeroStrikeNavigationEnvCfg

    def __init__(self, cfg: AeroStrikeNavigationEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._desired_velocity_b = torch.zeros_like(self._actions)
        self._desired_velocity_w = torch.zeros_like(self._actions)
        self._root_velocity_command_w = torch.zeros(self.num_envs, 6, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._min_ray_distance_m = torch.full(
            (self.num_envs,),
            AEROSTRIKE_RAY_SENSOR_SETTINGS.max_range_m,
            dtype=torch.float,
            device=self.device,
        )
        self._reached_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "progress",
                "forward_velocity",
                "proximity",
                "collision",
                "success",
                "instability",
                "action_smoothness",
                "alive",
            ]
        }

        self._prop_body_ids = self._robot.find_bodies("m.*_prop")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._hover_force_per_prop = (
            self._robot_mass * self._gravity_magnitude * self.cfg.hover_thrust_scale / len(self._prop_body_ids)
        )

    def _setup_scene(self) -> None:
        self._warehouse_layout = sample_warehouse_layout(
            seed=self.cfg.warehouse_layout_seed,
            root_prim_path=self.cfg.warehouse_root_prim_path,
        )
        spawn_warehouse_scene(self._warehouse_layout)

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._ray_caster = MultiMeshRayCaster(self._make_raycaster_cfg())
        self.scene.sensors["ray_caster"] = self._ray_caster

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/AmbientLight", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._desired_velocity_b = self._actions * self.cfg.action_velocity_limit_mps
        self._desired_velocity_w = quat_apply(self._robot.data.root_quat_w, self._desired_velocity_b)

    def _apply_action(self) -> None:
        self._root_velocity_command_w[:, :3] = self._desired_velocity_w
        self._root_velocity_command_w[:, 3:] = 0.0
        self._robot.write_root_velocity_to_sim(self._root_velocity_command_w)

        forces = torch.zeros(self.num_envs, len(self._prop_body_ids), 3, device=self.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = self._hover_force_per_prop
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=self._prop_body_ids,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        goal_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        goal_distance = torch.linalg.norm(goal_pos_b, dim=1, keepdim=True)
        goal_direction_b = goal_pos_b / goal_distance.clamp_min(1.0e-6)
        normalized_goal_distance = (goal_distance / self.cfg.goal_distance_normalizer_m).clamp(0.0, 1.0)
        ray_distances = self._get_normalized_ray_distances()
        height = self._robot.data.root_pos_w[:, 2:3] / self.cfg.max_height_m

        obs = torch.cat(
            [
                ray_distances,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                goal_direction_b,
                normalized_goal_distance,
                self._previous_actions,
                height.clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        goal_delta_w = self._desired_pos_w - self._robot.data.root_pos_w
        goal_distance = torch.linalg.norm(goal_delta_w, dim=1)
        goal_direction_w = goal_delta_w / goal_distance.unsqueeze(-1).clamp_min(1.0e-6)
        progress = self._previous_goal_distance - goal_distance
        forward_velocity = torch.sum(self._robot.data.root_lin_vel_w * goal_direction_w, dim=1).clamp_min(0.0)
        speed_score = (forward_velocity / self.cfg.target_speed_mps).clamp(0.0, 1.5)

        ray_distances_m = self._get_ray_distances_m()
        self._min_ray_distance_m = ray_distances_m.min(dim=1).values
        proximity_ratio = (
            (self.cfg.proximity_distance_m - self._min_ray_distance_m) / self.cfg.proximity_distance_m
        ).clamp(0.0, 1.0)
        self._collision = self._min_ray_distance_m <= self.cfg.collision_distance_m
        self._reached_goal = goal_distance <= self.cfg.goal_radius_m

        tilt_error = torch.linalg.norm(self._robot.data.projected_gravity_b[:, :2], dim=1)
        angular_speed = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)
        instability = tilt_error + 0.1 * angular_speed
        action_smoothness = torch.linalg.norm(self._actions - self._previous_actions, dim=1)

        rewards = {
            "progress": self.cfg.progress_weight * progress,
            "forward_velocity": self.cfg.forward_velocity_weight * speed_score * self.step_dt,
            "proximity": -self.cfg.proximity_penalty_weight * proximity_ratio.square() * self.step_dt,
            "collision": -self.cfg.collision_penalty * self._collision.float(),
            "success": self.cfg.success_bonus * self._reached_goal.float(),
            "instability": -self.cfg.instability_penalty_weight * instability * self.step_dt,
            "action_smoothness": -self.cfg.action_smoothness_penalty_weight * action_smoothness * self.step_dt,
            "alive": -torch.full((self.num_envs,), self.cfg.alive_penalty * self.step_dt, device=self.device),
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._previous_goal_distance[:] = goal_distance.detach()
        self._goal_distance[:] = goal_distance.detach()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_height = self._robot.data.root_pos_w[:, 2]
        out_of_height_bounds = torch.logical_or(root_height < self.cfg.min_height_m, root_height > self.cfg.max_height_m)
        ray_distances_m = self._get_ray_distances_m()
        self._min_ray_distance_m = ray_distances_m.min(dim=1).values
        self._collision = self._min_ray_distance_m <= self.cfg.collision_distance_m
        goal_distance = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        self._goal_distance = goal_distance.detach()
        self._reached_goal = goal_distance <= self.cfg.goal_radius_m
        return torch.logical_or(torch.logical_or(out_of_height_bounds, self._collision), self._reached_goal), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if hasattr(self, "_episode_sums"):
            extras = {
                f"Episode_Reward/{key}": torch.mean(value[env_ids]).item()
                for key, value in self._episode_sums.items()
            }
            extras.update(
                {
                    "Episode_Termination/collision": torch.count_nonzero(self._collision[env_ids]).item(),
                    "Episode_Termination/success": torch.count_nonzero(self._reached_goal[env_ids]).item(),
                    "Metrics/final_goal_distance": torch.mean(self._goal_distance[env_ids]).item(),
                    "Metrics/min_ray_distance_m": torch.mean(self._min_ray_distance_m[env_ids]).item(),
                }
            )
            self.extras["log"] = extras
            for value in self._episode_sums.values():
                value[env_ids] = 0.0
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._desired_velocity_b[env_ids] = 0.0
        self._desired_velocity_w[env_ids] = 0.0
        self._root_velocity_command_w[env_ids] = 0.0

        start = torch.tensor(self._warehouse_layout.start_position, device=self.device, dtype=torch.float)
        goal = torch.tensor(self._warehouse_layout.goal_position, device=self.device, dtype=torch.float)
        start = start.clone()
        goal = goal.clone()
        start[2] = self.cfg.start_height_m
        goal[2] = self.cfg.goal_height_m

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = self.scene.env_origins[env_ids] + start
        root_state[:, 7:] = 0.0
        self._desired_pos_w[env_ids] = self.scene.env_origins[env_ids] + goal

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._ray_caster.reset(env_ids)
        self._previous_goal_distance[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - root_state[:, :3],
            dim=1,
        )
        self._goal_distance[env_ids] = self._previous_goal_distance[env_ids]
        self._min_ray_distance_m[env_ids] = AEROSTRIKE_RAY_SENSOR_SETTINGS.max_range_m
        self._reached_goal[env_ids] = False
        self._collision[env_ids] = False

    def _make_raycaster_cfg(self) -> MultiMeshRayCasterCfg:
        sensor_settings = AEROSTRIKE_RAY_SENSOR_SETTINGS
        robot_body_prim = f"{self.cfg.robot.prim_path.rstrip('/')}/{sensor_settings.prim_suffix.lstrip('/')}"
        return MultiMeshRayCasterCfg(
            prim_path=robot_body_prim,
            ray_alignment="base",
            pattern_cfg=make_lidar_pattern_cfg(sensor_settings),
            debug_vis=sensor_settings.debug_vis or self.cfg.debug_vis,
            max_distance=sensor_settings.max_range_m,
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr=self.cfg.warehouse_mesh_prim_expr,
                    is_shared=False,
                    merge_prim_meshes=True,
                    track_mesh_transforms=False,
                )
            ],
        )

    def _get_normalized_ray_distances(self) -> torch.Tensor:
        distances = self._get_ray_distances_m()
        return distances / AEROSTRIKE_RAY_SENSOR_SETTINGS.max_range_m

    def _get_ray_distances_m(self) -> torch.Tensor:
        ray_hits_w = self._ray_caster.data.ray_hits_w
        if ray_hits_w is None or self._ray_caster.data.pos_w is None:
            return torch.ones(
                self.num_envs,
                AEROSTRIKE_RAY_SENSOR_SETTINGS.ray_count,
                dtype=torch.float,
                device=self.device,
            ) * AEROSTRIKE_RAY_SENSOR_SETTINGS.max_range_m
        ray_origins_w = self._ray_caster.data.pos_w.unsqueeze(1)
        distances = torch.linalg.norm(ray_hits_w - ray_origins_w, dim=-1)
        distances = torch.nan_to_num(
            distances,
            nan=AEROSTRIKE_RAY_SENSOR_SETTINGS.max_range_m,
            posinf=AEROSTRIKE_RAY_SENSOR_SETTINGS.max_range_m,
            neginf=AEROSTRIKE_RAY_SENSOR_SETTINGS.max_range_m,
        )
        distances = distances.clamp(
            min=AEROSTRIKE_RAY_SENSOR_SETTINGS.min_range_m,
            max=AEROSTRIKE_RAY_SENSOR_SETTINGS.max_range_m,
        )
        return distances


__all__ = ["AeroStrikeNavigationEnv"]
