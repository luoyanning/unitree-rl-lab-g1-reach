from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, sample_uniform, yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


POINT_GOAL_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/point_goal")
POINT_GOAL_MARKER_CFG.markers["frame"].scale = (0.16, 0.16, 0.16)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class PointGoalCommand(CommandTerm):
    cfg: PointGoalCommandCfg

    def __init__(self, cfg: PointGoalCommandCfg, env):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_heading_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["guidance_speed"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def set_goal_positions(self, env_ids: Sequence[int] | torch.Tensor, goal_pos_w: torch.Tensor):
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        self.goal_pos_w[env_ids, :2] = goal_pos_w[:, :2]
        self.goal_pos_w[env_ids, 2] = self._env.scene.env_origins[env_ids, 2] + self.cfg.target_height_offset
        self._update_guidance_command()

    def _goal_delta_body_xy(self) -> torch.Tensor:
        goal_delta_w = torch.zeros(self.num_envs, 3, device=self.device)
        goal_delta_w[:, :2] = self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        return quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), goal_delta_w)[:, :2]

    def _update_guidance_command(self):
        goal_delta_body_xy = self._goal_delta_body_xy()
        goal_delta_w_xy = self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        goal_distance = torch.linalg.norm(goal_delta_w_xy, dim=-1)
        goal_heading_error = _wrap_to_pi(torch.atan2(goal_delta_body_xy[:, 1], goal_delta_body_xy[:, 0]))
        heading_alignment = torch.clamp(torch.cos(goal_heading_error), min=0.0, max=1.0)
        heading_turn_gate = (torch.abs(goal_heading_error) < self.cfg.turn_in_place_threshold).float()

        distance_scale = torch.clamp(goal_distance / self.cfg.slow_down_distance, min=0.0, max=1.0)
        stop_scale = torch.clamp(goal_distance / self.cfg.stop_distance, min=0.0, max=1.0)

        lin_vel_x = torch.clamp(
            self.cfg.forward_gain * goal_delta_body_xy[:, 0],
            min=self.cfg.min_lin_vel_x,
            max=self.cfg.max_lin_vel_x,
        )
        lin_vel_y = torch.clamp(
            self.cfg.lateral_gain * goal_delta_body_xy[:, 1],
            min=-self.cfg.max_lin_vel_y,
            max=self.cfg.max_lin_vel_y,
        )
        ang_vel_z = torch.clamp(
            self.cfg.heading_gain * goal_heading_error,
            min=-self.cfg.max_ang_vel_z,
            max=self.cfg.max_ang_vel_z,
        )

        self._command[:, 0] = lin_vel_x * distance_scale * stop_scale * heading_alignment * heading_turn_gate
        self._command[:, 1] = lin_vel_y * distance_scale * stop_scale * heading_alignment * heading_turn_gate
        self._command[:, 2] = ang_vel_z * torch.clamp(goal_distance / self.cfg.heading_slow_down_distance, 0.0, 1.0)

        self.metrics["goal_distance"][:] = goal_distance
        self.metrics["goal_heading_error"][:] = torch.abs(goal_heading_error)
        self.metrics["guidance_speed"][:] = torch.linalg.norm(self._command[:, :2], dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_yaw_w = yaw_quat(self.robot.data.root_quat_w[env_ids])
        radius = sample_uniform(self.cfg.radius_range[0], self.cfg.radius_range[1], (len(env_ids),), device=self.device)
        angle = sample_uniform(self.cfg.angle_range[0], self.cfg.angle_range[1], (len(env_ids),), device=self.device)
        offset_local = torch.zeros(len(env_ids), 3, device=self.device)
        offset_local[:, 0] = radius * torch.cos(angle)
        offset_local[:, 1] = radius * torch.sin(angle)
        offset_w = quat_apply(root_yaw_w, offset_local)

        self.goal_pos_w[env_ids, :2] = root_pos_w[:, :2] + offset_w[:, :2]
        self.goal_pos_w[env_ids, 2] = self._env.scene.env_origins[env_ids, 2] + self.cfg.target_height_offset
        self._update_guidance_command()

    def _update_command(self):
        self._update_guidance_command()

    def _update_metrics(self):
        self._update_guidance_command()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_point_goal_visualizer"):
                self._point_goal_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
            self._point_goal_visualizer.set_visibility(True)
        elif hasattr(self, "_point_goal_visualizer"):
            self._point_goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized or not hasattr(self, "_point_goal_visualizer"):
            return
        target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        target_quat[:, 0] = 1.0
        self._point_goal_visualizer.visualize(self.goal_pos_w, target_quat)


@configclass
class PointGoalCommandCfg(CommandTermCfg):
    class_type: type = PointGoalCommand

    asset_name: str = MISSING

    radius_range: tuple[float, float] = (0.5, 2.5)
    angle_range: tuple[float, float] = (-math.pi, math.pi)

    forward_gain: float = 1.3
    lateral_gain: float = 1.8
    heading_gain: float = 1.2

    min_lin_vel_x: float = -0.35
    max_lin_vel_x: float = 0.9
    max_lin_vel_y: float = 0.35
    max_ang_vel_z: float = 0.8

    slow_down_distance: float = 0.8
    stop_distance: float = 0.2
    heading_slow_down_distance: float = 0.35
    turn_in_place_threshold: float = 0.60
    target_height_offset: float = 0.03

    target_visualizer_cfg: VisualizationMarkersCfg = POINT_GOAL_MARKER_CFG


def _point_goal_term(env, command_name: str = "base_velocity") -> PointGoalCommand:
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "goal_pos_w"):
        raise RuntimeError(f"Command term '{command_name}' is not a point-goal command.")
    return command_term


def _goal_delta_w_xy(env, command_name: str = "base_velocity") -> torch.Tensor:
    command_term = _point_goal_term(env, command_name=command_name)
    robot: Articulation = env.scene[command_term.cfg.asset_name]
    return command_term.goal_pos_w[:, :2] - robot.data.root_pos_w[:, :2]


def _goal_delta_body_xy(env, command_name: str = "base_velocity") -> torch.Tensor:
    command_term = _point_goal_term(env, command_name=command_name)
    robot: Articulation = env.scene[command_term.cfg.asset_name]
    goal_delta_w = torch.zeros(env.num_envs, 3, device=env.device)
    goal_delta_w[:, :2] = command_term.goal_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), goal_delta_w)[:, :2]


def point_goal_target_pos_env(env, command_name: str = "base_velocity") -> torch.Tensor:
    command_term = _point_goal_term(env, command_name=command_name)
    return command_term.goal_pos_w[:, :2] - env.scene.env_origins[:, :2]


def point_goal_root_pos_env(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]


def point_goal_rel_body_xy(env, command_name: str = "base_velocity") -> torch.Tensor:
    return _goal_delta_body_xy(env, command_name=command_name)


def point_goal_distance_obs(env, command_name: str = "base_velocity") -> torch.Tensor:
    return torch.linalg.norm(_goal_delta_w_xy(env, command_name=command_name), dim=-1, keepdim=True)


def point_goal_heading_error_obs(env, command_name: str = "base_velocity") -> torch.Tensor:
    goal_delta_body_xy = _goal_delta_body_xy(env, command_name=command_name)
    heading_error = _wrap_to_pi(torch.atan2(goal_delta_body_xy[:, 1], goal_delta_body_xy[:, 0]))
    return torch.stack((torch.sin(heading_error), torch.cos(heading_error)), dim=-1)


def _sync_point_goal_state(
    env,
    command_name: str,
    success_distance: float,
    success_hold_steps: int,
    stop_velocity_threshold: float,
    stop_yaw_rate_threshold: float,
):
    if getattr(env, "_point_goal_state_synced_step", None) == env.common_step_counter:
        return

    command_term = _point_goal_term(env, command_name=command_name)

    if not hasattr(env, "_point_goal_prev_distance"):
        env._point_goal_prev_distance = torch.zeros(env.num_envs, device=env.device)
        env._point_goal_current_distance = torch.zeros(env.num_envs, device=env.device)
        env._point_goal_initial_distance = torch.ones(env.num_envs, device=env.device)
        env._point_goal_progress = torch.zeros(env.num_envs, device=env.device)
        env._point_goal_completion = torch.zeros(env.num_envs, device=env.device)
        env._point_goal_min_distance = torch.full((env.num_envs,), float("inf"), device=env.device)
        env._point_goal_success_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._point_goal_just_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._point_goal_in_zone = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._point_goal_stop_quality = torch.zeros(env.num_envs, device=env.device)

    if hasattr(env, "episode_length_buf"):
        reset_ids = env.episode_length_buf == 0
    else:
        reset_ids = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if torch.any(reset_ids):
        command_term._resample_command(reset_ids.nonzero(as_tuple=False).squeeze(-1))

    robot: Articulation = env.scene["robot"]
    current_distance = torch.linalg.norm(_goal_delta_w_xy(env, command_name=command_name), dim=-1)
    base_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    yaw_rate = torch.abs(robot.data.root_ang_vel_w[:, 2])
    stop_quality = torch.exp(-base_speed / max(stop_velocity_threshold, 1.0e-6)) * torch.exp(
        -yaw_rate / max(stop_yaw_rate_threshold, 1.0e-6)
    )

    if torch.any(reset_ids):
        env._point_goal_prev_distance[reset_ids] = current_distance[reset_ids]
        env._point_goal_initial_distance[reset_ids] = torch.clamp(current_distance[reset_ids], min=success_distance)
        env._point_goal_min_distance[reset_ids] = current_distance[reset_ids]
        env._point_goal_success_steps[reset_ids] = 0
        env._point_goal_just_reached[reset_ids] = False
        env._point_goal_in_zone[reset_ids] = False
        env._point_goal_stop_quality[reset_ids] = 0.0

    env._point_goal_progress = env._point_goal_prev_distance - current_distance
    env._point_goal_current_distance = current_distance
    env._point_goal_completion = torch.clamp(
        1.0 - current_distance / torch.clamp(env._point_goal_initial_distance, min=success_distance),
        min=0.0,
        max=1.0,
    )
    env._point_goal_min_distance = torch.minimum(env._point_goal_min_distance, current_distance)

    success_zone = (
        (current_distance < success_distance)
        & (base_speed < stop_velocity_threshold)
        & (yaw_rate < stop_yaw_rate_threshold)
    )
    env._point_goal_success_steps = torch.where(
        success_zone,
        env._point_goal_success_steps + 1,
        torch.zeros_like(env._point_goal_success_steps),
    )
    env._point_goal_just_reached = env._point_goal_success_steps == success_hold_steps
    env._point_goal_in_zone = success_zone
    env._point_goal_stop_quality = stop_quality
    env._point_goal_prev_distance = current_distance
    env._point_goal_state_synced_step = env.common_step_counter


def point_goal_progress_reward(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    clip_value: float = 0.05,
    positive_scale: float = 4.0,
    regress_scale: float = 2.0,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
    )
    progress = torch.clamp(env._point_goal_progress, min=-clip_value, max=clip_value)
    return positive_scale * torch.clamp(progress, min=0.0) - regress_scale * torch.clamp(-progress, min=0.0)


def point_goal_completion_reward(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    exponent: float = 0.5,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
    )
    return torch.pow(env._point_goal_completion, exponent)


def point_goal_distance_reward(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    std: float = 0.35,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
    )
    return torch.exp(-env._point_goal_current_distance / std)


def point_goal_stop_reward(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    near_distance: float = 0.35,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
    )
    near_gate = torch.clamp(1.0 - env._point_goal_current_distance / max(near_distance, 1.0e-6), min=0.0, max=1.0)
    return env._point_goal_stop_quality * near_gate


def point_goal_success_bonus(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
    )
    return env._point_goal_just_reached.float()


def point_goal_success(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
    )
    return env._point_goal_just_reached
