from __future__ import annotations

import math
import os
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
ENABLE_POINT_GOAL_RESET_DEBUG = os.getenv("UTRL_POINT_GOAL_RESET_DEBUG", "0") == "1"
POINT_GOAL_RESET_DEBUG_MAX_GLOBAL_STEPS = int(os.getenv("UTRL_POINT_GOAL_RESET_DEBUG_STEPS", "8"))


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _rotate_xy(xy: torch.Tensor, angle: float) -> torch.Tensor:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    x = xy[:, 0]
    y = xy[:, 1]
    return torch.stack((cos_angle * x - sin_angle * y, sin_angle * x + cos_angle * y), dim=-1)


class PointGoalCommand(CommandTerm):
    cfg: PointGoalCommandCfg

    def __init__(self, cfg: PointGoalCommandCfg, env):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["min_goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_heading_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["guidance_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["recovery_turn_mode"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reverse_recovery_mode"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hold_mode"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_age_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["remaining_time_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reset_goal_heading_error_raw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reset_goal_heading_error_command"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reset_goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reset_goal_delta_raw_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reset_goal_delta_raw_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reset_goal_delta_command_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reset_goal_delta_command_y"] = torch.zeros(self.num_envs, device=self.device)

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

    def _goal_delta_body_xy_raw(self) -> torch.Tensor:
        goal_delta_w = torch.zeros(self.num_envs, 3, device=self.device)
        goal_delta_w[:, :2] = self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        return quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), goal_delta_w)[:, :2]

    def _goal_delta_body_xy(self) -> torch.Tensor:
        goal_delta_body_xy_raw = self._goal_delta_body_xy_raw()
        return _rotate_xy(goal_delta_body_xy_raw, self.cfg.frame_yaw_offset)

    def _record_reset_debug_metrics(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        goal_delta_body_xy_raw = self._goal_delta_body_xy_raw()[env_ids]
        goal_delta_body_xy = _rotate_xy(goal_delta_body_xy_raw, self.cfg.frame_yaw_offset)
        goal_distance = torch.linalg.norm(goal_delta_body_xy_raw, dim=-1)
        raw_heading_error = _wrap_to_pi(torch.atan2(goal_delta_body_xy_raw[:, 1], goal_delta_body_xy_raw[:, 0]))
        command_heading_error = _wrap_to_pi(torch.atan2(goal_delta_body_xy[:, 1], goal_delta_body_xy[:, 0]))

        self.metrics["reset_goal_heading_error_raw"][env_ids] = torch.abs(raw_heading_error)
        self.metrics["reset_goal_heading_error_command"][env_ids] = torch.abs(command_heading_error)
        self.metrics["reset_goal_distance"][env_ids] = goal_distance
        self.metrics["reset_goal_delta_raw_x"][env_ids] = goal_delta_body_xy_raw[:, 0]
        self.metrics["reset_goal_delta_raw_y"][env_ids] = goal_delta_body_xy_raw[:, 1]
        self.metrics["reset_goal_delta_command_x"][env_ids] = goal_delta_body_xy[:, 0]
        self.metrics["reset_goal_delta_command_y"][env_ids] = goal_delta_body_xy[:, 1]

        if ENABLE_POINT_GOAL_RESET_DEBUG and self._env.common_step_counter <= POINT_GOAL_RESET_DEBUG_MAX_GLOBAL_STEPS:
            env_id = int(env_ids[0].item())
            print(
                "[POINT_GOAL_DEBUG] "
                f"global_step={int(self._env.common_step_counter)} env_id={env_id} "
                f"frame_yaw_offset={self.cfg.frame_yaw_offset:.4f} "
                f"goal_distance={float(goal_distance[0].item()):.4f} "
                f"raw_xy=({float(goal_delta_body_xy_raw[0, 0].item()):.4f}, {float(goal_delta_body_xy_raw[0, 1].item()):.4f}) "
                f"cmd_xy=({float(goal_delta_body_xy[0, 0].item()):.4f}, {float(goal_delta_body_xy[0, 1].item()):.4f}) "
                f"raw_heading={float(raw_heading_error[0].item()):.4f} "
                f"cmd_heading={float(command_heading_error[0].item()):.4f}"
            )

    def _update_guidance_command(self):
        goal_delta_body_xy = self._goal_delta_body_xy()
        goal_delta_w_xy = self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        goal_distance = torch.linalg.norm(goal_delta_w_xy, dim=-1)
        goal_heading_error = _wrap_to_pi(torch.atan2(goal_delta_body_xy[:, 1], goal_delta_body_xy[:, 0]))
        goal_heading_abs = torch.abs(goal_heading_error)
        hold_mode = goal_distance < self.cfg.hold_position_distance
        recovery_turn_mode = (
            (~hold_mode)
            & (goal_distance < self.cfg.near_recovery_distance)
            & (goal_heading_abs > self.cfg.recovery_turn_threshold)
        )
        reverse_recovery_mode = (
            (~hold_mode)
            & (~recovery_turn_mode)
            & (goal_distance < self.cfg.reverse_recovery_distance)
            & (goal_delta_body_xy[:, 0] < -self.cfg.reverse_trigger_distance)
            & (goal_heading_abs < self.cfg.reverse_heading_threshold)
        )

        distance_scale = torch.clamp(goal_distance / self.cfg.slow_down_distance, min=0.0, max=1.0)
        stop_scale = torch.clamp(
            (goal_distance - self.cfg.hold_position_distance)
            / max(self.cfg.stop_distance - self.cfg.hold_position_distance, 1.0e-6),
            min=0.0,
            max=1.0,
        )
        heading_forward_gate = torch.clamp(
            1.0 - goal_heading_abs / max(self.cfg.heading_block_threshold, 1.0e-6),
            min=0.0,
            max=1.0,
        )

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

        lin_vel_x = lin_vel_x * distance_scale * stop_scale * heading_forward_gate
        lin_vel_y = lin_vel_y * distance_scale * stop_scale * heading_forward_gate

        reverse_lin_vel_x = -torch.clamp(
            self.cfg.reverse_gain * (-goal_delta_body_xy[:, 0]),
            min=0.0,
            max=self.cfg.max_reverse_lin_vel_x,
        )
        lin_vel_x = torch.where(reverse_recovery_mode, reverse_lin_vel_x, lin_vel_x)
        lin_vel_y = torch.where(reverse_recovery_mode, torch.zeros_like(lin_vel_y), lin_vel_y)

        ang_distance_scale = torch.clamp(goal_distance / self.cfg.heading_slow_down_distance, 0.0, 1.0)
        ang_vel_z = ang_vel_z * ang_distance_scale
        min_recovery_turn = torch.sign(goal_heading_error) * torch.maximum(
            torch.abs(ang_vel_z),
            torch.full_like(ang_vel_z, self.cfg.min_recovery_ang_vel_z),
        )
        ang_vel_z = torch.where(recovery_turn_mode, min_recovery_turn, ang_vel_z)

        lin_vel_x = torch.where(hold_mode | recovery_turn_mode, torch.zeros_like(lin_vel_x), lin_vel_x)
        lin_vel_y = torch.where(hold_mode | recovery_turn_mode, torch.zeros_like(lin_vel_y), lin_vel_y)
        ang_vel_z = torch.where(hold_mode, torch.zeros_like(ang_vel_z), ang_vel_z)

        self._command[:, 0] = lin_vel_x
        self._command[:, 1] = lin_vel_y
        self._command[:, 2] = ang_vel_z

        self.metrics["goal_distance"][:] = goal_distance
        self.metrics["goal_heading_error"][:] = torch.abs(goal_heading_error)
        self.metrics["guidance_speed"][:] = torch.linalg.norm(self._command[:, :2], dim=-1)
        self.metrics["recovery_turn_mode"][:] = recovery_turn_mode.float()
        self.metrics["reverse_recovery_mode"][:] = reverse_recovery_mode.float()
        self.metrics["hold_mode"][:] = hold_mode.float()

        if getattr(self._env, "_point_goal_use_policy_command", False) and hasattr(
            self._env, "_point_goal_policy_command"
        ):
            policy_command = self._env._point_goal_policy_command
            self._command[:] = policy_command
            self.metrics["guidance_speed"][:] = torch.linalg.norm(policy_command[:, :2], dim=-1)
            self.metrics.setdefault("policy_command_speed", torch.zeros(self.num_envs, device=self.device))
            self.metrics.setdefault("policy_command_turn", torch.zeros(self.num_envs, device=self.device))
            self.metrics["policy_command_speed"][:] = torch.linalg.norm(policy_command[:, :2], dim=-1)
            self.metrics["policy_command_turn"][:] = torch.abs(policy_command[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_yaw_w = yaw_quat(self.robot.data.root_quat_w[env_ids])
        radius = sample_uniform(self.cfg.radius_range[0], self.cfg.radius_range[1], (len(env_ids),), device=self.device)
        angle = sample_uniform(self.cfg.angle_range[0], self.cfg.angle_range[1], (len(env_ids),), device=self.device)
        offset_command_xy = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
        offset_body_xy_raw = _rotate_xy(offset_command_xy, -self.cfg.frame_yaw_offset)
        offset_local = torch.zeros(len(env_ids), 3, device=self.device)
        offset_local[:, :2] = offset_body_xy_raw
        offset_w = quat_apply(root_yaw_w, offset_local)

        self.goal_pos_w[env_ids, :2] = root_pos_w[:, :2] + offset_w[:, :2]
        self.goal_pos_w[env_ids, 2] = self._env.scene.env_origins[env_ids, 2] + self.cfg.target_height_offset
        self._record_reset_debug_metrics(env_ids)
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
    hold_position_distance: float = 0.22
    near_recovery_distance: float = 0.45
    recovery_turn_threshold: float = 0.40
    heading_block_threshold: float = 1.10
    min_recovery_ang_vel_z: float = 0.18
    reverse_recovery_distance: float = 0.30
    reverse_heading_threshold: float = 0.25
    reverse_trigger_distance: float = 0.04
    reverse_gain: float = 0.8
    max_reverse_lin_vel_x: float = 0.08
    terminal_slow_distance: float = 0.30
    terminal_latch_distance: float = 0.16
    terminal_max_lin_vel_x: float = 0.12
    terminal_max_lin_vel_y: float = 0.08
    terminal_max_ang_vel_z: float = 0.30
    terminal_settle_lin_vel_x: float = 0.05
    terminal_settle_reverse_lin_vel_x: float = 0.04
    frame_yaw_offset: float = 0.0
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
    goal_delta_body_xy_raw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), goal_delta_w)[:, :2]
    return _rotate_xy(goal_delta_body_xy_raw, command_term.cfg.frame_yaw_offset)


def _ensure_policy_command_state(env) -> torch.Tensor:
    if not hasattr(env, "_point_goal_policy_command"):
        env._point_goal_policy_command = torch.zeros(env.num_envs, 3, device=env.device)
    return env._point_goal_policy_command


def set_point_goal_policy_command(env, command: torch.Tensor):
    policy_command = _ensure_policy_command_state(env)
    policy_command[:] = command
    try:
        command_term = _point_goal_term(env, command_name="base_velocity")
    except RuntimeError:
        return
    command_term.metrics["guidance_speed"][:] = torch.linalg.norm(command[:, :2], dim=-1)
    command_term.metrics.setdefault("policy_command_speed", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics.setdefault("policy_command_turn", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics["policy_command_speed"][:] = torch.linalg.norm(command[:, :2], dim=-1)
    command_term.metrics["policy_command_turn"][:] = torch.abs(command[:, 2])


def point_goal_policy_command_obs(env) -> torch.Tensor:
    return _ensure_policy_command_state(env)


def track_policy_command_lin_vel_xy_exp(
    env,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    policy_command = _ensure_policy_command_state(env)
    asset: Articulation = env.scene[asset_cfg.name]
    vel_w = torch.zeros(env.num_envs, 3, device=env.device)
    vel_w[:, :2] = asset.data.root_lin_vel_w[:, :2]
    vel_body = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), vel_w)[:, :2]
    lin_vel_error = torch.sum(torch.square(policy_command[:, :2] - vel_body), dim=1)
    return torch.exp(-lin_vel_error / (std**2))


def track_policy_command_ang_vel_z_exp(
    env,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    policy_command = _ensure_policy_command_state(env)
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(policy_command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / (std**2))


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


def point_goal_target_levels(
    env,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    num_curriculum_episodes: int = 12,
    start_radius_range: tuple[float, float] = (0.30, 0.70),
    mid_radius_range: tuple[float, float] = (0.35, 1.00),
    final_radius_range: tuple[float, float] = (0.40, 1.50),
    start_angle_range: tuple[float, float] = (-math.pi / 12.0, math.pi / 12.0),
    mid_angle_range: tuple[float, float] = (-math.pi / 3.0, math.pi / 3.0),
    final_angle_range: tuple[float, float] = (-math.pi, math.pi),
    promote_success_threshold: float = 0.75,
):
    del env_ids
    command_term = env.command_manager.get_term(command_name)
    cfg = command_term.cfg

    progress, progress_tensor = _point_goal_curriculum_progress(
        env,
        num_curriculum_episodes,
        promote_success_threshold=promote_success_threshold,
    )
    halfway_tensor = torch.tensor(0.5, device=env.device)

    if env.common_step_counter % env.max_episode_length == 0:
        if progress <= 0.5:
            phase_progress = progress_tensor / halfway_tensor
            cfg.radius_range = torch.lerp(
                torch.tensor(start_radius_range, device=env.device),
                torch.tensor(mid_radius_range, device=env.device),
                phase_progress,
            ).tolist()
            cfg.angle_range = torch.lerp(
                torch.tensor(start_angle_range, device=env.device),
                torch.tensor(mid_angle_range, device=env.device),
                phase_progress,
            ).tolist()
        else:
            phase_progress = (progress_tensor - halfway_tensor) / halfway_tensor
            cfg.radius_range = torch.lerp(
                torch.tensor(mid_radius_range, device=env.device),
                torch.tensor(final_radius_range, device=env.device),
                phase_progress,
            ).tolist()
            cfg.angle_range = torch.lerp(
                torch.tensor(mid_angle_range, device=env.device),
                torch.tensor(final_angle_range, device=env.device),
                phase_progress,
            ).tolist()

    return progress_tensor


def _point_goal_curriculum_progress(
    env,
    num_curriculum_episodes: int,
    promote_success_threshold: float = 0.75,
) -> tuple[float, torch.Tensor]:
    if not hasattr(env, "_point_goal_curriculum_progress"):
        env._point_goal_curriculum_progress = 0.0
        env._point_goal_curriculum_success_ema = 0.0
        env._point_goal_curriculum_last_update_step = -1

    progress_step = 1.0 / max(num_curriculum_episodes, 1)
    should_update = env.common_step_counter % env.max_episode_length == 0
    if should_update and env._point_goal_curriculum_last_update_step != env.common_step_counter:
        if float(env._point_goal_curriculum_success_ema) >= promote_success_threshold:
            env._point_goal_curriculum_progress = min(env._point_goal_curriculum_progress + progress_step, 1.0)
        env._point_goal_curriculum_last_update_step = env.common_step_counter

    progress = float(env._point_goal_curriculum_progress)
    return progress, torch.tensor(progress, device=env.device)


def _lerp_scalar(start: float, final: float, progress: float) -> float:
    return (1.0 - progress) * float(start) + progress * float(final)


def point_goal_reward_levels(
    env,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    num_curriculum_episodes: int = 12,
    start_success_distance: float = 0.22,
    final_success_distance: float = 0.12,
    start_success_hold_steps: int = 2,
    final_success_hold_steps: int = 8,
    start_stop_velocity_threshold: float = 0.18,
    final_stop_velocity_threshold: float = 0.10,
    start_stop_yaw_rate_threshold: float = 0.45,
    final_stop_yaw_rate_threshold: float = 0.25,
    start_goal_stop_near_distance: float = 0.30,
    final_goal_stop_near_distance: float = 0.20,
    start_goal_progress_scale: float = 1.6,
    final_goal_progress_scale: float = 1.0,
    start_goal_stop_scale: float = 0.5,
    final_goal_stop_scale: float = 1.0,
    start_goal_success_scale: float = 2.0,
    final_goal_success_scale: float = 1.0,
    start_goal_time_penalty_scale: float = 0.8,
    final_goal_time_penalty_scale: float = 1.0,
    start_goal_timeout_penalty_scale: float = 0.75,
    final_goal_timeout_penalty_scale: float = 1.0,
    start_hold_position_distance: float = 0.18,
    final_hold_position_distance: float = 0.08,
    start_stop_distance: float = 0.30,
    final_stop_distance: float = 0.20,
    start_slow_down_distance: float = 0.45,
    final_slow_down_distance: float = 0.55,
    start_heading_slow_down_distance: float = 0.35,
    final_heading_slow_down_distance: float = 0.60,
    promote_success_threshold: float = 0.75,
):
    del env_ids
    progress, progress_tensor = _point_goal_curriculum_progress(
        env,
        num_curriculum_episodes,
        promote_success_threshold=promote_success_threshold,
    )

    env._point_goal_success_distance = _lerp_scalar(start_success_distance, final_success_distance, progress)
    env._point_goal_success_hold_steps = int(
        round(_lerp_scalar(start_success_hold_steps, final_success_hold_steps, progress))
    )
    env._point_goal_stop_velocity_threshold = _lerp_scalar(
        start_stop_velocity_threshold,
        final_stop_velocity_threshold,
        progress,
    )
    env._point_goal_stop_yaw_rate_threshold = _lerp_scalar(
        start_stop_yaw_rate_threshold,
        final_stop_yaw_rate_threshold,
        progress,
    )
    env._point_goal_goal_stop_near_distance = _lerp_scalar(
        start_goal_stop_near_distance,
        final_goal_stop_near_distance,
        progress,
    )
    env._point_goal_goal_progress_scale = _lerp_scalar(start_goal_progress_scale, final_goal_progress_scale, progress)
    env._point_goal_goal_stop_scale = _lerp_scalar(start_goal_stop_scale, final_goal_stop_scale, progress)
    env._point_goal_goal_success_scale = _lerp_scalar(start_goal_success_scale, final_goal_success_scale, progress)
    env._point_goal_goal_time_penalty_scale = _lerp_scalar(
        start_goal_time_penalty_scale,
        final_goal_time_penalty_scale,
        progress,
    )
    env._point_goal_goal_timeout_penalty_scale = _lerp_scalar(
        start_goal_timeout_penalty_scale,
        final_goal_timeout_penalty_scale,
        progress,
    )

    command_term = env.command_manager.get_term(command_name)
    scheduled_hold_position_distance = max(
        _lerp_scalar(start_hold_position_distance, final_hold_position_distance, progress),
        env._point_goal_success_distance + 0.01,
    )
    scheduled_stop_distance = max(
        _lerp_scalar(start_stop_distance, final_stop_distance, progress),
        scheduled_hold_position_distance + 0.07,
    )
    scheduled_slow_down_distance = max(
        _lerp_scalar(start_slow_down_distance, final_slow_down_distance, progress),
        scheduled_stop_distance + 0.18,
    )
    scheduled_heading_slow_down_distance = max(
        _lerp_scalar(start_heading_slow_down_distance, final_heading_slow_down_distance, progress),
        scheduled_stop_distance + 0.05,
    )
    scheduled_near_recovery_distance = scheduled_hold_position_distance + 0.02

    command_term.cfg.hold_position_distance = scheduled_hold_position_distance
    command_term.cfg.stop_distance = scheduled_stop_distance
    command_term.cfg.slow_down_distance = scheduled_slow_down_distance
    command_term.cfg.heading_slow_down_distance = scheduled_heading_slow_down_distance
    command_term.cfg.near_recovery_distance = scheduled_near_recovery_distance
    command_term.metrics.setdefault("scheduled_success_distance", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics.setdefault("scheduled_success_hold_steps", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics.setdefault("scheduled_stop_velocity_threshold", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics.setdefault("scheduled_stop_yaw_rate_threshold", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics.setdefault("scheduled_goal_stop_near_distance", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics.setdefault("scheduled_goal_success_scale", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics.setdefault("curriculum_success_ema", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics.setdefault("scheduled_hold_position_distance", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics.setdefault("scheduled_stop_distance", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics.setdefault("scheduled_slow_down_distance", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics.setdefault("scheduled_near_recovery_distance", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics["scheduled_success_distance"][:] = env._point_goal_success_distance
    command_term.metrics["scheduled_success_hold_steps"][:] = float(env._point_goal_success_hold_steps)
    command_term.metrics["scheduled_stop_velocity_threshold"][:] = env._point_goal_stop_velocity_threshold
    command_term.metrics["scheduled_stop_yaw_rate_threshold"][:] = env._point_goal_stop_yaw_rate_threshold
    command_term.metrics["scheduled_goal_stop_near_distance"][:] = env._point_goal_goal_stop_near_distance
    command_term.metrics["scheduled_goal_success_scale"][:] = env._point_goal_goal_success_scale
    command_term.metrics["curriculum_success_ema"][:] = float(env._point_goal_curriculum_success_ema)
    command_term.metrics["scheduled_hold_position_distance"][:] = scheduled_hold_position_distance
    command_term.metrics["scheduled_stop_distance"][:] = scheduled_stop_distance
    command_term.metrics["scheduled_slow_down_distance"][:] = scheduled_slow_down_distance
    command_term.metrics["scheduled_near_recovery_distance"][:] = scheduled_near_recovery_distance
    return progress_tensor


def _episode_length_buf(env):
    if hasattr(env, "episode_length_buf"):
        return env.episode_length_buf.clone()
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _compute_just_reset_mask(env):
    current_episode_length = _episode_length_buf(env)
    if not hasattr(env, "_point_goal_prev_episode_length_buf"):
        env._point_goal_prev_episode_length_buf = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    prev_episode_length = env._point_goal_prev_episode_length_buf
    just_reset = prev_episode_length < 0
    just_reset |= current_episode_length == 0
    just_reset |= current_episode_length < prev_episode_length
    return just_reset, current_episode_length


def _sync_point_goal_state(
    env,
    command_name: str,
    success_distance: float,
    success_hold_steps: int,
    stop_velocity_threshold: float,
    stop_yaw_rate_threshold: float,
    per_target_timeout_s: float,
):
    if getattr(env, "_point_goal_state_synced_step", None) == env.common_step_counter:
        return

    success_distance = float(getattr(env, "_point_goal_success_distance", success_distance))
    success_hold_steps = max(1, int(round(float(getattr(env, "_point_goal_success_hold_steps", success_hold_steps)))))
    stop_velocity_threshold = float(
        getattr(env, "_point_goal_stop_velocity_threshold", stop_velocity_threshold)
    )
    stop_yaw_rate_threshold = float(
        getattr(env, "_point_goal_stop_yaw_rate_threshold", stop_yaw_rate_threshold)
    )
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
        env._point_goal_terminal_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._point_goal_stop_quality = torch.zeros(env.num_envs, device=env.device)
        env._point_goal_target_age_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._point_goal_timed_out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._point_goal_remaining_time_fraction = torch.ones(env.num_envs, device=env.device)
    if not hasattr(env, "_point_goal_curriculum_success_ema"):
        env._point_goal_curriculum_success_ema = 0.0
        env._point_goal_curriculum_progress = 0.0
        env._point_goal_curriculum_last_update_step = -1

    reset_ids, current_episode_length = _compute_just_reset_mask(env)
    prev_episode_length = env._point_goal_prev_episode_length_buf
    completed_ids = reset_ids & (prev_episode_length >= 0)
    if torch.any(completed_ids):
        completed_success = env._point_goal_just_reached[completed_ids].float().mean().item()
        ema_decay = 0.9
        env._point_goal_curriculum_success_ema = (
            ema_decay * float(env._point_goal_curriculum_success_ema) + (1.0 - ema_decay) * completed_success
        )

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
        env._point_goal_terminal_latched[reset_ids] = False
        env._point_goal_stop_quality[reset_ids] = 0.0
        env._point_goal_target_age_steps[reset_ids] = 0
        env._point_goal_timed_out[reset_ids] = False
        env._point_goal_remaining_time_fraction[reset_ids] = 1.0
        if getattr(env, "_point_goal_use_policy_command", False) and hasattr(env, "_point_goal_policy_command"):
            env._point_goal_policy_command[reset_ids] = 0.0

    env._point_goal_target_age_steps += 1
    per_target_timeout_steps = max(1, int(round(per_target_timeout_s / env.step_dt)))
    env._point_goal_remaining_time_fraction = torch.clamp(
        (per_target_timeout_steps - env._point_goal_target_age_steps).float() / float(per_target_timeout_steps),
        min=0.0,
        max=1.0,
    )

    env._point_goal_progress = env._point_goal_prev_distance - current_distance
    env._point_goal_current_distance = current_distance
    env._point_goal_completion = torch.clamp(
        1.0 - current_distance / torch.clamp(env._point_goal_initial_distance, min=success_distance),
        min=0.0,
        max=1.0,
    )
    env._point_goal_min_distance = torch.minimum(env._point_goal_min_distance, current_distance)
    terminal_latch_distance = max(success_distance, min(0.18, success_distance + 0.04))
    terminal_settle_distance = max(success_distance, min(0.22, success_distance + 0.08))
    env._point_goal_terminal_latched |= env._point_goal_min_distance < terminal_latch_distance

    success_zone = (
        env._point_goal_terminal_latched
        & (current_distance < terminal_settle_distance)
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
    env._point_goal_timed_out = env._point_goal_target_age_steps >= per_target_timeout_steps
    env._point_goal_prev_distance = current_distance
    env._point_goal_prev_episode_length_buf = current_episode_length
    env._point_goal_state_synced_step = env.common_step_counter

    command_term.metrics["min_goal_distance"][:] = env._point_goal_min_distance
    command_term.metrics.setdefault("terminal_latched", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics["terminal_latched"][:] = env._point_goal_terminal_latched.float()
    command_term.metrics["target_age_s"][:] = env._point_goal_target_age_steps.float() * env.step_dt
    command_term.metrics["remaining_time_fraction"][:] = env._point_goal_remaining_time_fraction


def point_goal_progress_reward(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    per_target_timeout_s: float = 4.0,
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
        per_target_timeout_s=per_target_timeout_s,
    )
    progress = torch.clamp(env._point_goal_progress, min=-clip_value, max=clip_value)
    reward = positive_scale * torch.clamp(progress, min=0.0) - regress_scale * torch.clamp(-progress, min=0.0)
    return reward * float(getattr(env, "_point_goal_goal_progress_scale", 1.0))


def point_goal_completion_reward(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    per_target_timeout_s: float = 4.0,
    exponent: float = 0.5,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
        per_target_timeout_s=per_target_timeout_s,
    )
    return torch.pow(env._point_goal_completion, exponent)


def point_goal_distance_reward(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    per_target_timeout_s: float = 4.0,
    std: float = 0.35,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
        per_target_timeout_s=per_target_timeout_s,
    )
    return torch.exp(-env._point_goal_current_distance / std)


def point_goal_stop_reward(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    per_target_timeout_s: float = 4.0,
    near_distance: float = 0.35,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
        per_target_timeout_s=per_target_timeout_s,
    )
    near_distance = float(getattr(env, "_point_goal_goal_stop_near_distance", near_distance))
    near_gate = torch.clamp(1.0 - env._point_goal_current_distance / max(near_distance, 1.0e-6), min=0.0, max=1.0)
    return env._point_goal_stop_quality * near_gate * float(getattr(env, "_point_goal_goal_stop_scale", 1.0))


def point_goal_heading_alignment_reward(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    per_target_timeout_s: float = 4.0,
    near_distance: float = 0.45,
    std: float = 0.60,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
        per_target_timeout_s=per_target_timeout_s,
    )
    goal_delta_body_xy = _goal_delta_body_xy(env, command_name=command_name)
    heading_error = _wrap_to_pi(torch.atan2(goal_delta_body_xy[:, 1], goal_delta_body_xy[:, 0]))
    near_gate = torch.clamp(1.0 - env._point_goal_current_distance / max(near_distance, 1.0e-6), min=0.0, max=1.0)
    return near_gate * torch.exp(-torch.square(heading_error) / (std**2))


def point_goal_success_bonus(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    per_target_timeout_s: float = 4.0,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
        per_target_timeout_s=per_target_timeout_s,
    )
    reward = env._point_goal_just_reached.float() * (1.0 + env._point_goal_remaining_time_fraction)
    return reward * float(getattr(env, "_point_goal_goal_success_scale", 1.0))


def point_goal_time_penalty(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    per_target_timeout_s: float = 4.0,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
        per_target_timeout_s=per_target_timeout_s,
    )
    return torch.ones(env.num_envs, device=env.device) * float(
        getattr(env, "_point_goal_goal_time_penalty_scale", 1.0)
    )


def point_goal_timeout_penalty(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    per_target_timeout_s: float = 4.0,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
        per_target_timeout_s=per_target_timeout_s,
    )
    return env._point_goal_timed_out.float() * float(
        getattr(env, "_point_goal_goal_timeout_penalty_scale", 1.0)
    )


def point_goal_success(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    per_target_timeout_s: float = 4.0,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
        per_target_timeout_s=per_target_timeout_s,
    )
    return env._point_goal_just_reached


def point_goal_target_timeout(
    env,
    command_name: str = "base_velocity",
    success_distance: float = 0.10,
    success_hold_steps: int = 20,
    stop_velocity_threshold: float = 0.10,
    stop_yaw_rate_threshold: float = 0.15,
    per_target_timeout_s: float = 4.0,
):
    _sync_point_goal_state(
        env,
        command_name=command_name,
        success_distance=success_distance,
        success_hold_steps=success_hold_steps,
        stop_velocity_threshold=stop_velocity_threshold,
        stop_yaw_rate_threshold=stop_yaw_rate_threshold,
        per_target_timeout_s=per_target_timeout_s,
    )
    return env._point_goal_timed_out & ~env._point_goal_just_reached
