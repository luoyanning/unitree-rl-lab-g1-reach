from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, sample_uniform, yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from ..point_goal import point_goal_mdp as base_mdp
from ..point_goal.point_goal_mdp import PointGoalCommand, PointGoalCommandCfg


def _lerp_range(start: tuple[float, float], final: tuple[float, float], progress: float) -> tuple[float, float]:
    return (
        base_mdp._lerp_scalar(start[0], final[0], progress),
        base_mdp._lerp_scalar(start[1], final[1], progress),
    )


def _sample_signed_abs_angles(
    abs_angle_range: tuple[float, float],
    num_samples: int,
    device: torch.device | str,
    sample_both_sides: bool,
) -> torch.Tensor:
    abs_angles = sample_uniform(abs_angle_range[0], abs_angle_range[1], (num_samples,), device=device)
    if not sample_both_sides:
        return abs_angles
    side_selector = sample_uniform(0.0, 1.0, (num_samples,), device=device)
    side_sign = torch.where(side_selector < 0.5, -torch.ones_like(side_selector), torch.ones_like(side_selector))
    return abs_angles * side_sign


class RearTurnPointGoalCommand(PointGoalCommand):
    cfg: RearTurnPointGoalCommandCfg

    def __init__(self, cfg: "RearTurnPointGoalCommandCfg", env):
        super().__init__(cfg, env)
        self.metrics.setdefault("turn_in_place_mode", torch.zeros(self.num_envs, device=self.device))

    def _update_guidance_command(self):
        goal_delta_body_xy = self._goal_delta_body_xy()
        goal_delta_w_xy = self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        goal_distance = torch.linalg.norm(goal_delta_w_xy, dim=-1)
        goal_heading_error = base_mdp._wrap_to_pi(torch.atan2(goal_delta_body_xy[:, 1], goal_delta_body_xy[:, 0]))
        goal_heading_abs = torch.abs(goal_heading_error)

        hold_mode = goal_distance < self.cfg.hold_position_distance
        turn_in_place_mode = (
            (~hold_mode)
            & (goal_distance > self.cfg.turn_in_place_min_distance)
            & (goal_heading_abs > self.cfg.turn_in_place_threshold)
        )
        recovery_turn_mode = (
            (~hold_mode)
            & (~turn_in_place_mode)
            & (goal_distance < self.cfg.near_recovery_distance)
            & (goal_heading_abs > self.cfg.recovery_turn_threshold)
        )
        reverse_recovery_mode = (
            (~hold_mode)
            & (~turn_in_place_mode)
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
        min_turn_in_place = torch.sign(goal_heading_error) * torch.maximum(
            torch.abs(ang_vel_z),
            torch.full_like(ang_vel_z, self.cfg.min_turn_in_place_ang_vel_z),
        )
        min_recovery_turn = torch.sign(goal_heading_error) * torch.maximum(
            torch.abs(ang_vel_z),
            torch.full_like(ang_vel_z, self.cfg.min_recovery_ang_vel_z),
        )
        ang_vel_z = torch.where(turn_in_place_mode, min_turn_in_place, ang_vel_z)
        ang_vel_z = torch.where(recovery_turn_mode, min_recovery_turn, ang_vel_z)

        # Allow only a small amount of arc-like motion while turning so the
        # policy can still approach the goal without destabilizing itself.
        lin_vel_x = torch.where(turn_in_place_mode, lin_vel_x * 0.15, lin_vel_x)
        lin_vel_y = torch.where(turn_in_place_mode, lin_vel_y * 0.25, lin_vel_y)
        lin_vel_x = torch.where(hold_mode | recovery_turn_mode, torch.zeros_like(lin_vel_x), lin_vel_x)
        lin_vel_y = torch.where(hold_mode | recovery_turn_mode, torch.zeros_like(lin_vel_y), lin_vel_y)
        ang_vel_z = torch.where(hold_mode, torch.zeros_like(ang_vel_z), ang_vel_z)

        self._command[:, 0] = lin_vel_x
        self._command[:, 1] = lin_vel_y
        self._command[:, 2] = ang_vel_z

        self.metrics["goal_distance"][:] = goal_distance
        self.metrics["goal_heading_error"][:] = goal_heading_abs
        self.metrics["guidance_speed"][:] = torch.linalg.norm(self._command[:, :2], dim=-1)
        self.metrics["recovery_turn_mode"][:] = recovery_turn_mode.float()
        self.metrics["reverse_recovery_mode"][:] = reverse_recovery_mode.float()
        self.metrics["hold_mode"][:] = hold_mode.float()
        self.metrics["turn_in_place_mode"][:] = turn_in_place_mode.float()

        if getattr(self._env, "_point_goal_use_policy_command", False) and hasattr(self._env, "_point_goal_policy_command"):
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
        if self.cfg.abs_angle_range is not None:
            angle = _sample_signed_abs_angles(
                self.cfg.abs_angle_range,
                len(env_ids),
                self.device,
                sample_both_sides=self.cfg.sample_both_sides,
            )
        else:
            angle = sample_uniform(self.cfg.angle_range[0], self.cfg.angle_range[1], (len(env_ids),), device=self.device)

        offset_command_xy = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
        offset_body_xy_raw = base_mdp._rotate_xy(offset_command_xy, -self.cfg.frame_yaw_offset)
        offset_local = torch.zeros(len(env_ids), 3, device=self.device)
        offset_local[:, :2] = offset_body_xy_raw
        offset_w = quat_apply(root_yaw_w, offset_local)

        self.goal_pos_w[env_ids, :2] = root_pos_w[:, :2] + offset_w[:, :2]
        self.goal_pos_w[env_ids, 2] = self._env.scene.env_origins[env_ids, 2] + self.cfg.target_height_offset
        self._record_reset_debug_metrics(env_ids)
        self._update_guidance_command()


@configclass
class RearTurnPointGoalCommandCfg(PointGoalCommandCfg):
    class_type: type = RearTurnPointGoalCommand

    abs_angle_range: tuple[float, float] | None = (math.radians(90.0), math.pi)
    sample_both_sides: bool = True
    turn_in_place_min_distance: float = 0.35
    min_turn_in_place_ang_vel_z: float = 0.22


def _rear_turn_curriculum_progress(
    env,
    num_curriculum_episodes: int,
    early_promote_success_threshold: float = 0.68,
    early_demote_success_threshold: float = 0.50,
    mid_promote_success_threshold: float = 0.74,
    mid_demote_success_threshold: float = 0.56,
    late_promote_success_threshold: float = 0.78,
    late_demote_success_threshold: float = 0.62,
) -> tuple[float, torch.Tensor]:
    if not hasattr(env, "_point_goal_curriculum_progress"):
        env._point_goal_curriculum_progress = 0.0
        env._point_goal_curriculum_success_ema = 0.0
        env._point_goal_curriculum_last_update_step = -1

    progress_step = 1.0 / max(num_curriculum_episodes, 1)
    should_update = env.common_step_counter % env.max_episode_length == 0
    if should_update and env._point_goal_curriculum_last_update_step != env.common_step_counter:
        current_progress = float(env._point_goal_curriculum_progress)
        if current_progress < 0.25:
            promote_success_threshold = early_promote_success_threshold
            demote_success_threshold = early_demote_success_threshold
        elif current_progress < 0.60:
            promote_success_threshold = mid_promote_success_threshold
            demote_success_threshold = mid_demote_success_threshold
        else:
            promote_success_threshold = late_promote_success_threshold
            demote_success_threshold = late_demote_success_threshold

        success_ema = float(env._point_goal_curriculum_success_ema)
        if success_ema >= promote_success_threshold:
            env._point_goal_curriculum_progress = min(current_progress + progress_step, 1.0)
        elif success_ema <= demote_success_threshold:
            env._point_goal_curriculum_progress = max(current_progress - progress_step, 0.0)
        env._point_goal_curriculum_last_update_step = env.common_step_counter

    progress = float(env._point_goal_curriculum_progress)
    return progress, torch.tensor(progress, device=env.device)


def point_goal_turn_band_target_levels(
    env,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    num_curriculum_episodes: int = 48,
    start_radius_range: tuple[float, float] = (0.7, 0.95),
    angle_master_radius_range: tuple[float, float] = (0.8, 1.15),
    radius_expansion_range: tuple[float, float] = (1.0, 1.6),
    final_radius_range: tuple[float, float] = (1.5, 4.0),
    start_abs_angle_range: tuple[float, float] = (math.radians(90.0), math.radians(98.0)),
    mid_abs_angle_range: tuple[float, float] = (math.radians(90.0), math.radians(125.0)),
    late_abs_angle_range: tuple[float, float] = (math.radians(90.0), math.radians(170.0)),
    final_abs_angle_range: tuple[float, float] = (math.radians(90.0), math.pi),
):
    command_term = base_mdp._point_goal_term(env, command_name=command_name)
    progress, progress_tensor = _rear_turn_curriculum_progress(
        env,
        num_curriculum_episodes=num_curriculum_episodes,
    )

    # Phase 1: expand heading demand while keeping the target close enough that
    # the policy can focus on learning turn-first behavior.
    if progress < 0.60:
        stage_progress = progress / 0.60
        radius_range = _lerp_range(start_radius_range, angle_master_radius_range, stage_progress)
        if stage_progress < 0.50:
            abs_angle_range = _lerp_range(start_abs_angle_range, mid_abs_angle_range, stage_progress / 0.50)
        else:
            abs_angle_range = _lerp_range(mid_abs_angle_range, late_abs_angle_range, (stage_progress - 0.50) / 0.50)
    # Phase 2: keep the wide rear hemisphere but only grow distance moderately.
    elif progress < 0.85:
        stage_progress = (progress - 0.60) / 0.25
        radius_range = _lerp_range(angle_master_radius_range, radius_expansion_range, stage_progress)
        abs_angle_range = late_abs_angle_range
    else:
        stage_progress = (progress - 0.85) / 0.15
        radius_range = _lerp_range(radius_expansion_range, final_radius_range, stage_progress)
        abs_angle_range = _lerp_range(late_abs_angle_range, final_abs_angle_range, stage_progress)

    command_term.cfg.radius_range = radius_range
    command_term.cfg.abs_angle_range = abs_angle_range
    command_term.metrics.setdefault("curriculum_success_ema", torch.zeros(env.num_envs, device=env.device))
    command_term.metrics["curriculum_success_ema"][:] = float(getattr(env, "_point_goal_curriculum_success_ema", 0.0))
    return progress_tensor


point_goal_distance_obs = base_mdp.point_goal_distance_obs
point_goal_completion_reward = base_mdp.point_goal_completion_reward
point_goal_distance_reward = base_mdp.point_goal_distance_reward
point_goal_heading_alignment_reward = base_mdp.point_goal_heading_alignment_reward
point_goal_heading_error_obs = base_mdp.point_goal_heading_error_obs
point_goal_progress_reward = base_mdp.point_goal_progress_reward
point_goal_rel_body_xy = base_mdp.point_goal_rel_body_xy
point_goal_reward_levels = base_mdp.point_goal_reward_levels
point_goal_root_pos_env = base_mdp.point_goal_root_pos_env
point_goal_stop_reward = base_mdp.point_goal_stop_reward
point_goal_success = base_mdp.point_goal_success
point_goal_success_bonus = base_mdp.point_goal_success_bonus
point_goal_target_pos_env = base_mdp.point_goal_target_pos_env
point_goal_target_timeout = base_mdp.point_goal_target_timeout
point_goal_time_penalty = base_mdp.point_goal_time_penalty
point_goal_timeout_penalty = base_mdp.point_goal_timeout_penalty
