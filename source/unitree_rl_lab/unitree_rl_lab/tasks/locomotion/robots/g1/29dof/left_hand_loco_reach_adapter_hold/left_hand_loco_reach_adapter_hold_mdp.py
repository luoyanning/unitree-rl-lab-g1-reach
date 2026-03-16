from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from ..left_hand_loco_reach import left_hand_loco_reach_mdp as fixed_mdp


# Reuse the fixed-target task machinery and reward terms, but expose a 3D internal
# command so the policy interface stays aligned with the restore route (480/495 obs).
static_target_position_error = fixed_mdp.static_target_position_error
left_hand_target_pos_levels = fixed_mdp.left_hand_target_pos_levels
pre_stance_torso_lean_penalty = fixed_mdp.pre_stance_torso_lean_penalty
pre_stance_joint_deviation_penalty = fixed_mdp.pre_stance_joint_deviation_penalty
pre_stance_joint_limit_penalty = fixed_mdp.pre_stance_joint_limit_penalty
pre_stance_foot_motion_reward = fixed_mdp.pre_stance_foot_motion_reward
target_relative_base_stance_l2 = fixed_mdp.target_relative_base_stance_l2
target_relative_base_stance_ready = fixed_mdp.target_relative_base_stance_ready
target_relative_base_stance_progress = fixed_mdp.target_relative_base_stance_progress
gated_position_command_error_tanh = fixed_mdp.gated_position_command_error_tanh
target_completion_bonus = fixed_mdp.target_completion_bonus
target_hold_reward = fixed_mdp.target_hold_reward
success_posture_bonus = fixed_mdp.success_posture_bonus
target_quota_reached = fixed_mdp.target_quota_reached
target_timeout_reached = fixed_mdp.target_timeout_reached
near_target_action_rate_l2 = fixed_mdp.near_target_action_rate_l2
near_target_joint_deviation_l1 = fixed_mdp.near_target_joint_deviation_l1


def _pose_command_tensor(env, command_name: str):
    command_term = env.command_manager.get_term(command_name)
    for attr_name in ("_command", "command"):
        if hasattr(command_term, attr_name):
            command_tensor = getattr(command_term, attr_name)
            if isinstance(command_tensor, torch.Tensor) and command_tensor.ndim == 2:
                return command_tensor
    return None


def _ensure_adapter_state(env):
    if not hasattr(env, "_left_hand_adapter_command"):
        env._left_hand_adapter_command = torch.zeros(env.num_envs, 3, device=env.device)


def _adapter_command_range(sample_regimes, axis_name: str):
    return fixed_mdp._range_union(sample_regimes, axis_name)


def _compute_adapter_command(
    env,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    switch_phase_steps: int,
    sample_regimes,
    sample_weights,
    command_name: str,
    adapter_gate_std: float,
    adapter_post_switch_bias: float,
    adapter_min_z_blend: float,
    adapter_snap_to_target_radius: float,
):
    del sample_weights, command_name
    fixed_mdp._get_sampling_distribution_state(env, sample_regimes=sample_regimes, sample_weights=None)
    target_pos_base = fixed_mdp._active_target_pos_base_yaw(env)
    workspace_gate = fixed_mdp._workspace_ready_gate(
        env,
        command_name="left_hand_pose",
        x_range=x_range,
        y_range=y_range,
        gate_std=adapter_gate_std,
    )
    switch_phase = fixed_mdp._switch_phase_scale(env, switch_phase_steps)
    corridor_center = torch.tensor(
        [0.5 * (x_range[0] + x_range[1]), 0.5 * (y_range[0] + y_range[1])],
        device=env.device,
    )

    blend = torch.clamp(workspace_gate * (1.0 - adapter_post_switch_bias * switch_phase), min=0.0, max=1.0)
    near_success = getattr(env, "_left_hand_in_success_zone", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    near_target = fixed_mdp._static_target_position_error(env) <= adapter_snap_to_target_radius
    direct_target_mode = near_success | near_target
    blend = torch.where(direct_target_mode, torch.ones_like(blend), blend)

    adapted_xy = corridor_center.unsqueeze(0) + (target_pos_base[:, :2] - corridor_center.unsqueeze(0)) * blend.unsqueeze(-1)

    active_regimes, _ = fixed_mdp._get_sampling_distribution_state(env, sample_regimes=sample_regimes, sample_weights=None)
    x_cmd_range = _adapter_command_range(active_regimes, "pos_x")
    y_cmd_range = _adapter_command_range(active_regimes, "pos_y")
    z_cmd_range = _adapter_command_range(active_regimes, "pos_z")

    z_center = 0.5 * (z_cmd_range[0] + z_cmd_range[1])
    raw_z = torch.clamp(target_pos_base[:, 2], min=z_cmd_range[0], max=z_cmd_range[1])
    z_blend = torch.clamp(adapter_min_z_blend + (1.0 - adapter_min_z_blend) * blend, min=0.0, max=1.0)
    adapted_z = z_center + (raw_z - z_center) * z_blend

    adapter_command = torch.stack(
        (
            torch.clamp(adapted_xy[:, 0], min=x_cmd_range[0], max=x_cmd_range[1]),
            torch.clamp(adapted_xy[:, 1], min=y_cmd_range[0], max=y_cmd_range[1]),
            adapted_z,
        ),
        dim=-1,
    )
    adapter_command[direct_target_mode] = torch.stack(
        (
            torch.clamp(target_pos_base[direct_target_mode, 0], min=x_cmd_range[0], max=x_cmd_range[1]),
            torch.clamp(target_pos_base[direct_target_mode, 1], min=y_cmd_range[0], max=y_cmd_range[1]),
            torch.clamp(target_pos_base[direct_target_mode, 2], min=z_cmd_range[0], max=z_cmd_range[1]),
        ),
        dim=-1,
    )
    return adapter_command


def _sync_adapter_state(
    env,
    command_name: str,
    success_threshold: float,
    max_targets_per_episode: int,
    switch_phase_steps: int,
    static_target_hold_s: float,
    per_target_timeout_s: float,
    success_exit_radius: float,
    success_hold_steps: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None,
    sample_weights: dict[str, float] | None,
    adapter_gate_std: float = 0.04,
    adapter_post_switch_bias: float = 0.35,
    adapter_min_z_blend: float = 0.35,
    adapter_snap_to_target_radius: float = 0.12,
):
    fixed_mdp._sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    _ensure_adapter_state(env)
    adapter_command = _compute_adapter_command(
        env,
        x_range=x_range,
        y_range=y_range,
        switch_phase_steps=switch_phase_steps,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
        command_name=command_name,
        adapter_gate_std=adapter_gate_std,
        adapter_post_switch_bias=adapter_post_switch_bias,
        adapter_min_z_blend=adapter_min_z_blend,
        adapter_snap_to_target_radius=adapter_snap_to_target_radius,
    )
    env._left_hand_adapter_command[:] = adapter_command

    pose_command = _pose_command_tensor(env, command_name=command_name)
    if pose_command is not None and pose_command.shape[1] >= 3:
        pose_command[:, :3] = adapter_command
        if pose_command.shape[1] >= 6:
            pose_command[:, 3:6] = 0.0


def target_pos_command_obs(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    adapter_gate_std: float = 0.04,
    adapter_post_switch_bias: float = 0.35,
    adapter_min_z_blend: float = 0.35,
    adapter_snap_to_target_radius: float = 0.12,
):
    _sync_adapter_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
        adapter_gate_std=adapter_gate_std,
        adapter_post_switch_bias=adapter_post_switch_bias,
        adapter_min_z_blend=adapter_min_z_blend,
        adapter_snap_to_target_radius=adapter_snap_to_target_radius,
    )
    return env._left_hand_adapter_command


# Optional utility for future debugging or reward shaping against the internal adapter command.
def adapter_position_error(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    **kwargs,
):
    _sync_adapter_state(env, **kwargs)
    asset = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if ee_pos_w.ndim == 3:
        ee_pos_w = ee_pos_w[:, 0]
    return torch.linalg.norm(env._left_hand_adapter_command - ee_pos_w, dim=-1)
