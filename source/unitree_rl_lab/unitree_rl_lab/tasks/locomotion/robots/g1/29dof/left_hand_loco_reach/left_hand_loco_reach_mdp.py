from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp
from isaaclab.managers import SceneEntityCfg


def target_pos_command_obs(env, command_name: str = "left_hand_pose"):
    """Return the target position command as a 3D vector for policy/critic compatibility."""
    return env.command_manager.get_command(command_name)[:, :3]


def reach_success(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
    threshold: float = 0.05,
):
    """Terminate an episode when the left hand reaches the target threshold."""
    position_error = reach_mdp.position_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    return position_error < threshold


def left_hand_target_pos_levels(
    env,
    env_ids: Sequence[int],
    command_name: str = "left_hand_pose",
    num_curriculum_episodes: int = 40,
    near_pos_x: tuple[float, float] = (0.25, 0.48),
    posture_pos_x: tuple[float, float] = (0.35, 0.72),
    far_pos_x: tuple[float, float] = (0.50, 1.00),
    near_pos_y: tuple[float, float] = (0.08, 0.28),
    posture_pos_y: tuple[float, float] = (0.02, 0.38),
    far_pos_y: tuple[float, float] = (-0.05, 0.60),
    near_pos_z: tuple[float, float] = (0.18, 0.34),
    posture_pos_z: tuple[float, float] = (0.00, 0.22),
    far_pos_z: tuple[float, float] = (0.08, 0.30),
):
    """Expand explicit near, posture-heavy, and far-local loco-reach target regimes."""
    del env_ids
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges

    progress = min(env.common_step_counter / (env.max_episode_length * num_curriculum_episodes), 1.0)
    progress_tensor = torch.tensor(progress, device=env.device)
    third_tensor = torch.tensor(1.0 / 3.0, device=env.device)
    two_third_tensor = torch.tensor(2.0 / 3.0, device=env.device)

    if env.common_step_counter % env.max_episode_length == 0:
        if progress <= 1.0 / 3.0:
            phase_progress = progress_tensor / third_tensor
            ranges.pos_x = torch.lerp(
                torch.tensor(near_pos_x, device=env.device),
                torch.tensor(posture_pos_x, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_y = torch.lerp(
                torch.tensor(near_pos_y, device=env.device),
                torch.tensor(posture_pos_y, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_z = torch.lerp(
                torch.tensor(near_pos_z, device=env.device),
                torch.tensor(posture_pos_z, device=env.device),
                phase_progress,
            ).tolist()
        elif progress <= 2.0 / 3.0:
            phase_progress = (progress_tensor - third_tensor) / third_tensor
            ranges.pos_x = torch.lerp(
                torch.tensor(posture_pos_x, device=env.device),
                torch.tensor(far_pos_x, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_y = torch.lerp(
                torch.tensor(posture_pos_y, device=env.device),
                torch.tensor(far_pos_y, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_z = torch.lerp(
                torch.tensor(posture_pos_z, device=env.device),
                torch.tensor(far_pos_z, device=env.device),
                phase_progress,
            ).tolist()
        else:
            phase_progress = (progress_tensor - two_third_tensor) / third_tensor
            ranges.pos_x = torch.lerp(
                torch.tensor(far_pos_x, device=env.device),
                torch.tensor((near_pos_x[0], far_pos_x[1]), device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_y = torch.lerp(
                torch.tensor(far_pos_y, device=env.device),
                torch.tensor((far_pos_y[0], far_pos_y[1]), device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_z = torch.lerp(
                torch.tensor(far_pos_z, device=env.device),
                torch.tensor((posture_pos_z[0], near_pos_z[1]), device=env.device),
                phase_progress,
            ).tolist()

    return progress_tensor


def target_relative_base_stance_l2(
    env,
    command_name: str = "left_hand_pose",
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
):
    """Penalize target positions that lie outside a favorable left-hand reach corridor in body/base-yaw frame."""
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    x_error_low = torch.clamp(x_range[0] - target_pos[:, 0], min=0.0)
    x_error_high = torch.clamp(target_pos[:, 0] - x_range[1], min=0.0)
    y_error_low = torch.clamp(y_range[0] - target_pos[:, 1], min=0.0)
    y_error_high = torch.clamp(target_pos[:, 1] - y_range[1], min=0.0)
    return torch.square(x_error_low) + torch.square(x_error_high) + torch.square(y_error_low) + torch.square(y_error_high)


def position_command_progress_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
):
    """Reward step-to-step reduction in hand-to-target error."""
    position_error = reach_mdp.position_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    if not hasattr(env, "_left_hand_prev_error"):
        env._left_hand_prev_error = position_error.clone()
    if hasattr(env, "episode_length_buf"):
        reset_ids = env.episode_length_buf == 0
        env._left_hand_prev_error[reset_ids] = position_error[reset_ids]
    progress = env._left_hand_prev_error - position_error
    env._left_hand_prev_error = position_error.clone()
    return progress
