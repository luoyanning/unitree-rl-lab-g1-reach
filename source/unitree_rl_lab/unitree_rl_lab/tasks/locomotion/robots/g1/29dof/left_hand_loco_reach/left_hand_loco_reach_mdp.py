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
    start_pos_x: tuple[float, float] = (0.25, 0.50),
    mid_pos_x: tuple[float, float] = (0.40, 0.85),
    final_pos_x: tuple[float, float] = (0.35, 1.05),
    start_pos_y: tuple[float, float] = (0.05, 0.30),
    mid_pos_y: tuple[float, float] = (0.05, 0.45),
    final_pos_y: tuple[float, float] = (-0.05, 0.65),
    start_pos_z: tuple[float, float] = (0.15, 0.40),
    mid_pos_z: tuple[float, float] = (0.00, 0.35),
    final_pos_z: tuple[float, float] = (-0.10, 0.35),
):
    """Expand targets from in-place reach to posture-heavy and local-walking reach."""
    del env_ids
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges

    progress = min(env.common_step_counter / (env.max_episode_length * num_curriculum_episodes), 1.0)
    progress_tensor = torch.tensor(progress, device=env.device)
    half_tensor = torch.tensor(0.5, device=env.device)

    if env.common_step_counter % env.max_episode_length == 0:
        if progress <= 0.5:
            phase_progress = progress_tensor / half_tensor
            ranges.pos_x = torch.lerp(
                torch.tensor(start_pos_x, device=env.device),
                torch.tensor(mid_pos_x, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_y = torch.lerp(
                torch.tensor(start_pos_y, device=env.device),
                torch.tensor(mid_pos_y, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_z = torch.lerp(
                torch.tensor(start_pos_z, device=env.device),
                torch.tensor(mid_pos_z, device=env.device),
                phase_progress,
            ).tolist()
        else:
            phase_progress = (progress_tensor - half_tensor) / half_tensor
            ranges.pos_x = torch.lerp(
                torch.tensor(mid_pos_x, device=env.device),
                torch.tensor(final_pos_x, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_y = torch.lerp(
                torch.tensor(mid_pos_y, device=env.device),
                torch.tensor(final_pos_y, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_z = torch.lerp(
                torch.tensor(mid_pos_z, device=env.device),
                torch.tensor(final_pos_z, device=env.device),
                phase_progress,
            ).tolist()

    return progress_tensor


def target_relative_base_stance_l2(
    env,
    command_name: str = "left_hand_pose",
    desired_distance: float = 0.58,
    distance_band: float = 0.16,
):
    """Penalize target-to-base planar distance only when outside a local reach-friendly band."""
    target_pos_xy = env.command_manager.get_command(command_name)[:, :2]
    planar_distance = torch.norm(target_pos_xy, dim=-1)
    stance_error = torch.clamp(torch.abs(planar_distance - desired_distance) - distance_band, min=0.0)
    return torch.square(stance_error)
