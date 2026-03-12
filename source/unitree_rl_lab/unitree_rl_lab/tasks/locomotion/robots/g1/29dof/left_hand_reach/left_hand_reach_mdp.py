from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def position_command_error_obs(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
):
    """Return the end-effector position error as a single observation term."""
    return reach_mdp.position_command_error(env, asset_cfg=asset_cfg, command_name=command_name).unsqueeze(-1)


def reach_success(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
    threshold: float = 0.05,
):
    """Terminate an episode when the left hand reaches the target threshold."""
    position_error = reach_mdp.position_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    return position_error < threshold


def root_planar_drift_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Penalize horizontal drift away from the environment origin."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_xy = asset.data.root_pos_w[:, :2]
    origin_xy = env.scene.env_origins[:, :2]
    return torch.sum(torch.square(root_pos_xy - origin_xy), dim=-1)


def left_hand_target_pos_levels(
    env,
    env_ids: Sequence[int],
    command_name: str = "left_hand_pose",
    num_curriculum_episodes: int = 30,
    start_pos_x: tuple[float, float] = (0.25, 0.42),
    final_pos_x: tuple[float, float] = (0.25, 0.52),
    start_pos_y: tuple[float, float] = (0.10, 0.30),
    final_pos_y: tuple[float, float] = (0.08, 0.34),
    start_pos_z: tuple[float, float] = (0.18, 0.34),
    final_pos_z: tuple[float, float] = (0.15, 0.36),
):
    """Gradually expand the left-hand reach target range from easier to Stage-2 range."""
    del env_ids
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges

    progress = min(env.common_step_counter / (env.max_episode_length * num_curriculum_episodes), 1.0)
    progress_tensor = torch.tensor(progress, device=env.device)

    if env.common_step_counter % env.max_episode_length == 0:
        ranges.pos_x = torch.lerp(
            torch.tensor(start_pos_x, device=env.device),
            torch.tensor(final_pos_x, device=env.device),
            progress_tensor,
        ).tolist()
        ranges.pos_y = torch.lerp(
            torch.tensor(start_pos_y, device=env.device),
            torch.tensor(final_pos_y, device=env.device),
            progress_tensor,
        ).tolist()
        ranges.pos_z = torch.lerp(
            torch.tensor(start_pos_z, device=env.device),
            torch.tensor(final_pos_z, device=env.device),
            progress_tensor,
        ).tolist()

    return progress_tensor
