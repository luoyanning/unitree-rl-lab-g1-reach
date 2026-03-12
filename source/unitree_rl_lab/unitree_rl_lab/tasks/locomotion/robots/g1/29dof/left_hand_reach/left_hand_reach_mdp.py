from __future__ import annotations

import torch

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
