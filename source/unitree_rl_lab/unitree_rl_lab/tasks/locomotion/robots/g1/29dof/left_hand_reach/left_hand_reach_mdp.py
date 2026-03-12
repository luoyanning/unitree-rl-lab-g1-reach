import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp
from isaaclab.managers import SceneEntityCfg


def reach_success(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "ee_pose",
    threshold: float = 0.05,
):
    """左手末端进入目标阈值球即成功。"""
    pos_err = reach_mdp.position_command_error(
        env,
        asset_cfg=asset_cfg,
        command_name=command_name,
    )
    return pos_err < threshold