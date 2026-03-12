import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

from .left_hand_reach_mdp import reach_success

# ===== 按你本地真实类名改这里 =====
# 例如：from ..velocity_env_cfg import G1FlatEnvCfg
# 或者：from ..velocity_env_cfg import UnitreeG129DofFlatEnvCfg
from ..velocity_env_cfg import G1FlatEnvCfg


@configclass
class G1LeftHandReachEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------
        # 0) 任务目标：站着不动，左手去 reach 一个近处目标
        # ------------------------------------------------------------------
        # 把 locomotion 命令固定成“原地站立”
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        if hasattr(self.commands.base_velocity.ranges, "heading"):
            self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # ------------------------------------------------------------------
        # 1) 添加一个 Reach 目标命令 ee_pose
        #    第一版不放真实刚体球，先用 Reach 自带的 debug_vis 目标
        # ------------------------------------------------------------------
        self.commands.ee_pose = reach_mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name="left_wrist_yaw_link",
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            debug_vis=True,
            ranges=reach_mdp.UniformPoseCommandCfg.Ranges(
                # 这些范围是“近处、左前方、手臂可达”的第一版安全范围
                pos_x=(0.20, 0.40),
                pos_y=(0.15, 0.35),
                pos_z=(0.15, 0.40),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(-0.5, 0.5),
            ),
        )

        # ------------------------------------------------------------------
        # 2) 观测里加入目标命令
        #    官方 Reach 任务的 policy 观测里就有 pose_command
        # ------------------------------------------------------------------
        self.observations.policy.pose_command = ObsTerm(
            func=reach_mdp.generated_commands,
            params={"command_name": "ee_pose"},
        )

        # ------------------------------------------------------------------
        # 3) 动作：第一版强烈建议只开放 torso + 左臂
        #    这样先验证“抬左手不倒”，不要一开始就让全身乱动
        # ------------------------------------------------------------------
        # 按你本地动作字段名改这里；大多数 velocity 任务里就是 joint_pos
        if hasattr(self.actions, "joint_pos"):
            self.actions.joint_pos.joint_names = [
                "torso_joint",
                "left_.*shoulder.*_joint",
                "left_.*elbow.*_joint",
                "left_.*wrist.*_joint",
            ]
            self.actions.joint_pos.scale = 0.25

        # ------------------------------------------------------------------
        # 4) 去掉会和 reach 目标“打架”的 locomotion 奖励
        # ------------------------------------------------------------------
        # 不希望机器人为了 gait 奖励而抬脚/迈步
        if hasattr(self.rewards, "feet_air_time"):
            self.rewards.feet_air_time = None

        # 这两个是 locomotion 里“手臂尽量别动”的正则项；reach 第一版必须关掉
        if hasattr(self.rewards, "joint_deviation_arms"):
            self.rewards.joint_deviation_arms = None
        if hasattr(self.rewards, "joint_deviation_fingers"):
            self.rewards.joint_deviation_fingers = None

        # torso 还可以保留一个很弱的约束，防止过度扭腰
        if hasattr(self.rewards, "joint_deviation_torso") and self.rewards.joint_deviation_torso is not None:
            self.rewards.joint_deviation_torso.weight = -0.02

        # ------------------------------------------------------------------
        # 5) 加 Reach 奖励：粗粒度 + 细粒度
        #    直接复用官方 Reach 的位置 tracking 项
        # ------------------------------------------------------------------
        ee_cfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"])

        self.rewards.ee_pos_tracking = RewTerm(
            func=reach_mdp.position_command_error,
            weight=-0.4,
            params={
                "asset_cfg": ee_cfg,
                "command_name": "ee_pose",
            },
        )

        self.rewards.ee_pos_tracking_fine = RewTerm(
            func=reach_mdp.position_command_error_tanh,
            weight=3.0,
            params={
                "asset_cfg": ee_cfg,
                "command_name": "ee_pose",
                "std": 0.08,
            },
        )

        # 第一版先不加 orientation tracking
        # 因为你的目标是“碰到近处目标”，不是精确对齐手掌姿态

        # ------------------------------------------------------------------
        # 6) 保留平衡/稳定相关项
        # ------------------------------------------------------------------
        # 这些一般不要关：站稳、别摔、动作别太抖
        # 你现有 G1 locomotion 里通常已经有：
        # - flat_orientation_l2
        # - base_height
        # - action_rate_l2
        # - dof_acc_l2
        # - base_contact / bad_orientation termination
        #
        # 这里不做大改，只建议把动作抖动惩罚稍微留着
        if hasattr(self.rewards, "action_rate_l2") and self.rewards.action_rate_l2 is not None:
            self.rewards.action_rate_l2.weight = -0.01

        # ------------------------------------------------------------------
        # 7) 成功终止：末端进入 5cm 阈值球
        # ------------------------------------------------------------------
        self.terminations.reach_success = DoneTerm(
            func=reach_success,
            params={
                "asset_cfg": ee_cfg,
                "command_name": "ee_pose",
                "threshold": 0.05,
            },
        )


@configclass
class G1LeftHandReachEnvCfg_PLAY(G1LeftHandReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # 演示/录视频时关掉观测扰动
        self.observations.policy.enable_corruption = False

        # 去掉随机推搡
        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None