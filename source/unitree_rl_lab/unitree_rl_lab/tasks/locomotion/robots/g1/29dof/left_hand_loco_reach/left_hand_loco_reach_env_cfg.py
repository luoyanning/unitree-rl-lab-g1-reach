from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

from unitree_rl_lab.tasks.locomotion import mdp
from ..velocity_env_cfg import CurriculumCfg as BaseCurriculumCfg
from ..velocity_env_cfg import RobotEnvCfg

from .left_hand_loco_reach_mdp import (
    left_hand_target_pos_levels,
    reach_success,
    target_pos_command_obs,
    target_relative_base_stance_l2,
)


LEFT_HAND_BODY_NAME = "left_wrist_yaw_link"
LEFT_HAND_COMMAND_NAME = "left_hand_pose"
LOCO_REACH_START_POS_X = (0.25, 0.50)
LOCO_REACH_MID_POS_X = (0.40, 0.85)
LOCO_REACH_FINAL_POS_X = (0.35, 1.05)
LOCO_REACH_START_POS_Y = (0.05, 0.30)
LOCO_REACH_MID_POS_Y = (0.05, 0.45)
LOCO_REACH_FINAL_POS_Y = (-0.05, 0.65)
LOCO_REACH_START_POS_Z = (0.15, 0.40)
LOCO_REACH_MID_POS_Z = (0.00, 0.35)
LOCO_REACH_FINAL_POS_Z = (-0.10, 0.35)


@configclass
class LeftHandLocoReachCurriculumCfg(BaseCurriculumCfg):
    left_hand_target_levels = CurrTerm(
        func=left_hand_target_pos_levels,
        params={
            "command_name": LEFT_HAND_COMMAND_NAME,
            "num_curriculum_episodes": 40,
            "start_pos_x": LOCO_REACH_START_POS_X,
            "mid_pos_x": LOCO_REACH_MID_POS_X,
            "final_pos_x": LOCO_REACH_FINAL_POS_X,
            "start_pos_y": LOCO_REACH_START_POS_Y,
            "mid_pos_y": LOCO_REACH_MID_POS_Y,
            "final_pos_y": LOCO_REACH_FINAL_POS_Y,
            "start_pos_z": LOCO_REACH_START_POS_Z,
            "mid_pos_z": LOCO_REACH_MID_POS_Z,
            "final_pos_z": LOCO_REACH_FINAL_POS_Z,
        },
    )


@configclass
class RobotLeftHandLocoReachEnvCfg(RobotEnvCfg):
    """Aggressive local loco-reach task with full-body participation."""

    curriculum: LeftHandLocoReachCurriculumCfg = LeftHandLocoReachCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        ee_cfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME])

        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)

        self.commands.left_hand_pose = reach_mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name=LEFT_HAND_BODY_NAME,
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            ranges=reach_mdp.UniformPoseCommandCfg.Ranges(
                pos_x=LOCO_REACH_START_POS_X,
                pos_y=LOCO_REACH_START_POS_Y,
                pos_z=LOCO_REACH_START_POS_Z,
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(-1.0, 1.0),
            ),
        )

        self.actions.JointPositionAction.joint_names = [".*"]
        self.actions.JointPositionAction.scale = 0.25

        # Keep policy/critic tensor sizes aligned with the locomotion task for checkpoint warm-starting.
        self.observations.policy.velocity_commands = ObsTerm(
            func=target_pos_command_obs,
            params={"command_name": LEFT_HAND_COMMAND_NAME},
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=target_pos_command_obs,
            params={"command_name": LEFT_HAND_COMMAND_NAME},
        )

        self.events.base_external_force_torque = None
        self.events.reset_base.params["pose_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.8, 0.8)}

        self.rewards.track_lin_vel_xy = None
        self.rewards.track_ang_vel_z = None
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_legs.weight = -0.01
        self.rewards.joint_deviation_waists.weight = -0.45
        self.rewards.feet_slide.weight = -0.01
        self.rewards.action_rate.weight = -0.02
        self.rewards.base_target_stance = RewTerm(
            func=target_relative_base_stance_l2,
            weight=-0.45,
            params={
                "command_name": LEFT_HAND_COMMAND_NAME,
                "desired_distance": 0.58,
                "distance_band": 0.16,
            },
        )
        self.rewards.right_arm_balance_posture = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.02,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "right_shoulder_.*_joint",
                        "right_elbow_joint",
                        "right_wrist_.*_joint",
                    ],
                )
            },
        )
        self.rewards.left_hand_position_tracking = RewTerm(
            func=reach_mdp.position_command_error,
            weight=-0.25,
            params={
                "asset_cfg": ee_cfg,
                "command_name": LEFT_HAND_COMMAND_NAME,
            },
        )
        self.rewards.left_hand_position_tracking_fine = RewTerm(
            func=reach_mdp.position_command_error_tanh,
            weight=6.0,
            params={
                "asset_cfg": ee_cfg,
                "command_name": LEFT_HAND_COMMAND_NAME,
                "std": 0.14,
            },
        )

        self.terminations.reach_success = DoneTerm(
            func=reach_success,
            params={
                "asset_cfg": ee_cfg,
                "command_name": LEFT_HAND_COMMAND_NAME,
                "threshold": 0.06,
            },
        )

        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotLeftHandLocoReachPlayEnvCfg(RobotLeftHandLocoReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.6, -3.4, 1.9)
        self.viewer.lookat = (0.0, 0.0, 0.65)

        self.commands.left_hand_pose.ranges.pos_x = LOCO_REACH_FINAL_POS_X
        self.commands.left_hand_pose.ranges.pos_y = LOCO_REACH_FINAL_POS_Y
        self.commands.left_hand_pose.ranges.pos_z = LOCO_REACH_FINAL_POS_Z
        self.curriculum.left_hand_target_levels = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
