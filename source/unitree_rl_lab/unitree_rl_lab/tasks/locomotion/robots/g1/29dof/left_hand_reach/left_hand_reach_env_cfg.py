from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

from unitree_rl_lab.tasks.locomotion import mdp
from ..velocity_env_cfg import RobotEnvCfg

from .left_hand_reach_mdp import position_command_error_obs, reach_success, root_planar_drift_l2


LEFT_HAND_BODY_NAME = "left_wrist_yaw_link"
LEFT_HAND_COMMAND_NAME = "left_hand_pose"
LEFT_HAND_REACH_JOINTS = [
    "waist_.*_joint",
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "left_shoulder_.*_joint",
    "left_elbow_joint",
    "left_wrist_.*_joint",
]


@configclass
class RobotLeftHandReachEnvCfg(RobotEnvCfg):
    """Stage-2 standing reach task with limited balance-adjusting leg support."""

    def __post_init__(self):
        super().__post_init__()

        ee_cfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME])

        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.base_velocity.rel_standing_envs = 1.0
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
                pos_x=(0.25, 0.52),
                pos_y=(0.08, 0.34),
                pos_z=(0.15, 0.36),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(-0.75, 0.75),
            ),
        )

        self.actions.JointPositionAction.joint_names = LEFT_HAND_REACH_JOINTS
        self.actions.JointPositionAction.scale = 0.25

        self.observations.policy.left_hand_target_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": LEFT_HAND_COMMAND_NAME},
        )
        self.observations.policy.left_hand_target_distance = ObsTerm(
            func=position_command_error_obs,
            params={"asset_cfg": ee_cfg, "command_name": LEFT_HAND_COMMAND_NAME},
        )
        self.observations.critic.left_hand_target_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": LEFT_HAND_COMMAND_NAME},
        )
        self.observations.critic.left_hand_target_distance = ObsTerm(
            func=position_command_error_obs,
            params={"asset_cfg": ee_cfg, "command_name": LEFT_HAND_COMMAND_NAME},
        )

        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.5, 0.5)}

        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_legs.weight = -0.2
        self.rewards.joint_deviation_waists.weight = -0.2
        self.rewards.feet_slide.weight = -0.05
        self.rewards.action_rate.weight = -0.02
        self.rewards.base_planar_drift = RewTerm(
            func=root_planar_drift_l2,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        self.rewards.left_hand_position_tracking = RewTerm(
            func=reach_mdp.position_command_error,
            weight=-0.2,
            params={
                "asset_cfg": ee_cfg,
                "command_name": LEFT_HAND_COMMAND_NAME,
            },
        )
        self.rewards.left_hand_position_tracking_fine = RewTerm(
            func=reach_mdp.position_command_error_tanh,
            weight=4.0,
            params={
                "asset_cfg": ee_cfg,
                "command_name": LEFT_HAND_COMMAND_NAME,
                "std": 0.10,
            },
        )

        self.terminations.reach_success = DoneTerm(
            func=reach_success,
            params={
                "asset_cfg": ee_cfg,
                "command_name": LEFT_HAND_COMMAND_NAME,
                "threshold": 0.05,
            },
        )

        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotLeftHandReachPlayEnvCfg(RobotLeftHandReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.6, -3.4, 1.9)
        self.viewer.lookat = (0.0, 0.0, 0.65)

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
