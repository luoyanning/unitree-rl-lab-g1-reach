from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

from ..velocity_env_cfg import CurriculumCfg as BaseCurriculumCfg
from ..velocity_env_cfg import RobotEnvCfg

from .left_hand_loco_reach_adapter_hold_mdp import (
    gated_position_command_error_tanh,
    left_hand_target_pos_levels,
    near_target_action_rate_l2,
    near_target_joint_deviation_l1,
    pre_stance_foot_motion_reward,
    pre_stance_joint_deviation_penalty,
    pre_stance_joint_limit_penalty,
    pre_stance_torso_lean_penalty,
    static_target_position_error,
    success_posture_bonus,
    target_completion_bonus,
    target_hold_reward,
    target_pos_command_obs,
    target_quota_reached,
    target_timeout_reached,
    target_relative_base_stance_l2,
    target_relative_base_stance_progress,
    target_relative_base_stance_ready,
)


LEFT_HAND_BODY_NAME = "left_wrist_yaw_link"
LEFT_HAND_COMMAND_NAME = "left_hand_pose"
STATIC_TARGET_HOLD_S = 1.0e9
PER_TARGET_TIMEOUT_S = 4.0
MAX_TARGETS_PER_EPISODE = 6
POST_SWITCH_STEPS = 30
SUCCESS_ENTER_RADIUS = 0.06
SUCCESS_EXIT_RADIUS = 0.12
SUCCESS_HOLD_STEPS = 15
NEAR_SUCCESS_PENALTY_RADIUS = 0.14
NEAR_SUCCESS_PENALTY_SCALE = 0.05
ADAPTER_GATE_STD = 0.04
ADAPTER_POST_SWITCH_BIAS = 0.35
ADAPTER_MIN_Z_BLEND = 0.35
ADAPTER_SNAP_TO_TARGET_RADIUS = 0.14
LOCO_REACH_NEAR_POS_X = (0.25, 0.48)
LOCO_REACH_POSTURE_POS_X = (0.35, 0.72)
LOCO_REACH_FAR_POS_X = (0.50, 1.00)
LOCO_REACH_NEAR_POS_Y = (0.08, 0.28)
LOCO_REACH_POSTURE_POS_Y = (0.02, 0.38)
LOCO_REACH_FAR_POS_Y = (-0.05, 0.60)
LOCO_REACH_NEAR_POS_Z = (0.18, 0.34)
LOCO_REACH_POSTURE_POS_Z = (0.00, 0.20)
LOCO_REACH_FAR_POS_Z = (0.08, 0.24)
LOCO_REACH_SAMPLE_REGIMES = {
    "near": {
        "pos_x": LOCO_REACH_NEAR_POS_X,
        "pos_y": LOCO_REACH_NEAR_POS_Y,
        "pos_z": LOCO_REACH_NEAR_POS_Z,
    },
    "posture": {
        "pos_x": LOCO_REACH_POSTURE_POS_X,
        "pos_y": LOCO_REACH_POSTURE_POS_Y,
        "pos_z": LOCO_REACH_POSTURE_POS_Z,
    },
    "far": {
        "pos_x": LOCO_REACH_FAR_POS_X,
        "pos_y": LOCO_REACH_FAR_POS_Y,
        "pos_z": LOCO_REACH_FAR_POS_Z,
    },
}
LOCO_REACH_SAMPLE_WEIGHTS = {
    "near": 0.45,
    "posture": 0.30,
    "far": 0.25,
}


@configclass
class LeftHandLocoReachAdapterHoldCurriculumCfg(BaseCurriculumCfg):
    left_hand_target_levels = CurrTerm(
        func=left_hand_target_pos_levels,
        params={
            "command_name": LEFT_HAND_COMMAND_NAME,
            "num_curriculum_episodes": 40,
            "near_pos_x": LOCO_REACH_NEAR_POS_X,
            "posture_pos_x": LOCO_REACH_POSTURE_POS_X,
            "far_pos_x": LOCO_REACH_FAR_POS_X,
            "near_pos_y": LOCO_REACH_NEAR_POS_Y,
            "posture_pos_y": LOCO_REACH_POSTURE_POS_Y,
            "far_pos_y": LOCO_REACH_FAR_POS_Y,
            "near_pos_z": LOCO_REACH_NEAR_POS_Z,
            "posture_pos_z": LOCO_REACH_POSTURE_POS_Z,
            "far_pos_z": LOCO_REACH_FAR_POS_Z,
        },
    )


@configclass
class RobotLeftHandLocoReachAdapterHoldEnvCfg(RobotEnvCfg):
    curriculum: LeftHandLocoReachAdapterHoldCurriculumCfg = LeftHandLocoReachAdapterHoldCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        ee_cfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME])
        left_arm_cfg = SceneEntityCfg(
            "robot",
            joint_names=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
            ],
        )
        right_arm_cfg = SceneEntityCfg(
            "robot",
            joint_names=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
        )
        waist_yaw_cfg = SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])
        feet_cfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])
        stance_x_range = (0.36, 0.58)
        stance_y_range = (0.10, 0.26)
        stance_ready_std = 0.01
        long_horizon_params = {
            "command_name": LEFT_HAND_COMMAND_NAME,
            "success_threshold": SUCCESS_ENTER_RADIUS,
            "success_exit_radius": SUCCESS_EXIT_RADIUS,
            "success_hold_steps": SUCCESS_HOLD_STEPS,
            "max_targets_per_episode": MAX_TARGETS_PER_EPISODE,
            "switch_phase_steps": POST_SWITCH_STEPS,
            "static_target_hold_s": STATIC_TARGET_HOLD_S,
            "per_target_timeout_s": PER_TARGET_TIMEOUT_S,
            "x_range": stance_x_range,
            "y_range": stance_y_range,
            "sample_regimes": LOCO_REACH_SAMPLE_REGIMES,
            "sample_weights": LOCO_REACH_SAMPLE_WEIGHTS,
        }
        adapter_obs_params = {
            **long_horizon_params,
            "adapter_gate_std": ADAPTER_GATE_STD,
            "adapter_post_switch_bias": ADAPTER_POST_SWITCH_BIAS,
            "adapter_min_z_blend": ADAPTER_MIN_Z_BLEND,
            "adapter_snap_to_target_radius": ADAPTER_SNAP_TO_TARGET_RADIUS,
        }

        self.episode_length_s = 24.0
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
            resampling_time_range=(STATIC_TARGET_HOLD_S, STATIC_TARGET_HOLD_S),
            debug_vis=False,
            ranges=reach_mdp.UniformPoseCommandCfg.Ranges(
                pos_x=LOCO_REACH_NEAR_POS_X,
                pos_y=LOCO_REACH_NEAR_POS_Y,
                pos_z=LOCO_REACH_NEAR_POS_Z,
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(-1.0, 1.0),
            ),
        )

        self.actions.JointPositionAction.joint_names = [".*"]
        self.actions.JointPositionAction.scale = 0.25

        # Keep the observation tensor identical to Restore-v0 while swapping the command source
        # to an adapter command derived from the fixed world target.
        self.observations.policy.velocity_commands = ObsTerm(
            func=target_pos_command_obs,
            params=adapter_obs_params,
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=target_pos_command_obs,
            params=adapter_obs_params,
        )

        self.events.base_external_force_torque = None
        self.events.reset_base.params["pose_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.8, 0.8)}

        self.rewards.track_lin_vel_xy.weight = 1.0
        self.rewards.track_ang_vel_z.weight = 0.5
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_legs.weight = -0.005
        self.rewards.joint_deviation_waists.weight = -0.2
        self.rewards.feet_slide.weight = -0.01
        self.rewards.action_rate = RewTerm(
            func=near_target_action_rate_l2,
            weight=-0.02,
            params={
                **long_horizon_params,
                "near_success_penalty_radius": NEAR_SUCCESS_PENALTY_RADIUS,
                "near_success_penalty_scale": NEAR_SUCCESS_PENALTY_SCALE,
            },
        )
        self.rewards.base_target_stance = RewTerm(
            func=target_relative_base_stance_l2,
            weight=-2.5,
            params=long_horizon_params,
        )
        self.rewards.stance_ready = RewTerm(
            func=target_relative_base_stance_ready,
            weight=4.0,
            params={
                **long_horizon_params,
                "gate_std": stance_ready_std,
            },
        )
        self.rewards.stance_progress = RewTerm(
            func=target_relative_base_stance_progress,
            weight=5.0,
            params=long_horizon_params,
        )
        self.rewards.right_arm_balance_posture = RewTerm(
            func=near_target_joint_deviation_l1,
            weight=-0.02,
            params={
                **long_horizon_params,
                "asset_cfg": right_arm_cfg,
                "near_success_penalty_radius": NEAR_SUCCESS_PENALTY_RADIUS,
                "near_success_penalty_scale": NEAR_SUCCESS_PENALTY_SCALE,
            },
        )
        self.rewards.pre_stance_torso_lean = RewTerm(
            func=pre_stance_torso_lean_penalty,
            weight=-1.5,
            params={
                **long_horizon_params,
                "near_success_penalty_radius": NEAR_SUCCESS_PENALTY_RADIUS,
                "near_success_penalty_scale": NEAR_SUCCESS_PENALTY_SCALE,
            },
        )
        self.rewards.pre_stance_waist_twist = RewTerm(
            func=pre_stance_joint_deviation_penalty,
            weight=-0.8,
            params={
                **long_horizon_params,
                "asset_cfg": waist_yaw_cfg,
                "near_success_penalty_radius": NEAR_SUCCESS_PENALTY_RADIUS,
                "near_success_penalty_scale": NEAR_SUCCESS_PENALTY_SCALE,
            },
        )
        self.rewards.pre_stance_arm_extension = RewTerm(
            func=pre_stance_joint_limit_penalty,
            weight=-1.0,
            params={
                **long_horizon_params,
                "asset_cfg": left_arm_cfg,
                "margin_threshold": 0.18,
                "near_success_penalty_radius": NEAR_SUCCESS_PENALTY_RADIUS,
                "near_success_penalty_scale": NEAR_SUCCESS_PENALTY_SCALE,
            },
        )
        self.rewards.pre_stance_foot_motion = RewTerm(
            func=pre_stance_foot_motion_reward,
            weight=0.2,
            params={
                **long_horizon_params,
                "asset_cfg": feet_cfg,
            },
        )
        self.rewards.target_completion = RewTerm(
            func=target_completion_bonus,
            weight=4.0,
            params=long_horizon_params,
        )
        self.rewards.target_hold = RewTerm(
            func=target_hold_reward,
            weight=5.0,
            params={
                "asset_cfg": ee_cfg,
                **long_horizon_params,
                "hold_reward_std": 0.02,
            },
        )
        self.rewards.left_hand_position_tracking = RewTerm(
            func=static_target_position_error,
            weight=-0.12,
            params={
                "asset_cfg": ee_cfg,
                **long_horizon_params,
            },
        )
        self.rewards.left_hand_position_tracking_fine = RewTerm(
            func=gated_position_command_error_tanh,
            weight=9.0,
            params={
                "asset_cfg": ee_cfg,
                **long_horizon_params,
                "std": 0.12,
                "gate_std": stance_ready_std,
            },
        )
        self.rewards.success_posture_bonus = RewTerm(
            func=success_posture_bonus,
            weight=2.5,
            params={
                "asset_cfg": ee_cfg,
                "arm_joint_cfg": left_arm_cfg,
                **long_horizon_params,
                "gate_std": stance_ready_std,
            },
        )

        self.rewards.base_height.params["target_height"] = 0.74
        self.terminations.base_height.params["minimum_height"] = 0.12
        self.terminations.bad_orientation.params["limit_angle"] = 1.0
        self.terminations.reach_success = None
        self.terminations.target_quota = DoneTerm(
            func=target_quota_reached,
            params=long_horizon_params,
        )
        self.terminations.target_timeout = DoneTerm(
            func=target_timeout_reached,
            params=long_horizon_params,
        )

        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotLeftHandLocoReachAdapterHoldPlayEnvCfg(RobotLeftHandLocoReachAdapterHoldEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.6, -3.4, 1.9)
        self.viewer.lookat = (0.0, 0.0, 0.65)

        self.commands.left_hand_pose.ranges.pos_x = (LOCO_REACH_NEAR_POS_X[0], LOCO_REACH_FAR_POS_X[1])
        self.commands.left_hand_pose.ranges.pos_y = (LOCO_REACH_FAR_POS_Y[0], LOCO_REACH_FAR_POS_Y[1])
        self.commands.left_hand_pose.ranges.pos_z = (LOCO_REACH_POSTURE_POS_Z[0], LOCO_REACH_FAR_POS_Z[1])
        self.curriculum.left_hand_target_levels = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
