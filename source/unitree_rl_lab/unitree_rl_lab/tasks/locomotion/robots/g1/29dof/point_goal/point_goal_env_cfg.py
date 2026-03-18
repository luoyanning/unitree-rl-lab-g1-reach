import math
import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from ..velocity_env_cfg import CurriculumCfg as BaseCurriculumCfg
from ..velocity_env_cfg import RobotEnvCfg
from .point_goal_mdp import (
    PointGoalCommandCfg,
    point_goal_distance_obs,
    point_goal_heading_error_obs,
    point_goal_progress_reward,
    point_goal_rel_body_xy,
    point_goal_root_pos_env,
    point_goal_stop_reward,
    point_goal_success,
    point_goal_success_bonus,
    point_goal_target_levels,
    point_goal_target_timeout,
    point_goal_target_pos_env,
    point_goal_time_penalty,
    point_goal_timeout_penalty,
)


SUCCESS_DISTANCE = 0.12
SUCCESS_HOLD_STEPS = 8
STOP_VELOCITY_THRESHOLD = 0.10
STOP_YAW_RATE_THRESHOLD = 0.25
PER_TARGET_TIMEOUT_S = 4.5
POINT_GOAL_START_RADIUS = (0.30, 0.50)
POINT_GOAL_MID_RADIUS = (0.35, 0.75)
POINT_GOAL_FINAL_RADIUS = (0.40, 1.00)
POINT_GOAL_START_ANGLE = (0.0, 0.0)
POINT_GOAL_MID_ANGLE = (-math.radians(20.0), math.radians(20.0))
POINT_GOAL_FINAL_ANGLE = (-math.radians(90.0), math.radians(90.0))
POINT_GOAL_CURRICULUM_EPISODES = 24


@configclass
class PointGoalCurriculumCfg(BaseCurriculumCfg):
    point_goal_target_levels = CurrTerm(
        func=point_goal_target_levels,
        params={
            "command_name": "base_velocity",
            "num_curriculum_episodes": POINT_GOAL_CURRICULUM_EPISODES,
            "start_radius_range": POINT_GOAL_START_RADIUS,
            "mid_radius_range": POINT_GOAL_MID_RADIUS,
            "final_radius_range": POINT_GOAL_FINAL_RADIUS,
            "start_angle_range": POINT_GOAL_START_ANGLE,
            "mid_angle_range": POINT_GOAL_MID_ANGLE,
            "final_angle_range": POINT_GOAL_FINAL_ANGLE,
        },
    )


@configclass
class RobotPointGoalEnvCfg(RobotEnvCfg):
    curriculum: PointGoalCurriculumCfg = PointGoalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        frame_yaw_offset = float(os.getenv("UTRL_POINT_GOAL_FRAME_YAW_OFFSET", "0.0"))

        self.episode_length_s = 12.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.commands.base_velocity = PointGoalCommandCfg(
            asset_name="robot",
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            debug_vis=True,
            radius_range=POINT_GOAL_START_RADIUS,
            angle_range=POINT_GOAL_START_ANGLE,
            forward_gain=1.1,
            lateral_gain=0.4,
            heading_gain=1.0,
            min_lin_vel_x=-0.12,
            max_lin_vel_x=0.45,
            max_lin_vel_y=0.18,
            max_ang_vel_z=0.30,
            slow_down_distance=0.55,
            stop_distance=0.35,
            heading_slow_down_distance=0.6,
            hold_position_distance=0.08,
            near_recovery_distance=0.18,
            recovery_turn_threshold=1.05,
            heading_block_threshold=1.65,
            min_recovery_ang_vel_z=0.12,
            reverse_recovery_distance=0.25,
            reverse_heading_threshold=0.35,
            reverse_trigger_distance=0.02,
            reverse_gain=1.0,
            max_reverse_lin_vel_x=0.12,
            turn_in_place_threshold=1.20,
            terminal_slow_distance=0.30,
            terminal_latch_distance=0.16,
            terminal_max_lin_vel_x=0.10,
            terminal_max_lin_vel_y=0.06,
            terminal_max_ang_vel_z=0.25,
            terminal_settle_lin_vel_x=0.05,
            terminal_settle_reverse_lin_vel_x=0.04,
            frame_yaw_offset=frame_yaw_offset,
            target_height_offset=0.03,
        )
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        self.observations.critic.point_goal_target_world = ObsTerm(
            func=point_goal_target_pos_env,
            params={"command_name": "base_velocity"},
        )
        self.observations.critic.point_goal_root_world = ObsTerm(
            func=point_goal_root_pos_env,
        )
        self.observations.critic.point_goal_rel_body = ObsTerm(
            func=point_goal_rel_body_xy,
            params={"command_name": "base_velocity"},
        )
        self.observations.critic.point_goal_distance = ObsTerm(
            func=point_goal_distance_obs,
            params={"command_name": "base_velocity"},
        )
        self.observations.critic.point_goal_heading = ObsTerm(
            func=point_goal_heading_error_obs,
            params={"command_name": "base_velocity"},
        )

        self.rewards.track_lin_vel_xy = RewTerm(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=0.6,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        self.rewards.track_ang_vel_z = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.1,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.05)
        self.rewards.action_rate = None
        self.rewards.base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
        self.rewards.base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        self.rewards.joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
        self.rewards.joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
        self.rewards.dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
        self.rewards.energy = RewTerm(func=mdp.energy, weight=-2e-5)
        self.rewards.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
                )
            },
        )
        self.rewards.joint_deviation_waists = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.8,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
        )
        self.rewards.joint_deviation_legs = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.8,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
        )
        self.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
        self.rewards.base_height = RewTerm(func=mdp.base_height_l2, weight=-18.0, params={"target_height": 0.78})
        self.rewards.gait = RewTerm(
            func=mdp.feet_gait,
            weight=0.2,
            params={
                "period": 0.8,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            },
        )
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            },
        )
        self.rewards.feet_clearance = RewTerm(
            func=mdp.foot_clearance_reward,
            weight=0.2,
            params={
                "std": 0.05,
                "tanh_mult": 2.0,
                "target_height": 0.1,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            },
        )
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-3.0,
            params={
                "threshold": 1,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
            },
        )
        self.rewards.goal_progress = RewTerm(
            func=point_goal_progress_reward,
            weight=12.0,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "per_target_timeout_s": PER_TARGET_TIMEOUT_S,
                "clip_value": 0.08,
                "positive_scale": 2.5,
                "regress_scale": 1.5,
            },
        )
        self.rewards.goal_completion = None
        self.rewards.goal_distance = None
        self.rewards.goal_stop = RewTerm(
            func=point_goal_stop_reward,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "per_target_timeout_s": PER_TARGET_TIMEOUT_S,
                "near_distance": 0.20,
            },
        )
        self.rewards.goal_heading_align = None
        self.rewards.goal_time_penalty = RewTerm(
            func=point_goal_time_penalty,
            weight=-0.05,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "per_target_timeout_s": PER_TARGET_TIMEOUT_S,
            },
        )
        self.rewards.goal_timeout_penalty = RewTerm(
            func=point_goal_timeout_penalty,
            weight=-20.0,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "per_target_timeout_s": PER_TARGET_TIMEOUT_S,
            },
        )
        self.rewards.goal_success = RewTerm(
            func=point_goal_success_bonus,
            weight=60.0,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "per_target_timeout_s": PER_TARGET_TIMEOUT_S,
            },
        )

        self.events.push_robot = None
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.terrain_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
        self.terminations.point_goal_success = DoneTerm(
            func=point_goal_success,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "per_target_timeout_s": PER_TARGET_TIMEOUT_S,
            },
        )
        self.terminations.point_goal_timeout = DoneTerm(
            func=point_goal_target_timeout,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "per_target_timeout_s": PER_TARGET_TIMEOUT_S,
            },
        )


@configclass
class RobotPointGoalPlayEnvCfg(RobotPointGoalEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 10
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.curriculum.terrain_levels = None

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.6, -3.4, 1.9)
        self.viewer.lookat = (0.0, 0.0, 0.65)
