import os

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from ..velocity_env_cfg import RobotEnvCfg
from .point_goal_mdp import (
    PointGoalCommandCfg,
    point_goal_distance_obs,
    point_goal_heading_error_obs,
    point_goal_policy_command_obs,
    point_goal_progress_reward,
    point_goal_rel_body_xy,
    point_goal_root_pos_env,
    point_goal_success,
    point_goal_success_bonus,
    point_goal_target_timeout,
    point_goal_target_pos_env,
    point_goal_time_penalty,
    point_goal_timeout_penalty,
    track_policy_command_ang_vel_z_exp,
    track_policy_command_lin_vel_xy_exp,
)


SUCCESS_DISTANCE = 0.25
SUCCESS_HOLD_STEPS = 5
STOP_VELOCITY_THRESHOLD = 0.15
STOP_YAW_RATE_THRESHOLD = 0.35
PER_TARGET_TIMEOUT_S = 4.0


@configclass
class RobotPointGoalEnvCfg(RobotEnvCfg):
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
            radius_range=(0.30, 0.70),
            angle_range=(0.0, 0.0),
            forward_gain=0.8,
            lateral_gain=0.0,
            heading_gain=1.2,
            min_lin_vel_x=0.0,
            max_lin_vel_x=0.35,
            max_lin_vel_y=0.0,
            max_ang_vel_z=0.25,
            slow_down_distance=1.0,
            stop_distance=0.35,
            heading_slow_down_distance=0.6,
            hold_position_distance=0.22,
            near_recovery_distance=0.45,
            recovery_turn_threshold=0.40,
            heading_block_threshold=1.10,
            min_recovery_ang_vel_z=0.18,
            reverse_recovery_distance=0.30,
            reverse_heading_threshold=0.25,
            reverse_trigger_distance=0.04,
            reverse_gain=0.8,
            max_reverse_lin_vel_x=0.08,
            turn_in_place_threshold=0.60,
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
        self.observations.critic.policy_command = ObsTerm(
            func=point_goal_policy_command_obs,
        )

        # For hierarchical training the high-level policy should optimize task completion,
        # not "easy-to-track" low-level commands or survival bonuses.
        self.rewards.track_lin_vel_xy = None
        self.rewards.track_ang_vel_z = None
        self.rewards.alive = None
        self.rewards.action_rate = None
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.base_linear_velocity = None
        self.rewards.base_angular_velocity = None
        self.rewards.joint_vel = None
        self.rewards.joint_acc = None
        self.rewards.dof_pos_limits = None
        self.rewards.energy = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_waists = None
        self.rewards.joint_deviation_legs = None
        self.rewards.feet_slide = None
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.base_height.weight = -4.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.goal_progress = RewTerm(
            func=point_goal_progress_reward,
            weight=60.0,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "per_target_timeout_s": PER_TARGET_TIMEOUT_S,
                "clip_value": 0.08,
                "positive_scale": 2.0,
                "regress_scale": 2.0,
            },
        )
        self.rewards.goal_completion = None
        self.rewards.goal_distance = None
        self.rewards.goal_stop = None
        self.rewards.goal_time_penalty = RewTerm(
            func=point_goal_time_penalty,
            weight=-0.20,
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
            weight=-60.0,
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
            weight=120.0,
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
