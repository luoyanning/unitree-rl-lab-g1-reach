from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from ..velocity_env_cfg import RobotEnvCfg
from .point_goal_mdp import (
    PointGoalCommandCfg,
    point_goal_distance_obs,
    point_goal_distance_reward,
    point_goal_heading_error_obs,
    point_goal_progress_reward,
    point_goal_rel_body_xy,
    point_goal_root_pos_env,
    point_goal_success,
    point_goal_success_bonus,
    point_goal_stop_reward,
    point_goal_target_pos_env,
)


SUCCESS_DISTANCE = 0.10
SUCCESS_HOLD_STEPS = 20
STOP_VELOCITY_THRESHOLD = 0.10
STOP_YAW_RATE_THRESHOLD = 0.15


@configclass
class RobotPointGoalEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0
        self.commands.base_velocity = PointGoalCommandCfg(
            asset_name="robot",
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            debug_vis=True,
            radius_range=(0.4, 1.5),
            angle_range=(-1.5707963267948966, 1.5707963267948966),
            forward_gain=0.8,
            lateral_gain=0.5,
            heading_gain=0.2,
            min_lin_vel_x=0.0,
            max_lin_vel_x=0.6,
            max_lin_vel_y=0.1,
            max_ang_vel_z=0.2,
            slow_down_distance=0.6,
            stop_distance=0.2,
            heading_slow_down_distance=0.5,
        )

        self.observations.policy.point_goal_target_world = ObsTerm(
            func=point_goal_target_pos_env,
            params={"command_name": "base_velocity"},
        )
        self.observations.policy.point_goal_root_world = ObsTerm(
            func=point_goal_root_pos_env,
        )
        self.observations.policy.point_goal_rel_body = ObsTerm(
            func=point_goal_rel_body_xy,
            params={"command_name": "base_velocity"},
        )
        self.observations.policy.point_goal_distance = ObsTerm(
            func=point_goal_distance_obs,
            params={"command_name": "base_velocity"},
        )
        self.observations.policy.point_goal_heading = ObsTerm(
            func=point_goal_heading_error_obs,
            params={"command_name": "base_velocity"},
        )

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

        self.rewards.track_lin_vel_xy.weight = 1.0
        self.rewards.track_ang_vel_z.weight = 0.5
        self.rewards.goal_progress = RewTerm(
            func=point_goal_progress_reward,
            weight=8.0,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "clip_value": 0.05,
            },
        )
        self.rewards.goal_distance = RewTerm(
            func=point_goal_distance_reward,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "std": 0.35,
            },
        )
        self.rewards.goal_stop = RewTerm(
            func=point_goal_stop_reward,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "near_distance": 0.35,
            },
        )
        self.rewards.goal_success = RewTerm(
            func=point_goal_success_bonus,
            weight=12.0,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
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
            },
        )


@configclass
class RobotPointGoalPlayEnvCfg(RobotPointGoalEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
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
