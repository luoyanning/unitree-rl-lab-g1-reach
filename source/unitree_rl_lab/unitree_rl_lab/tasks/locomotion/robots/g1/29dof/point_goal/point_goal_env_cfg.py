from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from ..velocity_env_cfg import RobotEnvCfg
from .point_goal_mdp import (
    point_goal_completion_reward,
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


SUCCESS_DISTANCE = 0.25
SUCCESS_HOLD_STEPS = 5
STOP_VELOCITY_THRESHOLD = 0.15
STOP_YAW_RATE_THRESHOLD = 0.35


@configclass
class RobotPointGoalEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.commands.base_velocity = PointGoalCommandCfg(
            asset_name="robot",
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            debug_vis=True,
            radius_range=(0.35, 0.8),
            angle_range=(-0.17453292519943295, 0.17453292519943295),
            forward_gain=0.8,
            lateral_gain=0.0,
            heading_gain=0.2,
            min_lin_vel_x=0.0,
            max_lin_vel_x=0.35,
            max_lin_vel_y=0.0,
            max_ang_vel_z=0.15,
            slow_down_distance=1.0,
            stop_distance=0.35,
            heading_slow_down_distance=0.6,
            turn_in_place_threshold=1.20,
            target_height_offset=0.03,
        )
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

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

        self.rewards.track_lin_vel_xy.weight = 1.00
        self.rewards.track_ang_vel_z.weight = 0.20
        self.rewards.alive.weight = 0.15
        self.rewards.action_rate = None
        self.rewards.gait.weight = 0.40
        self.rewards.feet_clearance.weight = 0.50
        self.rewards.joint_deviation_arms.weight = -0.05
        self.rewards.joint_deviation_waists.weight = -0.80
        self.rewards.joint_deviation_legs.weight = -0.80
        self.rewards.flat_orientation_l2.weight = -10.0
        self.rewards.base_height.weight = -18.0
        self.rewards.undesired_contacts.weight = -3.0
        self.rewards.goal_progress = RewTerm(
            func=point_goal_progress_reward,
            weight=6.0,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "clip_value": 0.06,
                "positive_scale": 2.0,
                "regress_scale": 1.5,
            },
        )
        self.rewards.goal_completion = RewTerm(
            func=point_goal_completion_reward,
            weight=4.0,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "exponent": 0.5,
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
                "std": 0.50,
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
                "near_distance": 0.45,
            },
        )
        self.rewards.goal_success = RewTerm(
            func=point_goal_success_bonus,
            weight=20.0,
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
