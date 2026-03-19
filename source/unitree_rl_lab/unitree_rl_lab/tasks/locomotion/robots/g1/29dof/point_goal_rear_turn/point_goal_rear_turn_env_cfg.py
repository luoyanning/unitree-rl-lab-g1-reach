import math
import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..point_goal.point_goal_env_cfg import (
    GOAL_PROGRESS_SCALE_START,
    GOAL_STOP_NEAR_DISTANCE_START,
    GOAL_STOP_SCALE_START,
    GOAL_SUCCESS_SCALE_START,
    GOAL_TIMEOUT_PENALTY_SCALE_START,
    GOAL_TIME_PENALTY_SCALE_START,
    HOLD_POSITION_DISTANCE_START,
    HEADING_SLOW_DOWN_DISTANCE_START,
    RobotPointGoalEnvCfg,
    SUCCESS_DISTANCE,
    SUCCESS_DISTANCE_START,
    SUCCESS_HOLD_STEPS,
    SUCCESS_HOLD_STEPS_START,
    SLOW_DOWN_DISTANCE_START,
    STOP_DISTANCE_START,
    STOP_VELOCITY_THRESHOLD,
    STOP_VELOCITY_THRESHOLD_START,
    STOP_YAW_RATE_THRESHOLD,
    STOP_YAW_RATE_THRESHOLD_START,
)
from ..velocity_env_cfg import CurriculumCfg as BaseCurriculumCfg
from .point_goal_rear_turn_mdp import (
    RearTurnPointGoalCommandCfg,
    point_goal_completion_reward,
    point_goal_distance_reward,
    point_goal_heading_alignment_reward,
    point_goal_reward_levels,
    point_goal_turn_band_target_levels,
)


REAR_TURN_CURRICULUM_EPISODES = 48
REAR_TURN_PER_TARGET_TIMEOUT_S = 8.0
REAR_TURN_START_RADIUS = (0.7, 0.95)
REAR_TURN_ANGLE_MASTER_RADIUS = (0.8, 1.15)
REAR_TURN_RADIUS_EXPANSION = (1.0, 1.6)
REAR_TURN_FINAL_RADIUS = (1.5, 4.0)
REAR_TURN_START_ABS_ANGLE = (math.radians(90.0), math.radians(98.0))
REAR_TURN_MID_ABS_ANGLE = (math.radians(90.0), math.radians(125.0))
REAR_TURN_LATE_ABS_ANGLE = (math.radians(90.0), math.radians(170.0))
REAR_TURN_FINAL_ABS_ANGLE = (math.radians(90.0), math.pi)


@configclass
class PointGoalRearTurnCurriculumCfg(BaseCurriculumCfg):
    point_goal_target_levels = CurrTerm(
        func=point_goal_turn_band_target_levels,
        params={
            "command_name": "base_velocity",
            "num_curriculum_episodes": REAR_TURN_CURRICULUM_EPISODES,
            "start_radius_range": REAR_TURN_START_RADIUS,
            "angle_master_radius_range": REAR_TURN_ANGLE_MASTER_RADIUS,
            "radius_expansion_range": REAR_TURN_RADIUS_EXPANSION,
            "final_radius_range": REAR_TURN_FINAL_RADIUS,
            "start_abs_angle_range": REAR_TURN_START_ABS_ANGLE,
            "mid_abs_angle_range": REAR_TURN_MID_ABS_ANGLE,
            "late_abs_angle_range": REAR_TURN_LATE_ABS_ANGLE,
            "final_abs_angle_range": REAR_TURN_FINAL_ABS_ANGLE,
        },
    )
    point_goal_reward_levels = CurrTerm(
        func=point_goal_reward_levels,
        params={
            "command_name": "base_velocity",
            "num_curriculum_episodes": REAR_TURN_CURRICULUM_EPISODES,
            "start_success_distance": SUCCESS_DISTANCE_START,
            "final_success_distance": SUCCESS_DISTANCE,
            "start_success_hold_steps": SUCCESS_HOLD_STEPS_START,
            "final_success_hold_steps": SUCCESS_HOLD_STEPS,
            "start_stop_velocity_threshold": STOP_VELOCITY_THRESHOLD_START,
            "final_stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
            "start_stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD_START,
            "final_stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
            "start_goal_stop_near_distance": GOAL_STOP_NEAR_DISTANCE_START,
            "final_goal_stop_near_distance": 0.20,
            "start_goal_progress_scale": GOAL_PROGRESS_SCALE_START,
            "final_goal_progress_scale": 1.0,
            "start_goal_stop_scale": GOAL_STOP_SCALE_START,
            "final_goal_stop_scale": 1.0,
            "start_goal_success_scale": GOAL_SUCCESS_SCALE_START,
            "final_goal_success_scale": 1.0,
            "start_goal_time_penalty_scale": GOAL_TIME_PENALTY_SCALE_START,
            "final_goal_time_penalty_scale": 1.0,
            "start_goal_timeout_penalty_scale": GOAL_TIMEOUT_PENALTY_SCALE_START,
            "final_goal_timeout_penalty_scale": 1.0,
            "start_hold_position_distance": HOLD_POSITION_DISTANCE_START,
            "final_hold_position_distance": 0.08,
            "start_stop_distance": STOP_DISTANCE_START,
            "final_stop_distance": 0.20,
            "start_slow_down_distance": SLOW_DOWN_DISTANCE_START,
            "final_slow_down_distance": 0.55,
            "start_heading_slow_down_distance": HEADING_SLOW_DOWN_DISTANCE_START,
            "final_heading_slow_down_distance": 0.60,
            "promote_success_threshold": 0.76,
            "demote_success_threshold": 0.62,
        },
    )


@configclass
class RobotPointGoalRearTurnEnvCfg(RobotPointGoalEnvCfg):
    curriculum: PointGoalRearTurnCurriculumCfg = PointGoalRearTurnCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        frame_yaw_offset = float(os.getenv("UTRL_POINT_GOAL_FRAME_YAW_OFFSET", "0.0"))
        self.episode_length_s = 20.0
        self.commands.base_velocity = RearTurnPointGoalCommandCfg(
            asset_name="robot",
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            debug_vis=True,
            radius_range=REAR_TURN_START_RADIUS,
            angle_range=(-math.pi, math.pi),
            abs_angle_range=REAR_TURN_START_ABS_ANGLE,
            sample_both_sides=True,
            forward_gain=1.0,
            lateral_gain=0.24,
            heading_gain=1.25,
            min_lin_vel_x=-0.12,
            max_lin_vel_x=0.42,
            max_lin_vel_y=0.12,
            max_ang_vel_z=0.42,
            slow_down_distance=SLOW_DOWN_DISTANCE_START,
            stop_distance=STOP_DISTANCE_START,
            heading_slow_down_distance=max(HEADING_SLOW_DOWN_DISTANCE_START, 0.50),
            hold_position_distance=HOLD_POSITION_DISTANCE_START,
            near_recovery_distance=0.24,
            recovery_turn_threshold=0.80,
            heading_block_threshold=1.75,
            min_recovery_ang_vel_z=0.14,
            turn_in_place_threshold=1.35,
            turn_in_place_min_distance=0.35,
            min_turn_in_place_ang_vel_z=0.20,
            reverse_recovery_distance=0.25,
            reverse_heading_threshold=0.30,
            reverse_trigger_distance=0.02,
            reverse_gain=1.0,
            max_reverse_lin_vel_x=0.10,
            terminal_slow_distance=0.30,
            terminal_latch_distance=0.16,
            terminal_max_lin_vel_x=0.10,
            terminal_max_lin_vel_y=0.05,
            terminal_max_ang_vel_z=0.25,
            terminal_settle_lin_vel_x=0.05,
            terminal_settle_reverse_lin_vel_x=0.04,
            frame_yaw_offset=frame_yaw_offset,
            target_height_offset=0.03,
        )

        self.rewards.track_lin_vel_xy.weight = 0.45
        self.rewards.track_ang_vel_z.weight = 0.08
        self.rewards.goal_progress.weight = 14.0
        self.rewards.goal_progress.params["per_target_timeout_s"] = REAR_TURN_PER_TARGET_TIMEOUT_S
        self.rewards.goal_progress.params["positive_scale"] = 3.0
        self.rewards.goal_progress.params["regress_scale"] = 2.0
        self.rewards.goal_completion = None
        self.rewards.goal_distance = None
        self.rewards.goal_stop.params["per_target_timeout_s"] = REAR_TURN_PER_TARGET_TIMEOUT_S
        self.rewards.goal_time_penalty.params["per_target_timeout_s"] = REAR_TURN_PER_TARGET_TIMEOUT_S
        self.rewards.goal_time_penalty.weight = -0.05
        self.rewards.goal_timeout_penalty.params["per_target_timeout_s"] = REAR_TURN_PER_TARGET_TIMEOUT_S
        self.rewards.goal_success.params["per_target_timeout_s"] = REAR_TURN_PER_TARGET_TIMEOUT_S
        self.terminations.point_goal_success.params["per_target_timeout_s"] = REAR_TURN_PER_TARGET_TIMEOUT_S
        self.terminations.point_goal_timeout.params["per_target_timeout_s"] = REAR_TURN_PER_TARGET_TIMEOUT_S
        self.rewards.goal_heading_align = RewTerm(
            func=point_goal_heading_alignment_reward,
            weight=0.35,
            params={
                "command_name": "base_velocity",
                "success_distance": SUCCESS_DISTANCE,
                "success_hold_steps": SUCCESS_HOLD_STEPS,
                "stop_velocity_threshold": STOP_VELOCITY_THRESHOLD,
                "stop_yaw_rate_threshold": STOP_YAW_RATE_THRESHOLD,
                "per_target_timeout_s": REAR_TURN_PER_TARGET_TIMEOUT_S,
                "near_distance": 0.9,
                "std": 1.1,
            },
        )


@configclass
class RobotPointGoalRearTurnPlayEnvCfg(RobotPointGoalRearTurnEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
