from isaaclab.utils import configclass

from ..left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach.left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach_env_cfg import (
    LOCO_REACH_FAR_POS_X,
    LOCO_REACH_FAR_POS_Y,
    LOCO_REACH_FAR_POS_Z,
    LOCO_REACH_NEAR_POS_X,
    LOCO_REACH_POSTURE_POS_Z,
    RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachEnvCfg,
)


def _update_term_params(term_cfg, **updates):
    if term_cfg is None or getattr(term_cfg, "params", None) is None:
        return
    term_cfg.params.update(updates)


def _apply_settle_finetune(cfg):
    full_pos_x = (LOCO_REACH_NEAR_POS_X[0], LOCO_REACH_FAR_POS_X[1])
    full_pos_y = (LOCO_REACH_FAR_POS_Y[0], LOCO_REACH_FAR_POS_Y[1])
    full_pos_z = (LOCO_REACH_POSTURE_POS_Z[0], LOCO_REACH_FAR_POS_Z[1])

    cfg.episode_length_s = 36.0
    cfg.commands.base_velocity.resampling_time_range = (cfg.episode_length_s, cfg.episode_length_s)
    cfg.commands.left_hand_pose.ranges.pos_x = full_pos_x
    cfg.commands.left_hand_pose.ranges.pos_y = full_pos_y
    cfg.commands.left_hand_pose.ranges.pos_z = full_pos_z
    cfg.curriculum.left_hand_target_levels = None

    long_horizon_updates = {
        "success_exit_radius": 0.10,
        "success_hold_steps": 2,
        "post_success_dwell_steps": 25,
        "post_success_exit_radius": 0.12,
        "per_target_timeout_s": 5.0,
    }
    term_names = (
        ("observations", "policy", "velocity_commands"),
        ("observations", "critic", "velocity_commands"),
        ("rewards", "action_rate"),
        ("rewards", "base_target_stance"),
        ("rewards", "stance_ready"),
        ("rewards", "stance_progress"),
        ("rewards", "right_arm_balance_posture"),
        ("rewards", "ready_reach_right_arm_neutral"),
        ("rewards", "ready_reach_stationary"),
        ("rewards", "ready_reach_left_hand_stillness"),
        ("rewards", "ready_reach_left_hand_vertical_motion"),
        ("rewards", "ready_reach_left_arm_joint_acc"),
        ("rewards", "ready_reach_foot_shuffle"),
        ("rewards", "pre_stance_torso_lean"),
        ("rewards", "pre_stance_waist_twist"),
        ("rewards", "pre_stance_arm_extension"),
        ("rewards", "pre_stance_foot_motion"),
        ("rewards", "target_completion"),
        ("rewards", "target_hold"),
        ("rewards", "post_success_stay"),
        ("rewards", "near_target_left_hand_stillness"),
        ("rewards", "dwell_left_hand_stillness"),
        ("rewards", "left_hand_position_tracking"),
        ("rewards", "left_hand_position_tracking_fine"),
        ("rewards", "success_posture_bonus"),
        ("terminations", "target_quota"),
        ("terminations", "target_timeout"),
    )
    for group_name, term_name, *rest in term_names:
        group = getattr(cfg, group_name)
        term_cfg = getattr(group, term_name) if not rest else getattr(getattr(group, term_name), rest[0])
        _update_term_params(term_cfg, **long_horizon_updates)

    cfg.rewards.action_rate.weight = -0.03

    cfg.rewards.right_arm_balance_posture.weight = -0.03

    cfg.rewards.ready_reach_stationary.weight = 7.0
    cfg.rewards.ready_reach_stationary.params.update(
        {
            "base_lin_speed_scale": 0.05,
            "base_ang_speed_scale": 0.15,
            "foot_speed_scale": 0.035,
        }
    )

    cfg.rewards.ready_reach_left_hand_stillness.weight = 4.5
    cfg.rewards.ready_reach_left_hand_stillness.params.update({"hand_speed_scale": 0.08})

    cfg.rewards.target_hold.weight = 7.0
    cfg.rewards.target_hold.params.update({"hold_reward_std": 0.018})

    cfg.rewards.post_success_stay.weight = 10.0
    cfg.rewards.post_success_stay.params.update(
        {
            "stay_reward_std": 0.015,
            "hand_speed_scale": 0.07,
            "base_speed_scale": 0.10,
        }
    )

    cfg.rewards.near_target_left_hand_stillness.weight = 6.0
    cfg.rewards.near_target_left_hand_stillness.params.update(
        {
            "near_target_radius": 0.18,
            "hand_speed_scale": 0.06,
        }
    )

    cfg.rewards.dwell_left_hand_stillness.weight = 8.0
    cfg.rewards.dwell_left_hand_stillness.params.update({"hand_speed_scale": 0.03})

    cfg.rewards.success_posture_bonus.weight = 4.0


@configclass
class RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachSettleFinetuneEnvCfg(
    RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        _apply_settle_finetune(self)


@configclass
class RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachSettleFinetunePlayEnvCfg(
    RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachSettleFinetuneEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 10
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.6, -3.4, 1.9)
        self.viewer.lookat = (0.0, 0.0, 0.65)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
