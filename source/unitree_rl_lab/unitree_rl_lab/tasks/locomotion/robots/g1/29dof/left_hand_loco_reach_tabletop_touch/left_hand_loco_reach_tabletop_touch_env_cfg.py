from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from ..benchmark_v1.benchmark_env_cfg import _kinematic_cuboid
from ..left_hand_loco_reach_adapter_hold_stay.left_hand_loco_reach_adapter_hold_stay_env_cfg import (
    LEFT_HAND_COMMAND_NAME,
    STATIC_TARGET_HOLD_S,
    RobotLeftHandLocoReachAdapterHoldStayEnvCfg,
)
from . import left_hand_loco_reach_tabletop_touch_mdp as tabletop_touch_mdp
from ..velocity_env_cfg import RobotSceneCfg


TABLE_TOP_BLOCK_Z = 0.805
TABLETOP_BLOCK_NAMES = tuple(f"target_block_{index}" for index in range(6))
TABLETOP_MAX_TARGETS_PER_EPISODE = 1
TABLETOP_PER_TARGET_TIMEOUT_S = 6.0
TABLETOP_POST_SUCCESS_DWELL_STEPS = 10
TABLETOP_STANCE_X_RANGE = (0.40, 0.56)
TABLETOP_STANCE_Y_RANGE = (0.10, 0.26)
TABLETOP_SUPPORT_CONTACT_BODY_REGEX = (
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$).+$"
)
TABLETOP_HARD_SUPPORT_CONTACT_BODY_NAMES = ("torso_link", "waist.*")
TABLETOP_STAND_TOUCH_MAX_TARGETS_PER_EPISODE = 1
TABLETOP_STAND_TOUCH_PER_TARGET_TIMEOUT_S = 6.0
TABLETOP_STAND_TOUCH_STANCE_X_RANGE = (0.40, 0.54)
TABLETOP_STAND_TOUCH_STANCE_Y_RANGE = TABLETOP_STANCE_Y_RANGE
TABLETOP_NEAR_POS_X = (0.34, 0.46)
TABLETOP_POSTURE_POS_X = (0.40, 0.54)
TABLETOP_FAR_POS_X = (0.46, 0.62)
TABLETOP_NEAR_POS_Y = (0.08, 0.18)
TABLETOP_POSTURE_POS_Y = (0.08, 0.22)
TABLETOP_FAR_POS_Y = (0.06, 0.24)
TABLETOP_NEAR_POS_Z = (-0.02, 0.06)
TABLETOP_POSTURE_POS_Z = (-0.01, 0.08)
TABLETOP_FAR_POS_Z = (0.00, 0.10)
TABLETOP_SAMPLE_REGIMES = {
    "near": {
        "pos_x": TABLETOP_NEAR_POS_X,
        "pos_y": TABLETOP_NEAR_POS_Y,
        "pos_z": TABLETOP_NEAR_POS_Z,
    },
    "posture": {
        "pos_x": TABLETOP_POSTURE_POS_X,
        "pos_y": TABLETOP_POSTURE_POS_Y,
        "pos_z": TABLETOP_POSTURE_POS_Z,
    },
    "far": {
        "pos_x": TABLETOP_FAR_POS_X,
        "pos_y": TABLETOP_FAR_POS_Y,
        "pos_z": TABLETOP_FAR_POS_Z,
    },
}
TABLETOP_SAMPLE_WEIGHTS = {
    "near": 0.55,
    "posture": 0.30,
    "far": 0.15,
}
TABLETOP_BLOCK_LAYOUT = (
    ((0.58, 0.22, TABLE_TOP_BLOCK_Z), (0.90, 0.30, 0.24)),
    ((0.61, 0.10, TABLE_TOP_BLOCK_Z), (0.94, 0.56, 0.20)),
    ((0.65, 0.26, TABLE_TOP_BLOCK_Z), (0.95, 0.78, 0.20)),
    ((0.69, 0.15, TABLE_TOP_BLOCK_Z), (0.26, 0.74, 0.40)),
    ((0.72, 0.23, TABLE_TOP_BLOCK_Z), (0.22, 0.58, 0.90)),
    ((0.74, 0.09, TABLE_TOP_BLOCK_Z), (0.64, 0.38, 0.90)),
)


def _tabletop_block(
    prim_path: str,
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
):
    return _kinematic_cuboid(
        prim_path=prim_path,
        size=(0.055, 0.055, 0.055),
        pos=pos,
        color=color,
    )


def _retarget_term_params(term_cfg) -> None:
    params = getattr(term_cfg, "params", None)
    if not isinstance(params, dict):
        return
    if "command_name" in params and params["command_name"] == LEFT_HAND_COMMAND_NAME:
        if "x_range" in params:
            params["x_range"] = TABLETOP_STANCE_X_RANGE
        if "y_range" in params:
            params["y_range"] = TABLETOP_STANCE_Y_RANGE
    if "sample_regimes" in params:
        params["sample_regimes"] = TABLETOP_SAMPLE_REGIMES
    if "sample_weights" in params:
        params["sample_weights"] = TABLETOP_SAMPLE_WEIGHTS
    if "max_targets_per_episode" in params:
        params["max_targets_per_episode"] = TABLETOP_MAX_TARGETS_PER_EPISODE
    if "per_target_timeout_s" in params:
        params["per_target_timeout_s"] = TABLETOP_PER_TARGET_TIMEOUT_S
    if "post_success_dwell_steps" in params:
        params["post_success_dwell_steps"] = TABLETOP_POST_SUCCESS_DWELL_STEPS


def _override_stand_touch_term_params(term_cfg) -> None:
    params = getattr(term_cfg, "params", None)
    if not isinstance(params, dict):
        return
    if "command_name" in params and params["command_name"] == LEFT_HAND_COMMAND_NAME:
        if "x_range" in params:
            params["x_range"] = TABLETOP_STAND_TOUCH_STANCE_X_RANGE
        if "y_range" in params:
            params["y_range"] = TABLETOP_STAND_TOUCH_STANCE_Y_RANGE
    if "max_targets_per_episode" in params:
        params["max_targets_per_episode"] = TABLETOP_STAND_TOUCH_MAX_TARGETS_PER_EPISODE
    if "per_target_timeout_s" in params:
        params["per_target_timeout_s"] = TABLETOP_STAND_TOUCH_PER_TARGET_TIMEOUT_S


@configclass
class RobotLeftHandTableTopTouchSceneCfg(RobotSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.78, 0.78, 0.80),
            roughness=0.95,
            metallic=0.0,
        ),
        debug_vis=False,
    )

    table_top = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Table_top",
        size=(0.90, 0.60, 0.04),
        pos=(0.95, 0.0, 0.76),
        color=(0.56, 0.43, 0.30),
    )
    table_leg_front_left = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Table_leg_front_left",
        size=(0.05, 0.05, 0.74),
        pos=(0.58, 0.24, 0.37),
        color=(0.24, 0.24, 0.26),
    )
    table_leg_front_right = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Table_leg_front_right",
        size=(0.05, 0.05, 0.74),
        pos=(0.58, -0.24, 0.37),
        color=(0.24, 0.24, 0.26),
    )
    table_leg_back_left = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Table_leg_back_left",
        size=(0.05, 0.05, 0.74),
        pos=(1.32, 0.24, 0.37),
        color=(0.24, 0.24, 0.26),
    )
    table_leg_back_right = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Table_leg_back_right",
        size=(0.05, 0.05, 0.74),
        pos=(1.32, -0.24, 0.37),
        color=(0.24, 0.24, 0.26),
    )

    target_block_0 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_0", *TABLETOP_BLOCK_LAYOUT[0])
    target_block_1 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_1", *TABLETOP_BLOCK_LAYOUT[1])
    target_block_2 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_2", *TABLETOP_BLOCK_LAYOUT[2])
    target_block_3 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_3", *TABLETOP_BLOCK_LAYOUT[3])
    target_block_4 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_4", *TABLETOP_BLOCK_LAYOUT[4])
    target_block_5 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_5", *TABLETOP_BLOCK_LAYOUT[5])


@configclass
class RobotLeftHandLocoReachTableTopTouchEnvCfg(
    RobotLeftHandLocoReachAdapterHoldStayEnvCfg
):
    """Table-aligned hold-stay reach task for natural tabletop touch pretraining."""

    scene: RobotLeftHandTableTopTouchSceneCfg = RobotLeftHandTableTopTouchSceneCfg(num_envs=2048, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()

        ee_cfg = SceneEntityCfg("robot", body_names=[tabletop_touch_mdp.LEFT_HAND_BODY_NAME])
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
        task_params = {
            "success_threshold": 0.05,
            "success_hold_steps": 10,
            "per_target_timeout_s": TABLETOP_PER_TARGET_TIMEOUT_S,
            "pretouch_height": 0.08,
            "pretouch_backoff_x": 0.04,
            "touch_height_offset": 0.02,
            "base_speed_threshold": 0.06,
            "hand_speed_threshold": 0.10,
            "pre_target_switch_radius": 0.08,
            "x_range": TABLETOP_STANCE_X_RANGE,
            "y_range": TABLETOP_STANCE_Y_RANGE,
        }

        self.scene.robot.init_state.pos = (0.18, 0.0, 0.8)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.env_spacing = 4.0
        self.episode_length_s = 24.0
        self.left_hand_scene_target_names = TABLETOP_BLOCK_NAMES
        self.left_hand_scene_target_randomize_order = True

        self.commands.left_hand_pose.resampling_time_range = (STATIC_TARGET_HOLD_S, STATIC_TARGET_HOLD_S)
        self.commands.left_hand_pose.ranges.pos_x = (
            min(TABLETOP_NEAR_POS_X[0], TABLETOP_POSTURE_POS_X[0], TABLETOP_FAR_POS_X[0]),
            max(TABLETOP_NEAR_POS_X[1], TABLETOP_POSTURE_POS_X[1], TABLETOP_FAR_POS_X[1]),
        )
        self.commands.left_hand_pose.ranges.pos_y = (
            min(TABLETOP_NEAR_POS_Y[0], TABLETOP_POSTURE_POS_Y[0], TABLETOP_FAR_POS_Y[0]),
            max(TABLETOP_NEAR_POS_Y[1], TABLETOP_POSTURE_POS_Y[1], TABLETOP_FAR_POS_Y[1]),
        )
        self.commands.left_hand_pose.ranges.pos_z = (
            min(TABLETOP_NEAR_POS_Z[0], TABLETOP_POSTURE_POS_Z[0], TABLETOP_FAR_POS_Z[0]),
            max(TABLETOP_NEAR_POS_Z[1], TABLETOP_POSTURE_POS_Z[1], TABLETOP_FAR_POS_Z[1]),
        )

        self.curriculum.left_hand_target_levels = None

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params["pose_range"] = {"x": (-0.015, 0.015), "y": (-0.02, 0.02), "yaw": (-0.08, 0.08)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        self.observations.policy.velocity_commands = ObsTerm(
            func=tabletop_touch_mdp.target_pos_command_obs,
            params=task_params,
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=tabletop_touch_mdp.target_pos_command_obs,
            params=task_params,
        )

        for stale_reward_name in (
            "post_success_stay",
            "dwell_left_hand_stillness",
            "ready_reach_right_arm_neutral",
            "ready_reach_stationary",
            "ready_reach_left_hand_stillness",
            "ready_reach_left_hand_vertical_motion",
            "ready_reach_left_arm_joint_acc",
            "ready_reach_foot_shuffle",
        ):
            if hasattr(self.rewards, stale_reward_name):
                setattr(self.rewards, stale_reward_name, None)

        self.terminations.reach_success = None
        self.rewards.base_target_stance = RewTerm(
            func=tabletop_touch_mdp.target_relative_base_stance_l2,
            weight=-1.5,
            params=task_params,
        )
        self.rewards.stance_ready = RewTerm(
            func=tabletop_touch_mdp.target_relative_base_stance_ready,
            weight=1.5,
            params={**task_params, "gate_std": 0.01},
        )
        self.rewards.stance_progress = RewTerm(
            func=tabletop_touch_mdp.target_relative_base_stance_progress,
            weight=2.0,
            params=task_params,
        )
        self.rewards.right_arm_balance_posture = RewTerm(
            func=tabletop_touch_mdp.pre_stance_joint_deviation_penalty,
            weight=-0.01,
            params={**task_params, "asset_cfg": right_arm_cfg},
        )
        self.rewards.pre_stance_torso_lean = RewTerm(
            func=tabletop_touch_mdp.pre_stance_torso_lean_penalty,
            weight=-1.5,
            params=task_params,
        )
        self.rewards.pre_stance_waist_twist = RewTerm(
            func=tabletop_touch_mdp.pre_stance_joint_deviation_penalty,
            weight=-0.8,
            params={**task_params, "asset_cfg": waist_yaw_cfg},
        )
        self.rewards.pre_stance_arm_extension = RewTerm(
            func=tabletop_touch_mdp.pre_stance_joint_limit_penalty,
            weight=-1.0,
            params={**task_params, "asset_cfg": left_arm_cfg, "margin_threshold": 0.18},
        )
        self.rewards.pre_stance_foot_motion = RewTerm(
            func=tabletop_touch_mdp.pre_stance_foot_motion_reward,
            weight=0.05,
            params={**task_params, "asset_cfg": feet_cfg},
        )
        self.rewards.hand_phase_progress = RewTerm(
            func=tabletop_touch_mdp.hand_phase_progress_reward,
            weight=10.0,
            params={**task_params, "progress_scale": 0.04},
        )
        self.rewards.pretouch_ready_bonus = RewTerm(
            func=tabletop_touch_mdp.pretouch_ready_bonus,
            weight=2.5,
            params=task_params,
        )
        self.rewards.target_completion = RewTerm(
            func=tabletop_touch_mdp.target_completion_bonus,
            weight=12.0,
            params=task_params,
        )
        self.rewards.target_hold = RewTerm(
            func=tabletop_touch_mdp.target_hold_reward,
            weight=6.0,
            params={**task_params, "asset_cfg": ee_cfg, "hold_reward_std": 0.02},
        )
        self.rewards.near_target_left_hand_stillness = RewTerm(
            func=tabletop_touch_mdp.near_target_left_hand_stillness_reward,
            weight=1.0,
            params={**task_params, "asset_cfg": ee_cfg, "near_target_radius": 0.10, "hand_speed_scale": 0.08},
        )
        self.rewards.left_hand_position_tracking = RewTerm(
            func=tabletop_touch_mdp.static_target_position_error,
            weight=-0.08,
            params={**task_params, "asset_cfg": ee_cfg},
        )
        self.rewards.left_hand_position_tracking_fine = RewTerm(
            func=tabletop_touch_mdp.gated_position_command_error_tanh,
            weight=12.0,
            params={**task_params, "asset_cfg": ee_cfg, "std": 0.10, "gate_std": 0.01},
        )
        self.rewards.success_posture_bonus = RewTerm(
            func=tabletop_touch_mdp.success_posture_bonus,
            weight=4.0,
            params={**task_params, "asset_cfg": ee_cfg, "arm_joint_cfg": left_arm_cfg},
        )

        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[TABLETOP_SUPPORT_CONTACT_BODY_REGEX],
        )
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts.weight = -1.0

        self.terminations.base_height.params["minimum_height"] = 0.16
        self.terminations.bad_orientation.params["limit_angle"] = 0.8
        self.terminations.target_quota = DoneTerm(
            func=tabletop_touch_mdp.target_success_reached,
            params=task_params,
        )
        self.terminations.target_timeout = DoneTerm(
            func=tabletop_touch_mdp.target_timeout_reached,
            params=task_params,
        )
        self.terminations.body_support_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=list(TABLETOP_HARD_SUPPORT_CONTACT_BODY_NAMES),
                ),
                "threshold": 5.0,
            },
        )

        self.viewer.origin_type = "world"
        self.viewer.eye = (2.8, -2.8, 1.9)
        self.viewer.lookat = (0.84, 0.08, 0.88)


@configclass
class RobotLeftHandLocoReachTableTopTouchPlayEnvCfg(RobotLeftHandLocoReachTableTopTouchEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


@configclass
class RobotLeftHandLocoReachTableTopTouchStandTouchEnvCfg(RobotLeftHandLocoReachTableTopTouchEnvCfg):
    """Stage-2 tabletop finetune: single-target standing touch without body support on the table."""

    def __post_init__(self):
        super().__post_init__()

        for term_cfg in (
            self.observations.policy.velocity_commands,
            self.observations.critic.velocity_commands,
        ):
            _override_stand_touch_term_params(term_cfg)

        for term_cfg in vars(self.rewards).values():
            _override_stand_touch_term_params(term_cfg)
        for term_cfg in vars(self.terminations).values():
            _override_stand_touch_term_params(term_cfg)

        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[TABLETOP_SUPPORT_CONTACT_BODY_REGEX],
        )
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts.weight = -1.0

        self.terminations.body_support_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=list(TABLETOP_HARD_SUPPORT_CONTACT_BODY_NAMES),
                ),
                "threshold": 5.0,
            },
        )


@configclass
class RobotLeftHandLocoReachTableTopTouchStandTouchPlayEnvCfg(RobotLeftHandLocoReachTableTopTouchStandTouchEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
