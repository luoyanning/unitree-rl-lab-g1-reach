from __future__ import annotations

from isaaclab.utils import configclass

from ..benchmark_v1.benchmark_env_cfg import RobotBenchmarkSceneCfg, _kinematic_cuboid
from ..left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach.left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach_env_cfg import (
    LEFT_HAND_COMMAND_NAME,
    STATIC_TARGET_HOLD_S,
    RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachEnvCfg,
)


TABLE_TOP_BLOCK_Z = 0.805
TABLETOP_STANCE_X_RANGE = (0.38, 0.56)
TABLETOP_STANCE_Y_RANGE = (0.10, 0.24)
TABLETOP_NEAR_POS_X = (0.34, 0.50)
TABLETOP_POSTURE_POS_X = (0.44, 0.60)
TABLETOP_FAR_POS_X = (0.54, 0.70)
TABLETOP_NEAR_POS_Y = (0.08, 0.24)
TABLETOP_POSTURE_POS_Y = (0.08, 0.24)
TABLETOP_FAR_POS_Y = (0.06, 0.22)
TABLETOP_NEAR_POS_Z = (-0.01, 0.08)
TABLETOP_POSTURE_POS_Z = (0.00, 0.10)
TABLETOP_FAR_POS_Z = (0.02, 0.12)
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
    "near": 0.45,
    "posture": 0.35,
    "far": 0.20,
}


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


@configclass
class RobotLeftHandTableTopTouchSceneCfg(RobotBenchmarkSceneCfg):
    target_block_0 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_0", (0.74, 0.10, TABLE_TOP_BLOCK_Z), (0.90, 0.30, 0.24))
    target_block_1 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_1", (0.80, 0.20, TABLE_TOP_BLOCK_Z), (0.94, 0.56, 0.20))
    target_block_2 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_2", (0.88, 0.14, TABLE_TOP_BLOCK_Z), (0.95, 0.78, 0.20))
    target_block_3 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_3", (0.94, 0.22, TABLE_TOP_BLOCK_Z), (0.26, 0.74, 0.40))
    target_block_4 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_4", (1.00, 0.08, TABLE_TOP_BLOCK_Z), (0.22, 0.58, 0.90))
    target_block_5 = _tabletop_block("{ENV_REGEX_NS}/TargetBlock_5", (1.06, 0.18, TABLE_TOP_BLOCK_Z), (0.64, 0.38, 0.90))


@configclass
class RobotLeftHandLocoReachTableTopTouchEnvCfg(
    RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachEnvCfg
):
    """Table-aligned freeze-base reach task for pre-benchmark tabletop touch training."""

    scene: RobotLeftHandTableTopTouchSceneCfg = RobotLeftHandTableTopTouchSceneCfg(num_envs=2048, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.init_state.pos = (0.36, 0.0, 0.8)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.env_spacing = 4.0
        self.episode_length_s = 24.0

        self.commands.left_hand_pose.resampling_time_range = (STATIC_TARGET_HOLD_S, STATIC_TARGET_HOLD_S)
        self.commands.left_hand_pose.ranges.pos_x = (TABLETOP_NEAR_POS_X[0], TABLETOP_FAR_POS_X[1])
        self.commands.left_hand_pose.ranges.pos_y = (TABLETOP_FAR_POS_Y[0], TABLETOP_NEAR_POS_Y[1])
        self.commands.left_hand_pose.ranges.pos_z = (TABLETOP_NEAR_POS_Z[0], TABLETOP_FAR_POS_Z[1])

        self.curriculum.left_hand_target_levels = None

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params["pose_range"] = {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.15, 0.15)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        for term_cfg in (
            self.observations.policy.velocity_commands,
            self.observations.critic.velocity_commands,
        ):
            _retarget_term_params(term_cfg)

        for term_cfg in vars(self.rewards).values():
            _retarget_term_params(term_cfg)
        for term_cfg in vars(self.terminations).values():
            _retarget_term_params(term_cfg)

        self.rewards.base_height.params["target_height"] = 0.78

        self.viewer.origin_type = "world"
        self.viewer.eye = (2.8, -2.8, 1.9)
        self.viewer.lookat = (0.92, 0.06, 0.90)


@configclass
class RobotLeftHandLocoReachTableTopTouchPlayEnvCfg(RobotLeftHandLocoReachTableTopTouchEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
