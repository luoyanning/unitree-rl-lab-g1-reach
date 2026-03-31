from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from ..benchmark_v1.benchmark_env_cfg import _kinematic_cuboid
from ..left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach.left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach_env_cfg import (
    LEFT_HAND_COMMAND_NAME,
    STATIC_TARGET_HOLD_S,
    RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachEnvCfg,
)
from ..left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach import (
    left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach_mdp as freeze_base_reach_mdp,
)
from ..velocity_env_cfg import RobotSceneCfg


TABLE_TOP_BLOCK_Z = 0.805
TABLETOP_BLOCK_NAMES = tuple(f"target_block_{index}" for index in range(6))
TABLETOP_MAX_TARGETS_PER_EPISODE = 3
TABLETOP_PER_TARGET_TIMEOUT_S = 5.0
TABLETOP_POST_SUCCESS_DWELL_STEPS = 6
TABLETOP_STANCE_X_RANGE = (0.40, 0.54)
TABLETOP_STANCE_Y_RANGE = (0.10, 0.24)
TABLETOP_SUPPORT_CONTACT_BODY_REGEX = (
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$).+$"
)
TABLETOP_HARD_SUPPORT_CONTACT_BODY_NAMES = ("torso_link", "waist.*")
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
    ((0.70, 0.10, TABLE_TOP_BLOCK_Z), (0.90, 0.30, 0.24)),
    ((0.76, 0.18, TABLE_TOP_BLOCK_Z), (0.94, 0.56, 0.20)),
    ((0.82, 0.12, TABLE_TOP_BLOCK_Z), (0.95, 0.78, 0.20)),
    ((0.86, 0.22, TABLE_TOP_BLOCK_Z), (0.26, 0.74, 0.40)),
    ((0.90, 0.08, TABLE_TOP_BLOCK_Z), (0.22, 0.58, 0.90)),
    ((0.94, 0.16, TABLE_TOP_BLOCK_Z), (0.64, 0.38, 0.90)),
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


def _tabletop_support_contact_force(
    env,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return torch.amax(net_forces, dim=1)


def _tabletop_stand_then_touch_gate(
    env,
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    support_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=[TABLETOP_SUPPORT_CONTACT_BODY_REGEX],
    ),
    support_force_threshold: float = 1.0,
    command_name: str = LEFT_HAND_COMMAND_NAME,
    x_range: tuple[float, float] = TABLETOP_STANCE_X_RANGE,
    y_range: tuple[float, float] = TABLETOP_STANCE_Y_RANGE,
    switch_phase_steps: int = 30,
    gate_std: float = 0.01,
    base_lin_speed_scale: float = 0.10,
    base_ang_speed_scale: float = 0.28,
    foot_speed_scale: float = 0.08,
    **_,
) -> torch.Tensor:
    robot = env.scene["robot"]
    base_lin_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    base_ang_speed = torch.linalg.norm(robot.data.root_ang_vel_w[:, :3], dim=-1)
    foot_vel_xy = robot.data.body_lin_vel_w[:, feet_cfg.body_ids, :2] - robot.data.root_lin_vel_w[:, None, :2]
    foot_speed = torch.linalg.norm(foot_vel_xy, dim=-1).mean(dim=1)
    support_clear = (_tabletop_support_contact_force(env, support_sensor_cfg=support_sensor_cfg) < support_force_threshold).float()
    reach_gate = freeze_base_reach_mdp._ready_reach_gate(
        env,
        command_name=command_name,
        x_range=x_range,
        y_range=y_range,
        switch_phase_steps=switch_phase_steps,
        gate_std=gate_std,
    )
    stability = (
        torch.exp(-base_lin_speed / base_lin_speed_scale)
        * torch.exp(-base_ang_speed / base_ang_speed_scale)
        * torch.exp(-foot_speed / foot_speed_scale)
    )
    return reach_gate * stability * support_clear


def tabletop_success_posture_bonus(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    arm_joint_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
        ],
    ),
    command_name: str = LEFT_HAND_COMMAND_NAME,
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = TABLETOP_POST_SUCCESS_DWELL_STEPS,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = TABLETOP_MAX_TARGETS_PER_EPISODE,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = STATIC_TARGET_HOLD_S,
    per_target_timeout_s: float = TABLETOP_PER_TARGET_TIMEOUT_S,
    x_range: tuple[float, float] = TABLETOP_STANCE_X_RANGE,
    y_range: tuple[float, float] = TABLETOP_STANCE_Y_RANGE,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    gate_std: float = 0.01,
    reach_std: float = 0.08,
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    support_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=[TABLETOP_SUPPORT_CONTACT_BODY_REGEX],
    ),
    support_force_threshold: float = 1.0,
    base_lin_speed_scale: float = 0.10,
    base_ang_speed_scale: float = 0.28,
    foot_speed_scale: float = 0.08,
) -> torch.Tensor:
    posture_bonus = freeze_base_reach_mdp.success_posture_bonus(
        env,
        asset_cfg=asset_cfg,
        arm_joint_cfg=arm_joint_cfg,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
        gate_std=gate_std,
        reach_std=reach_std,
    )
    stand_then_touch_gate = _tabletop_stand_then_touch_gate(
        env,
        feet_cfg=feet_cfg,
        support_sensor_cfg=support_sensor_cfg,
        support_force_threshold=support_force_threshold,
        command_name=command_name,
        x_range=x_range,
        y_range=y_range,
        switch_phase_steps=switch_phase_steps,
        gate_std=gate_std,
        base_lin_speed_scale=base_lin_speed_scale,
        base_ang_speed_scale=base_ang_speed_scale,
        foot_speed_scale=foot_speed_scale,
    )
    return posture_bonus * stand_then_touch_gate


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
    RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachEnvCfg
):
    """Table-aligned freeze-base reach task for pre-benchmark tabletop touch training."""

    scene: RobotLeftHandTableTopTouchSceneCfg = RobotLeftHandTableTopTouchSceneCfg(num_envs=2048, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.init_state.pos = (0.26, 0.0, 0.8)
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
        self.events.reset_base.params["pose_range"] = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.12, 0.12)}
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

        self.rewards.base_target_stance.weight = -1.6
        self.rewards.stance_ready.weight = 2.0
        self.rewards.stance_progress.weight = 2.5
        self.rewards.ready_reach_stationary.weight = 2.0
        self.rewards.ready_reach_left_hand_stillness.weight = 0.8
        self.rewards.ready_reach_left_hand_vertical_motion.weight = -1.0
        self.rewards.ready_reach_foot_shuffle.weight = -0.8
        self.rewards.target_completion.func = freeze_base_reach_mdp.target_completion_bonus
        self.rewards.target_completion.weight = 8.0
        self.rewards.target_hold.func = freeze_base_reach_mdp.target_hold_reward
        self.rewards.target_hold.weight = 7.5
        self.rewards.near_target_left_hand_stillness.weight = 2.0
        self.rewards.dwell_left_hand_stillness.weight = 2.0
        self.rewards.left_hand_position_tracking_fine.weight = 10.0
        self.rewards.success_posture_bonus.func = tabletop_success_posture_bonus
        self.rewards.success_posture_bonus.weight = 6.0
        # Only the supporting feet and the active left hand are allowed to touch geometry.
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[TABLETOP_SUPPORT_CONTACT_BODY_REGEX],
        )
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts.weight = -1.0

        self.rewards.base_height.params["target_height"] = 0.78
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
