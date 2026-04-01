from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from ..benchmark_v1.benchmark_env_cfg import _kinematic_cuboid
from ..velocity_env_cfg import RobotEnvCfg
from ..velocity_env_cfg import RobotSceneCfg
from . import left_hand_loco_reach_tabletop_clean_mdp as tabletop_clean_mdp


LEFT_HAND_BODY_NAME = tabletop_clean_mdp.LEFT_HAND_BODY_NAME
LEFT_HAND_COMMAND_NAME = "left_hand_pose"
STATIC_TARGET_HOLD_S = 1.0e9
TABLE_TOP_BLOCK_Z = 0.805
TABLETOP_BLOCK_NAMES = tuple(f"target_block_{index}" for index in range(4))
TABLETOP_BLOCK_LAYOUT = (
    ((0.62, 0.18, TABLE_TOP_BLOCK_Z), (0.90, 0.38, 0.24)),
    ((0.66, 0.08, TABLE_TOP_BLOCK_Z), (0.24, 0.68, 0.90)),
    ((0.72, 0.25, TABLE_TOP_BLOCK_Z), (0.26, 0.64, 0.40)),
    ((0.74, 0.12, TABLE_TOP_BLOCK_Z), (0.84, 0.58, 0.26)),
)
SUPPORT_CONTACT_BODY_REGEX = (
    r"^(pelvis|torso_link|waist.*|left_elbow_link|left_wrist_(roll|pitch)_link)$"
)
SUPPORT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=[SUPPORT_CONTACT_BODY_REGEX])
STANCE_ANCHOR_XY = (0.18, 0.0)
READY_LOCAL_POS = (0.20, 0.18, 0.18)


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


@configclass
class RobotLeftHandTableTopCleanSceneCfg(RobotSceneCfg):
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
            diffuse_color=(0.80, 0.80, 0.82),
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


def _make_task_params(
    *,
    mode: str,
    scene_target_names: tuple[str, ...],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    pretouch_backoff_x: float,
    pretouch_height: float,
    touch_height_offset: float,
    support_force_threshold: float,
):
    return {
        "mode": mode,
        "scene_target_names": scene_target_names,
        "randomize_order": randomize_order,
        "max_targets_per_episode": max_targets_per_episode,
        "per_target_timeout_s": per_target_timeout_s,
        "stance_anchor_xy": STANCE_ANCHOR_XY,
        "stance_anchor_std": 0.05,
        "stance_anchor_tolerance": 0.03,
        "base_speed_threshold": 0.10,
        "torso_lean_threshold": 0.30,
        "stability_speed_scale": 0.10,
        "stability_lean_scale": 0.22,
        "ready_local_pos": READY_LOCAL_POS,
        "balance_radius": 0.07,
        "balance_hold_steps": 10,
        "pretouch_backoff_x": pretouch_backoff_x,
        "pretouch_height": pretouch_height,
        "pretouch_radius": 0.07,
        "pretouch_hold_steps": 6,
        "pretouch_stability_gate": 0.30,
        "touch_height_offset": touch_height_offset,
        "touch_radius": 0.05,
        "touch_hold_steps": 5,
        "touch_stability_gate": 0.50,
        "recover_radius": 0.08,
        "recover_hold_steps": 8,
        "recover_stability_gate": 0.55,
        "hand_speed_threshold": 0.18,
        "support_sensor_cfg": SUPPORT_SENSOR_CFG,
        "support_force_threshold": support_force_threshold,
    }


@configclass
class RobotLeftHandLocoReachTableTopCleanBaseEnvCfg(RobotEnvCfg):
    scene: RobotLeftHandTableTopCleanSceneCfg = RobotLeftHandTableTopCleanSceneCfg(num_envs=2048, env_spacing=4.0)

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
        waist_cfg = SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])

        task_params = _make_task_params(
            mode="balance",
            scene_target_names=("target_block_0",),
            randomize_order=False,
            max_targets_per_episode=1,
            per_target_timeout_s=8.0,
            pretouch_backoff_x=0.04,
            pretouch_height=0.10,
            touch_height_offset=0.03,
            support_force_threshold=1.0,
        )

        self.scene.robot.init_state.pos = (0.18, 0.0, 0.8)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.episode_length_s = 12.0
        self.scene.env_spacing = 4.0

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
                pos_x=(0.15, 0.65),
                pos_y=(0.02, 0.30),
                pos_z=(0.02, 0.30),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(-0.5, 0.5),
            ),
        )

        self.actions.JointPositionAction.joint_names = [".*"]
        self.actions.JointPositionAction.scale = 0.25

        self.observations.policy.velocity_commands = ObsTerm(
            func=tabletop_clean_mdp.target_pos_command_obs,
            params=task_params,
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=tabletop_clean_mdp.target_pos_command_obs,
            params=task_params,
        )

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params["pose_range"] = {"x": (-0.01, 0.01), "y": (-0.015, 0.015), "yaw": (-0.05, 0.05)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None

        self.rewards.track_lin_vel_xy.weight = 0.10
        self.rewards.track_ang_vel_z.weight = 0.05
        self.rewards.alive.weight = 0.05
        self.rewards.base_linear_velocity.weight = -0.5
        self.rewards.base_angular_velocity.weight = -0.05
        self.rewards.joint_vel.weight = -0.0005
        self.rewards.joint_acc.weight = -1.0e-7
        self.rewards.action_rate.weight = -0.015
        self.rewards.dof_pos_limits.weight = -2.0
        self.rewards.energy.weight = -1.0e-5
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_waists = None
        self.rewards.joint_deviation_legs.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.base_height.weight = -3.0
        self.rewards.base_height.params["target_height"] = 0.74
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.feet_slide.weight = -0.02
        self.rewards.undesired_contacts = None

        self.rewards.stance_anchor = RewTerm(
            func=tabletop_clean_mdp.stance_anchor_penalty,
            weight=-1.5,
            params=task_params,
        )
        self.rewards.stance_stability = RewTerm(
            func=tabletop_clean_mdp.stance_stability_reward,
            weight=2.5,
            params=task_params,
        )
        self.rewards.phase_progress = RewTerm(
            func=tabletop_clean_mdp.phase_progress_reward,
            weight=2.0,
            params={**task_params, "progress_scale": 0.04},
        )
        self.rewards.phase_tracking = RewTerm(
            func=tabletop_clean_mdp.phase_target_tracking_reward,
            weight=4.0,
            params={**task_params, "std": 0.08},
        )
        self.rewards.phase_hold = RewTerm(
            func=tabletop_clean_mdp.phase_hold_reward,
            weight=1.5,
            params={**task_params, "hold_reward_std": 0.03, "hand_speed_scale": 0.10},
        )
        self.rewards.lift_intent = RewTerm(
            func=tabletop_clean_mdp.lift_intent_reward,
            weight=0.5,
            params={**task_params, "lift_reference_z": 0.08, "lift_scale": 0.08},
        )
        self.rewards.pretouch_bonus = RewTerm(
            func=tabletop_clean_mdp.pretouch_bonus,
            weight=2.0,
            params=task_params,
        )
        self.rewards.touch_bonus = RewTerm(
            func=tabletop_clean_mdp.touch_bonus,
            weight=4.0,
            params=task_params,
        )
        self.rewards.target_completion = RewTerm(
            func=tabletop_clean_mdp.target_completion_bonus,
            weight=8.0,
            params=task_params,
        )
        self.rewards.support_contact = RewTerm(
            func=tabletop_clean_mdp.support_contact_penalty,
            weight=-2.0,
            params={**task_params, "force_scale": 8.0},
        )
        self.rewards.torso_lean = RewTerm(
            func=tabletop_clean_mdp.torso_lean_penalty,
            weight=-1.0,
            params=task_params,
        )
        self.rewards.waist_twist = RewTerm(
            func=tabletop_clean_mdp.joint_deviation_penalty,
            weight=-0.2,
            params={**task_params, "asset_cfg": waist_cfg},
        )
        self.rewards.right_arm_balance = RewTerm(
            func=tabletop_clean_mdp.joint_deviation_penalty,
            weight=-0.01,
            params={**task_params, "asset_cfg": right_arm_cfg},
        )
        self.rewards.left_arm_limit = RewTerm(
            func=tabletop_clean_mdp.joint_limit_penalty,
            weight=-0.4,
            params={**task_params, "asset_cfg": left_arm_cfg, "margin_threshold": 0.18},
        )
        self.rewards.left_hand_tracking = RewTerm(
            func=tabletop_clean_mdp.phase_target_tracking_reward,
            weight=1.0,
            params={**task_params, "std": 0.05},
        )

        self.terminations.base_height.params["minimum_height"] = 0.18
        self.terminations.bad_orientation.params["limit_angle"] = 0.8
        self.terminations.target_quota = DoneTerm(
            func=tabletop_clean_mdp.task_success_reached,
            params=task_params,
        )
        self.terminations.target_timeout = DoneTerm(
            func=tabletop_clean_mdp.task_timeout_reached,
            params=task_params,
        )

        self.viewer.origin_type = "world"
        self.viewer.eye = (2.8, -2.6, 1.9)
        self.viewer.lookat = (0.84, 0.08, 0.86)


def _set_task_params(env_cfg: RobotLeftHandLocoReachTableTopCleanBaseEnvCfg, **updates):
    for term_cfg in (
        env_cfg.observations.policy.velocity_commands,
        env_cfg.observations.critic.velocity_commands,
        env_cfg.rewards.stance_anchor,
        env_cfg.rewards.stance_stability,
        env_cfg.rewards.phase_progress,
        env_cfg.rewards.phase_tracking,
        env_cfg.rewards.phase_hold,
        env_cfg.rewards.lift_intent,
        env_cfg.rewards.pretouch_bonus,
        env_cfg.rewards.touch_bonus,
        env_cfg.rewards.target_completion,
        env_cfg.rewards.support_contact,
        env_cfg.rewards.torso_lean,
        env_cfg.rewards.waist_twist,
        env_cfg.rewards.right_arm_balance,
        env_cfg.rewards.left_arm_limit,
        env_cfg.rewards.left_hand_tracking,
        env_cfg.terminations.target_quota,
        env_cfg.terminations.target_timeout,
    ):
        term_cfg.params.update(updates)


@configclass
class RobotLeftHandLocoReachTableTopBalanceCleanEnvCfg(RobotLeftHandLocoReachTableTopCleanBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_task_params(
            self,
            mode="balance",
            scene_target_names=("target_block_0",),
            randomize_order=False,
            max_targets_per_episode=1,
            per_target_timeout_s=6.0,
            stance_anchor_std=0.12,
            stance_anchor_tolerance=0.08,
            base_speed_threshold=0.25,
            torso_lean_threshold=0.55,
            stability_speed_scale=0.25,
            stability_lean_scale=0.45,
            balance_hold_steps=15,
        )
        self.episode_length_s = 10.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.rewards.stance_anchor.weight = -2.0
        self.rewards.stance_stability.weight = 5.0
        self.rewards.phase_progress.weight = 0.0
        self.rewards.phase_tracking.weight = 0.0
        self.rewards.phase_hold.weight = 0.0
        self.rewards.lift_intent.weight = 0.0
        self.rewards.pretouch_bonus.weight = 0.0
        self.rewards.touch_bonus.weight = 0.0
        self.rewards.target_completion.weight = 12.0
        self.rewards.support_contact.weight = 0.0
        self.rewards.left_hand_tracking.weight = 0.0


@configclass
class RobotLeftHandLocoReachTableTopPreTouchCleanEnvCfg(RobotLeftHandLocoReachTableTopCleanBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_task_params(
            self,
            mode="pretouch",
            scene_target_names=("target_block_0",),
            randomize_order=False,
            max_targets_per_episode=1,
            per_target_timeout_s=10.0,
            stance_anchor_std=0.22,
            stance_anchor_tolerance=0.12,
            base_speed_threshold=0.35,
            torso_lean_threshold=0.65,
            stability_speed_scale=0.35,
            stability_lean_scale=0.55,
            pretouch_backoff_x=0.04,
            pretouch_height=0.10,
            pretouch_radius=0.12,
            pretouch_hold_steps=2,
            pretouch_stability_gate=0.05,
            touch_height_offset=0.03,
            recover_radius=0.14,
            recover_hold_steps=3,
            recover_stability_gate=0.10,
            hand_speed_threshold=0.45,
            support_force_threshold=4.0,
        )
        self.episode_length_s = 12.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.rewards.stance_anchor.weight = -0.4
        self.rewards.stance_stability.weight = 1.5
        self.rewards.phase_progress.weight = 6.0
        self.rewards.phase_tracking.weight = 5.0
        self.rewards.phase_hold.weight = 2.5
        self.rewards.lift_intent.weight = 2.0
        self.rewards.pretouch_bonus.weight = 10.0
        self.rewards.touch_bonus.weight = 0.0
        self.rewards.target_completion.weight = 16.0
        self.rewards.support_contact.weight = -0.05
        self.rewards.left_hand_tracking.weight = 2.5


@configclass
class RobotLeftHandLocoReachTableTopTouchCleanEnvCfg(RobotLeftHandLocoReachTableTopCleanBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_task_params(
            self,
            mode="touch",
            scene_target_names=("target_block_0",),
            randomize_order=False,
            max_targets_per_episode=1,
            per_target_timeout_s=10.0,
            stance_anchor_std=0.20,
            stance_anchor_tolerance=0.12,
            base_speed_threshold=0.33,
            torso_lean_threshold=0.63,
            stability_speed_scale=0.33,
            stability_lean_scale=0.53,
            pretouch_backoff_x=0.04,
            pretouch_height=0.10,
            pretouch_radius=0.10,
            pretouch_hold_steps=2,
            pretouch_stability_gate=0.05,
            touch_height_offset=0.03,
            touch_radius=0.08,
            touch_hold_steps=2,
            touch_stability_gate=0.08,
            recover_radius=0.14,
            recover_hold_steps=3,
            recover_stability_gate=0.12,
            hand_speed_threshold=0.45,
            support_force_threshold=4.0,
        )
        self.episode_length_s = 14.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.rewards.stance_anchor.weight = -0.4
        self.rewards.stance_stability.weight = 1.5
        self.rewards.phase_progress.weight = 6.0
        self.rewards.phase_tracking.weight = 5.0
        self.rewards.phase_hold.weight = 2.5
        self.rewards.lift_intent.weight = 1.5
        self.rewards.pretouch_bonus.weight = 4.0
        self.rewards.touch_bonus.weight = 10.0
        self.rewards.target_completion.weight = 18.0
        self.rewards.support_contact.weight = -0.1
        self.rewards.left_hand_tracking.weight = 2.5


@configclass
class RobotLeftHandLocoReachTableTopTouchSpreadCleanEnvCfg(RobotLeftHandLocoReachTableTopCleanBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_task_params(
            self,
            mode="touch",
            scene_target_names=("target_block_0", "target_block_1", "target_block_2"),
            randomize_order=True,
            max_targets_per_episode=1,
            per_target_timeout_s=11.0,
            stance_anchor_std=0.20,
            stance_anchor_tolerance=0.12,
            base_speed_threshold=0.33,
            torso_lean_threshold=0.63,
            stability_speed_scale=0.33,
            stability_lean_scale=0.53,
            pretouch_backoff_x=0.04,
            pretouch_height=0.10,
            pretouch_radius=0.10,
            pretouch_hold_steps=2,
            pretouch_stability_gate=0.06,
            touch_height_offset=0.03,
            touch_radius=0.08,
            touch_hold_steps=2,
            touch_stability_gate=0.10,
            recover_radius=0.14,
            recover_hold_steps=3,
            recover_stability_gate=0.12,
            hand_speed_threshold=0.45,
            support_force_threshold=4.0,
        )
        self.episode_length_s = 14.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.rewards.stance_anchor.weight = -0.4
        self.rewards.stance_stability.weight = 1.5
        self.rewards.phase_progress.weight = 5.5
        self.rewards.phase_tracking.weight = 4.5
        self.rewards.phase_hold.weight = 2.5
        self.rewards.lift_intent.weight = 1.2
        self.rewards.pretouch_bonus.weight = 4.0
        self.rewards.touch_bonus.weight = 9.0
        self.rewards.target_completion.weight = 16.0
        self.rewards.support_contact.weight = -0.12
        self.rewards.left_hand_tracking.weight = 2.0


@configclass
class RobotLeftHandLocoReachTableTopMultiTouchCleanEnvCfg(RobotLeftHandLocoReachTableTopCleanBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_task_params(
            self,
            mode="touch",
            scene_target_names=TABLETOP_BLOCK_NAMES,
            randomize_order=True,
            max_targets_per_episode=3,
            per_target_timeout_s=7.0,
            pretouch_backoff_x=0.05,
            pretouch_height=0.11,
            touch_height_offset=0.03,
            support_force_threshold=1.0,
        )
        self.episode_length_s = 18.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.rewards.phase_progress.weight = 5.5
        self.rewards.phase_tracking.weight = 4.0
        self.rewards.phase_hold.weight = 3.0
        self.rewards.lift_intent.weight = 0.8
        self.rewards.pretouch_bonus.weight = 2.5
        self.rewards.touch_bonus.weight = 5.0
        self.rewards.target_completion.weight = 10.0
        self.rewards.support_contact.weight = -4.0
        self.terminations.body_support_contact = DoneTerm(
            func=tabletop_clean_mdp.support_contact_termination,
            params={
                **self.observations.policy.velocity_commands.params,
                "termination_force_threshold": 5.0,
            },
        )


@configclass
class RobotLeftHandLocoReachTableTopCleanPlayBaseEnvCfg(RobotLeftHandLocoReachTableTopCleanBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


@configclass
class RobotLeftHandLocoReachTableTopBalanceCleanPlayEnvCfg(
    RobotLeftHandLocoReachTableTopBalanceCleanEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


@configclass
class RobotLeftHandLocoReachTableTopPreTouchCleanPlayEnvCfg(
    RobotLeftHandLocoReachTableTopPreTouchCleanEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


@configclass
class RobotLeftHandLocoReachTableTopTouchCleanPlayEnvCfg(
    RobotLeftHandLocoReachTableTopTouchCleanEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


@configclass
class RobotLeftHandLocoReachTableTopTouchSpreadCleanPlayEnvCfg(
    RobotLeftHandLocoReachTableTopTouchSpreadCleanEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


@configclass
class RobotLeftHandLocoReachTableTopMultiTouchCleanPlayEnvCfg(
    RobotLeftHandLocoReachTableTopMultiTouchCleanEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
