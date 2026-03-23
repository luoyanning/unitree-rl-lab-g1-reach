from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_HOMIE_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


LOWER_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

UPPER_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ALL_JOINT_NAMES = LOWER_JOINT_NAMES + UPPER_JOINT_NAMES

MIRROR_JOINT_PAIRS = [
    ("left_hip_yaw_joint", "right_hip_yaw_joint"),
    ("left_hip_roll_joint", "right_hip_roll_joint"),
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_knee_joint", "right_knee_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    ("left_ankle_roll_joint", "right_ankle_roll_joint"),
    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
    ("left_shoulder_roll_joint", "right_shoulder_roll_joint"),
    ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint"),
    ("left_elbow_joint", "right_elbow_joint"),
    ("left_wrist_roll_joint", "right_wrist_roll_joint"),
    ("left_wrist_pitch_joint", "right_wrist_pitch_joint"),
    ("left_wrist_yaw_joint", "right_wrist_yaw_joint"),
]

MIRROR_NEGATE_JOINT_NAMES = (
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
LEFT_FOOT_SURFACE_BODY_NAMES = ["left_foot_front_link", "left_foot_mid_link", "left_foot_hind_link"]
RIGHT_FOOT_SURFACE_BODY_NAMES = ["right_foot_front_link", "right_foot_mid_link", "right_foot_hind_link"]
KNEE_BODY_NAMES = ["left_knee_link", "left_hip_yaw_link", "right_knee_link", "right_hip_yaw_link"]
TERMINATION_CONTACT_BODY_NAMES = ["torso_link"]
LEFT_HAND_BODY_NAME = "left_hand_palm_link"
RIGHT_HAND_BODY_NAME = "right_hand_palm_link"
ANKLE_SOLE_DISTANCE = 0.02
LOWER_TORQUE_LIMITS = [88.0, 139.0, 88.0, 139.0, 50.0, 50.0, 88.0, 139.0, 88.0, 139.0, 50.0, 50.0]
LOWER_VELOCITY_LIMITS = [32.0, 20.0, 32.0, 20.0, 37.0, 37.0, 32.0, 20.0, 32.0, 20.0, 37.0, 37.0]

COMMAND_DIM = 4
ANG_VEL_DIM = 3
GRAVITY_DIM = 3
ALL_JOINT_DIM = len(ALL_JOINT_NAMES)
LOWER_ACTION_DIM = len(LOWER_JOINT_NAMES)
UPPER_JOINT_DIM = len(UPPER_JOINT_NAMES)
POLICY_FRAME_DIM = COMMAND_DIM + ANG_VEL_DIM + GRAVITY_DIM + 2 * ALL_JOINT_DIM + LOWER_ACTION_DIM
CRITIC_EXTRA_DIM = 3


@configclass
class EventCfg:
    """Minimal domain randomization for the Direct RL workflow."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 3.0),
            "dynamic_friction_range": (0.1, 3.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 32,
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={"velocity_range": {"x": (-0.50, 0.50), "y": (-0.50, 0.50)}},
    )


@configclass
class G1HomieLowerBodyEnvCfg(DirectRLEnvCfg):
    """HOMIE-inspired lower-body humanoid task built on Isaac Lab Direct RL."""

    decimation = 4
    episode_length_s = 20.0

    action_space = LOWER_ACTION_DIM
    observation_space = 0
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    events: EventCfg = EventCfg()

    lower_joint_names = LOWER_JOINT_NAMES
    upper_joint_names = UPPER_JOINT_NAMES
    all_joint_names = ALL_JOINT_NAMES
    mirror_joint_pairs = MIRROR_JOINT_PAIRS
    mirror_negate_joint_names = MIRROR_NEGATE_JOINT_NAMES
    foot_body_names = FOOT_BODY_NAMES
    left_foot_surface_body_names = LEFT_FOOT_SURFACE_BODY_NAMES
    right_foot_surface_body_names = RIGHT_FOOT_SURFACE_BODY_NAMES
    knee_body_names = KNEE_BODY_NAMES
    termination_contact_body_names = TERMINATION_CONTACT_BODY_NAMES

    history_length = 6
    action_scale = 0.25
    clip_joint_targets_to_soft_limits = False
    upper_soft_limit_factor = 0.9

    torso_body_name = "torso_link"
    imu_body_name = "imu_in_pelvis"
    left_hand_body_name = LEFT_HAND_BODY_NAME
    right_hand_body_name = RIGHT_HAND_BODY_NAME
    termination_body_height_threshold = 0.20
    foot_contact_height_threshold = 0.06

    command_resample_interval_s = 4.0
    command_transition_duration_s = 0.0
    command_vx_range = (-0.8, 1.2)
    command_vy_range = (-0.5, 0.5)
    command_yaw_rate_range = (-0.8, 0.8)
    base_height_target = 0.74
    command_height_offset_range = (-0.5, 0.0)
    height_command_probability = 1.0 / 3.0
    velocity_command_probability = 0.5

    upper_body_resample_interval_s = 1.0
    upper_curriculum_init = 0.0
    upper_curriculum_step = 0.05
    upper_curriculum_demote_step = 0.0
    upper_curriculum_promote_threshold = 0.80
    upper_curriculum_demote_threshold = 0.0
    upper_curriculum_max_progress = 1.0
    upper_curriculum_eval_fixed = False

    reset_xy_noise = 1.0
    reset_yaw_noise = math.pi
    reset_root_z_noise = (0.0, 0.10)
    reset_root_velocity_noise = 0.5
    reset_joint_position_scale = (0.8, 1.2)
    reset_joint_position_offset = (-0.10, 0.10)
    reset_joint_velocity_noise = 0.12

    obs_command_lin_vel_scale = 2.0
    obs_command_ang_vel_scale = 0.5
    obs_base_lin_vel_scale = 2.0
    obs_ang_vel_scale = 0.5
    obs_joint_vel_scale = 0.05
    obs_joint_pos_noise = 0.02
    obs_joint_vel_noise = 0.10
    obs_ang_vel_noise = 0.15
    obs_gravity_noise = 0.05

    tracking_sigma = 0.25
    soft_dof_pos_limit = 0.975
    soft_dof_vel_limit = 0.80
    soft_torque_limit = 0.95
    max_contact_force = 400.0
    least_feet_distance_lateral = 0.2
    most_feet_distance_lateral = 0.35
    most_knee_distance_lateral = 0.35
    least_knee_distance_lateral = 0.2
    clearance_height_target = 0.14

    randomize_joint_injection = True
    joint_injection_range = (-0.05, 0.05)
    randomize_actuation_offset = True
    actuation_offset_range = (-0.05, 0.05)
    randomize_payload_mass = True
    payload_mass_range = (-5.0, 10.0)
    hand_payload_mass_range = (-0.1, 0.3)
    randomize_com_displacement = False
    com_displacement_range = (-0.1, 0.1)
    randomize_body_displacement = True
    body_displacement_range = (-0.1, 0.1)
    randomize_link_mass = True
    link_mass_range = (0.8, 1.2)
    randomize_kp = True
    kp_range = (0.9, 1.1)
    randomize_kd = True
    kd_range = (0.9, 1.1)
    delay = True

    rew_scale_tracking_x_vel = 1.5
    rew_scale_tracking_y_vel = 1.0
    rew_scale_tracking_ang_vel = 2.0
    rew_scale_lin_vel_z = -0.5
    rew_scale_ang_vel_xy = -0.025
    rew_scale_orientation = -1.5
    rew_scale_action_rate = -0.01
    rew_scale_tracking_base_height = 2.0
    rew_scale_deviation_hip_joint = -0.2
    rew_scale_deviation_ankle_joint = -0.5
    rew_scale_deviation_knee_joint = -0.75
    rew_scale_dof_acc = -2.5e-7
    rew_scale_dof_pos_limits = -2.0
    rew_scale_feet_air_time = 0.05
    rew_scale_feet_clearance = -0.25
    rew_scale_feet_distance_lateral = 0.5
    rew_scale_knee_distance_lateral = 1.0
    rew_scale_feet_ground_parallel = -2.0
    rew_scale_feet_parallel = -3.0
    rew_scale_smoothness = -0.05
    rew_scale_joint_power = -2.0e-5
    rew_scale_feet_stumble = -1.5
    rew_scale_torques = -2.5e-6
    rew_scale_dof_vel = -1.0e-4
    rew_scale_dof_vel_limits = -2.0e-3
    rew_scale_torque_limits = -0.1
    rew_scale_no_fly = 0.75
    rew_scale_joint_tracking_error = -0.1
    rew_scale_feet_slip = -0.25
    rew_scale_feet_contact_forces = -2.5e-4
    rew_scale_contact_momentum = 2.5e-4
    rew_scale_action_vanish = -1.0
    rew_scale_stand_still = -0.15

    def __post_init__(self):
        self.observation_space = self.history_length * POLICY_FRAME_DIM
        self.state_space = POLICY_FRAME_DIM + CRITIC_EXTRA_DIM


@configclass
class G1HomieLowerBodyPlayEnvCfg(G1HomieLowerBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.upper_curriculum_init = 1.0
        self.upper_curriculum_max_progress = 1.0
        self.upper_curriculum_eval_fixed = True
        self.events.push_robot = None
        self.events.physics_material = None


class G1HomieLowerBodyEnv(DirectRLEnv):
    """Direct RL humanoid lower-body controller under changing upper-body posture targets."""

    cfg: G1HomieLowerBodyEnvCfg

    def __init__(self, cfg: G1HomieLowerBodyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve all joint groups by name so articulation ordering changes do not silently break the task.
        self._lower_joint_ids = self._find_joint_ids(self.cfg.lower_joint_names)
        self._upper_joint_ids = self._find_joint_ids(self.cfg.upper_joint_names)
        self._all_joint_ids = self._find_joint_ids(self.cfg.all_joint_names)
        self._torso_body_id = int(self.robot.find_bodies([self.cfg.torso_body_name], preserve_order=True)[0][0])
        try:
            self._imu_body_id = int(self.robot.find_bodies([self.cfg.imu_body_name], preserve_order=True)[0][0])
        except Exception:
            self._imu_body_id = self._torso_body_id
        self._foot_body_ids = self._find_body_ids_safe(self.cfg.foot_body_names)
        self._left_foot_surface_ids = self._find_body_ids_safe(self.cfg.left_foot_surface_body_names)
        self._right_foot_surface_ids = self._find_body_ids_safe(self.cfg.right_foot_surface_body_names)
        self._knee_body_ids = self._find_body_ids_safe(self.cfg.knee_body_names)
        if self._left_foot_surface_ids.numel() == 0 and self._foot_body_ids.numel() > 0:
            self._left_foot_surface_ids = self._foot_body_ids[:1].repeat(3)
        if self._right_foot_surface_ids.numel() == 0 and self._foot_body_ids.numel() > 1:
            self._right_foot_surface_ids = self._foot_body_ids[1:2].repeat(3)
        self._termination_contact_body_ids = self._find_body_ids_safe(self.cfg.termination_contact_body_names)
        if self._termination_contact_body_ids.numel() == 0:
            self._termination_contact_body_ids = self._find_body_ids_safe([self.cfg.torso_body_name])
        self._left_hand_body_id = self._find_body_ids_safe([self.cfg.left_hand_body_name])
        self._right_hand_body_id = self._find_body_ids_safe([self.cfg.right_hand_body_name])

        self._hard_joint_limits = self.robot.data.joint_pos_limits[0].clone()
        self._soft_joint_limits = self.robot.data.soft_joint_pos_limits[0].clone()
        self._joint_vel_limits = torch.tensor(LOWER_VELOCITY_LIMITS, device=self.device, dtype=torch.float)
        self._torque_limits = torch.tensor(LOWER_TORQUE_LIMITS, device=self.device, dtype=torch.float)
        self._default_joint_pos = self.robot.data.default_joint_pos[0, self._all_joint_ids].clone()
        self._default_lower_joint_pos = self.robot.data.default_joint_pos[0, self._lower_joint_ids].clone()
        self._default_upper_joint_pos = self.robot.data.default_joint_pos[0, self._upper_joint_ids].clone()
        self._homie_joint_pos_limits = self._compute_homie_soft_limits(self._lower_joint_ids)

        self._lower_joint_min = self._hard_joint_limits[self._lower_joint_ids, 0]
        self._lower_joint_max = self._hard_joint_limits[self._lower_joint_ids, 1]
        self._upper_joint_min = self._default_upper_joint_pos + self.cfg.upper_soft_limit_factor * (
            self._soft_joint_limits[self._upper_joint_ids, 0] - self._default_upper_joint_pos
        )
        self._upper_joint_max = self._default_upper_joint_pos + self.cfg.upper_soft_limit_factor * (
            self._soft_joint_limits[self._upper_joint_ids, 1] - self._default_upper_joint_pos
        )
        self._upper_action_min = (self._upper_joint_min - self._default_upper_joint_pos) / self.cfg.action_scale
        self._upper_action_max = (self._upper_joint_max - self._default_upper_joint_pos) / self.cfg.action_scale
        self._lower_action_min = (self._lower_joint_min - self._default_lower_joint_pos) / self.cfg.action_scale
        self._lower_action_max = (self._lower_joint_max - self._default_lower_joint_pos) / self.cfg.action_scale
        self._p_gains = self._build_lower_pd_gains(stiffness=True)
        self._d_gains = self._build_lower_pd_gains(stiffness=False)

        self._gravity_vec_w = self._expand_gravity(self.robot.data.GRAVITY_VEC_W)

        self._command_resample_interval_steps = max(1, int(round(self.cfg.command_resample_interval_s / self.step_dt)))
        self._command_transition_steps = max(1, int(round(self.cfg.command_transition_duration_s / self.step_dt)))
        self._upper_resample_interval_steps = max(
            1, int(round(self.cfg.upper_body_resample_interval_s / self.step_dt))
        )

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._last_last_actions = torch.zeros_like(self._actions)
        self._origin_actions = torch.zeros_like(self._actions)
        self._prev_lower_joint_vel = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._lower_joint_targets = self._default_lower_joint_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self._torques = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._kp_factors = torch.ones(self.num_envs, self.cfg.action_space, device=self.device)
        self._kd_factors = torch.ones(self.num_envs, self.cfg.action_space, device=self.device)
        self._joint_injection = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._actuation_offset = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        self._current_command = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        self._command_start = torch.zeros_like(self._current_command)
        self._command_goal = torch.zeros_like(self._current_command)
        self._command_interp_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._upper_pose_current = self._default_upper_joint_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self._upper_pose_start = self._upper_pose_current.clone()
        self._upper_pose_goal = self._upper_pose_current.clone()
        self._upper_interp_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._upper_curriculum_progress = float(self.cfg.upper_curriculum_init)

        self._policy_history = torch.zeros(
            self.num_envs, self.cfg.history_length, POLICY_FRAME_DIM, device=self.device
        )

        self._reward_names = [
            "tracking_x_vel",
            "tracking_y_vel",
            "tracking_ang_vel",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "action_rate",
            "tracking_base_height",
            "deviation_hip_joint",
            "deviation_ankle_joint",
            "deviation_knee_joint",
            "dof_acc",
            "dof_pos_limits",
            "feet_air_time",
            "feet_clearance",
            "feet_distance_lateral",
            "knee_distance_lateral",
            "feet_ground_parallel",
            "feet_parallel",
            "smoothness",
            "joint_power",
            "feet_stumble",
            "torques",
            "dof_vel",
            "dof_vel_limits",
            "torque_limits",
            "no_fly",
            "joint_tracking_error",
            "feet_slip",
            "feet_contact_forces",
            "contact_momentum",
            "action_vanish",
            "stand_still",
            "symmetry_joint",
        ]
        self._reward_scales = {
            "tracking_x_vel": self.cfg.rew_scale_tracking_x_vel,
            "tracking_y_vel": self.cfg.rew_scale_tracking_y_vel,
            "tracking_ang_vel": self.cfg.rew_scale_tracking_ang_vel,
            "lin_vel_z": self.cfg.rew_scale_lin_vel_z,
            "ang_vel_xy": self.cfg.rew_scale_ang_vel_xy,
            "orientation": self.cfg.rew_scale_orientation,
            "action_rate": self.cfg.rew_scale_action_rate,
            "tracking_base_height": self.cfg.rew_scale_tracking_base_height,
            "deviation_hip_joint": self.cfg.rew_scale_deviation_hip_joint,
            "deviation_ankle_joint": self.cfg.rew_scale_deviation_ankle_joint,
            "deviation_knee_joint": self.cfg.rew_scale_deviation_knee_joint,
            "dof_acc": self.cfg.rew_scale_dof_acc,
            "dof_pos_limits": self.cfg.rew_scale_dof_pos_limits,
            "feet_air_time": self.cfg.rew_scale_feet_air_time,
            "feet_clearance": self.cfg.rew_scale_feet_clearance,
            "feet_distance_lateral": self.cfg.rew_scale_feet_distance_lateral,
            "knee_distance_lateral": self.cfg.rew_scale_knee_distance_lateral,
            "feet_ground_parallel": self.cfg.rew_scale_feet_ground_parallel,
            "feet_parallel": self.cfg.rew_scale_feet_parallel,
            "smoothness": self.cfg.rew_scale_smoothness,
            "joint_power": self.cfg.rew_scale_joint_power,
            "feet_stumble": self.cfg.rew_scale_feet_stumble,
            "torques": self.cfg.rew_scale_torques,
            "dof_vel": self.cfg.rew_scale_dof_vel,
            "dof_vel_limits": self.cfg.rew_scale_dof_vel_limits,
            "torque_limits": self.cfg.rew_scale_torque_limits,
            "no_fly": self.cfg.rew_scale_no_fly,
            "joint_tracking_error": self.cfg.rew_scale_joint_tracking_error,
            "feet_slip": self.cfg.rew_scale_feet_slip,
            "feet_contact_forces": self.cfg.rew_scale_feet_contact_forces,
            "contact_momentum": self.cfg.rew_scale_contact_momentum,
            "action_vanish": self.cfg.rew_scale_action_vanish,
            "stand_still": self.cfg.rew_scale_stand_still,
            "symmetry_joint": 0.0,
        }
        self._reward_scales = {name: scale * self.step_dt for name, scale in self._reward_scales.items()}
        self._metric_episode_sums = {
            "linear_velocity_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "forward_velocity_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "lateral_velocity_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "yaw_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "height_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "symmetry_joint_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "lin_vel_x_tracking_score": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "lin_vel_y_tracking_score": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "yaw_tracking_score": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "height_tracking_score": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }
        self._reward_episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for name in self._reward_names
        }
        self._last_episode_metrics = {
            "linear_velocity_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "forward_velocity_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "lateral_velocity_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "yaw_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "height_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "symmetry_joint_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "episode_length": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "survival_time": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

        self._base_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._base_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._torso_projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        self._all_joint_pos = torch.zeros(self.num_envs, ALL_JOINT_DIM, device=self.device)
        self._all_joint_vel = torch.zeros(self.num_envs, ALL_JOINT_DIM, device=self.device)
        self._lower_joint_acc = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._base_height = torch.zeros(self.num_envs, device=self.device)
        self._feet_pos_w = torch.zeros(self.num_envs, len(self.cfg.foot_body_names), 3, device=self.device)
        self._feet_vel_w = torch.zeros_like(self._feet_pos_w)
        self._foot_contact_forces = torch.zeros_like(self._feet_pos_w)
        self._contact_filt = torch.zeros(self.num_envs, len(self.cfg.foot_body_names), dtype=torch.bool, device=self.device)
        self._current_contacts = torch.zeros_like(self._contact_filt)
        self._last_contacts = torch.zeros_like(self._contact_filt)
        self._first_contacts = torch.zeros_like(self._contact_filt)
        self._feet_air_time = torch.zeros(self.num_envs, len(self.cfg.foot_body_names), device=self.device)
        self._feet_max_height = torch.zeros(self.num_envs, len(self.cfg.foot_body_names), device=self.device)

        self._all_joint_mirror_index, self._all_joint_mirror_sign = self._build_joint_mirror(self.cfg.all_joint_names)
        self._lower_joint_mirror_index, self._lower_joint_mirror_sign = self._build_joint_mirror(
            self.cfg.lower_joint_names
        )
        self._upper_joint_mirror_index, self._upper_joint_mirror_sign = self._build_joint_mirror(
            self.cfg.upper_joint_names
        )

        self._default_body_masses = self.robot.root_physx_view.get_masses().clone().cpu()
        self._default_body_coms = self.robot.root_physx_view.get_coms().clone().cpu()
        self._randomize_reset_control_props(self.robot._ALL_INDICES)
        self._randomize_reset_rigid_body_props(self.robot._ALL_INDICES)

        self._compute_intermediate_values()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._origin_actions = actions.clone()
        self._actions = torch.clamp(actions.clone(), -1.0, 1.0)
        if self.cfg.randomize_joint_injection:
            self._joint_injection = sample_uniform(
                self.cfg.joint_injection_range[0],
                self.cfg.joint_injection_range[1],
                (self.num_envs, self.cfg.action_space),
                self.device,
            ) * self._torque_limits.unsqueeze(0)

        resample_command_env_ids = torch.nonzero(
            (self.episode_length_buf > 0) & (self.episode_length_buf % self._command_resample_interval_steps == 0),
            as_tuple=False,
        ).flatten()
        if len(resample_command_env_ids) > 0:
            self._resample_commands(resample_command_env_ids)

        resample_upper_env_ids = torch.nonzero(
            (self.episode_length_buf > 0) & (self.episode_length_buf % self._upper_resample_interval_steps == 0),
            as_tuple=False,
        ).flatten()
        if len(resample_upper_env_ids) > 0:
            self._resample_upper_body_targets(resample_upper_env_ids)

        self._update_command_targets()
        self._update_upper_body_targets()

        self._lower_joint_targets = self._default_lower_joint_pos.unsqueeze(0) + self.cfg.action_scale * self._actions
        if self.cfg.clip_joint_targets_to_soft_limits:
            self._lower_joint_targets = torch.clamp(
                self._lower_joint_targets,
                min=self._lower_joint_min.unsqueeze(0),
                max=self._lower_joint_max.unsqueeze(0),
            )

    def _apply_action(self) -> None:
        lower_joint_pos = self._all_joint_pos[:, : self.cfg.action_space]
        lower_joint_vel = self._all_joint_vel[:, : self.cfg.action_space]
        self._torques = (
            self._p_gains.unsqueeze(0) * self._kp_factors * (self._lower_joint_targets - lower_joint_pos)
            - self._d_gains.unsqueeze(0) * self._kd_factors * lower_joint_vel
        )
        self._torques = self._torques + self._actuation_offset + self._joint_injection
        self._torques = torch.clamp(self._torques, min=-self._torque_limits.unsqueeze(0), max=self._torque_limits.unsqueeze(0))
        # Lower body follows OpenHomie's M-controller: env-computed torques with an explicit effort actuator path.
        self.robot.set_joint_position_target(self._lower_joint_targets, joint_ids=self._lower_joint_ids)
        self.robot.set_joint_effort_target(self._torques, joint_ids=self._lower_joint_ids)
        self.robot.set_joint_position_target(self._upper_pose_current, joint_ids=self._upper_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        current_frame = self._build_policy_frame(apply_noise=True)
        critic_frame = self._build_policy_frame(apply_noise=False)
        self._policy_history = torch.roll(self._policy_history, shifts=-1, dims=1)
        self._policy_history[:, -1, :] = current_frame

        policy_obs = self._policy_history.reshape(self.num_envs, -1)
        critic_obs = torch.cat(
            (
                critic_frame,
                self._base_lin_vel_b * self.cfg.obs_base_lin_vel_scale,
            ),
            dim=-1,
        )

        current_metrics = self.get_current_eval_metrics()
        self.extras["track/linear_velocity_tracking_error"] = current_metrics["linear_velocity_tracking_error"].detach()
        self.extras["track/forward_velocity_tracking_error"] = current_metrics["forward_velocity_tracking_error"].detach()
        self.extras["track/lateral_velocity_tracking_error"] = current_metrics["lateral_velocity_tracking_error"].detach()
        self.extras["track/yaw_tracking_error"] = current_metrics["yaw_tracking_error"].detach()
        self.extras["track/height_tracking_error"] = current_metrics["height_tracking_error"].detach()
        self.extras["track/symmetry_joint_error"] = current_metrics["symmetry_joint_error"].detach()

        self._last_last_actions.copy_(self._prev_actions)
        self._prev_actions.copy_(self._actions)
        self._prev_lower_joint_vel.copy_(self._all_joint_vel[:, : self.cfg.action_space])
        self._last_contacts.copy_(self._current_contacts)
        self._feet_air_time *= (~self._contact_filt).float()
        self._feet_max_height *= (~self._contact_filt).float()

        return {"policy": policy_obs, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        reward_terms = self._compute_reward_terms()
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for name, term in reward_terms.items():
            reward += term
            self._reward_episode_sums[name] += term

        current_metrics = self.get_current_eval_metrics()
        for name in (
            "linear_velocity_tracking_error",
            "forward_velocity_tracking_error",
            "lateral_velocity_tracking_error",
            "yaw_tracking_error",
            "height_tracking_error",
            "symmetry_joint_error",
        ):
            self._metric_episode_sums[name] += current_metrics[name]
        self._metric_episode_sums["lin_vel_x_tracking_score"] += torch.exp(
            -torch.square(self._current_command[:, 0] - self._base_lin_vel_b[:, 0]) / self.cfg.tracking_sigma
        )
        self._metric_episode_sums["lin_vel_y_tracking_score"] += torch.exp(
            -torch.square(self._current_command[:, 1] - self._base_lin_vel_b[:, 1]) / self.cfg.tracking_sigma
        )
        self._metric_episode_sums["yaw_tracking_score"] += torch.exp(
            -torch.square(self._current_command[:, 2] - self._base_ang_vel_b[:, 2]) / self.cfg.tracking_sigma
        )
        self._metric_episode_sums["height_tracking_score"] += torch.exp(
            -torch.abs(self._current_command[:, 3] - self._base_height) * 4.0
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.extras["episode"] = {}
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        termination_contact = torch.any(
            self.robot.data.body_pos_w[:, self._termination_contact_body_ids, 2] < self.cfg.termination_body_height_threshold,
            dim=1,
        )
        died = termination_contact
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        completed_episode_length = torch.clamp(self.episode_length_buf[env_ids].float(), min=1.0)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._update_upper_body_curriculum(env_ids, completed_episode_length)
        self._log_and_reset_episode_summaries(env_ids, completed_episode_length)

        num_resets = len(env_ids)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, :2] += sample_uniform(-self.cfg.reset_xy_noise, self.cfg.reset_xy_noise, (num_resets, 2), self.device)

        yaw = sample_uniform(-self.cfg.reset_yaw_noise, self.cfg.reset_yaw_noise, (num_resets,), self.device)
        # Isaac Lab / Isaac Sim quaternions are wxyz.
        root_state[:, 3:7] = quat_from_euler_xyz(
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            yaw,
        )
        root_state[:, 2] += sample_uniform(
            self.cfg.reset_root_z_noise[0],
            self.cfg.reset_root_z_noise[1],
            (num_resets,),
            self.device,
        )
        root_state[:, 7:13] = sample_uniform(
            -self.cfg.reset_root_velocity_noise,
            self.cfg.reset_root_velocity_noise,
            (num_resets, 6),
            self.device,
        )

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_scale = sample_uniform(
            self.cfg.reset_joint_position_scale[0],
            self.cfg.reset_joint_position_scale[1],
            (num_resets, joint_pos.shape[1]),
            self.device,
        )
        joint_offset = sample_uniform(
            self.cfg.reset_joint_position_offset[0],
            self.cfg.reset_joint_position_offset[1],
            (num_resets, joint_pos.shape[1]),
            self.device,
        )
        joint_pos = torch.clamp(
            joint_pos * joint_scale + joint_offset,
            min=self._hard_joint_limits[:, 0].unsqueeze(0),
            max=self._hard_joint_limits[:, 1].unsqueeze(0),
        )
        joint_vel += sample_uniform(
            -self.cfg.reset_joint_velocity_noise,
            self.cfg.reset_joint_velocity_noise,
            (num_resets, joint_vel.shape[1]),
            self.device,
        )

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._actions[env_ids] = 0.0
        self._origin_actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._last_last_actions[env_ids] = 0.0
        self._prev_lower_joint_vel[env_ids] = 0.0
        self._lower_joint_targets[env_ids] = self._default_lower_joint_pos
        self._torques[env_ids] = 0.0
        self._joint_injection[env_ids] = 0.0
        self._actuation_offset[env_ids] = 0.0
        self._current_contacts[env_ids] = False
        self._last_contacts[env_ids] = False
        self._contact_filt[env_ids] = False
        self._first_contacts[env_ids] = False
        self._feet_air_time[env_ids] = 0.0
        self._feet_max_height[env_ids] = 0.0

        self._randomize_reset_control_props(env_ids)
        self._randomize_reset_rigid_body_props(env_ids)

        self._resample_commands(env_ids, initialize=True)
        self._resample_upper_body_targets(env_ids, initialize=True)

        self._compute_intermediate_values()
        init_frames = self._build_policy_frame(apply_noise=False)
        self._policy_history[env_ids] = init_frames[env_ids].unsqueeze(1).repeat(1, self.cfg.history_length, 1)

    def _build_policy_frame(self, apply_noise: bool) -> torch.Tensor:
        frame = torch.cat(
            (
                torch.cat(
                    (
                        self._current_command[:, :2] * self.cfg.obs_command_lin_vel_scale,
                        self._current_command[:, 2:3] * self.cfg.obs_command_ang_vel_scale,
                        self._current_command[:, 3:4],
                    ),
                    dim=-1,
                ),
                self._base_ang_vel_b * self.cfg.obs_ang_vel_scale,
                self._torso_projected_gravity,
                self._all_joint_pos - self._default_joint_pos.unsqueeze(0),
                self._all_joint_vel * self.cfg.obs_joint_vel_scale,
                self._prev_actions,
            ),
            dim=-1,
        )
        if apply_noise:
            frame = frame.clone()
            frame[:, COMMAND_DIM : COMMAND_DIM + ANG_VEL_DIM] += sample_uniform(
                -self.cfg.obs_ang_vel_noise,
                self.cfg.obs_ang_vel_noise,
                (self.num_envs, ANG_VEL_DIM),
                self.device,
            )
            gravity_start = COMMAND_DIM + ANG_VEL_DIM
            gravity_end = gravity_start + GRAVITY_DIM
            frame[:, gravity_start:gravity_end] += sample_uniform(
                -self.cfg.obs_gravity_noise,
                self.cfg.obs_gravity_noise,
                (self.num_envs, GRAVITY_DIM),
                self.device,
            )
            joint_pos_start = gravity_end
            joint_pos_end = joint_pos_start + ALL_JOINT_DIM
            frame[:, joint_pos_start:joint_pos_end] += sample_uniform(
                -self.cfg.obs_joint_pos_noise,
                self.cfg.obs_joint_pos_noise,
                (self.num_envs, ALL_JOINT_DIM),
                self.device,
            )
            joint_vel_start = joint_pos_end
            joint_vel_end = joint_vel_start + ALL_JOINT_DIM
            frame[:, joint_vel_start:joint_vel_end] += sample_uniform(
                -self.cfg.obs_joint_vel_noise,
                self.cfg.obs_joint_vel_noise,
                (self.num_envs, ALL_JOINT_DIM),
                self.device,
            )
        return frame

    def _compute_intermediate_values(self):
        self._base_lin_vel_b = self.robot.data.root_lin_vel_b
        imu_quat_w = self.robot.data.body_quat_w[:, self._imu_body_id]
        imu_ang_vel_w = self.robot.data.body_ang_vel_w[:, self._imu_body_id]
        self._base_ang_vel_b = quat_apply_inverse(imu_quat_w, imu_ang_vel_w)
        self._torso_projected_gravity = quat_apply_inverse(imu_quat_w, self._gravity_vec_w)
        self._all_joint_pos = self.robot.data.joint_pos[:, self._all_joint_ids]
        self._all_joint_vel = self.robot.data.joint_vel[:, self._all_joint_ids]
        lower_joint_vel = self._all_joint_vel[:, : self.cfg.action_space]
        self._lower_joint_acc = (lower_joint_vel - self._prev_lower_joint_vel) / self.step_dt
        self._feet_pos_w = self.robot.data.body_pos_w[:, self._foot_body_ids]
        self._feet_vel_w = self.robot.data.body_lin_vel_w[:, self._foot_body_ids]
        support_height = torch.max(self._feet_pos_w[:, 0, 2], self._feet_pos_w[:, 1, 2])
        self._base_height = self.robot.data.root_pos_w[:, 2] - support_height + ANKLE_SOLE_DISTANCE
        self._current_contacts = self._feet_pos_w[:, :, 2] < self.cfg.foot_contact_height_threshold
        pseudo_force_z = self._current_contacts.float() * (
            50.0 + 200.0 * torch.clamp(-self._feet_vel_w[:, :, 2], min=0.0)
        )
        pseudo_force_xy = self._current_contacts.float() * 60.0 * torch.linalg.norm(self._feet_vel_w[:, :, :2], dim=-1)
        self._foot_contact_forces = torch.stack(
            (pseudo_force_xy, torch.zeros_like(pseudo_force_xy), pseudo_force_z),
            dim=-1,
        )
        self._contact_filt = torch.logical_or(self._current_contacts, self._last_contacts)
        self._first_contacts = (self._feet_air_time >= self.step_dt) & self._contact_filt
        self._feet_air_time += self.step_dt
        feet_height, _ = self._get_feet_heights()
        self._feet_max_height = torch.maximum(self._feet_max_height, feet_height)

    def _compute_reward_terms(self) -> dict[str, torch.Tensor]:
        command_height = self._current_command[:, 3]
        standing_mask = (command_height >= 0.735).float()

        lin_vel_error_x = torch.square(self._current_command[:, 0] - self._base_lin_vel_b[:, 0])
        lin_vel_error_y = torch.square(self._current_command[:, 1] - self._base_lin_vel_b[:, 1])
        ang_vel_error = torch.square(self._current_command[:, 2] - self._base_ang_vel_b[:, 2])
        tracking_x_vel = torch.exp(-lin_vel_error_x / self.cfg.tracking_sigma)
        tracking_y_vel = torch.exp(-lin_vel_error_y / self.cfg.tracking_sigma)
        tracking_ang_vel = torch.exp(-ang_vel_error / self.cfg.tracking_sigma)

        base_height_error = torch.abs(self._base_height - command_height)
        tracking_base_height = torch.exp(-base_height_error * 4.0)

        lower_joint_pos = self._all_joint_pos[:, : self.cfg.action_space]
        lower_joint_vel = self._all_joint_vel[:, : self.cfg.action_space]
        lower_joint_delta = lower_joint_pos - self._default_lower_joint_pos.unsqueeze(0)
        hip_indices = torch.tensor([0, 1, 2, 6, 7, 8], device=self.device, dtype=torch.long)
        ankle_indices = torch.tensor([5, 11], device=self.device, dtype=torch.long)
        knee_indices = torch.tensor([3, 9], device=self.device, dtype=torch.long)
        deviation_hip_joint = torch.sum(torch.square(lower_joint_delta[:, hip_indices]), dim=-1) * standing_mask
        deviation_ankle_joint = torch.sum(torch.square(lower_joint_delta[:, ankle_indices]), dim=-1) * standing_mask
        knee_action_min = self._default_lower_joint_pos[knee_indices].unsqueeze(0) + self.cfg.action_scale * self._lower_action_min[knee_indices].unsqueeze(0)
        knee_action_max = self._default_lower_joint_pos[knee_indices].unsqueeze(0) + self.cfg.action_scale * self._lower_action_max[knee_indices].unsqueeze(0)
        knee_joint_deviation = (lower_joint_pos[:, knee_indices] - knee_action_min) / torch.clamp(
            knee_action_max - knee_action_min, min=1.0e-6
        )
        deviation_knee_joint = torch.sum(
            torch.abs((knee_joint_deviation - 0.5) * (self.robot.data.root_pos_w[:, 2] - command_height).unsqueeze(-1)),
            dim=-1,
        )

        feet_height, feet_height_var = self._get_feet_heights()
        cur_footvel_translated = self._feet_vel_w - self.robot.data.root_lin_vel_w.unsqueeze(1)
        feetvel_in_body = quat_apply_inverse(
            self.robot.data.root_quat_w.unsqueeze(1).repeat(1, cur_footvel_translated.shape[1], 1).reshape(-1, 4),
            cur_footvel_translated.reshape(-1, 3),
        ).view_as(cur_footvel_translated)
        feet_clearance = torch.sum(
            torch.square(feet_height - self.cfg.clearance_height_target)
            * torch.sqrt(torch.sum(torch.square(feetvel_in_body[:, :, :2]), dim=2)),
            dim=1,
        ) * (command_height >= 0.71).float()

        foot_pos_body = quat_apply_inverse(
            self.robot.data.root_quat_w.unsqueeze(1).repeat(1, self._feet_pos_w.shape[1], 1).reshape(-1, 4),
            (self._feet_pos_w - self.robot.data.root_pos_w.unsqueeze(1)).reshape(-1, 3),
        ).view_as(self._feet_pos_w)
        foot_lateral_dis = torch.abs(foot_pos_body[:, 0, 1] - foot_pos_body[:, 1, 1])
        feet_distance_lateral = (
            torch.clamp(foot_lateral_dis - self.cfg.least_feet_distance_lateral, max=0.0)
            + torch.clamp(-foot_lateral_dis + self.cfg.most_feet_distance_lateral, max=0.0)
        ) * standing_mask

        knee_pos_body = quat_apply_inverse(
            self.robot.data.root_quat_w.unsqueeze(1).repeat(1, self._knee_body_ids.numel(), 1).reshape(-1, 4),
            (self.robot.data.body_pos_w[:, self._knee_body_ids, :] - self.robot.data.root_pos_w.unsqueeze(1)).reshape(-1, 3),
        ).view(self.num_envs, self._knee_body_ids.numel(), 3)
        knee_lateral_dis = torch.abs(knee_pos_body[:, 0, 1] - knee_pos_body[:, 2, 1]) + torch.abs(
            knee_pos_body[:, 1, 1] - knee_pos_body[:, 3, 1]
        )
        knee_distance_lateral = (
            torch.clamp(knee_lateral_dis - self.cfg.least_knee_distance_lateral * 2.0, max=0.0)
            + torch.clamp(-knee_lateral_dis + self.cfg.most_knee_distance_lateral * 2.0, max=0.0)
        ) * standing_mask

        continue_contact = (self._feet_air_time >= 3.0 * self.step_dt) & self._contact_filt
        feet_ground_parallel = torch.sum(feet_height_var * continue_contact.float(), dim=1)
        left_foot_pos = self.robot.data.body_pos_w[:, self._left_foot_surface_ids]
        right_foot_pos = self.robot.data.body_pos_w[:, self._right_foot_surface_ids]
        feet_parallel = torch.var(torch.norm(left_foot_pos - right_foot_pos, dim=2), dim=1) * standing_mask

        lower_contact_forces = self._foot_contact_forces
        feet_stumble = torch.any(
            torch.linalg.norm(lower_contact_forces[:, :, :2], dim=2) > 3.0 * torch.abs(lower_contact_forces[:, :, 2]),
            dim=1,
        ).float()
        dof_pos_limits = torch.sum(
            torch.clamp(self._homie_joint_pos_limits[:, 0].unsqueeze(0) - lower_joint_pos, max=0.0).abs()
            + torch.clamp(lower_joint_pos - self._homie_joint_pos_limits[:, 1].unsqueeze(0), min=0.0),
            dim=1,
        )
        dof_vel_limits = torch.sum(
            (torch.abs(lower_joint_vel) - self._joint_vel_limits.unsqueeze(0) * self.cfg.soft_dof_vel_limit).clamp(min=0.0),
            dim=1,
        )
        torque_limits = torch.sum(
            (torch.abs(self._torques) - self._torque_limits.unsqueeze(0) * self.cfg.soft_torque_limit).clamp(min=0.0),
            dim=1,
        )
        no_fly = (torch.sum(lower_contact_forces[:, :, 2] > 0.5, dim=1) == 1).float()
        no_fly = torch.maximum(no_fly, (torch.linalg.norm(self._current_command[:, :3], dim=1) < 0.1).float())
        feet_slip = torch.sum(torch.linalg.norm(self._feet_vel_w[:, :, :2], dim=2) * (lower_contact_forces[:, :, 2] > 1.0), dim=1)
        feet_contact_forces = torch.sum(
            (torch.linalg.norm(lower_contact_forces, dim=-1) - self.cfg.max_contact_force).clamp(min=0.0), dim=1
        )
        contact_momentum = torch.sum(
            torch.clamp(self._feet_vel_w[:, :, 2], max=0.0) * torch.clamp(lower_contact_forces[:, :, 2] - 50.0, min=0.0),
            dim=1,
        )
        action_vanish = torch.sum(
            torch.clamp(self._origin_actions - self._lower_action_max.unsqueeze(0), min=0.0)
            + torch.clamp(self._lower_action_min.unsqueeze(0) - self._origin_actions, min=0.0),
            dim=1,
        )
        stand_still = (
            torch.sum(lower_contact_forces[:, :, 2] < 0.1, dim=-1).float()
            * standing_mask
            * (torch.linalg.norm(self._current_command[:, :3], dim=1) < 0.1).float()
        )

        raw_terms = {
            "tracking_x_vel": tracking_x_vel,
            "tracking_y_vel": tracking_y_vel,
            "tracking_ang_vel": tracking_ang_vel,
            "lin_vel_z": torch.square(self._base_lin_vel_b[:, 2]) * standing_mask,
            "ang_vel_xy": torch.sum(torch.square(self._base_ang_vel_b[:, :2]), dim=1),
            "orientation": torch.sum(torch.square(self._torso_projected_gravity[:, :2]), dim=1),
            "action_rate": torch.sum(torch.square(self._prev_actions - self._actions), dim=1),
            "tracking_base_height": tracking_base_height,
            "deviation_hip_joint": deviation_hip_joint,
            "deviation_ankle_joint": deviation_ankle_joint,
            "deviation_knee_joint": deviation_knee_joint,
            "dof_acc": torch.sum(torch.square(self._lower_joint_acc), dim=1),
            "dof_pos_limits": dof_pos_limits,
            "feet_air_time": torch.sum((self._feet_air_time - 0.5) * self._first_contacts.float(), dim=1)
            * (torch.linalg.norm(self._current_command[:, :3], dim=1) > 0.1).float(),
            "feet_clearance": feet_clearance,
            "feet_distance_lateral": feet_distance_lateral,
            "knee_distance_lateral": knee_distance_lateral,
            "feet_ground_parallel": feet_ground_parallel,
            "feet_parallel": feet_parallel,
            "smoothness": torch.sum(torch.square(self._actions - 2.0 * self._prev_actions + self._last_last_actions), dim=1),
            "joint_power": torch.sum(torch.abs(lower_joint_vel) * torch.abs(self._torques), dim=1)
            / torch.clamp(
                torch.sum(torch.square(self._current_command[:, 0:2]), dim=-1) + 0.2 * torch.square(self._current_command[:, 2]),
                min=0.1,
            ),
            "feet_stumble": feet_stumble,
            "torques": torch.sum(torch.square(self._torques / torch.clamp(self._p_gains.unsqueeze(0), min=1.0e-6)), dim=1),
            "dof_vel": torch.sum(torch.square(lower_joint_vel), dim=1),
            "dof_vel_limits": dof_vel_limits,
            "torque_limits": torque_limits,
            "no_fly": no_fly,
            "joint_tracking_error": torch.sum(torch.square(self._lower_joint_targets - lower_joint_pos), dim=-1),
            "feet_slip": feet_slip,
            "feet_contact_forces": feet_contact_forces,
            "contact_momentum": contact_momentum,
            "action_vanish": action_vanish,
            "stand_still": stand_still,
            "symmetry_joint": self._compute_symmetry_joint_error(),
        }
        return {name: raw_terms[name] * self._reward_scales[name] for name in self._reward_names}

    def _resample_commands(self, env_ids: torch.Tensor, initialize: bool = False):
        if len(env_ids) == 0:
            return

        num_envs = len(env_ids)
        mode_selector = torch.rand(num_envs, device=self.device)
        height_mode = mode_selector < self.cfg.height_command_probability
        velocity_mode = mode_selector > (1.0 - self.cfg.velocity_command_probability)

        vx = sample_uniform(
            self.cfg.command_vx_range[0], self.cfg.command_vx_range[1], (num_envs,), self.device
        )
        vy = sample_uniform(
            self.cfg.command_vy_range[0], self.cfg.command_vy_range[1], (num_envs,), self.device
        )
        yaw_rate = sample_uniform(
            self.cfg.command_yaw_rate_range[0], self.cfg.command_yaw_rate_range[1], (num_envs,), self.device
        )
        vx = torch.where(velocity_mode, vx, torch.zeros_like(vx))
        vy = torch.where(velocity_mode, vy, torch.zeros_like(vy))
        yaw_rate = torch.where(velocity_mode, yaw_rate, torch.zeros_like(yaw_rate))

        target_height = torch.where(
            height_mode,
            self.cfg.base_height_target
            + sample_uniform(
                self.cfg.command_height_offset_range[0],
                self.cfg.command_height_offset_range[1],
                (num_envs,),
                self.device,
            ),
            torch.full((num_envs,), self.cfg.base_height_target, device=self.device),
        )

        new_goal = torch.stack((vx, vy, yaw_rate, target_height), dim=-1)

        if initialize:
            self._current_command[env_ids] = new_goal
            self._command_start[env_ids] = new_goal
            self._command_goal[env_ids] = new_goal
            self._command_interp_step[env_ids] = self._command_transition_steps
        else:
            self._command_start[env_ids] = self._current_command[env_ids]
            self._command_goal[env_ids] = new_goal
            self._command_interp_step[env_ids] = 0

    def _update_command_targets(self):
        self._command_interp_step = torch.clamp(self._command_interp_step + 1, max=self._command_transition_steps)
        alpha = (self._command_interp_step.float() / float(self._command_transition_steps)).unsqueeze(-1)
        self._current_command = self._command_start + alpha * (self._command_goal - self._command_start)

    def _resample_upper_body_targets(self, env_ids: torch.Tensor, initialize: bool = False):
        if len(env_ids) == 0:
            return

        goal = self._sample_upper_body_pose_targets(len(env_ids))
        if initialize:
            default_upper = self._default_upper_joint_pos.unsqueeze(0).repeat(len(env_ids), 1)
            self._upper_pose_current[env_ids] = default_upper
            self._upper_pose_start[env_ids] = default_upper
            self._upper_pose_goal[env_ids] = goal
            self._upper_interp_step[env_ids] = 0
        else:
            self._upper_pose_start[env_ids] = self._upper_pose_current[env_ids]
            self._upper_pose_goal[env_ids] = goal
            self._upper_interp_step[env_ids] = 0

    def _update_upper_body_targets(self):
        self._upper_interp_step = torch.clamp(self._upper_interp_step + 1, max=self._upper_resample_interval_steps)
        alpha = (self._upper_interp_step.float() / float(self._upper_resample_interval_steps)).unsqueeze(-1)
        self._upper_pose_current = self._upper_pose_start + alpha * (self._upper_pose_goal - self._upper_pose_start)

    def _sample_upper_body_pose_targets(self, batch_size: int) -> torch.Tensor:
        progress = 1.0 if self.cfg.upper_curriculum_eval_fixed else min(
            self._upper_curriculum_progress, self.cfg.upper_curriculum_max_progress
        )
        uu = torch.rand(batch_size, UPPER_JOINT_DIM, device=self.device)
        progress_tensor = torch.full_like(uu, progress)
        scaled_ratio = -1.0 / (20.0 * (1.0 - progress_tensor * 0.99)) * torch.log(
            1.0 - uu + uu * math.exp(-20.0 * (1.0 - progress * 0.99))
        )
        random_joint_ratio = scaled_ratio * torch.rand(batch_size, UPPER_JOINT_DIM, device=self.device)
        rand_pos = torch.rand(batch_size, UPPER_JOINT_DIM, device=self.device) - 0.5
        sampled_upper_actions = torch.where(
            rand_pos >= 0.0,
            self._upper_action_min.unsqueeze(0),
            self._upper_action_max.unsqueeze(0),
        ) * random_joint_ratio
        return self._default_upper_joint_pos.unsqueeze(0) + self.cfg.action_scale * sampled_upper_actions

    def _update_upper_body_curriculum(self, env_ids: torch.Tensor, completed_episode_length: torch.Tensor):
        if self.cfg.upper_curriculum_eval_fixed:
            self._upper_curriculum_progress = 1.0
            return

        mean_tracking_x_reward = float(
            (self._reward_episode_sums["tracking_x_vel"][env_ids] / completed_episode_length).mean().item()
        )
        # Corresponds to HOMIE's tracking-driven action curriculum: expand upper-body disturbance
        # only when forward velocity tracking stays above 80% of its maximum scale.
        if mean_tracking_x_reward > self.cfg.upper_curriculum_promote_threshold * self._reward_scales["tracking_x_vel"]:
            self._upper_curriculum_progress = min(
                self.cfg.upper_curriculum_max_progress,
                self._upper_curriculum_progress + self.cfg.upper_curriculum_step,
            )

    def _log_and_reset_episode_summaries(self, env_ids: torch.Tensor, completed_episode_length: torch.Tensor):
        if len(env_ids) == 0:
            return

        self.extras["episode"] = {}
        for name, values in self._reward_episode_sums.items():
            self.extras["episode"][f"rew_{name}"] = torch.mean(values[env_ids] / completed_episode_length)
            values[env_ids] = 0.0
        for name, values in self._metric_episode_sums.items():
            per_env_metric = values[env_ids] / completed_episode_length
            if name in self._last_episode_metrics:
                self._last_episode_metrics[name][env_ids] = per_env_metric
            self.extras["episode"][name] = torch.mean(per_env_metric)
            values[env_ids] = 0.0
        per_env_survival = completed_episode_length * self.step_dt
        self._last_episode_metrics["episode_length"][env_ids] = completed_episode_length
        self._last_episode_metrics["survival_time"][env_ids] = per_env_survival
        self.extras["episode"]["episode_length"] = torch.mean(completed_episode_length)
        self.extras["episode"]["survival_time"] = torch.mean(per_env_survival)
        curriculum_value = torch.tensor(self._upper_curriculum_progress, device=self.device)
        self.extras["episode"]["upper_curriculum_progress"] = curriculum_value
        self.extras["episode"]["action_curriculum_ratio"] = curriculum_value

    def _find_joint_ids(self, joint_names: list[str]) -> torch.Tensor:
        return torch.as_tensor(
            self.robot.find_joints(joint_names, preserve_order=True)[0],
            device=self.device,
            dtype=torch.long,
        )

    def _find_body_ids_safe(self, body_names: list[str] | tuple[str, ...]) -> torch.Tensor:
        try:
            return torch.as_tensor(
                self.robot.find_bodies(list(body_names), preserve_order=True)[0],
                device=self.device,
                dtype=torch.long,
            )
        except Exception:
            return torch.zeros(0, device=self.device, dtype=torch.long)

    def _compute_homie_soft_limits(self, joint_ids: torch.Tensor) -> torch.Tensor:
        joint_limits = self._hard_joint_limits[joint_ids]
        center = 0.5 * (joint_limits[:, 0] + joint_limits[:, 1])
        width = joint_limits[:, 1] - joint_limits[:, 0]
        return torch.stack(
            (
                center - 0.5 * width * self.cfg.soft_dof_pos_limit,
                center + 0.5 * width * self.cfg.soft_dof_pos_limit,
            ),
            dim=-1,
        )

    def _build_lower_pd_gains(self, stiffness: bool) -> torch.Tensor:
        gains = []
        for name in self.cfg.lower_joint_names:
            if "knee" in name:
                gains.append(150.0 if stiffness else 4.0)
            elif "ankle" in name:
                gains.append(40.0 if stiffness else 2.0)
            else:
                gains.append(100.0 if stiffness else 2.0)
        return torch.tensor(gains, device=self.device, dtype=torch.float)

    def _randomize_reset_control_props(self, env_ids: torch.Tensor):
        num_envs = len(env_ids)
        if self.cfg.randomize_kp:
            self._kp_factors[env_ids] = sample_uniform(
                self.cfg.kp_range[0], self.cfg.kp_range[1], (num_envs, self.cfg.action_space), self.device
            )
        else:
            self._kp_factors[env_ids] = 1.0
        if self.cfg.randomize_kd:
            self._kd_factors[env_ids] = sample_uniform(
                self.cfg.kd_range[0], self.cfg.kd_range[1], (num_envs, self.cfg.action_space), self.device
            )
        else:
            self._kd_factors[env_ids] = 1.0
        if self.cfg.randomize_actuation_offset:
            self._actuation_offset[env_ids] = sample_uniform(
                self.cfg.actuation_offset_range[0],
                self.cfg.actuation_offset_range[1],
                (num_envs, self.cfg.action_space),
                self.device,
            ) * self._torque_limits.unsqueeze(0)
        else:
            self._actuation_offset[env_ids] = 0.0

    def _randomize_reset_rigid_body_props(self, env_ids: torch.Tensor):
        env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
        masses = self._default_body_masses.clone()
        coms = self._default_body_coms.clone()
        if self.cfg.randomize_link_mass:
            scale = sample_uniform(
                self.cfg.link_mass_range[0],
                self.cfg.link_mass_range[1],
                (len(env_ids), masses.shape[1]),
                self.device,
            ).cpu()
            masses[env_ids_cpu] = masses[env_ids_cpu] * scale
            masses[env_ids_cpu, 0] = self._default_body_masses[env_ids_cpu, 0]
        if self.cfg.randomize_payload_mass:
            torso_mass_delta = sample_uniform(
                self.cfg.payload_mass_range[0], self.cfg.payload_mass_range[1], (len(env_ids),), self.device
            ).cpu()
            masses[env_ids_cpu, self._torso_body_id] = self._default_body_masses[env_ids_cpu, self._torso_body_id] + torso_mass_delta
            if self._left_hand_body_id.numel() > 0:
                left_hand_delta = sample_uniform(
                    self.cfg.hand_payload_mass_range[0], self.cfg.hand_payload_mass_range[1], (len(env_ids),), self.device
                ).cpu()
                masses[env_ids_cpu, int(self._left_hand_body_id[0])] = (
                    self._default_body_masses[env_ids_cpu, int(self._left_hand_body_id[0])] + left_hand_delta
                )
            if self._right_hand_body_id.numel() > 0:
                right_hand_delta = sample_uniform(
                    self.cfg.hand_payload_mass_range[0], self.cfg.hand_payload_mass_range[1], (len(env_ids),), self.device
                ).cpu()
                masses[env_ids_cpu, int(self._right_hand_body_id[0])] = (
                    self._default_body_masses[env_ids_cpu, int(self._right_hand_body_id[0])] + right_hand_delta
                )
        if self.cfg.randomize_com_displacement:
            delta = sample_uniform(
                self.cfg.com_displacement_range[0], self.cfg.com_displacement_range[1], (len(env_ids), 3), self.device
            ).cpu()
            coms[env_ids_cpu, 0, :3] = self._default_body_coms[env_ids_cpu, 0, :3] + delta
        if self.cfg.randomize_body_displacement:
            delta = sample_uniform(
                self.cfg.body_displacement_range[0], self.cfg.body_displacement_range[1], (len(env_ids), 3), self.device
            ).cpu()
            coms[env_ids_cpu, self._torso_body_id, :3] = self._default_body_coms[env_ids_cpu, self._torso_body_id, :3] + delta
        try:
            self.robot.root_physx_view.set_masses(masses, env_ids_cpu)
            self.robot.root_physx_view.set_coms(coms, env_ids_cpu)
        except AttributeError:
            pass

    def _get_feet_heights(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_foot_pos = self.robot.data.body_pos_w[:, self._left_foot_surface_ids, :3]
        right_foot_pos = self.robot.data.body_pos_w[:, self._right_foot_surface_ids, :3]
        left_height = torch.mean(left_foot_pos[:, :, 2], dim=-1, keepdim=True)
        left_var = torch.var(left_foot_pos[:, :, 2], dim=-1, keepdim=True)
        right_height = torch.mean(right_foot_pos[:, :, 2], dim=-1, keepdim=True)
        right_var = torch.var(right_foot_pos[:, :, 2], dim=-1, keepdim=True)
        return torch.cat((left_height, right_height), dim=-1), torch.cat((left_var, right_var), dim=-1)

    def _build_joint_mirror(self, joint_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        paired = {left: right for left, right in self.cfg.mirror_joint_pairs}
        paired.update({right: left for left, right in self.cfg.mirror_joint_pairs})

        mirror_index = []
        mirror_sign = []
        for name in joint_names:
            mirrored_name = paired.get(name, name)
            mirror_index.append(joint_names.index(mirrored_name))
            mirror_sign.append(-1.0 if name in self.cfg.mirror_negate_joint_names else 1.0)
        return (
            torch.tensor(mirror_index, device=self.device, dtype=torch.long),
            torch.tensor(mirror_sign, device=self.device, dtype=torch.float),
        )

    def _expand_gravity(self, gravity: torch.Tensor) -> torch.Tensor:
        if gravity.ndim == 1:
            return gravity.unsqueeze(0).repeat(self.num_envs, 1)
        if gravity.shape[0] == 1:
            return gravity.repeat(self.num_envs, 1)
        return gravity

    def _compute_symmetry_joint_error(self) -> torch.Tensor:
        lower_joint_pos = self._all_joint_pos[:, : self.cfg.action_space] - self._default_joint_pos[: self.cfg.action_space].unsqueeze(0)
        mirrored = lower_joint_pos[:, self._lower_joint_mirror_index] * self._lower_joint_mirror_sign.unsqueeze(0)
        return torch.mean(torch.square(lower_joint_pos - mirrored), dim=-1)

    def get_current_eval_metrics(self) -> dict[str, torch.Tensor]:
        lin_vel_error_xy = self._current_command[:, :2] - self._base_lin_vel_b[:, :2]
        return {
            "linear_velocity_tracking_error": torch.linalg.norm(lin_vel_error_xy, dim=-1),
            "forward_velocity_tracking_error": torch.abs(self._current_command[:, 0] - self._base_lin_vel_b[:, 0]),
            "lateral_velocity_tracking_error": torch.abs(self._current_command[:, 1] - self._base_lin_vel_b[:, 1]),
            "yaw_tracking_error": torch.abs(self._current_command[:, 2] - self._base_ang_vel_b[:, 2]),
            "height_tracking_error": torch.abs(self._current_command[:, 3] - self._base_height),
            "symmetry_joint_error": self._compute_symmetry_joint_error(),
        }

    def get_last_episode_metrics(self) -> dict[str, torch.Tensor]:
        return {name: values.clone() for name, values in self._last_episode_metrics.items()}

    def mirror_lower_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions[:, self._lower_joint_mirror_index] * self._lower_joint_mirror_sign.unsqueeze(0)

    def _mirror_policy_frames(self, frames: torch.Tensor) -> torch.Tensor:
        mirrored = frames.clone()
        mirrored[..., 0] = frames[..., 0]
        mirrored[..., 1] = -frames[..., 1]
        mirrored[..., 2] = -frames[..., 2]
        mirrored[..., 3] = frames[..., 3]

        ang_start = COMMAND_DIM
        mirrored[..., ang_start + 0] = -frames[..., ang_start + 0]
        mirrored[..., ang_start + 1] = frames[..., ang_start + 1]
        mirrored[..., ang_start + 2] = -frames[..., ang_start + 2]

        gravity_start = ang_start + ANG_VEL_DIM
        mirrored[..., gravity_start + 0] = frames[..., gravity_start + 0]
        mirrored[..., gravity_start + 1] = -frames[..., gravity_start + 1]
        mirrored[..., gravity_start + 2] = frames[..., gravity_start + 2]

        joint_pos_start = gravity_start + GRAVITY_DIM
        joint_pos_end = joint_pos_start + ALL_JOINT_DIM
        mirrored[..., joint_pos_start:joint_pos_end] = (
            frames[..., joint_pos_start:joint_pos_end][..., self._all_joint_mirror_index]
            * self._all_joint_mirror_sign.view(*([1] * (frames.ndim - 1)), -1)
        )

        joint_vel_start = joint_pos_end
        joint_vel_end = joint_vel_start + ALL_JOINT_DIM
        mirrored[..., joint_vel_start:joint_vel_end] = (
            frames[..., joint_vel_start:joint_vel_end][..., self._all_joint_mirror_index]
            * self._all_joint_mirror_sign.view(*([1] * (frames.ndim - 1)), -1)
        )

        action_start = joint_vel_end
        mirrored[..., action_start:] = (
            frames[..., action_start:][..., self._lower_joint_mirror_index]
            * self._lower_joint_mirror_sign.view(*([1] * (frames.ndim - 1)), -1)
        )
        return mirrored

    def mirror_policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs_frames = obs.view(-1, self.cfg.history_length, POLICY_FRAME_DIM)
        return self._mirror_policy_frames(obs_frames).view(obs.shape[0], -1)

    def mirror_critic_obs(self, obs: torch.Tensor) -> torch.Tensor:
        mirrored = obs.clone()
        frame_dim = POLICY_FRAME_DIM
        mirrored[:, :frame_dim] = self._mirror_policy_frames(obs[:, :frame_dim])

        offset = frame_dim
        mirrored[:, offset + 0] = obs[:, offset + 0]
        mirrored[:, offset + 1] = -obs[:, offset + 1]
        mirrored[:, offset + 2] = obs[:, offset + 2]
        return mirrored
