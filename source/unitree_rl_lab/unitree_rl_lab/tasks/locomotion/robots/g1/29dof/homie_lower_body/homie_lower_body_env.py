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

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
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
ANKLE_SOLE_DISTANCE = 0.02

COMMAND_DIM = 4
ANG_VEL_DIM = 3
GRAVITY_DIM = 3
ALL_JOINT_DIM = len(ALL_JOINT_NAMES)
LOWER_ACTION_DIM = len(LOWER_JOINT_NAMES)
UPPER_JOINT_DIM = len(UPPER_JOINT_NAMES)
POLICY_FRAME_DIM = COMMAND_DIM + ANG_VEL_DIM + GRAVITY_DIM + 2 * ALL_JOINT_DIM + LOWER_ACTION_DIM
CRITIC_EXTRA_DIM = 3 + UPPER_JOINT_DIM + 1


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
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 10.0),
            "operation": "add",
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

    history_length = 6
    action_scale = 0.25
    clip_joint_targets_to_soft_limits = True
    upper_soft_limit_factor = 0.9

    torso_body_name = "torso_link"
    lower_body_height_termination_m = 0.40
    upper_body_height_termination_m = 1.10
    undesired_body_height_termination_m = 0.03
    max_projected_gravity_xy = 0.78

    command_resample_interval_s = 4.0
    command_transition_duration_s = 0.75
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
    upper_curriculum_demote_step = 0.04
    upper_curriculum_promote_threshold = 0.80
    upper_curriculum_demote_threshold = 0.55
    upper_curriculum_max_progress = 0.60
    upper_curriculum_eval_fixed = False

    reset_xy_noise = 0.15
    reset_yaw_noise = math.pi
    reset_root_z_noise = (-0.02, 0.03)
    reset_root_velocity_noise = 0.15
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

    lin_vel_tracking_sigma = 0.25
    yaw_rate_tracking_sigma = 0.25
    height_tracking_alpha = 4.0
    knee_guidance_sigma = 0.10
    foot_clearance_target = 0.14
    foot_clearance_sigma = 0.05
    foot_clearance_tanh_mult = 2.0

    rew_scale_track_lin_vel = 1.5
    rew_scale_track_lin_vel_y = 1.0
    rew_scale_track_yaw_rate = 2.0
    rew_scale_track_height = 2.0
    rew_scale_knee_guidance = 0.8
    rew_scale_alive = 0.05
    rew_scale_foot_clearance = 0.25
    rew_scale_orientation = -1.5
    rew_scale_vertical_vel = -0.5
    rew_scale_ang_vel_xy = -0.025
    rew_scale_action_rate = -0.01
    rew_scale_joint_vel = -1.0e-4
    rew_scale_joint_acc = -2.5e-7
    rew_scale_power = -2.0e-5
    rew_scale_joint_limit = -2.0
    rew_scale_feet_slip = -0.25
    rew_scale_symmetry_joint = -0.05

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
        self.events.add_base_mass = None


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
        self._foot_body_ids = torch.as_tensor(
            self.robot.find_bodies(self.cfg.foot_body_names, preserve_order=True)[0],
            device=self.device,
            dtype=torch.long,
        )

        self._soft_joint_limits = self.robot.data.soft_joint_pos_limits[0].clone()
        self._default_joint_pos = self.robot.data.default_joint_pos[0, self._all_joint_ids].clone()
        self._default_lower_joint_pos = self.robot.data.default_joint_pos[0, self._lower_joint_ids].clone()
        self._default_upper_joint_pos = self.robot.data.default_joint_pos[0, self._upper_joint_ids].clone()

        self._lower_joint_min = self._soft_joint_limits[self._lower_joint_ids, 0]
        self._lower_joint_max = self._soft_joint_limits[self._lower_joint_ids, 1]
        self._upper_joint_min = self._default_upper_joint_pos + self.cfg.upper_soft_limit_factor * (
            self._soft_joint_limits[self._upper_joint_ids, 0] - self._default_upper_joint_pos
        )
        self._upper_joint_max = self._default_upper_joint_pos + self.cfg.upper_soft_limit_factor * (
            self._soft_joint_limits[self._upper_joint_ids, 1] - self._default_upper_joint_pos
        )

        self._gravity_vec_w = self._expand_gravity(self.robot.data.GRAVITY_VEC_W)

        self._command_resample_interval_steps = max(1, int(round(self.cfg.command_resample_interval_s / self.step_dt)))
        self._command_transition_steps = max(1, int(round(self.cfg.command_transition_duration_s / self.step_dt)))
        self._upper_resample_interval_steps = max(
            1, int(round(self.cfg.upper_body_resample_interval_s / self.step_dt))
        )

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._prev_lower_joint_vel = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._lower_joint_targets = self._default_lower_joint_pos.unsqueeze(0).repeat(self.num_envs, 1)

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
            "track_lin_vel",
            "track_lin_vel_y",
            "track_yaw_rate",
            "track_height",
            "knee_guidance",
            "alive",
            "foot_clearance",
            "orientation",
            "vertical_vel",
            "ang_vel_xy",
            "action_rate",
            "joint_vel",
            "joint_acc",
            "power",
            "joint_limit",
            "feet_slip",
            "symmetry_joint",
        ]
        self._reward_episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for name in self._reward_names
        }
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

        self._all_joint_mirror_index, self._all_joint_mirror_sign = self._build_joint_mirror(self.cfg.all_joint_names)
        self._lower_joint_mirror_index, self._lower_joint_mirror_sign = self._build_joint_mirror(
            self.cfg.lower_joint_names
        )
        self._upper_joint_mirror_index, self._upper_joint_mirror_sign = self._build_joint_mirror(
            self.cfg.upper_joint_names
        )

        self._compute_intermediate_values()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = torch.clamp(actions.clone(), -1.0, 1.0)

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
        self.robot.set_joint_position_target(self._lower_joint_targets, joint_ids=self._lower_joint_ids)
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
                self._upper_pose_current - self._default_upper_joint_pos.unsqueeze(0),
                torch.full((self.num_envs, 1), self._upper_curriculum_progress, device=self.device),
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

        self._prev_actions.copy_(self._actions)
        self._prev_lower_joint_vel.copy_(self._all_joint_vel[:, : self.cfg.action_space])

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
            -torch.square(self._current_command[:, 0] - self._base_lin_vel_b[:, 0]) / self.cfg.lin_vel_tracking_sigma
        )
        self._metric_episode_sums["lin_vel_y_tracking_score"] += torch.exp(
            -torch.square(self._current_command[:, 1] - self._base_lin_vel_b[:, 1]) / self.cfg.lin_vel_tracking_sigma
        )
        self._metric_episode_sums["yaw_tracking_score"] += torch.exp(
            -torch.square(self._current_command[:, 2] - self._base_ang_vel_b[:, 2]) / self.cfg.yaw_rate_tracking_sigma
        )
        self._metric_episode_sums["height_tracking_score"] += torch.exp(
            -torch.abs(self._current_command[:, 3] - self._base_height) * self.cfg.height_tracking_alpha
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.extras["episode"] = {}
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_too_low = self._base_height < self.cfg.lower_body_height_termination_m
        root_too_high = self._base_height > self.cfg.upper_body_height_termination_m
        bad_orientation = torch.linalg.norm(self._torso_projected_gravity[:, :2], dim=-1) > self.cfg.max_projected_gravity_xy
        undesired_body_low = self._min_non_foot_body_height() < self.cfg.undesired_body_height_termination_m

        died = root_too_low | root_too_high | bad_orientation | undesired_body_low
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
        root_state[:, :2] += sample_uniform(
            -self.cfg.reset_xy_noise,
            self.cfg.reset_xy_noise,
            (num_resets, 2),
            self.device,
        )

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
            min=self._soft_joint_limits[:, 0].unsqueeze(0),
            max=self._soft_joint_limits[:, 1].unsqueeze(0),
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
        self._prev_actions[env_ids] = 0.0
        self._prev_lower_joint_vel[env_ids] = 0.0
        self._lower_joint_targets[env_ids] = self._default_lower_joint_pos

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
        self._base_ang_vel_b = self.robot.data.root_ang_vel_b
        torso_quat_w = self.robot.data.body_quat_w[:, self._torso_body_id]
        self._torso_projected_gravity = quat_apply_inverse(torso_quat_w, self._gravity_vec_w)
        self._all_joint_pos = self.robot.data.joint_pos[:, self._all_joint_ids]
        self._all_joint_vel = self.robot.data.joint_vel[:, self._all_joint_ids]
        lower_joint_vel = self._all_joint_vel[:, : self.cfg.action_space]
        self._lower_joint_acc = (lower_joint_vel - self._prev_lower_joint_vel) / self.step_dt
        self._feet_pos_w = self.robot.data.body_pos_w[:, self._foot_body_ids]
        self._feet_vel_w = self.robot.data.body_lin_vel_w[:, self._foot_body_ids]
        support_height = torch.max(self._feet_pos_w[:, 0, 2], self._feet_pos_w[:, 1, 2])
        self._base_height = self.robot.data.root_pos_w[:, 2] - support_height + ANKLE_SOLE_DISTANCE

    def _compute_reward_terms(self) -> dict[str, torch.Tensor]:
        command_vx = self._current_command[:, 0]
        command_vy = self._current_command[:, 1]
        command_yaw = self._current_command[:, 2]
        command_height = self._current_command[:, 3]

        lin_vel_error = command_vx - self._base_lin_vel_b[:, 0]
        lin_vel_y_error = command_vy - self._base_lin_vel_b[:, 1]
        yaw_rate_error = command_yaw - self._base_ang_vel_b[:, 2]
        height_error = command_height - self._base_height

        track_lin_vel = self.cfg.rew_scale_track_lin_vel * torch.exp(
            -torch.square(lin_vel_error) / self.cfg.lin_vel_tracking_sigma
        )
        track_lin_vel_y = self.cfg.rew_scale_track_lin_vel_y * torch.exp(
            -torch.square(lin_vel_y_error) / self.cfg.lin_vel_tracking_sigma
        )
        track_yaw_rate = self.cfg.rew_scale_track_yaw_rate * torch.exp(
            -torch.square(yaw_rate_error) / self.cfg.yaw_rate_tracking_sigma
        )
        track_height = self.cfg.rew_scale_track_height * torch.exp(-torch.abs(height_error) * self.cfg.height_tracking_alpha)

        knee_pos = self._all_joint_pos[:, [3, 9]]
        knee_default = self._default_joint_pos[[3, 9]].unsqueeze(0)
        knee_upper = self._soft_joint_limits[self._lower_joint_ids[[3, 9]], 1].unsqueeze(0)
        desired_squat_ratio = torch.clamp(
            (self.cfg.base_height_target - command_height) / max(abs(self.cfg.command_height_offset_range[0]), 1.0e-6),
            min=0.0,
            max=1.0,
        )
        actual_squat_ratio = torch.clamp((knee_pos - knee_default) / torch.clamp(knee_upper - knee_default, min=1.0e-6), 0.0, 1.0)
        knee_guidance = self.cfg.rew_scale_knee_guidance * torch.exp(
            -torch.square(actual_squat_ratio.mean(dim=-1) - desired_squat_ratio) / self.cfg.knee_guidance_sigma
        )

        foot_pos_body = self._feet_pos_w - self.robot.data.root_pos_w.unsqueeze(1)
        foot_vel_body = self._feet_vel_w - self.robot.data.root_lin_vel_w.unsqueeze(1)
        root_quat_repeat = self.robot.data.root_quat_w.unsqueeze(1).repeat(1, foot_pos_body.shape[1], 1).reshape(-1, 4)
        foot_pos_body = quat_apply_inverse(root_quat_repeat, foot_pos_body.reshape(-1, 3)).view_as(foot_pos_body)
        foot_vel_body = quat_apply_inverse(root_quat_repeat, foot_vel_body.reshape(-1, 3)).view_as(foot_vel_body)
        foot_height_error = torch.square(foot_pos_body[:, :, 2] - self.cfg.foot_clearance_target)
        foot_speed_xy = torch.tanh(
            self.cfg.foot_clearance_tanh_mult * torch.linalg.norm(foot_vel_body[:, :, :2], dim=-1)
        )
        moving_mask = (torch.abs(command_vx) + torch.abs(command_vy) + torch.abs(command_yaw)) > 0.05
        foot_clearance = self.cfg.rew_scale_foot_clearance * torch.exp(
            -torch.sum(foot_height_error * foot_speed_xy, dim=-1) / self.cfg.foot_clearance_sigma
        ) * moving_mask.float()

        orientation = self.cfg.rew_scale_orientation * torch.sum(torch.square(self._torso_projected_gravity[:, :2]), dim=-1)
        vertical_vel = self.cfg.rew_scale_vertical_vel * torch.square(self._base_lin_vel_b[:, 2])
        ang_vel_xy = self.cfg.rew_scale_ang_vel_xy * torch.sum(torch.square(self._base_ang_vel_b[:, :2]), dim=-1)
        action_rate = self.cfg.rew_scale_action_rate * torch.sum(torch.square(self._actions - self._prev_actions), dim=-1)
        joint_vel = self.cfg.rew_scale_joint_vel * torch.sum(
            torch.square(self._all_joint_vel[:, : self.cfg.action_space]), dim=-1
        )
        joint_acc = self.cfg.rew_scale_joint_acc * torch.sum(torch.square(self._lower_joint_acc), dim=-1)
        power = self.cfg.rew_scale_power * torch.sum(
            torch.abs(self.robot.data.applied_torque[:, self._lower_joint_ids] * self._all_joint_vel[:, : self.cfg.action_space]),
            dim=-1,
        )
        joint_limit = self.cfg.rew_scale_joint_limit * self._joint_limit_penalty()
        feet_slip = self.cfg.rew_scale_feet_slip * self._feet_slip_penalty()
        symmetry_joint = self.cfg.rew_scale_symmetry_joint * self._compute_symmetry_joint_error()
        alive = torch.full((self.num_envs,), self.cfg.rew_scale_alive, device=self.device)

        return {
            "track_lin_vel": track_lin_vel,
            "track_lin_vel_y": track_lin_vel_y,
            "track_yaw_rate": track_yaw_rate,
            "track_height": track_height,
            "knee_guidance": knee_guidance,
            "alive": alive,
            "foot_clearance": foot_clearance,
            "orientation": orientation,
            "vertical_vel": vertical_vel,
            "ang_vel_xy": ang_vel_xy,
            "action_rate": action_rate,
            "joint_vel": joint_vel,
            "joint_acc": joint_acc,
            "power": power,
            "joint_limit": joint_limit,
            "feet_slip": feet_slip,
            "symmetry_joint": symmetry_joint,
        }

    def _joint_limit_penalty(self) -> torch.Tensor:
        lower_joint_pos = self._all_joint_pos[:, : self.cfg.action_space]
        lower_soft_limits = self._soft_joint_limits[self._lower_joint_ids]
        lower_violation = torch.clamp(lower_soft_limits[:, 0].unsqueeze(0) - lower_joint_pos, min=0.0)
        upper_violation = torch.clamp(lower_joint_pos - lower_soft_limits[:, 1].unsqueeze(0), min=0.0)
        return torch.sum(lower_violation + upper_violation, dim=-1)

    def _feet_slip_penalty(self) -> torch.Tensor:
        near_ground = self._feet_pos_w[:, :, 2] < 0.05
        feet_speed_xy = torch.linalg.norm(self._feet_vel_w[:, :, :2], dim=-1)
        return torch.sum(feet_speed_xy * near_ground.float(), dim=-1)

    def _min_non_foot_body_height(self) -> torch.Tensor:
        body_pos_w = self.robot.data.body_pos_w
        keep_mask = torch.ones(body_pos_w.shape[1], dtype=torch.bool, device=self.device)
        keep_mask[self._foot_body_ids] = False
        return body_pos_w[:, keep_mask, 2].amin(dim=-1)

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
            self._upper_pose_current[env_ids] = goal
            self._upper_pose_start[env_ids] = goal
            self._upper_pose_goal[env_ids] = goal
            self._upper_interp_step[env_ids] = self._upper_resample_interval_steps
        else:
            self._upper_pose_start[env_ids] = self._upper_pose_current[env_ids]
            self._upper_pose_goal[env_ids] = goal
            self._upper_interp_step[env_ids] = 0

    def _update_upper_body_targets(self):
        self._upper_interp_step = torch.clamp(self._upper_interp_step + 1, max=self._upper_resample_interval_steps)
        alpha = (self._upper_interp_step.float() / float(self._upper_resample_interval_steps)).unsqueeze(-1)
        self._upper_pose_current = self._upper_pose_start + alpha * (self._upper_pose_goal - self._upper_pose_start)

    def _sample_upper_body_pose_targets(self, batch_size: int) -> torch.Tensor:
        lower_delta = self._upper_joint_min - self._default_upper_joint_pos
        upper_delta = self._upper_joint_max - self._default_upper_joint_pos
        progress = 1.0 if self.cfg.upper_curriculum_eval_fixed else min(
            self._upper_curriculum_progress, self.cfg.upper_curriculum_max_progress
        )
        low = (
            self._default_upper_joint_pos.unsqueeze(0)
            + progress * lower_delta.unsqueeze(0)
        ).repeat(batch_size, 1)
        high = (
            self._default_upper_joint_pos.unsqueeze(0)
            + progress * upper_delta.unsqueeze(0)
        ).repeat(batch_size, 1)
        rand = torch.rand(batch_size, UPPER_JOINT_DIM, device=self.device)
        return low + rand * (high - low)

    def _update_upper_body_curriculum(self, env_ids: torch.Tensor, completed_episode_length: torch.Tensor):
        if self.cfg.upper_curriculum_eval_fixed:
            self._upper_curriculum_progress = 1.0
            return

        self._upper_curriculum_progress = min(self._upper_curriculum_progress, self.cfg.upper_curriculum_max_progress)
        tracking_score = (
            self._metric_episode_sums["lin_vel_x_tracking_score"][env_ids]
            + self._metric_episode_sums["lin_vel_y_tracking_score"][env_ids]
            + self._metric_episode_sums["yaw_tracking_score"][env_ids]
            + self._metric_episode_sums["height_tracking_score"][env_ids]
        ) / (4.0 * completed_episode_length)
        mean_score = float(tracking_score.mean().item())

        # Corresponds to HOMIE's tracking-driven action curriculum, but the expanded range is the
        # internally generated upper-body target trajectory instead of the policy action envelope.
        if mean_score > self.cfg.upper_curriculum_promote_threshold:
            self._upper_curriculum_progress = min(
                self.cfg.upper_curriculum_max_progress,
                self._upper_curriculum_progress + self.cfg.upper_curriculum_step,
            )
        elif mean_score < self.cfg.upper_curriculum_demote_threshold:
            self._upper_curriculum_progress = max(
                self.cfg.upper_curriculum_init,
                self._upper_curriculum_progress - self.cfg.upper_curriculum_demote_step,
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
        self.extras["episode"]["upper_curriculum_progress"] = torch.tensor(
            self._upper_curriculum_progress, device=self.device
        )

    def _find_joint_ids(self, joint_names: list[str]) -> torch.Tensor:
        return torch.as_tensor(
            self.robot.find_joints(joint_names, preserve_order=True)[0],
            device=self.device,
            dtype=torch.long,
        )

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
        offset += 3

        mirrored[:, offset : offset + UPPER_JOINT_DIM] = (
            obs[:, offset : offset + UPPER_JOINT_DIM][:, self._upper_joint_mirror_index]
            * self._upper_joint_mirror_sign.unsqueeze(0)
        )
        return mirrored
