import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..benchmark_v1.benchmark_env_cfg import RobotBenchmarkSceneCfg, _kinematic_sphere
from ..velocity_env_cfg import RobotEnvCfg
from .left_hand_pick_place_local_mdp import (
    BALL_CENTER_Z,
    gated_position_command_error_tanh,
    near_target_left_hand_stillness_reward,
    pre_stance_foot_motion_reward,
    pre_stance_joint_deviation_penalty,
    pre_stance_joint_limit_penalty,
    pre_stance_torso_lean_penalty,
    static_target_position_error,
    success_posture_bonus,
    target_completion_bonus,
    target_hold_reward,
    target_pos_command_obs,
    target_relative_base_stance_l2,
    target_relative_base_stance_progress,
    target_relative_base_stance_ready,
    target_success_reached,
    target_timeout_reached,
)


LEFT_HAND_BODY_NAME = "left_wrist_yaw_link"


def _dynamic_sphere(
    prim_path: str,
    radius: float,
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    mass: float = 0.18,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.SphereCfg(
            radius=radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=3.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.3,
                metallic=0.05,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class RobotLeftHandPickPlaceLocalSceneCfg(RobotBenchmarkSceneCfg):
    ball = _dynamic_sphere(
        prim_path="{ENV_REGEX_NS}/Ball",
        radius=0.04,
        pos=(0.84, 0.16, BALL_CENTER_Z),
        color=(0.95, 0.43, 0.16),
    )
    place_target = _kinematic_sphere(
        prim_path="{ENV_REGEX_NS}/PlaceTarget",
        radius=0.045,
        pos=(1.00, 0.12, BALL_CENTER_Z),
        color=(0.16, 0.85, 0.48),
    )


@configclass
class RobotLeftHandPickPlaceLocalEnvCfg(RobotEnvCfg):
    scene: RobotLeftHandPickPlaceLocalSceneCfg = RobotLeftHandPickPlaceLocalSceneCfg(num_envs=4096, env_spacing=4.0)

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
        waist_yaw_cfg = SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])
        feet_cfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])

        task_params = {
            "place_episode_ratio": 0.45,
            "success_threshold": 0.06,
            "success_hold_steps": 10,
            "per_target_timeout_s": 6.0,
            "pregrasp_height": 0.08,
            "preplace_height": 0.08,
            "base_speed_threshold": 0.10,
            "hand_speed_threshold": 0.14,
            "pre_target_switch_radius": 0.12,
            "x_range": (0.36, 0.58),
            "y_range": (0.10, 0.26),
        }

        self.episode_length_s = 12.0
        self.scene.robot.init_state.pos = (0.34, 0.0, 0.8)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.terrain.terrain_generator = None

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

        self.observations.policy.velocity_commands = ObsTerm(
            func=target_pos_command_obs,
            params=task_params,
        )
        self.observations.critic.velocity_commands = ObsTerm(
            func=target_pos_command_obs,
            params=task_params,
        )

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params["pose_range"] = {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "yaw": (-0.35, 0.35)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        self.rewards.track_lin_vel_xy.weight = 0.8
        self.rewards.track_ang_vel_z.weight = 0.2
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_legs.weight = -0.005
        self.rewards.joint_deviation_waists.weight = -0.2
        self.rewards.feet_slide.weight = -0.01
        self.rewards.action_rate.weight = -0.02
        self.rewards.base_target_stance = RewTerm(
            func=target_relative_base_stance_l2,
            weight=-2.5,
            params=task_params,
        )
        self.rewards.stance_ready = RewTerm(
            func=target_relative_base_stance_ready,
            weight=4.0,
            params={**task_params, "gate_std": 0.01},
        )
        self.rewards.stance_progress = RewTerm(
            func=target_relative_base_stance_progress,
            weight=5.0,
            params=task_params,
        )
        self.rewards.pre_stance_torso_lean = RewTerm(
            func=pre_stance_torso_lean_penalty,
            weight=-1.5,
            params=task_params,
        )
        self.rewards.pre_stance_waist_twist = RewTerm(
            func=pre_stance_joint_deviation_penalty,
            weight=-0.8,
            params={**task_params, "asset_cfg": waist_yaw_cfg},
        )
        self.rewards.pre_stance_arm_extension = RewTerm(
            func=pre_stance_joint_limit_penalty,
            weight=-1.0,
            params={**task_params, "asset_cfg": left_arm_cfg, "margin_threshold": 0.18},
        )
        self.rewards.pre_stance_foot_motion = RewTerm(
            func=pre_stance_foot_motion_reward,
            weight=0.2,
            params={**task_params, "asset_cfg": feet_cfg},
        )
        self.rewards.target_completion = RewTerm(
            func=target_completion_bonus,
            weight=6.0,
            params=task_params,
        )
        self.rewards.target_hold = RewTerm(
            func=target_hold_reward,
            weight=4.0,
            params={**task_params, "asset_cfg": ee_cfg, "hold_reward_std": 0.02},
        )
        self.rewards.near_target_left_hand_stillness = RewTerm(
            func=near_target_left_hand_stillness_reward,
            weight=2.0,
            params={**task_params, "asset_cfg": ee_cfg, "near_target_radius": 0.12, "hand_speed_scale": 0.08},
        )
        self.rewards.left_hand_position_tracking = RewTerm(
            func=static_target_position_error,
            weight=-0.12,
            params={**task_params, "asset_cfg": ee_cfg},
        )
        self.rewards.left_hand_position_tracking_fine = RewTerm(
            func=gated_position_command_error_tanh,
            weight=9.0,
            params={**task_params, "asset_cfg": ee_cfg, "std": 0.14, "gate_std": 0.01},
        )
        self.rewards.success_posture_bonus = RewTerm(
            func=success_posture_bonus,
            weight=2.0,
            params={**task_params, "asset_cfg": ee_cfg, "arm_joint_cfg": left_arm_cfg},
        )

        self.rewards.base_height.params["target_height"] = 0.78
        self.terminations.base_height.params["minimum_height"] = 0.12
        self.terminations.bad_orientation.params["limit_angle"] = 1.0
        self.terminations.task_success = DoneTerm(func=target_success_reached, params=task_params)
        self.terminations.task_timeout = DoneTerm(func=target_timeout_reached, params=task_params)

        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotLeftHandPickPlaceLocalPlayEnvCfg(RobotLeftHandPickPlaceLocalEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 8
        self.viewer.origin_type = "world"
        self.viewer.eye = (2.8, -2.6, 1.8)
        self.viewer.lookat = (0.95, 0.0, 0.82)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
