import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..velocity_env_cfg import RobotSceneCfg
from ..left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach.left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach_env_cfg import (
    LEFT_HAND_BODY_NAME,
    MAX_TARGETS_PER_EPISODE,
    PER_TARGET_TIMEOUT_S,
    RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachEnvCfg,
)

BENCHMARK_BLOCK_NAMES = tuple(f"target_block_{index}" for index in range(6))


def _benchmark_block(
    prim_path: str,
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(0.055, 0.055, 0.055),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.45,
                metallic=0.05,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class FreezeBaseReachBenchmarkSceneCfg(RobotSceneCfg):
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

    target_block_0 = _benchmark_block("{ENV_REGEX_NS}/TargetBlock_0", (0.36, 0.14, 0.26), (0.88, 0.28, 0.24))
    target_block_1 = _benchmark_block("{ENV_REGEX_NS}/TargetBlock_1", (0.44, 0.20, 0.24), (0.93, 0.52, 0.23))
    target_block_2 = _benchmark_block("{ENV_REGEX_NS}/TargetBlock_2", (0.54, 0.28, 0.18), (0.94, 0.78, 0.26))
    target_block_3 = _benchmark_block("{ENV_REGEX_NS}/TargetBlock_3", (0.66, 0.10, 0.18), (0.24, 0.80, 0.48))
    target_block_4 = _benchmark_block("{ENV_REGEX_NS}/TargetBlock_4", (0.82, 0.22, 0.16), (0.23, 0.63, 0.92))
    target_block_5 = _benchmark_block("{ENV_REGEX_NS}/TargetBlock_5", (0.94, 0.34, 0.14), (0.57, 0.38, 0.90))


@configclass
class RobotLeftHandLocoReachFreezeBaseReachBenchmarkEnvCfg(
    RobotLeftHandLocoReachAdapterAcquireTightStayNaturalReachSettleShortFreezeBaseReachEnvCfg
):
    scene: FreezeBaseReachBenchmarkSceneCfg = FreezeBaseReachBenchmarkSceneCfg(num_envs=128, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.8)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.env_spacing = 4.0

        self.curriculum.left_hand_target_levels = None
        self.observations.policy.enable_corruption = False

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        self.terminations.target_quota = None
        self.terminations.target_timeout = None
        self.terminations.time_out = None
        self.terminations.base_height = None
        self.terminations.bad_orientation = None

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.6, -3.4, 1.9)
        self.viewer.lookat = (0.0, 0.0, 0.65)


@configclass
class RobotLeftHandLocoReachFreezeBaseReachBenchmarkPlayEnvCfg(RobotLeftHandLocoReachFreezeBaseReachBenchmarkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
