import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG

from ..velocity_env_cfg import RobotEnvCfg


def _kinematic_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """Create a fixed box-shaped rigid object for simple benchmark furniture."""

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.55,
                metallic=0.05,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _kinematic_sphere(
    prim_path: str,
    radius: float,
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.SphereCfg(
            radius=radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.35,
                metallic=0.1,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _kinematic_cylinder(
    prim_path: str,
    radius: float,
    height: float,
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.45,
                metallic=0.05,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class RobotBenchmarkSceneCfg(InteractiveSceneCfg):
    """Scene with a flat floor, a table surrogate, a shelf surrogate, and static targets."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.78, 0.78, 0.8),
            roughness=0.95,
            metallic=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

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

    shelf_left_post_front = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Shelf_left_post_front",
        size=(0.05, 0.05, 1.30),
        pos=(1.34, -0.63, 0.65),
        color=(0.30, 0.34, 0.38),
    )
    shelf_left_post_back = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Shelf_left_post_back",
        size=(0.05, 0.05, 1.30),
        pos=(1.34, -0.97, 0.65),
        color=(0.30, 0.34, 0.38),
    )
    shelf_right_post_front = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Shelf_right_post_front",
        size=(0.05, 0.05, 1.30),
        pos=(1.82, -0.63, 0.65),
        color=(0.30, 0.34, 0.38),
    )
    shelf_right_post_back = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Shelf_right_post_back",
        size=(0.05, 0.05, 1.30),
        pos=(1.82, -0.97, 0.65),
        color=(0.30, 0.34, 0.38),
    )
    shelf_level_low = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Shelf_level_low",
        size=(0.56, 0.36, 0.04),
        pos=(1.58, -0.80, 0.40),
        color=(0.46, 0.49, 0.52),
    )
    shelf_level_mid = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Shelf_level_mid",
        size=(0.56, 0.36, 0.04),
        pos=(1.58, -0.80, 0.84),
        color=(0.46, 0.49, 0.52),
    )
    shelf_level_top = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Shelf_level_top",
        size=(0.56, 0.36, 0.04),
        pos=(1.58, -0.80, 1.28),
        color=(0.46, 0.49, 0.52),
    )

    target_table_front_left = _kinematic_sphere(
        prim_path="{ENV_REGEX_NS}/Target_table_front_left",
        radius=0.04,
        pos=(0.78, 0.18, 0.82),
        color=(0.93, 0.24, 0.21),
    )
    target_table_front_right = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Target_table_front_right",
        size=(0.06, 0.06, 0.06),
        pos=(0.78, -0.18, 0.81),
        color=(0.15, 0.55, 0.87),
    )
    target_table_back_left = _kinematic_cylinder(
        prim_path="{ENV_REGEX_NS}/Target_table_back_left",
        radius=0.035,
        height=0.10,
        pos=(1.12, 0.18, 0.83),
        color=(0.97, 0.76, 0.12),
    )
    target_table_back_right = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Target_table_back_right",
        size=(0.07, 0.07, 0.05),
        pos=(1.12, -0.16, 0.805),
        color=(0.27, 0.77, 0.42),
    )
    target_shelf_low_inner = _kinematic_sphere(
        prim_path="{ENV_REGEX_NS}/Target_shelf_low_inner",
        radius=0.04,
        pos=(1.48, -0.80, 0.46),
        color=(0.88, 0.38, 0.75),
    )
    target_shelf_mid_inner = _kinematic_cylinder(
        prim_path="{ENV_REGEX_NS}/Target_shelf_mid_inner",
        radius=0.03,
        height=0.10,
        pos=(1.58, -0.72, 0.91),
        color=(0.96, 0.58, 0.15),
    )
    target_shelf_mid_outer = _kinematic_cuboid(
        prim_path="{ENV_REGEX_NS}/Target_shelf_mid_outer",
        size=(0.07, 0.05, 0.08),
        pos=(1.69, -0.88, 0.90),
        color=(0.20, 0.79, 0.79),
    )
    target_shelf_top_center = _kinematic_sphere(
        prim_path="{ENV_REGEX_NS}/Target_shelf_top_center",
        radius=0.045,
        pos=(1.58, -0.80, 1.345),
        color=(0.91, 0.20, 0.32),
    )

    key_light = AssetBaseCfg(
        prim_path="/World/keyLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=3500.0,
            color=(0.96, 0.95, 0.90),
        ),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=900.0,
            color=(0.92, 0.94, 1.0),
        ),
    )


@configclass
class RobotBenchmarkEnvCfg(RobotEnvCfg):
    """Fixed-scene benchmark environment for standing and video capture."""

    scene: RobotBenchmarkSceneCfg = RobotBenchmarkSceneCfg(num_envs=1, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.8)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
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

        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.observations.policy.enable_corruption = False
        if hasattr(self.observations.critic, "enable_corruption"):
            self.observations.critic.enable_corruption = False

        self.viewer.origin_type = "world"
        self.viewer.eye = (3.7, -4.3, 2.4)
        self.viewer.lookat = (1.0, -0.20, 1.0)

        self.rewards.track_lin_vel_xy = None
        self.rewards.track_ang_vel_z = None
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.feet_slide = None
        self.rewards.energy = None


@configclass
class RobotBenchmarkPlayEnvCfg(RobotBenchmarkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.viewer.eye = (3.5, -4.0, 2.25)
        self.viewer.lookat = (0.95, -0.18, 0.98)
