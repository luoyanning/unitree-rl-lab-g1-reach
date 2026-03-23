from __future__ import annotations

from pathlib import Path

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg


REPO_ROOT = Path(__file__).resolve().parents[8]
OPENHOMIE_G1_DIR = REPO_ROOT / "OpenHomie" / "HomieRL" / "legged_gym" / "resources" / "robots" / "g1_description"


OPENHOMIE_G1_LOWER_JOINT_NAMES = [
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

OPENHOMIE_G1_SDK_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
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


OPENHOMIE_G1_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=str(OPENHOMIE_G1_DIR / "g1.urdf"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0,
            ".*_wrist_.*": 0.0,
            "waist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "homie_lower_effort": IdealPDActuatorCfg(
            joint_names_expr=OPENHOMIE_G1_LOWER_JOINT_NAMES,
            effort_limit={
                "left_hip_yaw_joint": 88.0,
                "left_hip_roll_joint": 139.0,
                "left_hip_pitch_joint": 88.0,
                "left_knee_joint": 139.0,
                "left_ankle_pitch_joint": 50.0,
                "left_ankle_roll_joint": 50.0,
                "right_hip_yaw_joint": 88.0,
                "right_hip_roll_joint": 139.0,
                "right_hip_pitch_joint": 88.0,
                "right_knee_joint": 139.0,
                "right_ankle_pitch_joint": 50.0,
                "right_ankle_roll_joint": 50.0,
            },
            velocity_limit={
                "left_hip_yaw_joint": 32.0,
                "left_hip_roll_joint": 20.0,
                "left_hip_pitch_joint": 32.0,
                "left_knee_joint": 20.0,
                "left_ankle_pitch_joint": 37.0,
                "left_ankle_roll_joint": 37.0,
                "right_hip_yaw_joint": 32.0,
                "right_hip_roll_joint": 20.0,
                "right_hip_pitch_joint": 32.0,
                "right_knee_joint": 20.0,
                "right_ankle_pitch_joint": 37.0,
                "right_ankle_roll_joint": 37.0,
            },
            stiffness=0.0,
            damping=0.0,
            armature=0.01,
        ),
        "homie_waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=300.0,
            damping=5.0,
            armature=0.01,
        ),
        "homie_arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness={
                ".*_shoulder_.*": 200.0,
                ".*_elbow_joint": 100.0,
            },
            damping={
                ".*_shoulder_.*": 4.0,
                ".*_elbow_joint": 1.0,
            },
            armature=0.01,
        ),
        "homie_wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness=20.0,
            damping=0.5,
            armature=0.01,
        ),
    },
    joint_sdk_names=OPENHOMIE_G1_SDK_JOINT_NAMES,
)
