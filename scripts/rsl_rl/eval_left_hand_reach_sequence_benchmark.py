"""Benchmark a FreezeBaseReach checkpoint on fixed world-coordinate block sequences."""

import argparse
import csv
import importlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


FIXED_LOCAL_SEQUENCES = {
    "easy": (
        (0.36, 0.14, 0.26),
        (0.42, 0.20, 0.28),
        (0.46, 0.16, 0.24),
        (0.40, 0.24, 0.22),
        (0.34, 0.18, 0.20),
        (0.44, 0.12, 0.26),
    ),
    "medium": (
        (0.48, 0.10, 0.18),
        (0.58, 0.18, 0.16),
        (0.66, 0.26, 0.14),
        (0.62, 0.08, 0.20),
        (0.54, 0.30, 0.12),
        (0.70, 0.16, 0.18),
    ),
    "hard": (
        (0.62, 0.06, 0.16),
        (0.78, 0.18, 0.14),
        (0.90, 0.30, 0.16),
        (0.72, 0.40, 0.12),
        (0.96, 0.12, 0.20),
        (0.82, 0.34, 0.10),
    ),
}

parser = argparse.ArgumentParser(description="Benchmark a left-hand loco-reach checkpoint on fixed block sequences.")
parser.add_argument("--video", action="store_true", default=False, help="Record a benchmark video.")
parser.add_argument("--video_length", type=int, default=2400, help="Recorded video length in simulation steps.")
parser.add_argument(
    "--task",
    type=str,
    default="Unitree-G1-29dof-LeftHand-LocoReach-AdapterAcquireTightStay-NaturalReachSettleShort-FreezeBaseReach-v0",
    help="Benchmark task name.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of benchmark environments to batch.")
parser.add_argument(
    "--difficulty",
    type=str,
    default="all",
    choices=["easy", "medium", "hard", "all"],
    help="Benchmark difficulty group.",
)
parser.add_argument(
    "--mode",
    type=str,
    default="main",
    choices=["main", "stability"],
    help="Benchmark mode. main=10 exact repeats, stability=100 perturbed repeats.",
)
parser.add_argument("--main_repeats", type=int, default=10, help="Repeats per difficulty in main mode.")
parser.add_argument("--stability_repeats", type=int, default=100, help="Repeats per difficulty in stability mode.")
parser.add_argument("--repeats", type=int, default=None, help="Optional explicit repeat override.")
parser.add_argument(
    "--stability_xy_perturb_m",
    type=float,
    default=0.02,
    help="Stability mode reset perturbation magnitude for x/y in meters.",
)
parser.add_argument(
    "--stability_yaw_perturb_deg",
    type=float,
    default=2.0,
    help="Stability mode reset perturbation magnitude for yaw in degrees.",
)
parser.add_argument(
    "--near_target_radius",
    type=float,
    default=0.14,
    help="Radius used to summarize near-target motion quality.",
)
parser.add_argument(
    "--fall_height_threshold",
    type=float,
    default=0.20,
    help="Root height threshold below which the robot is considered fallen.",
)
parser.add_argument(
    "--fall_gravity_threshold",
    type=float,
    default=-0.70,
    help="Projected gravity z threshold above which the robot is considered fallen.",
)
parser.add_argument(
    "--sequence_timeout_s",
    type=float,
    default=40.0,
    help="Hard guard timeout for one full benchmark sequence.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for benchmark outputs. Defaults to <run_dir>/reach_benchmark/<timestamp>.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

freeze_base_reach_env_cfg = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof"
    ".left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach"
    ".left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach_env_cfg"
)
freeze_base_reach_mdp = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof"
    ".left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach"
    ".left_hand_loco_reach_adapter_acquire_tight_stay_natural_reach_settle_short_freeze_base_reach_mdp"
)
fixed_target_mdp = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.left_hand_loco_reach.left_hand_loco_reach_mdp"
)

SETTLE_GRACE_S = float(freeze_base_reach_mdp.SETTLE_GRACE_S)
PER_TARGET_TIMEOUT_S = float(freeze_base_reach_env_cfg.PER_TARGET_TIMEOUT_S)
MAX_TARGETS_PER_EPISODE = int(freeze_base_reach_env_cfg.MAX_TARGETS_PER_EPISODE)

_ORIGINAL_SPAWN_NEW_FIXED_TARGETS = fixed_target_mdp._spawn_new_fixed_targets


SEQUENCE_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/left_hand_loco_reach_benchmark_sequence")
SEQUENCE_MARKER_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


def _sequence_marker_quat(base_env, count: int) -> torch.Tensor:
    cache_key = "_reach_benchmark_marker_quat"
    quat = getattr(base_env, cache_key, None)
    if quat is None or quat.shape[0] != count:
        quat = torch.zeros(count, 4, device=base_env.device)
        quat[:, 0] = 1.0
        setattr(base_env, cache_key, quat)
    return quat


def _update_sequence_debug_visualization(base_env, sequence_w_env0: torch.Tensor):
    if not hasattr(base_env, "_reach_benchmark_sequence_visualizer"):
        base_env._reach_benchmark_sequence_visualizer = VisualizationMarkers(SEQUENCE_MARKER_CFG)
    base_env._reach_benchmark_sequence_visualizer.visualize(
        sequence_w_env0, _sequence_marker_quat(base_env, sequence_w_env0.shape[0])
    )


def _prime_benchmark_target_state(base_env):
    if not hasattr(base_env, "_reach_benchmark_sequence_w"):
        return
    sequence_w = base_env._reach_benchmark_sequence_w
    base_env._left_hand_completed_targets.zero_()
    base_env._left_hand_target_index.zero_()
    base_env._left_hand_target_age_steps.zero_()
    base_env._left_hand_post_switch_steps.zero_()
    base_env._left_hand_prev_success.zero_()
    base_env._left_hand_recent_success.zero_()
    base_env._left_hand_in_success_zone.zero_()
    base_env._left_hand_success_hold_counter.zero_()
    base_env._left_hand_success_zone_time.zero_()
    base_env._left_hand_held_success_count.zero_()
    base_env._left_hand_completion_after_hold.zero_()
    base_env._left_hand_target_switched_this_step.zero_()
    base_env._left_hand_has_active_target[:] = True
    base_env._left_hand_active_target_w[:] = sequence_w[:, 0]
    base_env._left_hand_prev_target_w[:] = sequence_w[:, 0]
    if hasattr(base_env, "_left_hand_distance_at_completion"):
        base_env._left_hand_distance_at_completion.zero_()
    if hasattr(base_env, "_left_hand_foot_motion_before_contact"):
        base_env._left_hand_foot_motion_before_contact.zero_()
    if hasattr(base_env, "_left_hand_workspace_error_at_contact"):
        base_env._left_hand_workspace_error_at_contact.zero_()
    if hasattr(base_env, "_left_hand_torso_lean_at_contact"):
        base_env._left_hand_torso_lean_at_contact.zero_()
    if hasattr(base_env, "_left_hand_arm_extension_at_contact"):
        base_env._left_hand_arm_extension_at_contact.zero_()




def _maybe_tuple_obs(reset_result):
    return reset_result[0] if isinstance(reset_result, tuple) else reset_result


def _difficulty_names() -> list[str]:
    if args_cli.difficulty == "all":
        return ["easy", "medium", "hard"]
    return [args_cli.difficulty]


def _num_repeats() -> int:
    if args_cli.repeats is not None:
        return int(args_cli.repeats)
    return int(args_cli.main_repeats if args_cli.mode == "main" else args_cli.stability_repeats)


def _mode_pose_range() -> dict[str, tuple[float, float]]:
    if args_cli.mode == "stability":
        yaw = math.radians(args_cli.stability_yaw_perturb_deg)
        return {
            "x": (-args_cli.stability_xy_perturb_m, args_cli.stability_xy_perturb_m),
            "y": (-args_cli.stability_xy_perturb_m, args_cli.stability_xy_perturb_m),
            "yaw": (-yaw, yaw),
        }
    return {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


def _resolve_checkpoint_path(agent_cfg: RslRlOnPolicyRunnerCfg) -> str:
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _get_output_dir(resume_path: str) -> str:
    if args_cli.output_dir:
        return os.path.abspath(args_cli.output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(os.path.dirname(resume_path), "reach_benchmark", timestamp)


def _configure_benchmark_env(env_cfg):
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(args_cli.sequence_timeout_s))
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        if hasattr(env_cfg.curriculum, "left_hand_target_levels"):
            env_cfg.curriculum.left_hand_target_levels = None
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    if hasattr(env_cfg, "events"):
        for attr_name in ("physics_material", "add_base_mass", "push_robot", "base_external_force_torque"):
            if hasattr(env_cfg.events, attr_name):
                setattr(env_cfg.events, attr_name, None)
        if hasattr(env_cfg.events, "reset_base"):
            env_cfg.events.reset_base.params["pose_range"] = _mode_pose_range()
            env_cfg.events.reset_base.params["velocity_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        if hasattr(env_cfg.events, "reset_robot_joints"):
            env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    if hasattr(env_cfg, "terminations"):
        for attr_name in ("target_quota", "target_timeout", "time_out", "base_height", "bad_orientation"):
            if hasattr(env_cfg.terminations, attr_name):
                setattr(env_cfg.terminations, attr_name, None)


def _safe_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    return sum(finite) / float(len(finite))


def _benchmark_spawn_new_fixed_targets(env, env_ids, sample_regimes=None, sample_weights=None):
    del sample_regimes, sample_weights
    if len(env_ids) == 0:
        return
    if not hasattr(env, "_reach_benchmark_sequence_w"):
        _ORIGINAL_SPAWN_NEW_FIXED_TARGETS(env, env_ids, sample_regimes=None, sample_weights=None)
        return
    target_indices = torch.clamp(
        env._left_hand_completed_targets[env_ids],
        min=0,
        max=env._reach_benchmark_sequence_w.shape[1] - 1,
    )
    env._left_hand_active_target_w[env_ids] = env._reach_benchmark_sequence_w[env_ids, target_indices]
    env._left_hand_has_active_target[env_ids] = True


def _nominal_root_pos_w(base_env) -> torch.Tensor:
    root_init_pos = getattr(base_env.scene["robot"].cfg.init_state, "pos", (0.0, 0.0, 0.8))
    root_init_pos = torch.tensor(root_init_pos, dtype=torch.float32, device=base_env.device)
    return base_env.scene.env_origins[:, :3] + root_init_pos.unsqueeze(0)


def _sequence_tensor(base_env, difficulty: str) -> torch.Tensor:
    sequence_local = torch.tensor(FIXED_LOCAL_SEQUENCES[difficulty], dtype=torch.float32, device=base_env.device)
    nominal_root_pos_w = _nominal_root_pos_w(base_env)
    return nominal_root_pos_w.unsqueeze(1) + sequence_local.unsqueeze(0)


def _hand_pos_w(robot, hand_body_id: int) -> torch.Tensor:
    return robot.data.body_pos_w[:, hand_body_id]


def _hand_vel_w(robot, hand_body_id: int) -> torch.Tensor:
    return robot.data.body_lin_vel_w[:, hand_body_id]


def _foot_shuffle_metric(robot, foot_body_ids: torch.Tensor) -> torch.Tensor:
    foot_vel_xy = robot.data.body_lin_vel_w[:, foot_body_ids, :2] - robot.data.root_lin_vel_w[:, None, :2]
    return torch.linalg.norm(foot_vel_xy, dim=-1).mean(dim=1)


def _right_arm_deviation(robot, right_arm_joint_ids: torch.Tensor) -> torch.Tensor:
    return torch.sum(
        torch.abs(robot.data.joint_pos[:, right_arm_joint_ids] - robot.data.default_joint_pos[:, right_arm_joint_ids]),
        dim=1,
    )


def _waist_deviation(robot, waist_joint_ids: torch.Tensor) -> torch.Tensor:
    return torch.sum(
        torch.abs(robot.data.joint_pos[:, waist_joint_ids] - robot.data.default_joint_pos[:, waist_joint_ids]),
        dim=1,
    )


def _build_summary(
    records: list[dict],
    difficulty: str,
    sequence_world_xyz_env0: list[list[float]],
) -> dict[str, float | list[float] | str]:
    summary: dict[str, float | list[float] | str] = {
        "difficulty": difficulty,
        "episodes": len(records),
        "sequence_length": MAX_TARGETS_PER_EPISODE,
        "sequence_completion_rate": _safe_mean([float(record["sequence_completed"]) for record in records]),
        "mean_blocks_completed": _safe_mean([record["blocks_completed"] for record in records]),
        "target_timeout_rate": _safe_mean([float(record["failure_reason"] == "target_timeout") for record in records]),
        "fall_rate": _safe_mean([float(record["failure_reason"] == "fall") for record in records]),
        "mean_total_time_s": _safe_mean([record["total_time_s"] for record in records]),
        "mean_final_position_error_m": _safe_mean([record["final_position_error_m"] for record in records]),
        "mean_near_target_hand_speed_mps": _safe_mean([record["mean_near_target_hand_speed_mps"] for record in records]),
        "mean_near_target_base_speed_mps": _safe_mean([record["mean_near_target_base_speed_mps"] for record in records]),
        "mean_near_target_yaw_rate_rps": _safe_mean([record["mean_near_target_yaw_rate_rps"] for record in records]),
        "mean_near_target_foot_shuffle_mps": _safe_mean([record["mean_near_target_foot_shuffle_mps"] for record in records]),
        "mean_near_target_right_arm_deviation": _safe_mean([record["mean_near_target_right_arm_deviation"] for record in records]),
        "mean_near_target_waist_deviation": _safe_mean([record["mean_near_target_waist_deviation"] for record in records]),
        "sequence_local_xyz": [list(point) for point in FIXED_LOCAL_SEQUENCES[difficulty]],
        "sequence_world_xyz_env0": sequence_world_xyz_env0,
    }
    for block_index in range(MAX_TARGETS_PER_EPISODE):
        summary[f"block_{block_index}_success_rate"] = _safe_mean(
            [float(record[f"block_{block_index}_success"]) for record in records]
        )
        summary[f"block_{block_index}_timeout_rate"] = _safe_mean(
            [
                float(
                    (record["failure_reason"] == "target_timeout")
                    and (record["failure_block_index"] == block_index)
                )
                for record in records
            ]
        )
        summary[f"block_{block_index}_time_mean_s"] = _safe_mean(
            [record[f"block_{block_index}_time_s"] for record in records]
        )
    return summary


def main():
    fixed_target_mdp._spawn_new_fixed_targets = _benchmark_spawn_new_fixed_targets
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
            entry_point_key="play_env_cfg_entry_point",
        )
        _configure_benchmark_env(env_cfg)

        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        resume_path = _resolve_checkpoint_path(agent_cfg)
        output_dir = _get_output_dir(resume_path)
        os.makedirs(output_dir, exist_ok=True)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(output_dir, "videos"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=vec_env.device)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        step_dt = float(base_env.step_dt)
        num_envs = base_env.num_envs
        max_steps_per_sequence = max(1, int(round(args_cli.sequence_timeout_s / step_dt)))
        per_target_timeout_steps = max(1, int(round(PER_TARGET_TIMEOUT_S / step_dt)))
        settle_grace_steps = max(0, int(round(SETTLE_GRACE_S / step_dt)))

        hand_body_id = robot.find_bodies([freeze_base_reach_env_cfg.LEFT_HAND_BODY_NAME], preserve_order=True)[0][0]
        foot_body_ids = torch.tensor(
            robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0],
            dtype=torch.long,
            device=base_env.device,
        )
        right_arm_joint_ids = torch.tensor(
            robot.find_joints(
                [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                preserve_order=True,
            )[0],
            dtype=torch.long,
            device=base_env.device,
        )
        waist_joint_ids = torch.tensor(
            robot.find_joints(["waist_yaw_joint"], preserve_order=True)[0],
            dtype=torch.long,
            device=base_env.device,
        )

        print("[INFO] Benchmark configuration:")
        print(f"  checkpoint: {resume_path}")
        print(f"  output_dir: {output_dir}")
        print(f"  mode: {args_cli.mode}")
        print(f"  difficulty: {args_cli.difficulty}")
        print(f"  repeats_per_difficulty: {_num_repeats()}")
        print(f"  reset_pose_range: {_mode_pose_range()}")
        print(f"  sequence_timeout_s: {args_cli.sequence_timeout_s:.2f}")
        print(f"  per_target_timeout_s: {PER_TARGET_TIMEOUT_S:.2f}")
        print(f"  settle_grace_s: {SETTLE_GRACE_S:.2f}")
        print(f"  near_target_radius: {args_cli.near_target_radius:.3f}")

        all_records: list[dict] = []
        summaries: dict[str, dict] = {}
        difficulty_names = _difficulty_names()
        repeats = _num_repeats()

        for difficulty in difficulty_names:
            difficulty_records: list[dict] = []
            sequence_world_xyz_env0: list[list[float]] | None = None
            remaining = repeats
            batch_index = 0
            while remaining > 0:
                batch_size = min(remaining, num_envs)
                active_mask = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
                active_mask[:batch_size] = True

                base_env._reach_benchmark_sequence_w = _sequence_tensor(base_env, difficulty)
                sequence_world_xyz_env0 = base_env._reach_benchmark_sequence_w[0].detach().cpu().tolist()
                print(
                    "[REACH_BENCH] "
                    f"difficulty={difficulty} "
                    f"sequence_local_first={list(FIXED_LOCAL_SEQUENCES[difficulty][0])} "
                    f"sequence_world_env0_first={sequence_world_xyz_env0[0]}"
                )
                with torch.inference_mode():
                    obs = _maybe_tuple_obs(vec_env.reset())
                    _prime_benchmark_target_state(base_env)
                    _update_sequence_debug_visualization(base_env, base_env._reach_benchmark_sequence_w[0])
                    fixed_target_mdp._update_target_debug_visualization(base_env)

                completed_prev = base_env._left_hand_completed_targets.clone()
                block_start_time_s = torch.zeros(num_envs, device=base_env.device)
                sequence_completed = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
                failure_reason = ["" for _ in range(num_envs)]
                failure_block_index = torch.full((num_envs,), -1, dtype=torch.long, device=base_env.device)
                done_mask = ~active_mask.clone()
                total_time_s = torch.zeros(num_envs, device=base_env.device)

                near_target_count = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
                near_target_hand_speed_sum = torch.zeros(num_envs, device=base_env.device)
                near_target_base_speed_sum = torch.zeros(num_envs, device=base_env.device)
                near_target_yaw_rate_sum = torch.zeros(num_envs, device=base_env.device)
                near_target_foot_shuffle_sum = torch.zeros(num_envs, device=base_env.device)
                near_target_right_arm_dev_sum = torch.zeros(num_envs, device=base_env.device)
                near_target_waist_dev_sum = torch.zeros(num_envs, device=base_env.device)

                block_success = torch.zeros((num_envs, MAX_TARGETS_PER_EPISODE), dtype=torch.bool, device=base_env.device)
                block_time_s = torch.full(
                    (num_envs, MAX_TARGETS_PER_EPISODE),
                    float("nan"),
                    device=base_env.device,
                )

                final_position_error = torch.full((num_envs,), float("nan"), device=base_env.device)

                for step in range(max_steps_per_sequence):
                    start_time = time.time()
                    with torch.inference_mode():
                        actions = policy(obs)
                        actions = actions.clone()
                        actions[~active_mask] = 0.0
                        obs, _, _, _ = vec_env.step(actions)

                    hand_pos_w = _hand_pos_w(robot, hand_body_id)
                    hand_vel_w = _hand_vel_w(robot, hand_body_id)
                    position_error = torch.linalg.norm(base_env._left_hand_active_target_w - hand_pos_w, dim=-1)
                    hand_speed = torch.linalg.norm(hand_vel_w, dim=-1)
                    base_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
                    yaw_rate = torch.abs(robot.data.root_ang_vel_w[:, 2])
                    foot_shuffle = _foot_shuffle_metric(robot, foot_body_ids)
                    right_arm_dev = _right_arm_deviation(robot, right_arm_joint_ids)
                    waist_dev = _waist_deviation(robot, waist_joint_ids)
                    root_height = robot.data.root_pos_w[:, 2]
                    projected_gravity_z = robot.data.projected_gravity_b[:, 2]

                    tracked_mask = active_mask & (~done_mask)
                    near_target_mask = tracked_mask & (position_error <= args_cli.near_target_radius)
                    near_target_count += near_target_mask.long()
                    near_target_hand_speed_sum += hand_speed * near_target_mask.float()
                    near_target_base_speed_sum += base_speed * near_target_mask.float()
                    near_target_yaw_rate_sum += yaw_rate * near_target_mask.float()
                    near_target_foot_shuffle_sum += foot_shuffle * near_target_mask.float()
                    near_target_right_arm_dev_sum += right_arm_dev * near_target_mask.float()
                    near_target_waist_dev_sum += waist_dev * near_target_mask.float()

                    completed_now = base_env._left_hand_completed_targets.clone()
                    newly_completed = tracked_mask & (completed_now > completed_prev)
                    if torch.any(newly_completed):
                        done_ids = torch.nonzero(newly_completed, as_tuple=False).squeeze(-1)
                        now_time_s = (step + 1) * step_dt
                        for env_id in done_ids.tolist():
                            block_index = int(completed_now[env_id].item()) - 1
                            if 0 <= block_index < MAX_TARGETS_PER_EPISODE:
                                block_success[env_id, block_index] = True
                                block_time_s[env_id, block_index] = now_time_s - block_start_time_s[env_id]
                                block_start_time_s[env_id] = now_time_s
                        completed_prev = torch.maximum(completed_prev, completed_now)

                    effective_timeout_steps = per_target_timeout_steps + settle_grace_steps * (
                        base_env._left_hand_success_zone_time > 0
                    ).long()
                    sequence_success_mask = tracked_mask & (completed_now >= MAX_TARGETS_PER_EPISODE)
                    fall_mask = tracked_mask & (
                        (root_height < args_cli.fall_height_threshold)
                        | (projected_gravity_z > args_cli.fall_gravity_threshold)
                    )
                    timeout_mask = tracked_mask & (base_env._left_hand_target_age_steps >= effective_timeout_steps)

                    if torch.any(sequence_success_mask):
                        env_ids = torch.nonzero(sequence_success_mask, as_tuple=False).squeeze(-1)
                        sequence_completed[env_ids] = True
                        final_position_error[env_ids] = position_error[env_ids]
                        total_time_s[env_ids] = (step + 1) * step_dt
                        done_mask[env_ids] = True
                        active_mask[env_ids] = False

                    unresolved_fall = fall_mask & (~done_mask)
                    if torch.any(unresolved_fall):
                        env_ids = torch.nonzero(unresolved_fall, as_tuple=False).squeeze(-1)
                        final_position_error[env_ids] = position_error[env_ids]
                        total_time_s[env_ids] = (step + 1) * step_dt
                        failure_block_index[env_ids] = completed_now[env_ids]
                        done_mask[env_ids] = True
                        active_mask[env_ids] = False
                        for env_id in env_ids.tolist():
                            failure_reason[env_id] = "fall"

                    unresolved_timeout = timeout_mask & (~done_mask)
                    if torch.any(unresolved_timeout):
                        env_ids = torch.nonzero(unresolved_timeout, as_tuple=False).squeeze(-1)
                        final_position_error[env_ids] = position_error[env_ids]
                        total_time_s[env_ids] = (step + 1) * step_dt
                        failure_block_index[env_ids] = completed_now[env_ids]
                        done_mask[env_ids] = True
                        active_mask[env_ids] = False
                        for env_id in env_ids.tolist():
                            failure_reason[env_id] = "target_timeout"

                    if torch.all(done_mask[:batch_size]):
                        break

                    sleep_time = step_dt - (time.time() - start_time)
                    if args_cli.real_time and sleep_time > 0.0:
                        time.sleep(sleep_time)

                for env_id in range(batch_size):
                    if not math.isfinite(float(final_position_error[env_id].item())):
                        final_position_error[env_id] = position_error[env_id]
                        total_time_s[env_id] = max(float(total_time_s[env_id].item()), (step + 1) * step_dt)
                    near_count = max(int(near_target_count[env_id].item()), 1)
                    record = {
                        "difficulty": difficulty,
                        "mode": args_cli.mode,
                        "repeat_index": len(difficulty_records),
                        "sequence_completed": bool(sequence_completed[env_id].item()),
                        "blocks_completed": int(completed_prev[env_id].item()),
                        "failure_reason": failure_reason[env_id],
                        "failure_block_index": int(failure_block_index[env_id].item()),
                        "total_time_s": float(total_time_s[env_id].item()),
                        "final_position_error_m": float(final_position_error[env_id].item()),
                        "mean_near_target_hand_speed_mps": float(near_target_hand_speed_sum[env_id].item() / near_count),
                        "mean_near_target_base_speed_mps": float(near_target_base_speed_sum[env_id].item() / near_count),
                        "mean_near_target_yaw_rate_rps": float(near_target_yaw_rate_sum[env_id].item() / near_count),
                        "mean_near_target_foot_shuffle_mps": float(near_target_foot_shuffle_sum[env_id].item() / near_count),
                        "mean_near_target_right_arm_deviation": float(near_target_right_arm_dev_sum[env_id].item() / near_count),
                        "mean_near_target_waist_deviation": float(near_target_waist_dev_sum[env_id].item() / near_count),
                    }
                    for block_index in range(MAX_TARGETS_PER_EPISODE):
                        record[f"block_{block_index}_success"] = bool(block_success[env_id, block_index].item())
                        record[f"block_{block_index}_time_s"] = float(block_time_s[env_id, block_index].item())
                    difficulty_records.append(record)

                remaining -= batch_size
                batch_index += 1
                print(
                    "[REACH_BENCH] "
                    f"difficulty={difficulty} batch={batch_index} remaining={remaining} "
                    f"completed={sum(int(record['sequence_completed']) for record in difficulty_records)}/{len(difficulty_records)}"
                )

            summaries[difficulty] = _build_summary(
                difficulty_records,
                difficulty,
                sequence_world_xyz_env0 if sequence_world_xyz_env0 is not None else [],
            )
            all_records.extend(difficulty_records)

        overall_summary = {
            "task": args_cli.task,
            "mode": args_cli.mode,
            "checkpoint": resume_path,
            "difficulty": args_cli.difficulty,
            "repeats_per_difficulty": repeats,
            "num_envs": args_cli.num_envs,
            "reset_pose_range": _mode_pose_range(),
            "per_target_timeout_s": PER_TARGET_TIMEOUT_S,
            "settle_grace_s": SETTLE_GRACE_S,
            "near_target_radius": args_cli.near_target_radius,
            "summaries": summaries,
            "overall": {
                "episodes": len(all_records),
                "sequence_completion_rate": _safe_mean([float(record["sequence_completed"]) for record in all_records]),
                "mean_blocks_completed": _safe_mean([record["blocks_completed"] for record in all_records]),
                "target_timeout_rate": _safe_mean([float(record["failure_reason"] == "target_timeout") for record in all_records]),
                "fall_rate": _safe_mean([float(record["failure_reason"] == "fall") for record in all_records]),
                "mean_total_time_s": _safe_mean([record["total_time_s"] for record in all_records]),
                "mean_final_position_error_m": _safe_mean([record["final_position_error_m"] for record in all_records]),
            },
        }

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(overall_summary, file, indent=2)

        csv_path = os.path.join(output_dir, "episodes.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(all_records[0].keys()) if all_records else [])
            if all_records:
                writer.writeheader()
                writer.writerows(all_records)

        print(f"[REACH_BENCH] Summary written to: {summary_path}")
        print(f"[REACH_BENCH] Episode records written to: {csv_path}")
    finally:
        fixed_target_mdp._spawn_new_fixed_targets = _ORIGINAL_SPAWN_NEW_FIXED_TARGETS


if __name__ == "__main__":
    main()
    simulation_app.close()
