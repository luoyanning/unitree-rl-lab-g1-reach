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
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


FIXED_ENV0_WORLD_SEQUENCES = {
    "easy": (
        (0.45, 0.00, 1.02),
        (0.95, 0.00, 1.02),
        (1.45, 0.00, 1.02),
        (1.95, 0.00, 1.02),
        (2.45, 0.00, 1.02),
        (2.95, 0.00, 1.02),
    ),
    "medium": (
        (0.45, 0.00, 0.90),
        (1.015685, 0.00, 1.10),
        (1.581370, 0.00, 0.90),
        (2.147055, 0.00, 1.10),
        (2.712740, 0.00, 0.90),
        (3.278425, 0.00, 1.10),
    ),
    "hard": (
        (0.45, 0.00, 0.90),
        (1.113325, 0.00, 1.25),
        (1.776650, 0.00, 0.90),
        (2.439975, 0.00, 1.25),
        (3.103300, 0.00, 0.90),
        (3.766625, 0.00, 1.25),
    ),
}

BENCHMARK_BLOCK_COLORS = (
    (0.92, 0.30, 0.24),
    (0.95, 0.56, 0.20),
    (0.92, 0.80, 0.22),
    (0.30, 0.76, 0.36),
    (0.24, 0.56, 0.92),
    (0.66, 0.34, 0.90),
)

BENCHMARK_BLOCK_VISUAL_SIZE_M = 0.14
BENCHMARK_BLOCK_TOUCH_MARGIN_M = 0.02
DEFAULT_BENCHMARK_PER_TARGET_TIMEOUT_S = 10.0
DEFAULT_BENCHMARK_HOLD_TIME_S = 2.0
DEFAULT_BENCHMARK_HOLD_MARGIN_M = 0.04
BENCHMARK_ENV_TASK = "Unitree-G1-29dof-LeftHand-LocoReach-FreezeBaseReach-Benchmark-v0"

parser = argparse.ArgumentParser(description="Benchmark a left-hand loco-reach checkpoint on fixed block sequences.")
parser.add_argument("--video", action="store_true", default=False, help="Record a benchmark video.")
parser.add_argument("--video_length", type=int, default=2400, help="Recorded video length in simulation steps.")
parser.add_argument("--video_fps", type=int, default=20, help="Encoded benchmark video FPS.")
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
    "--benchmark_per_target_timeout_s",
    type=float,
    default=None,
    help="Optional benchmark-only per-target timeout override in seconds. Default uses difficulty-specific values.",
)
parser.add_argument(
    "--benchmark_hold_time_s",
    type=float,
    default=DEFAULT_BENCHMARK_HOLD_TIME_S,
    help="Continuous hold time required after first touch before a block counts as stabilized.",
)
parser.add_argument(
    "--benchmark_hold_margin_m",
    type=float,
    default=DEFAULT_BENCHMARK_HOLD_MARGIN_M,
    help="Extra half-extent margin used for the post-touch hold zone around a block.",
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

import omni.usd
from pxr import Gf, UsdGeom

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
TRAINING_PER_TARGET_TIMEOUT_S = float(freeze_base_reach_env_cfg.PER_TARGET_TIMEOUT_S)
MAX_TARGETS_PER_EPISODE = int(freeze_base_reach_env_cfg.MAX_TARGETS_PER_EPISODE)

_ORIGINAL_SPAWN_NEW_FIXED_TARGETS = fixed_target_mdp._spawn_new_fixed_targets
_ORIGINAL_SYNC_LONG_HORIZON_STATE = fixed_target_mdp._sync_long_horizon_state
_ORIGINAL_FREEZE_SYNC_ADAPTER_HOLD_STAY_STATE = freeze_base_reach_mdp._sync_adapter_hold_stay_state


def _pairwise_distances(points_xyz: list[list[float]]) -> list[float]:
    distances: list[float] = []
    for index in range(len(points_xyz) - 1):
        distances.append(math.dist(points_xyz[index], points_xyz[index + 1]))
    return distances


def _touch_half_extent_m() -> float:
    return 0.5 * BENCHMARK_BLOCK_VISUAL_SIZE_M + BENCHMARK_BLOCK_TOUCH_MARGIN_M


def _hand_touches_block(hand_pos_w: torch.Tensor, block_pos_w: torch.Tensor) -> torch.Tensor:
    half_extent = _touch_half_extent_m()
    return torch.all(torch.abs(hand_pos_w - block_pos_w) <= half_extent, dim=-1)


def _hold_half_extent_m() -> float:
    return 0.5 * BENCHMARK_BLOCK_VISUAL_SIZE_M + float(args_cli.benchmark_hold_margin_m)


def _hand_in_hold_zone(hand_pos_w: torch.Tensor, block_pos_w: torch.Tensor) -> torch.Tensor:
    half_extent = _hold_half_extent_m()
    return torch.all(torch.abs(hand_pos_w - block_pos_w) <= half_extent, dim=-1)


def _update_sequence_block_visualization(base_env, sequence_w_env0: torch.Tensor):
    stage = omni.usd.get_context().get_stage()
    block_prims = getattr(base_env, "_reach_benchmark_block_prims", None)
    if block_prims is None:
        block_prims = []
        for block_index, color in enumerate(BENCHMARK_BLOCK_COLORS):
            prim_path = f"/Visuals/Command/left_hand_loco_reach_benchmark_blocks/block_{block_index}"
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(BENCHMARK_BLOCK_VISUAL_SIZE_M)
            cube_prim = cube.GetPrim()
            cube_xform = UsdGeom.XformCommonAPI(cube_prim)
            cube_xform.SetScale(Gf.Vec3f(1.0, 1.0, 1.0))
            UsdGeom.Gprim(cube_prim).CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
                [Gf.Vec3f(*color)]
            )
            block_prims.append(cube_prim)
        base_env._reach_benchmark_block_prims = block_prims
    for block_prim, position in zip(block_prims, sequence_w_env0.tolist(), strict=True):
        cube_xform = UsdGeom.XformCommonAPI(block_prim)
        cube_xform.SetTranslate(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))


def _activate_benchmark_block(base_env, env_ids: torch.Tensor, block_indices: torch.Tensor):
    if len(env_ids) == 0:
        return
    with torch.inference_mode():
        valid_mask = block_indices < MAX_TARGETS_PER_EPISODE
        if torch.any(valid_mask):
            valid_env_ids = env_ids[valid_mask]
            valid_block_indices = block_indices[valid_mask]
            base_env._left_hand_has_active_target[valid_env_ids] = True
            base_env._left_hand_active_target_w[valid_env_ids] = base_env._reach_benchmark_sequence_w[
                valid_env_ids, valid_block_indices
            ]
            base_env._left_hand_prev_target_w[valid_env_ids] = base_env._left_hand_active_target_w[valid_env_ids]
            base_env._left_hand_target_age_steps[valid_env_ids] = 0
            base_env._left_hand_post_switch_steps[valid_env_ids] = 0
            base_env._left_hand_prev_success[valid_env_ids] = False
            base_env._left_hand_recent_success[valid_env_ids] = False
            base_env._left_hand_in_success_zone[valid_env_ids] = False
            base_env._left_hand_success_hold_counter[valid_env_ids] = 0
            base_env._left_hand_success_zone_time[valid_env_ids] = 0
            base_env._left_hand_held_success_count[valid_env_ids] = 0
            base_env._left_hand_completion_after_hold[valid_env_ids] = 0
            base_env._left_hand_target_switched_this_step[valid_env_ids] = True
        if torch.any(~valid_mask):
            done_env_ids = env_ids[~valid_mask]
            base_env._left_hand_has_active_target[done_env_ids] = False


def _benchmark_sync_long_horizon_state(
    env,
    command_name: str,
    success_threshold: float,
    max_targets_per_episode: int,
    switch_phase_steps: int,
    static_target_hold_s: float,
    per_target_timeout_s: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    sample_regimes,
    sample_weights,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
):
    del success_threshold, static_target_hold_s, per_target_timeout_s, sample_regimes, sample_weights, success_exit_radius, success_hold_steps
    fixed_target_mdp._ensure_long_horizon_state(
        env,
        command_name=command_name,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
    )
    freeze_base_reach_mdp._ensure_adapter_state(env)
    if env._left_hand_state_synced_step == env.common_step_counter:
        return

    reset_ids, current_episode_length, prev_episode_length = fixed_target_mdp._compute_just_reset_mask(env)
    env._left_hand_prev_episode_length_buf = current_episode_length.clone()
    env._left_hand_just_reset_this_step[:] = reset_ids
    env._left_hand_recent_success.zero_()
    env._left_hand_completion_after_hold.zero_()
    env._left_hand_target_switched_this_step.zero_()

    if torch.any(reset_ids):
        env._left_hand_completed_targets[reset_ids] = 0
        env._left_hand_held_success_count[reset_ids] = 0
        env._left_hand_target_index[reset_ids] = 0
        env._left_hand_post_switch_steps[reset_ids] = switch_phase_steps
        env._left_hand_target_age_steps[reset_ids] = 0
        env._left_hand_prev_success[reset_ids] = False
        env._left_hand_recent_success[reset_ids] = False
        env._left_hand_in_success_zone[reset_ids] = False
        env._left_hand_success_hold_counter[reset_ids] = 0
        env._left_hand_success_zone_time[reset_ids] = 0
        env._left_hand_completion_after_hold[reset_ids] = False
        env._left_hand_has_active_target[reset_ids] = False
        env._left_hand_in_post_success_dwell[reset_ids] = False
        env._left_hand_post_success_dwell_counter[reset_ids] = 0
        env._left_hand_recent_dwell_completion[reset_ids] = False

    switch_detected = torch.norm(env._left_hand_active_target_w - env._left_hand_prev_target_w, dim=-1) > 1.0e-5
    switch_detected |= reset_ids
    env._left_hand_post_switch_steps = torch.clamp(env._left_hand_post_switch_steps - 1, min=0)
    env._left_hand_target_age_steps += env._left_hand_has_active_target.long()
    env._left_hand_target_index[:] = torch.clamp(env._left_hand_completed_targets, max=max_targets_per_episode - 1)
    env._left_hand_post_switch_steps[switch_detected] = switch_phase_steps
    env._left_hand_target_age_steps[switch_detected] = 0
    env._left_hand_in_success_zone[switch_detected] = False
    env._left_hand_success_hold_counter[switch_detected] = 0
    env._left_hand_success_zone_time[switch_detected] = 0
    env._left_hand_target_switched_this_step[:] = switch_detected

    fixed_target_mdp._set_base_velocity_guidance_command(env, x_range=x_range, y_range=y_range)

    command_term = env.command_manager.get_term(command_name)
    if hasattr(command_term, "metrics"):
        command_term.metrics["success_zone_flag"][:] = env._left_hand_in_success_zone.float()
        command_term.metrics["success_hold_counter"][:] = env._left_hand_success_hold_counter.float()
        command_term.metrics["success_zone_time"][:] = env._left_hand_success_zone_time.float()
        command_term.metrics["held_success_count"][:] = env._left_hand_held_success_count.float()
        command_term.metrics["completion_distance"][:] = 0.0
        command_term.metrics["completion_after_hold"][:] = env._left_hand_completion_after_hold.float()
        command_term.metrics.setdefault("post_success_dwell_flag", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("post_success_dwell_counter", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("completion_after_dwell", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics["post_success_dwell_flag"][:] = env._left_hand_in_post_success_dwell.float()
        command_term.metrics["post_success_dwell_counter"][:] = env._left_hand_post_success_dwell_counter.float()
        command_term.metrics["completion_after_dwell"][:] = env._left_hand_recent_dwell_completion.float()

    adapter_command = fixed_target_mdp._active_target_pos_base_yaw(env)
    env._left_hand_adapter_command[:] = adapter_command
    pose_command = freeze_base_reach_mdp._command_tensor(env, command_name)
    if pose_command is not None and pose_command.shape[1] >= 3:
        pose_command[:, :3] = adapter_command
        if pose_command.shape[1] >= 6:
            pose_command[:, 3:6] = 0.0

    env._left_hand_prev_target_w = env._left_hand_active_target_w.clone()
    env._left_hand_prev_success.zero_()
    env._left_hand_state_synced_step = env.common_step_counter


def _benchmark_sync_adapter_hold_stay_state(
    env,
    command_name: str,
    success_threshold: float,
    max_targets_per_episode: int,
    switch_phase_steps: int,
    static_target_hold_s: float,
    per_target_timeout_s: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    sample_regimes,
    sample_weights,
    **kwargs,
):
    del kwargs
    _benchmark_sync_long_horizon_state(
        env=env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
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


def _benchmark_per_target_timeout_s(difficulty: str) -> float:
    if args_cli.benchmark_per_target_timeout_s is not None:
        return float(args_cli.benchmark_per_target_timeout_s)
    return float(DEFAULT_BENCHMARK_PER_TARGET_TIMEOUT_S)


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
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    if os.path.isdir(resume_path):
        raise ValueError(
            f"--checkpoint resolved to a directory, not a model file: {resume_path}. "
            "Pass an explicit model_XXXX.pt path."
        )
    return resume_path


def _benchmark_env_task_name() -> str:
    return BENCHMARK_ENV_TASK


def _get_output_dir(resume_path: str) -> str:
    if args_cli.output_dir:
        return os.path.abspath(args_cli.output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(os.path.dirname(resume_path), "reach_benchmark", timestamp)


def _latest_video_path(output_dir: str) -> str | None:
    video_dir = os.path.join(output_dir, "videos")
    if not os.path.isdir(video_dir):
        return None
    video_paths: list[str] = []
    for root, _, files in os.walk(video_dir):
        for file_name in files:
            if file_name.endswith(".mp4"):
                video_paths.append(os.path.join(root, file_name))
    if not video_paths:
        return None
    return sorted(video_paths)[-1]


def _init_manual_video_writer(output_dir: str, difficulty: str, repeat_index: int):
    if imageio is None:
        raise RuntimeError("imageio is required for --video but is not available in the current environment.")
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{difficulty}_repeat_{repeat_index:03d}.mp4")
    writer = imageio.get_writer(
        video_path,
        fps=int(args_cli.video_fps),
        codec="libx264",
        macro_block_size=None,
    )
    return writer, video_path


def _render_rgb_frame(env):
    frame = env.render()
    if isinstance(frame, (list, tuple)):
        frame = frame[0]
    return frame


def _configure_benchmark_env(env_cfg):
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(args_cli.sequence_timeout_s))
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "terrain"):
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        if hasattr(env_cfg.scene, "env_spacing"):
            env_cfg.scene.env_spacing = 12.0
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (8.5, -4.8, 2.8)
        env_cfg.viewer.lookat = (1.8, 0.0, 1.0)
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        if hasattr(env_cfg.curriculum, "left_hand_target_levels"):
            env_cfg.curriculum.left_hand_target_levels = None
    if hasattr(env_cfg, "rewards") and env_cfg.rewards is not None:
        for attr_name in list(vars(env_cfg.rewards).keys()):
            if not attr_name.startswith("_"):
                setattr(env_cfg.rewards, attr_name, None)
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


def _sequence_tensor(base_env, difficulty: str) -> torch.Tensor:
    sequence_local = torch.tensor(
        FIXED_ENV0_WORLD_SEQUENCES[difficulty], dtype=torch.float32, device=base_env.device
    )
    env_origins = base_env.scene.env_origins[:, :3]
    return env_origins.unsqueeze(1) + sequence_local.unsqueeze(0)


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
        "benchmark_per_target_timeout_s": _benchmark_per_target_timeout_s(difficulty),
        "benchmark_hold_time_s": float(args_cli.benchmark_hold_time_s),
        "sequence_completion_rate": _safe_mean([float(record["sequence_completed"]) for record in records]),
        "mean_blocks_touched": _safe_mean([record["blocks_touched"] for record in records]),
        "mean_blocks_stabilized": _safe_mean([record["blocks_stabilized"] for record in records]),
        "mean_blocks_completed": _safe_mean([record["blocks_completed"] for record in records]),
        "target_timeout_rate": _safe_mean(
            [
                float(
                    any(
                        record[f"block_{idx}_status"] in ("timeout", "touched_timeout")
                        for idx in range(MAX_TARGETS_PER_EPISODE)
                    )
                )
                for record in records
            ]
        ),
        "fall_rate": _safe_mean([float(record["failure_reason"] == "fall") for record in records]),
        "mean_total_time_s": _safe_mean([record["total_time_s"] for record in records]),
        "mean_final_position_error_m": _safe_mean([record["final_position_error_m"] for record in records]),
        "mean_near_target_hand_speed_mps": _safe_mean([record["mean_near_target_hand_speed_mps"] for record in records]),
        "mean_near_target_base_speed_mps": _safe_mean([record["mean_near_target_base_speed_mps"] for record in records]),
        "mean_near_target_yaw_rate_rps": _safe_mean([record["mean_near_target_yaw_rate_rps"] for record in records]),
        "mean_near_target_foot_shuffle_mps": _safe_mean([record["mean_near_target_foot_shuffle_mps"] for record in records]),
        "mean_near_target_right_arm_deviation": _safe_mean([record["mean_near_target_right_arm_deviation"] for record in records]),
        "mean_near_target_waist_deviation": _safe_mean([record["mean_near_target_waist_deviation"] for record in records]),
        "sequence_world_xyz_env0": sequence_world_xyz_env0,
        "sequence_pairwise_distances_m": _pairwise_distances(sequence_world_xyz_env0),
    }
    for block_index in range(MAX_TARGETS_PER_EPISODE):
        summary[f"block_{block_index}_touch_rate"] = _safe_mean(
            [float(record[f"block_{block_index}_touched"]) for record in records]
        )
        summary[f"block_{block_index}_stabilize_rate"] = _safe_mean(
            [float(record[f"block_{block_index}_stabilized"]) for record in records]
        )
        summary[f"block_{block_index}_success_rate"] = _safe_mean(
            [float(record[f"block_{block_index}_success"]) for record in records]
        )
        summary[f"block_{block_index}_timeout_rate"] = _safe_mean(
            [
                float(record[f"block_{block_index}_status"] in ("timeout", "touched_timeout"))
                for record in records
            ]
        )
        summary[f"block_{block_index}_touch_time_mean_s"] = _safe_mean(
            [record[f"block_{block_index}_touch_time_s"] for record in records]
        )
        summary[f"block_{block_index}_stabilize_time_mean_s"] = _safe_mean(
            [record[f"block_{block_index}_stabilize_time_s"] for record in records]
        )
        summary[f"block_{block_index}_max_hold_mean_s"] = _safe_mean(
            [record[f"block_{block_index}_max_hold_s"] for record in records]
        )
        summary[f"block_{block_index}_elapsed_mean_s"] = _safe_mean(
            [record[f"block_{block_index}_elapsed_s"] for record in records]
        )
    return summary


def main():
    fixed_target_mdp._spawn_new_fixed_targets = _benchmark_spawn_new_fixed_targets
    fixed_target_mdp._sync_long_horizon_state = _benchmark_sync_long_horizon_state
    freeze_base_reach_mdp._sync_adapter_hold_stay_state = _benchmark_sync_adapter_hold_stay_state
    env = None
    vec_env = None
    video_writer = None
    try:
        benchmark_env_task = _benchmark_env_task_name()
        env_cfg = parse_env_cfg(
            benchmark_env_task,
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

        env = gym.make(benchmark_env_task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=vec_env.device)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        step_dt = float(base_env.step_dt)
        num_envs = base_env.num_envs
        base_max_steps_per_sequence = max(1, int(round(args_cli.sequence_timeout_s / step_dt)))
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
        print(f"  policy_task: {args_cli.task}")
        print(f"  benchmark_env_task: {benchmark_env_task}")
        print(f"  mode: {args_cli.mode}")
        print(f"  difficulty: {args_cli.difficulty}")
        print(f"  repeats_per_difficulty: {_num_repeats()}")
        print(f"  reset_pose_range: {_mode_pose_range()}")
        print(f"  sequence_timeout_s: {args_cli.sequence_timeout_s:.2f}")
        if args_cli.video:
            print(f"  video_fps: {args_cli.video_fps}")
        print(f"  training_per_target_timeout_s: {TRAINING_PER_TARGET_TIMEOUT_S:.2f}")
        print(f"  benchmark_per_target_timeout_default_s: {DEFAULT_BENCHMARK_PER_TARGET_TIMEOUT_S:.2f}")
        if args_cli.benchmark_per_target_timeout_s is not None:
            print(f"  benchmark_per_target_timeout_override_s: {args_cli.benchmark_per_target_timeout_s:.2f}")
        print(f"  benchmark_hold_time_s: {args_cli.benchmark_hold_time_s:.2f}")
        print(f"  benchmark_hold_margin_m: {args_cli.benchmark_hold_margin_m:.3f}")
        print(f"  settle_grace_s: {SETTLE_GRACE_S:.2f}")
        print(f"  near_target_radius: {args_cli.near_target_radius:.3f}")

        all_records: list[dict] = []
        summaries: dict[str, dict] = {}
        difficulty_names = _difficulty_names()
        repeats = _num_repeats()

        for difficulty in difficulty_names:
            difficulty_records: list[dict] = []
            sequence_world_xyz_env0: list[list[float]] | None = None
            benchmark_per_target_timeout_s = _benchmark_per_target_timeout_s(difficulty)
            per_target_timeout_steps = max(1, int(round(benchmark_per_target_timeout_s / step_dt)))
            hold_steps_required = max(1, int(round(float(args_cli.benchmark_hold_time_s) / step_dt)))
            max_steps_per_sequence = max(base_max_steps_per_sequence, MAX_TARGETS_PER_EPISODE * per_target_timeout_steps)
            remaining = repeats
            batch_index = 0
            while remaining > 0:
                batch_size = min(remaining, num_envs)
                active_mask = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
                active_mask[:batch_size] = True

                base_env._reach_benchmark_sequence_w = _sequence_tensor(base_env, difficulty)
                sequence_world_xyz_env0 = base_env._reach_benchmark_sequence_w[0].detach().cpu().tolist()
                sequence_local_spec = [list(point) for point in FIXED_ENV0_WORLD_SEQUENCES[difficulty]]
                env_origin_env0 = base_env.scene.env_origins[0, :3].detach().cpu().tolist()
                sequence_pairwise_distances = _pairwise_distances(sequence_world_xyz_env0)
                print(
                    "[REACH_BENCH] "
                    f"difficulty={difficulty} "
                    f"env_origin_env0={env_origin_env0}"
                )
                print(
                    "[REACH_BENCH] "
                    f"difficulty={difficulty} "
                    f"sequence_local_spec={sequence_local_spec}"
                )
                print(
                    "[REACH_BENCH] "
                    f"difficulty={difficulty} "
                    f"sequence_world_env0={sequence_world_xyz_env0}"
                )
                print(
                    "[REACH_BENCH] "
                    f"difficulty={difficulty} "
                    f"sequence_pairwise_distances_m={sequence_pairwise_distances}"
                )
                print(
                    "[REACH_BENCH] "
                    f"difficulty={difficulty} "
                    f"benchmark_per_target_timeout_s={benchmark_per_target_timeout_s:.2f}"
                )
                print(
                    "[REACH_BENCH] "
                    f"difficulty={difficulty} "
                    f"benchmark_hold_time_s={float(args_cli.benchmark_hold_time_s):.2f}"
                )
                with torch.inference_mode():
                    obs = _maybe_tuple_obs(vec_env.reset())
                    _prime_benchmark_target_state(base_env)
                    _update_sequence_block_visualization(base_env, base_env._reach_benchmark_sequence_w[0])
                    active_env_ids = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
                    _activate_benchmark_block(
                        base_env,
                        active_env_ids,
                        torch.zeros(batch_size, dtype=torch.long, device=base_env.device),
                    )
                robot_root_env0 = robot.data.root_pos_w[0].detach().cpu().tolist()
                hand_pos_env0 = robot.data.body_pos_w[0, hand_body_id].detach().cpu().tolist()
                initial_target_env0 = base_env._left_hand_active_target_w[0].detach().cpu().tolist()
                initial_position_error_env0 = float(
                    torch.linalg.norm(base_env._left_hand_active_target_w[0] - robot.data.body_pos_w[0, hand_body_id]).item()
                )
                print(
                    "[REACH_BENCH] "
                    f"difficulty={difficulty} "
                    f"robot_root_env0={robot_root_env0}"
                )
                print(
                    "[REACH_BENCH] "
                    f"difficulty={difficulty} "
                    f"hand_pos_env0={hand_pos_env0} initial_target_env0={initial_target_env0} "
                    f"initial_position_error_env0={initial_position_error_env0:.3f}"
                )
                video_capture_stride = max(1, int(round((1.0 / step_dt) / float(args_cli.video_fps))))
                if args_cli.video:
                    if batch_size != 1:
                        raise RuntimeError("--video currently requires --num_envs 1 for benchmark capture.")
                    video_writer, video_path = _init_manual_video_writer(
                        output_dir=output_dir,
                        difficulty=difficulty,
                        repeat_index=len(difficulty_records),
                    )
                    print(f"[REACH_BENCH] Manual video path: {video_path}")
                    video_writer.append_data(_render_rgb_frame(env))

                current_block_index = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
                blocks_touched = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
                blocks_stabilized = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
                current_block_elapsed_steps = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
                current_block_touched = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
                current_block_hold_steps = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
                current_block_max_hold_steps = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
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
                block_touched = torch.zeros((num_envs, MAX_TARGETS_PER_EPISODE), dtype=torch.bool, device=base_env.device)
                block_stabilized = torch.zeros((num_envs, MAX_TARGETS_PER_EPISODE), dtype=torch.bool, device=base_env.device)
                block_touch_time_s = torch.full(
                    (num_envs, MAX_TARGETS_PER_EPISODE),
                    float("nan"),
                    device=base_env.device,
                )
                block_stabilize_time_s = torch.full(
                    (num_envs, MAX_TARGETS_PER_EPISODE),
                    float("nan"),
                    device=base_env.device,
                )
                block_max_hold_s = torch.full(
                    (num_envs, MAX_TARGETS_PER_EPISODE),
                    float("nan"),
                    device=base_env.device,
                )
                block_elapsed_s = torch.full(
                    (num_envs, MAX_TARGETS_PER_EPISODE),
                    float("nan"),
                    device=base_env.device,
                )
                block_statuses = [["not_reached" for _ in range(MAX_TARGETS_PER_EPISODE)] for _ in range(num_envs)]

                final_position_error = torch.full((num_envs,), float("nan"), device=base_env.device)
                position_error = torch.full((num_envs,), float("nan"), device=base_env.device)

                for step in range(max_steps_per_sequence):
                    start_time = time.time()
                    if step > 0 and step % 100 == 0:
                        print(
                            "[REACH_BENCH] "
                            f"difficulty={difficulty} batch={batch_index + 1} "
                            f"step={step}/{max_steps_per_sequence} "
                            f"active={int((active_mask & (~done_mask)).sum().item())}/{batch_size} "
                            f"completed={sum(int(record['sequence_completed']) for record in difficulty_records)}/{len(difficulty_records)}"
                        )
                    with torch.inference_mode():
                        actions = policy(obs)
                        actions = actions.clone()
                        actions[~active_mask] = 0.0
                        obs, _, _, _ = vec_env.step(actions)
                    if args_cli.video and step % video_capture_stride == 0:
                        video_writer.append_data(_render_rgb_frame(env))

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

                    current_block_elapsed_steps += tracked_mask.long()
                    touch_mask = tracked_mask & (~current_block_touched) & _hand_touches_block(
                        hand_pos_w, base_env._left_hand_active_target_w
                    )
                    fall_mask = tracked_mask & (
                        (root_height < args_cli.fall_height_threshold)
                        | (projected_gravity_z > args_cli.fall_gravity_threshold)
                    )
                    timeout_mask = tracked_mask & (current_block_elapsed_steps >= per_target_timeout_steps)

                    if torch.any(touch_mask):
                        env_ids = torch.nonzero(touch_mask, as_tuple=False).squeeze(-1)
                        elapsed_s = current_block_elapsed_steps[env_ids].float() * step_dt
                        for local_idx, env_id in enumerate(env_ids.tolist()):
                            block_index = int(current_block_index[env_id].item())
                            block_touched[env_id, block_index] = True
                            block_touch_time_s[env_id, block_index] = elapsed_s[local_idx]
                            block_elapsed_s[env_id, block_index] = elapsed_s[local_idx]
                            block_statuses[env_id][block_index] = "touched"
                            if env_id == 0:
                                print(
                                    "[REACH_BENCH] "
                                    f"difficulty={difficulty} env=0 block={block_index} "
                                    f"event=touched elapsed_s={float(elapsed_s[local_idx]):.2f}"
                                )
                        current_block_touched[env_ids] = True
                        current_block_hold_steps[env_ids] = 0
                        current_block_max_hold_steps[env_ids] = 0
                        blocks_touched[env_ids] += 1

                    hold_zone_mask = tracked_mask & current_block_touched & _hand_in_hold_zone(
                        hand_pos_w, base_env._left_hand_active_target_w
                    )
                    current_block_hold_steps = torch.where(
                        hold_zone_mask,
                        current_block_hold_steps + 1,
                        torch.where(
                            tracked_mask & current_block_touched,
                            torch.zeros_like(current_block_hold_steps),
                            current_block_hold_steps,
                        ),
                    )
                    current_block_max_hold_steps = torch.maximum(current_block_max_hold_steps, current_block_hold_steps)
                    stabilize_mask = tracked_mask & current_block_touched & (current_block_hold_steps >= hold_steps_required)

                    if torch.any(stabilize_mask):
                        env_ids = torch.nonzero(stabilize_mask, as_tuple=False).squeeze(-1)
                        elapsed_s = current_block_elapsed_steps[env_ids].float() * step_dt
                        hold_s = current_block_max_hold_steps[env_ids].float() * step_dt
                        for local_idx, env_id in enumerate(env_ids.tolist()):
                            block_index = int(current_block_index[env_id].item())
                            block_stabilized[env_id, block_index] = True
                            block_success[env_id, block_index] = True
                            block_stabilize_time_s[env_id, block_index] = elapsed_s[local_idx]
                            block_max_hold_s[env_id, block_index] = hold_s[local_idx]
                            block_elapsed_s[env_id, block_index] = elapsed_s[local_idx]
                            block_statuses[env_id][block_index] = "stabilized"
                            if env_id == 0:
                                print(
                                    "[REACH_BENCH] "
                                    f"difficulty={difficulty} env=0 block={block_index} "
                                    f"event=stabilized elapsed_s={float(elapsed_s[local_idx]):.2f} "
                                    f"hold_s={float(hold_s[local_idx]):.2f}"
                                )
                        blocks_stabilized[env_ids] += 1
                        current_block_index[env_ids] += 1
                        with torch.inference_mode():
                            base_env._left_hand_completed_targets[env_ids] = current_block_index[env_ids]
                        current_block_elapsed_steps[env_ids] = 0
                        current_block_touched[env_ids] = False
                        current_block_hold_steps[env_ids] = 0
                        current_block_max_hold_steps[env_ids] = 0
                        total_time_s[env_ids] = (step + 1) * step_dt
                        reached_end_mask = current_block_index[env_ids] >= MAX_TARGETS_PER_EPISODE
                        if torch.any(reached_end_mask):
                            done_env_ids = env_ids[reached_end_mask]
                            sequence_completed[done_env_ids] = True
                            final_position_error[done_env_ids] = position_error[done_env_ids]
                            done_mask[done_env_ids] = True
                            active_mask[done_env_ids] = False
                            with torch.inference_mode():
                                base_env._left_hand_has_active_target[done_env_ids] = False
                            for env_id in done_env_ids.tolist():
                                if env_id == 0:
                                    print(
                                        "[REACH_BENCH] "
                                        f"difficulty={difficulty} env=0 event=sequence_completed "
                                        f"blocks_stabilized={int(blocks_stabilized[env_id].item())}"
                                    )
                        if torch.any(~reached_end_mask):
                            next_env_ids = env_ids[~reached_end_mask]
                            with torch.inference_mode():
                                _activate_benchmark_block(base_env, next_env_ids, current_block_index[next_env_ids])

                    unresolved_fall = fall_mask & (~done_mask) & (~stabilize_mask)
                    if torch.any(unresolved_fall):
                        env_ids = torch.nonzero(unresolved_fall, as_tuple=False).squeeze(-1)
                        final_position_error[env_ids] = position_error[env_ids]
                        total_time_s[env_ids] = (step + 1) * step_dt
                        failure_block_index[env_ids] = current_block_index[env_ids]
                        done_mask[env_ids] = True
                        active_mask[env_ids] = False
                        elapsed_s = current_block_elapsed_steps[env_ids].float() * step_dt
                        hold_s = current_block_max_hold_steps[env_ids].float() * step_dt
                        for local_idx, env_id in enumerate(env_ids.tolist()):
                            failure_reason[env_id] = "fall"
                            block_index = int(current_block_index[env_id].item())
                            if 0 <= block_index < MAX_TARGETS_PER_EPISODE:
                                if current_block_touched[env_id]:
                                    block_max_hold_s[env_id, block_index] = hold_s[local_idx]
                                block_statuses[env_id][block_index] = "fall"
                                block_elapsed_s[env_id, block_index] = elapsed_s[local_idx]
                                if env_id == 0:
                                    print(
                                        "[REACH_BENCH] "
                                        f"difficulty={difficulty} env=0 block={block_index} "
                                        f"event=fall elapsed_s={float(elapsed_s[local_idx]):.2f}"
                                    )

                    unresolved_timeout = timeout_mask & (~done_mask) & (~stabilize_mask)
                    if torch.any(unresolved_timeout):
                        env_ids = torch.nonzero(unresolved_timeout, as_tuple=False).squeeze(-1)
                        timed_out_block_indices = current_block_index[env_ids].clone()
                        elapsed_s = current_block_elapsed_steps[env_ids].float() * step_dt
                        hold_s = current_block_max_hold_steps[env_ids].float() * step_dt
                        for local_idx, env_id in enumerate(env_ids.tolist()):
                            block_index = int(current_block_index[env_id].item())
                            if 0 <= block_index < MAX_TARGETS_PER_EPISODE:
                                if current_block_touched[env_id]:
                                    block_statuses[env_id][block_index] = "touched_timeout"
                                    block_max_hold_s[env_id, block_index] = hold_s[local_idx]
                                else:
                                    block_statuses[env_id][block_index] = "timeout"
                                block_elapsed_s[env_id, block_index] = elapsed_s[local_idx]
                                if env_id == 0:
                                    print(
                                        "[REACH_BENCH] "
                                        f"difficulty={difficulty} env=0 block={block_index} "
                                        f"event=timeout elapsed_s={float(elapsed_s[local_idx]):.2f}"
                                    )
                        current_block_index[env_ids] += 1
                        with torch.inference_mode():
                            base_env._left_hand_completed_targets[env_ids] = current_block_index[env_ids]
                        current_block_elapsed_steps[env_ids] = 0
                        current_block_touched[env_ids] = False
                        current_block_hold_steps[env_ids] = 0
                        current_block_max_hold_steps[env_ids] = 0
                        total_time_s[env_ids] = (step + 1) * step_dt
                        reached_end_mask = current_block_index[env_ids] >= MAX_TARGETS_PER_EPISODE
                        if torch.any(reached_end_mask):
                            done_env_ids = env_ids[reached_end_mask]
                            final_position_error[done_env_ids] = position_error[done_env_ids]
                            done_mask[done_env_ids] = True
                            active_mask[done_env_ids] = False
                            failure_block_index[done_env_ids] = timed_out_block_indices[reached_end_mask]
                            with torch.inference_mode():
                                base_env._left_hand_has_active_target[done_env_ids] = False
                            for env_id in done_env_ids.tolist():
                                failure_reason[env_id] = "sequence_end"
                        if torch.any(~reached_end_mask):
                            next_env_ids = env_ids[~reached_end_mask]
                            with torch.inference_mode():
                                _activate_benchmark_block(base_env, next_env_ids, current_block_index[next_env_ids])

                    if torch.all(done_mask[:batch_size]):
                        print(
                            "[REACH_BENCH] "
                            f"difficulty={difficulty} batch={batch_index + 1} "
                            f"break_step={step} done_mask={done_mask[:batch_size].detach().cpu().tolist()} "
                            f"current_block_index={current_block_index[:batch_size].detach().cpu().tolist()} "
                            f"sequence_completed={sequence_completed[:batch_size].detach().cpu().tolist()} "
                            f"failure_reason={failure_reason[:batch_size]}"
                        )
                        break

                    sleep_time = step_dt - (time.time() - start_time)
                    if args_cli.real_time and sleep_time > 0.0:
                        time.sleep(sleep_time)

                guard_unfinished = active_mask & (~done_mask)
                if torch.any(guard_unfinished):
                    env_ids = torch.nonzero(guard_unfinished, as_tuple=False).squeeze(-1)
                    elapsed_s = current_block_elapsed_steps[env_ids].float() * step_dt
                    hold_s = current_block_max_hold_steps[env_ids].float() * step_dt
                    final_position_error[env_ids] = position_error[env_ids]
                    total_time_s[env_ids] = max_steps_per_sequence * step_dt
                    failure_block_index[env_ids] = current_block_index[env_ids]
                    done_mask[env_ids] = True
                    active_mask[env_ids] = False
                    for local_idx, env_id in enumerate(env_ids.tolist()):
                        failure_reason[env_id] = "guard_timeout"
                        block_index = int(current_block_index[env_id].item())
                        if 0 <= block_index < MAX_TARGETS_PER_EPISODE:
                            if current_block_touched[env_id]:
                                block_statuses[env_id][block_index] = "touched_timeout"
                                block_max_hold_s[env_id, block_index] = hold_s[local_idx]
                            else:
                                block_statuses[env_id][block_index] = "guard_timeout"
                            block_elapsed_s[env_id, block_index] = elapsed_s[local_idx]

                for env_id in range(batch_size):
                    if not math.isfinite(float(final_position_error[env_id].item())):
                        final_position_error[env_id] = position_error[env_id]
                        total_time_s[env_id] = max(float(total_time_s[env_id].item()), (step + 1) * step_dt)
                    touched_blocks = int(blocks_touched[env_id].item())
                    stabilized_blocks = int(blocks_stabilized[env_id].item())
                    failed_block_index = int(failure_block_index[env_id].item())
                    failure_reason_env = failure_reason[env_id]
                    failed_block_elapsed_s = float("nan")
                    if failure_reason_env and 0 <= failed_block_index < MAX_TARGETS_PER_EPISODE:
                        failed_block_elapsed_s = float(block_elapsed_s[env_id, failed_block_index].item())
                    near_count = max(int(near_target_count[env_id].item()), 1)
                    record = {
                        "difficulty": difficulty,
                        "mode": args_cli.mode,
                        "repeat_index": len(difficulty_records),
                        "sequence_completed": bool(sequence_completed[env_id].item()),
                        "blocks_touched": touched_blocks,
                        "blocks_stabilized": stabilized_blocks,
                        "blocks_completed": stabilized_blocks,
                        "failure_reason": failure_reason_env,
                        "failure_block_index": failed_block_index,
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
                        block_completed = bool(block_success[env_id, block_index].item())
                        block_status = block_statuses[env_id][block_index]
                        block_elapsed_until_failure_s = (
                            failed_block_elapsed_s
                            if block_status in ("timeout", "touched_timeout", "fall", "guard_timeout")
                            and block_index == failed_block_index
                            else float("nan")
                        )
                        record[f"block_{block_index}_success"] = block_completed
                        record[f"block_{block_index}_touched"] = bool(block_touched[env_id, block_index].item())
                        record[f"block_{block_index}_stabilized"] = bool(block_stabilized[env_id, block_index].item())
                        record[f"block_{block_index}_time_s"] = float(block_stabilize_time_s[env_id, block_index].item())
                        record[f"block_{block_index}_touch_time_s"] = float(block_touch_time_s[env_id, block_index].item())
                        record[f"block_{block_index}_stabilize_time_s"] = float(block_stabilize_time_s[env_id, block_index].item())
                        record[f"block_{block_index}_max_hold_s"] = float(block_max_hold_s[env_id, block_index].item())
                        record[f"block_{block_index}_elapsed_s"] = float(block_elapsed_s[env_id, block_index].item())
                        record[f"block_{block_index}_status"] = block_status
                        record[f"block_{block_index}_elapsed_until_failure_s"] = block_elapsed_until_failure_s
                    difficulty_records.append(record)
                if args_cli.video and video_writer is not None:
                    video_writer.append_data(_render_rgb_frame(env))
                    video_writer.close()
                    video_writer = None

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
            "training_per_target_timeout_s": TRAINING_PER_TARGET_TIMEOUT_S,
            "benchmark_per_target_timeouts_s": DEFAULT_BENCHMARK_PER_TARGET_TIMEOUT_S,
            "benchmark_per_target_timeout_override_s": args_cli.benchmark_per_target_timeout_s,
            "benchmark_hold_time_s": args_cli.benchmark_hold_time_s,
            "benchmark_hold_margin_m": args_cli.benchmark_hold_margin_m,
            "settle_grace_s": SETTLE_GRACE_S,
            "near_target_radius": args_cli.near_target_radius,
            "summaries": summaries,
            "overall": {
                "episodes": len(all_records),
                "sequence_completion_rate": _safe_mean([float(record["sequence_completed"]) for record in all_records]),
                "mean_blocks_touched": _safe_mean([record["blocks_touched"] for record in all_records]),
                "mean_blocks_stabilized": _safe_mean([record["blocks_stabilized"] for record in all_records]),
                "mean_blocks_completed": _safe_mean([record["blocks_completed"] for record in all_records]),
                "target_timeout_rate": _safe_mean(
                    [
                        float(
                            any(
                                record[f"block_{idx}_status"] in ("timeout", "touched_timeout")
                                for idx in range(MAX_TARGETS_PER_EPISODE)
                            )
                        )
                        for record in all_records
                    ]
                ),
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
        if args_cli.video:
            latest_video_path = _latest_video_path(output_dir)
            if latest_video_path is not None:
                print(f"[REACH_BENCH] Latest video: {latest_video_path}")
            else:
                print("[REACH_BENCH] Latest video: not found under output_dir/videos")
    except BaseException as exc:
        print(f"[REACH_BENCH] ERROR: {type(exc).__name__}: {exc}")
        raise
    finally:
        fixed_target_mdp._spawn_new_fixed_targets = _ORIGINAL_SPAWN_NEW_FIXED_TARGETS
        fixed_target_mdp._sync_long_horizon_state = _ORIGINAL_SYNC_LONG_HORIZON_STATE
        freeze_base_reach_mdp._sync_adapter_hold_stay_state = _ORIGINAL_FREEZE_SYNC_ADAPTER_HOLD_STAY_STATE
        if video_writer is not None:
            try:
                print("[REACH_BENCH] Closing manual video writer...")
                video_writer.close()
            except Exception as exc:
                print(f"[REACH_BENCH] Warning: video writer close failed: {exc}")
        if vec_env is not None:
            try:
                print("[REACH_BENCH] Closing vec env...")
                vec_env.close()
            except Exception as exc:
                print(f"[REACH_BENCH] Warning: vec env close failed: {exc}")
        if env is not None:
            try:
                print("[REACH_BENCH] Closing env...")
                env.close()
            except Exception as exc:
                print(f"[REACH_BENCH] Warning: env close failed: {exc}")
        if args_cli.video:
            video_dir = os.path.join(output_dir if 'output_dir' in locals() else "", "videos")
            print(f"[REACH_BENCH] Video directory: {video_dir}")
            latest_video_path = _latest_video_path(output_dir) if 'output_dir' in locals() else None
            if latest_video_path is not None:
                print(f"[REACH_BENCH] Latest video after close: {latest_video_path}")
            else:
                print("[REACH_BENCH] Latest video after close: not found")


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[REACH_BENCH] Closing simulation app...")
        simulation_app.close()
        print("[REACH_BENCH] Shutdown complete.")
