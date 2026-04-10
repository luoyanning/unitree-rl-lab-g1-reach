"""Benchmark the G1 point-goal policy on fixed single-target cases."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import importlib
import json
import math
import os
import statistics
import time
from html import escape
from collections import defaultdict
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from rsl_rl_compat import sanitize_runner_cfg  # isort: skip


parser = argparse.ArgumentParser(description="Benchmark a low-level G1 point-goal checkpoint on fixed targets.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during benchmark playback.")
parser.add_argument("--video_length", type=int, default=2000, help="Recorded video length in steps.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-PointGoal-v0", help="Task name.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--episodes_per_case", type=int, default=20, help="Episodes per distance-angle case.")
parser.add_argument(
    "--layout",
    type=str,
    default="rings",
    choices=["rings", "grid"],
    help="Benchmark layout. 'rings' generates evenly spaced points on multiple circles.",
)
parser.add_argument("--case_distance", type=float, default=None, help="If set, evaluate only this one target distance.")
parser.add_argument("--case_angle_deg", type=float, default=None, help="If set, evaluate only this one target angle.")
parser.add_argument(
    "--ring_radii",
    type=str,
    default="0.5,1.0,1.5,2.0,3.0,4.0,5.0",
    help="Comma-separated benchmark ring radii in meters.",
)
parser.add_argument("--points_per_ring", type=int, default=24, help="Number of equally spaced points on each ring.")
parser.add_argument(
    "--ring_angle_offset_deg",
    type=float,
    default=0.0,
    help="Angular offset applied to all ring points.",
)
parser.add_argument(
    "--distances",
    type=str,
    default="0.4,0.6,0.8,1.2,1.8,2.5,3.5,5.0",
    help="Comma-separated target distances in meters.",
)
parser.add_argument(
    "--angles_deg",
    type=str,
    default="0,-15,15,-30,30,-45,45,-60,60,-90,90",
    help="Comma-separated target angles in degrees.",
)
parser.add_argument("--timeout_s", type=float, default=20.0, help="Per-episode benchmark timeout in seconds.")
parser.add_argument(
    "--benchmark_progress",
    type=float,
    default=1.0,
    help="Fixed command-scheduling progress in [0, 1]. 1.0 means the final strict stage.",
)
parser.add_argument("--reach_distance", type=float, default=0.20, help="Distance threshold for reach@20cm-style metrics.")
parser.add_argument(
    "--precise_distance",
    type=float,
    default=0.10,
    help="Distance threshold for precise_stop@10cm-style metrics.",
)
parser.add_argument(
    "--stop_hold_s",
    type=float,
    default=0.40,
    help="Required hold duration to count as a stable stop.",
)
parser.add_argument(
    "--stop_lin_vel_threshold",
    type=float,
    default=0.10,
    help="Linear speed threshold for a stable stop.",
)
parser.add_argument(
    "--stop_yaw_rate_threshold",
    type=float,
    default=0.25,
    help="Yaw-rate threshold for a stable stop.",
)
parser.add_argument(
    "--heading_align_deg",
    type=float,
    default=15.0,
    help="Heading-alignment threshold used to summarize turning ability.",
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
    help="Projected-gravity z threshold above which the robot is considered fallen.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for benchmark outputs. Defaults to <run_dir>/benchmark_v1/<timestamp>.",
)
parser.add_argument(
    "--world_camera",
    action="store_true",
    default=False,
    help="Use a fixed world camera instead of following the robot.",
)
parser.add_argument(
    "--save_trajectory_plots",
    action="store_true",
    default=False,
    help="Save one top-down XY trajectory plot per evaluated episode.",
)
parser.add_argument(
    "--trajectory_plot_format",
    type=str,
    default="svg",
    choices=["png", "pdf", "svg"],
    help="File format for --save_trajectory_plots outputs.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

point_goal_env_cfg = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.point_goal.point_goal_env_cfg"
)
GOAL_STOP_NEAR_DISTANCE_START = point_goal_env_cfg.GOAL_STOP_NEAR_DISTANCE_START
HEADING_SLOW_DOWN_DISTANCE_START = point_goal_env_cfg.HEADING_SLOW_DOWN_DISTANCE_START
HOLD_POSITION_DISTANCE_START = point_goal_env_cfg.HOLD_POSITION_DISTANCE_START
POINT_GOAL_CURRICULUM_PROMOTE_SUCCESS = point_goal_env_cfg.POINT_GOAL_CURRICULUM_PROMOTE_SUCCESS
SLOW_DOWN_DISTANCE_START = point_goal_env_cfg.SLOW_DOWN_DISTANCE_START
STOP_DISTANCE_START = point_goal_env_cfg.STOP_DISTANCE_START
STOP_VELOCITY_THRESHOLD = point_goal_env_cfg.STOP_VELOCITY_THRESHOLD
STOP_VELOCITY_THRESHOLD_START = point_goal_env_cfg.STOP_VELOCITY_THRESHOLD_START
STOP_YAW_RATE_THRESHOLD = point_goal_env_cfg.STOP_YAW_RATE_THRESHOLD
STOP_YAW_RATE_THRESHOLD_START = point_goal_env_cfg.STOP_YAW_RATE_THRESHOLD_START
SUCCESS_DISTANCE = point_goal_env_cfg.SUCCESS_DISTANCE
SUCCESS_DISTANCE_START = point_goal_env_cfg.SUCCESS_DISTANCE_START
SUCCESS_HOLD_STEPS = point_goal_env_cfg.SUCCESS_HOLD_STEPS
SUCCESS_HOLD_STEPS_START = point_goal_env_cfg.SUCCESS_HOLD_STEPS_START

_PYPLOT = None


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _lerp_scalar(start: float, end: float, progress: float) -> float:
    return float(start + (end - start) * progress)


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp_progress(progress: float) -> float:
    return max(0.0, min(1.0, float(progress)))


def _percentile(values: list[float], q: float) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    ordered = sorted(finite_values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_weight = upper_index - position
    upper_weight = position - lower_index
    return ordered[lower_index] * lower_weight + ordered[upper_index] * upper_weight


def _mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite_values) if finite_values else float("nan")


def _rate(records: list[dict], key: str) -> float:
    return sum(bool(record[key]) for record in records) / max(len(records), 1)


def _numeric_summary(records: list[dict], key: str) -> dict[str, float]:
    values = [float(record[key]) for record in records if math.isfinite(float(record[key]))]
    return {
        f"{key}_mean": _mean(values),
        f"{key}_p50": _percentile(values, 0.50),
        f"{key}_p90": _percentile(values, 0.90),
    }


def _summarize_records(records: list[dict]) -> dict[str, float]:
    summary = {
        "episodes": float(len(records)),
        "reach@20cm": _rate(records, "reach_20cm"),
        "stop@20cm": _rate(records, "stop_20cm"),
        "precise_stop@10cm": _rate(records, "precise_stop_10cm"),
        "heading_align@15deg": _rate(records, "heading_align_15deg"),
        "fall_rate": _rate(records, "fall"),
        "timeout_rate": _rate(records, "timeout"),
        "overshoot_rate": _rate(records, "overshoot"),
    }
    numeric_keys = [
        "min_error_m",
        "final_error_m",
        "final_stop_error_m",
        "final_radius_error_m",
        "final_stop_radius_error_m",
        "final_target_bearing_error_deg",
        "final_stop_target_bearing_error_deg",
        "path_efficiency",
        "path_length_m",
        "time_to_reach_20cm_s",
        "time_to_stop_20cm_s",
        "time_to_precise_stop_10cm_s",
        "time_to_heading_align_15deg_s",
        "min_heading_error_deg",
        "final_heading_error_deg",
        "final_stop_heading_error_deg",
        "final_along_track_error_m",
        "final_cross_track_error_m",
        "final_stop_along_track_error_m",
        "final_stop_cross_track_error_m",
    ]
    for key in numeric_keys:
        summary.update(_numeric_summary(records, key))
    return summary


def _format_metric(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def _print_summary(title: str, summary: dict[str, float]):
    print(title)
    for key, value in summary.items():
        print(f"  {key}: {_format_metric(value)}")


def _get_pyplot():
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _PYPLOT = plt
    return _PYPLOT


def _trajectory_plot_path(output_dir: str, case_index: int, case: dict[str, float], episode_index: int) -> str:
    angle_token = f"{case['angle_deg']:+07.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    distance_token = f"{case['distance_m']:.2f}".replace(".", "p")
    filename = (
        f"case_{case_index:04d}_episode_{episode_index:03d}"
        f"_r{distance_token}_a{angle_token}.{args_cli.trajectory_plot_format}"
    )
    return os.path.join(output_dir, "trajectory_plots", filename)


def _save_trajectory_plot(
    output_dir: str,
    case_index: int,
    case: dict[str, float],
    episode_index: int,
    trajectory_xy: list[tuple[float, float]],
    record: dict,
):
    if len(trajectory_xy) < 2:
        return
    plot_dir = os.path.join(output_dir, "trajectory_plots")
    os.makedirs(plot_dir, exist_ok=True)

    xs = [point[0] for point in trajectory_xy]
    ys = [point[1] for point in trajectory_xy]
    goal_x = float(case["goal_x_m"])
    goal_y = float(case["goal_y_m"])
    margin = 0.35
    extent = max(max(abs(value) for value in xs + [goal_x]), max(abs(value) for value in ys + [goal_y]), 0.5) + margin
    plot_path = _trajectory_plot_path(output_dir, case_index, case, episode_index)
    if args_cli.trajectory_plot_format == "svg":
        width = 800
        height = 800
        padding = 70
        scale = (width - 2 * padding) / (2.0 * extent)

        def project(point_x: float, point_y: float) -> tuple[float, float]:
            pixel_x = padding + (point_x + extent) * scale
            pixel_y = height - (padding + (point_y + extent) * scale)
            return pixel_x, pixel_y

        polyline_points = " ".join(f"{project(x, y)[0]:.2f},{project(x, y)[1]:.2f}" for x, y in trajectory_xy)
        start_px = project(0.0, 0.0)
        final_px = project(xs[-1], ys[-1])
        goal_px = project(goal_x, goal_y)
        reach_radius_px = args_cli.reach_distance * scale
        title = (
            f"r={case['distance_m']:.2f}m, angle={case['angle_deg']:+.1f}deg | "
            f"final={record['final_error_m']:.3f}m, stop={record['stop_20cm']}, fall={record['fall']}"
        )
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.1f}" y="32" text-anchor="middle" font-family="monospace" font-size="18">{escape(title)}</text>
  <line x1="{padding}" y1="{height / 2:.2f}" x2="{width - padding}" y2="{height / 2:.2f}" stroke="#dddddd" stroke-width="1"/>
  <line x1="{width / 2:.2f}" y1="{padding}" x2="{width / 2:.2f}" y2="{height - padding}" stroke="#dddddd" stroke-width="1"/>
  <line x1="{start_px[0]:.2f}" y1="{start_px[1]:.2f}" x2="{goal_px[0]:.2f}" y2="{goal_px[1]:.2f}" stroke="#d62728" stroke-width="2" stroke-dasharray="7 6" opacity="0.45"/>
  <circle cx="{goal_px[0]:.2f}" cy="{goal_px[1]:.2f}" r="{reach_radius_px:.2f}" fill="none" stroke="#d62728" stroke-width="2" stroke-dasharray="3 5" opacity="0.55"/>
  <polyline points="{polyline_points}" fill="none" stroke="#1f77b4" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{start_px[0]:.2f}" cy="{start_px[1]:.2f}" r="8" fill="#2ca02c"/>
  <circle cx="{final_px[0]:.2f}" cy="{final_px[1]:.2f}" r="7" fill="#1f77b4"/>
  <polygon points="{goal_px[0]:.2f},{goal_px[1] - 14:.2f} {goal_px[0] + 4:.2f},{goal_px[1] - 4:.2f} {goal_px[0] + 14:.2f},{goal_px[1] - 4:.2f} {goal_px[0] + 6:.2f},{goal_px[1] + 3:.2f} {goal_px[0] + 9:.2f},{goal_px[1] + 14:.2f} {goal_px[0]:.2f},{goal_px[1] + 8:.2f} {goal_px[0] - 9:.2f},{goal_px[1] + 14:.2f} {goal_px[0] - 6:.2f},{goal_px[1] + 3:.2f} {goal_px[0] - 14:.2f},{goal_px[1] - 4:.2f} {goal_px[0] - 4:.2f},{goal_px[1] - 4:.2f}" fill="#d62728"/>
  <text x="{padding}" y="{height - 28}" font-family="monospace" font-size="14" fill="#555555">green=start, blue=final, red=target, dotted=reach radius</text>
</svg>
"""
        with open(plot_path, "w", encoding="utf-8") as plot_file:
            plot_file.write(svg)
        return

    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot(xs, ys, color="#1f77b4", linewidth=1.8, label="trajectory")
    ax.scatter([0.0], [0.0], color="#2ca02c", s=45, label="start", zorder=3)
    ax.scatter([xs[-1]], [ys[-1]], color="#1f77b4", s=35, label="final", zorder=3)
    ax.scatter([goal_x], [goal_y], color="#d62728", marker="*", s=140, label="target", zorder=4)
    ax.plot([0.0, goal_x], [0.0, goal_y], color="#d62728", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.add_patch(plt.Circle((goal_x, goal_y), args_cli.reach_distance, color="#d62728", fill=False, linestyle=":", alpha=0.55))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("x from reset origin (m)")
    ax.set_ylabel("y from reset origin (m)")
    ax.set_title(
        f"r={case['distance_m']:.2f}m, angle={case['angle_deg']:+.1f}deg | "
        f"final={record['final_error_m']:.3f}m, stop={record['stop_20cm']}, fall={record['fall']}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _configure_benchmark_env(env_cfg, timeout_s: float):
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), timeout_s)
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        for attr_name in [
            "terrain_levels",
            "lin_vel_cmd_levels",
            "point_goal_target_levels",
            "point_goal_reward_levels",
        ]:
            if hasattr(env_cfg.curriculum, attr_name):
                setattr(env_cfg.curriculum, attr_name, None)
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.curriculum = False
    for attr_name in [
        "point_goal_success",
        "point_goal_timeout",
        "base_height",
        "bad_orientation",
        "time_out",
    ]:
        if hasattr(env_cfg.terminations, attr_name):
            setattr(env_cfg.terminations, attr_name, None)
    if getattr(args_cli, "world_camera", False):
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = (3.7, -4.3, 2.4)
        env_cfg.viewer.lookat = (1.0, -0.2, 1.0)


def _apply_benchmark_progress(env, progress: float, command_name: str = "base_velocity") -> dict[str, float]:
    progress = _clamp_progress(progress)
    env._point_goal_success_distance = _lerp_scalar(SUCCESS_DISTANCE_START, SUCCESS_DISTANCE, progress)
    env._point_goal_success_hold_steps = int(round(_lerp_scalar(SUCCESS_HOLD_STEPS_START, SUCCESS_HOLD_STEPS, progress)))
    env._point_goal_stop_velocity_threshold = _lerp_scalar(
        STOP_VELOCITY_THRESHOLD_START,
        STOP_VELOCITY_THRESHOLD,
        progress,
    )
    env._point_goal_stop_yaw_rate_threshold = _lerp_scalar(
        STOP_YAW_RATE_THRESHOLD_START,
        STOP_YAW_RATE_THRESHOLD,
        progress,
    )
    env._point_goal_goal_stop_near_distance = _lerp_scalar(GOAL_STOP_NEAR_DISTANCE_START, 0.20, progress)
    command_term = env.command_manager.get_term(command_name)
    scheduled_hold_position_distance = max(
        _lerp_scalar(HOLD_POSITION_DISTANCE_START, 0.08, progress),
        env._point_goal_success_distance + 0.01,
    )
    scheduled_stop_distance = max(
        _lerp_scalar(STOP_DISTANCE_START, 0.20, progress),
        scheduled_hold_position_distance + 0.07,
    )
    scheduled_slow_down_distance = max(
        _lerp_scalar(SLOW_DOWN_DISTANCE_START, 0.55, progress),
        scheduled_stop_distance + 0.18,
    )
    scheduled_heading_slow_down_distance = max(
        _lerp_scalar(HEADING_SLOW_DOWN_DISTANCE_START, 0.60, progress),
        scheduled_stop_distance + 0.05,
    )
    scheduled_near_recovery_distance = scheduled_hold_position_distance + 0.02
    command_term.cfg.hold_position_distance = scheduled_hold_position_distance
    command_term.cfg.stop_distance = scheduled_stop_distance
    command_term.cfg.slow_down_distance = scheduled_slow_down_distance
    command_term.cfg.heading_slow_down_distance = scheduled_heading_slow_down_distance
    command_term.cfg.near_recovery_distance = scheduled_near_recovery_distance
    return {
        "benchmark_progress": progress,
        "scheduled_success_distance": env._point_goal_success_distance,
        "scheduled_success_hold_steps": float(env._point_goal_success_hold_steps),
        "scheduled_stop_velocity_threshold": env._point_goal_stop_velocity_threshold,
        "scheduled_stop_yaw_rate_threshold": env._point_goal_stop_yaw_rate_threshold,
        "scheduled_goal_stop_near_distance": env._point_goal_goal_stop_near_distance,
        "scheduled_hold_position_distance": scheduled_hold_position_distance,
        "scheduled_stop_distance": scheduled_stop_distance,
        "scheduled_slow_down_distance": scheduled_slow_down_distance,
        "scheduled_heading_slow_down_distance": scheduled_heading_slow_down_distance,
        "scheduled_near_recovery_distance": scheduled_near_recovery_distance,
        "curriculum_promote_success_threshold": POINT_GOAL_CURRICULUM_PROMOTE_SUCCESS,
    }


def _resolve_checkpoint_path(agent_cfg: RslRlOnPolicyRunnerCfg) -> str:
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if getattr(args_cli, "use_pretrained_checkpoint", False):
        raise ValueError("Pretrained checkpoints are not supported for this benchmark.")
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _to_goal_frame_errors(root_pos_xy: torch.Tensor, env_origins_xy: torch.Tensor, unit_dir: torch.Tensor, distance: float):
    rel_pos = root_pos_xy - env_origins_xy
    along_progress = torch.sum(rel_pos * unit_dir, dim=-1)
    cross_progress = rel_pos[:, 0] * unit_dir[:, 1] - rel_pos[:, 1] * unit_dir[:, 0]
    along_error = distance - along_progress
    cross_error = torch.abs(cross_progress)
    return along_error, cross_error


def _build_cases(distances: list[float], angles_deg: list[float]) -> list[dict[str, float]]:
    cases = []
    for distance in distances:
        for point_index, angle_deg in enumerate(angles_deg):
            angle_rad = math.radians(angle_deg)
            cases.append(
                {
                    "distance_m": float(distance),
                    "angle_deg": float(angle_deg),
                    "angle_rad": angle_rad,
                    "point_index": float(point_index),
                    "goal_x_m": float(distance * math.cos(angle_rad)),
                    "goal_y_m": float(distance * math.sin(angle_rad)),
                }
            )
    return cases


def _build_ring_cases(radii: list[float], points_per_ring: int, angle_offset_deg: float) -> list[dict[str, float]]:
    if points_per_ring <= 0:
        raise ValueError("points_per_ring must be positive.")
    angles_deg = []
    for point_index in range(points_per_ring):
        raw_angle_deg = angle_offset_deg + point_index * (360.0 / float(points_per_ring))
        wrapped_angle_deg = math.degrees(_wrap_to_pi(math.radians(raw_angle_deg)))
        angles_deg.append((point_index, wrapped_angle_deg))

    cases = []
    for radius in radii:
        for point_index, angle_deg in angles_deg:
            angle_rad = math.radians(angle_deg)
            cases.append(
                {
                    "distance_m": float(radius),
                    "angle_deg": float(angle_deg),
                    "angle_rad": angle_rad,
                    "point_index": float(point_index),
                    "goal_x_m": float(radius * math.cos(angle_rad)),
                    "goal_y_m": float(radius * math.sin(angle_rad)),
                }
            )
    return cases


def _maybe_tuple_obs(reset_result):
    return reset_result[0] if isinstance(reset_result, tuple) else reset_result


def _get_output_dir(resume_path: str) -> str:
    if args_cli.output_dir:
        return os.path.abspath(args_cli.output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(os.path.dirname(resume_path), "benchmark_v1", timestamp)


def _polar_errors(root_pos_xy: torch.Tensor, env_origins_xy: torch.Tensor, target_radius: float, target_angle_rad: float):
    rel_pos = root_pos_xy - env_origins_xy
    radius = torch.linalg.norm(rel_pos, dim=-1)
    radius_error = torch.abs(radius - target_radius)
    azimuth = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
    bearing_error = torch.abs(torch.atan2(torch.sin(azimuth - target_angle_rad), torch.cos(azimuth - target_angle_rad)))
    return radius_error, torch.rad2deg(bearing_error)


def main():
    if args_cli.case_distance is not None or args_cli.case_angle_deg is not None:
        if args_cli.case_distance is None or args_cli.case_angle_deg is None:
            raise ValueError("--case_distance and --case_angle_deg must be set together.")
        distances = [float(args_cli.case_distance)]
        angles_deg = [float(args_cli.case_angle_deg)]
        benchmark_cases = _build_cases(distances, angles_deg)
    elif args_cli.layout == "rings":
        distances = _parse_float_list(args_cli.ring_radii)
        angles_deg = [math.degrees(_wrap_to_pi(math.radians(args_cli.ring_angle_offset_deg + index * (360.0 / float(args_cli.points_per_ring))))) for index in range(args_cli.points_per_ring)]
        benchmark_cases = _build_ring_cases(distances, args_cli.points_per_ring, args_cli.ring_angle_offset_deg)
    else:
        distances = _parse_float_list(args_cli.distances)
        angles_deg = _parse_float_list(args_cli.angles_deg)
        benchmark_cases = _build_cases(distances, angles_deg)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    _configure_benchmark_env(env_cfg, args_cli.timeout_s)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    resume_path = _resolve_checkpoint_path(agent_cfg)
    output_dir = _get_output_dir(resume_path)
    os.makedirs(output_dir, exist_ok=True)
    if args_cli.save_trajectory_plots:
        os.makedirs(os.path.join(output_dir, "trajectory_plots"), exist_ok=True)

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
    agent_cfg_dict = sanitize_runner_cfg(agent_cfg.to_dict())
    runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=vec_env.device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    command_term = base_env.command_manager.get_term("base_velocity")
    if hasattr(command_term.cfg, "resampling_time_range"):
        command_term.cfg.resampling_time_range = (1.0e9, 1.0e9)
    # Keep fixed benchmark targets fixed after reset; otherwise the task sync logic
    # will resample a random target on the first rollout step.
    if hasattr(command_term, "_resample_command"):
        command_term._resample_command = lambda env_ids: None
    applied_schedule = _apply_benchmark_progress(base_env, args_cli.benchmark_progress, command_name="base_velocity")

    step_dt = float(base_env.step_dt)
    step_budget = int(round(args_cli.timeout_s / step_dt))
    stop_hold_steps = max(1, int(round(args_cli.stop_hold_s / step_dt)))
    heading_align_rad = math.radians(args_cli.heading_align_deg)
    num_envs = base_env.num_envs
    env_ids = torch.arange(num_envs, dtype=torch.long, device=base_env.device)
    env_origins_xy = base_env.scene.env_origins[:, :2]

    print("[INFO] Benchmark configuration:")
    print(f"  checkpoint: {resume_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  layout: {args_cli.layout}")
    print(f"  episodes_per_case: {args_cli.episodes_per_case}")
    print(f"  distances_m: {distances}")
    print(f"  angles_deg: {angles_deg}")
    if args_cli.layout == "rings" and args_cli.case_distance is None:
        print(f"  points_per_ring: {args_cli.points_per_ring}")
        print(f"  ring_angle_offset_deg: {args_cli.ring_angle_offset_deg:.2f}")
    if args_cli.save_trajectory_plots:
        print(f"  trajectory_plots: {os.path.join(output_dir, 'trajectory_plots')}")
        print(f"  trajectory_plot_format: {args_cli.trajectory_plot_format}")
    print(f"  timeout_s: {args_cli.timeout_s:.2f}")
    print(f"  benchmark_progress: {args_cli.benchmark_progress:.3f}")
    for key, value in applied_schedule.items():
        print(f"  {key}: {_format_metric(value)}")

    all_records: list[dict] = []

    for case_index, case in enumerate(benchmark_cases):
        case_records: list[dict] = []
        remaining = args_cli.episodes_per_case
        angle_rad = case["angle_rad"]
        goal_offset_xy = torch.tensor(
            [case["goal_x_m"], case["goal_y_m"]],
            dtype=torch.float32,
            device=base_env.device,
        ).repeat(num_envs, 1)
        target_unit_dir = torch.tensor(
            [math.cos(angle_rad), math.sin(angle_rad)],
            dtype=torch.float32,
            device=base_env.device,
        ).repeat(num_envs, 1)

        while remaining > 0:
            batch_size = min(remaining, num_envs)
            episode_base_index = args_cli.episodes_per_case - remaining
            active_mask = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
            active_mask[:batch_size] = True
            obs = _maybe_tuple_obs(vec_env.reset())

            goal_pos_w = torch.zeros(num_envs, 3, device=base_env.device)
            goal_pos_w[:, :2] = env_origins_xy + goal_offset_xy
            command_term.set_goal_positions(env_ids, goal_pos_w)

            last_root_pos = robot.data.root_pos_w[:, :2].clone()
            start_distance = torch.linalg.norm(goal_pos_w[:, :2] - last_root_pos, dim=-1)
            current_heading_error = torch.abs(command_term.metrics["goal_heading_error"]).clone()
            trajectories_xy = None
            if args_cli.save_trajectory_plots:
                local_root_pos = (last_root_pos[:batch_size] - env_origins_xy[:batch_size]).detach().cpu().tolist()
                trajectories_xy = [[(float(point[0]), float(point[1]))] for point in local_root_pos]

            path_length = torch.zeros(num_envs, device=base_env.device)
            min_error = start_distance.clone()
            final_error = start_distance.clone()
            min_heading_error = current_heading_error.clone()
            final_heading_error = current_heading_error.clone()

            hold_20cm = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
            hold_10cm = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
            hold_any_stop = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)

            reach_20cm = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
            stop_20cm = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
            precise_stop_10cm = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
            heading_align_15deg = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
            fall = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)

            time_to_reach_20cm = torch.full((num_envs,), float("nan"), device=base_env.device)
            time_to_stop_20cm = torch.full((num_envs,), float("nan"), device=base_env.device)
            time_to_precise_stop_10cm = torch.full((num_envs,), float("nan"), device=base_env.device)
            time_to_heading_align_15deg = torch.full((num_envs,), float("nan"), device=base_env.device)

            final_stop_error = torch.full((num_envs,), float("nan"), device=base_env.device)
            final_stop_heading_error = torch.full((num_envs,), float("nan"), device=base_env.device)
            final_stop_along_error = torch.full((num_envs,), float("nan"), device=base_env.device)
            final_stop_cross_error = torch.full((num_envs,), float("nan"), device=base_env.device)
            final_stop_radius_error = torch.full((num_envs,), float("nan"), device=base_env.device)
            final_stop_target_bearing_error = torch.full((num_envs,), float("nan"), device=base_env.device)

            for step in range(step_budget):
                start_time = time.time()
                with torch.no_grad():
                    actions = policy(obs)
                    obs, _, _, _ = vec_env.step(actions)

                root_pos = robot.data.root_pos_w[:, :2].clone()
                if trajectories_xy is not None:
                    local_root_pos = (root_pos[:batch_size] - env_origins_xy[:batch_size]).detach().cpu().tolist()
                    alive_for_plot = (~fall[:batch_size]).detach().cpu().tolist()
                    for env_id, point in enumerate(local_root_pos):
                        if alive_for_plot[env_id]:
                            trajectories_xy[env_id].append((float(point[0]), float(point[1])))

                root_lin_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
                yaw_rate = torch.abs(robot.data.root_ang_vel_w[:, 2])
                root_height = robot.data.root_pos_w[:, 2]
                projected_gravity_z = robot.data.projected_gravity_b[:, 2]
                current_error = torch.linalg.norm(goal_pos_w[:, :2] - root_pos, dim=-1)
                current_heading_error = torch.abs(command_term.metrics["goal_heading_error"]).clone()
                along_error, cross_error = _to_goal_frame_errors(
                    root_pos,
                    env_origins_xy,
                    target_unit_dir,
                    case["distance_m"],
                )
                radius_error, target_bearing_error = _polar_errors(
                    root_pos,
                    env_origins_xy,
                    case["distance_m"],
                    angle_rad,
                )

                tracking_mask = active_mask & (~fall)
                path_length += torch.linalg.norm(root_pos - last_root_pos, dim=-1) * tracking_mask.float()
                min_error = torch.minimum(min_error, torch.where(tracking_mask, current_error, min_error))
                min_heading_error = torch.minimum(
                    min_heading_error,
                    torch.where(tracking_mask, current_heading_error, min_heading_error),
                )
                final_error = torch.where(tracking_mask, current_error, final_error)
                final_heading_error = torch.where(tracking_mask, current_heading_error, final_heading_error)
                last_root_pos = root_pos

                reached_now = tracking_mask & (current_error <= args_cli.reach_distance)
                newly_reached = reached_now & (~reach_20cm)
                reach_20cm |= reached_now
                time_to_reach_20cm[newly_reached] = (step + 1) * step_dt

                heading_ok = tracking_mask & (current_heading_error <= heading_align_rad)
                newly_heading_ok = heading_ok & (~heading_align_15deg)
                heading_align_15deg |= heading_ok
                time_to_heading_align_15deg[newly_heading_ok] = (step + 1) * step_dt

                stable_stop = tracking_mask & (root_lin_speed <= args_cli.stop_lin_vel_threshold) & (
                    yaw_rate <= args_cli.stop_yaw_rate_threshold
                )
                in_stop_20cm = stable_stop & (current_error <= args_cli.reach_distance)
                in_stop_10cm = stable_stop & (current_error <= args_cli.precise_distance)

                hold_any_stop = torch.where(stable_stop, hold_any_stop + 1, torch.zeros_like(hold_any_stop))
                hold_20cm = torch.where(in_stop_20cm, hold_20cm + 1, torch.zeros_like(hold_20cm))
                hold_10cm = torch.where(in_stop_10cm, hold_10cm + 1, torch.zeros_like(hold_10cm))

                any_stop_mask = stable_stop & (hold_any_stop >= stop_hold_steps)
                final_stop_error[any_stop_mask] = current_error[any_stop_mask]
                final_stop_heading_error[any_stop_mask] = torch.rad2deg(current_heading_error[any_stop_mask])
                final_stop_along_error[any_stop_mask] = along_error[any_stop_mask]
                final_stop_cross_error[any_stop_mask] = cross_error[any_stop_mask]
                final_stop_radius_error[any_stop_mask] = radius_error[any_stop_mask]
                final_stop_target_bearing_error[any_stop_mask] = target_bearing_error[any_stop_mask]

                newly_stop_20cm = (~stop_20cm) & (hold_20cm >= stop_hold_steps)
                stop_20cm |= newly_stop_20cm
                time_to_stop_20cm[newly_stop_20cm] = (step + 1) * step_dt

                newly_precise_10cm = (~precise_stop_10cm) & (hold_10cm >= stop_hold_steps)
                precise_stop_10cm |= newly_precise_10cm
                time_to_precise_stop_10cm[newly_precise_10cm] = (step + 1) * step_dt

                newly_fall = tracking_mask & (
                    (root_height < args_cli.fall_height_threshold)
                    | (projected_gravity_z > args_cli.fall_gravity_threshold)
                )
                fall |= newly_fall

                if torch.all(fall[active_mask]):
                    break

                sleep_time = step_dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0.0:
                    time.sleep(sleep_time)

            timeout = active_mask & (~fall) & (~stop_20cm)
            overshoot = active_mask & reach_20cm & (~stop_20cm)
            final_along_error, final_cross_error = _to_goal_frame_errors(
                last_root_pos,
                env_origins_xy,
                target_unit_dir,
                case["distance_m"],
            )
            final_radius_error, final_target_bearing_error = _polar_errors(
                last_root_pos,
                env_origins_xy,
                case["distance_m"],
                angle_rad,
            )

            for env_id in range(batch_size):
                path_length_value = float(path_length[env_id].item())
                start_distance_value = float(start_distance[env_id].item())
                path_efficiency = min(start_distance_value / max(path_length_value, 1.0e-6), 1.0) if path_length_value > 0.0 else 0.0
                record = {
                    "case_index": float(case_index),
                    "point_index": float(case["point_index"]),
                    "distance_m": float(case["distance_m"]),
                    "angle_deg": float(case["angle_deg"]),
                    "start_distance_m": start_distance_value,
                    "reach_20cm": bool(reach_20cm[env_id].item()),
                    "stop_20cm": bool(stop_20cm[env_id].item()),
                    "precise_stop_10cm": bool(precise_stop_10cm[env_id].item()),
                    "heading_align_15deg": bool(heading_align_15deg[env_id].item()),
                    "fall": bool(fall[env_id].item()),
                    "timeout": bool(timeout[env_id].item()),
                    "overshoot": bool(overshoot[env_id].item()),
                    "min_error_m": float(min_error[env_id].item()),
                    "final_error_m": float(final_error[env_id].item()),
                    "final_stop_error_m": float(final_stop_error[env_id].item()),
                    "final_radius_error_m": float(final_radius_error[env_id].item()),
                    "final_stop_radius_error_m": float(final_stop_radius_error[env_id].item()),
                    "final_target_bearing_error_deg": float(final_target_bearing_error[env_id].item()),
                    "final_stop_target_bearing_error_deg": float(final_stop_target_bearing_error[env_id].item()),
                    "time_to_reach_20cm_s": float(time_to_reach_20cm[env_id].item()),
                    "time_to_stop_20cm_s": float(time_to_stop_20cm[env_id].item()),
                    "time_to_precise_stop_10cm_s": float(time_to_precise_stop_10cm[env_id].item()),
                    "time_to_heading_align_15deg_s": float(time_to_heading_align_15deg[env_id].item()),
                    "path_length_m": path_length_value,
                    "path_efficiency": float(path_efficiency),
                    "min_heading_error_deg": float(torch.rad2deg(min_heading_error[env_id]).item()),
                    "final_heading_error_deg": float(torch.rad2deg(final_heading_error[env_id]).item()),
                    "final_stop_heading_error_deg": float(final_stop_heading_error[env_id].item()),
                    "final_along_track_error_m": float(final_along_error[env_id].item()),
                    "final_cross_track_error_m": float(final_cross_error[env_id].item()),
                    "final_stop_along_track_error_m": float(final_stop_along_error[env_id].item()),
                    "final_stop_cross_track_error_m": float(final_stop_cross_error[env_id].item()),
                }
                case_records.append(record)
                if trajectories_xy is not None:
                    _save_trajectory_plot(
                        output_dir,
                        case_index,
                        case,
                        episode_base_index + env_id,
                        trajectories_xy[env_id],
                        record,
                    )

            remaining -= batch_size

        case_summary = _summarize_records(case_records)
        _print_summary(f"[CASE] distance={case['distance_m']:.2f}m angle={case['angle_deg']:+.0f}deg", case_summary)
        all_records.extend(case_records)

    overall_summary = _summarize_records(all_records)

    by_distance_records: dict[float, list[dict]] = defaultdict(list)
    by_angle_records: dict[float, list[dict]] = defaultdict(list)
    by_case_records: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for record in all_records:
        by_distance_records[record["distance_m"]].append(record)
        by_angle_records[record["angle_deg"]].append(record)
        by_case_records[(record["distance_m"], record["angle_deg"])].append(record)

    distance_summaries = {
        f"{distance:.2f}m": _summarize_records(records) for distance, records in sorted(by_distance_records.items())
    }
    angle_summaries = {
        f"{angle:+.0f}deg": _summarize_records(records) for angle, records in sorted(by_angle_records.items())
    }
    case_summaries = {
        f"{distance:.2f}m_{angle:+.0f}deg": _summarize_records(records)
        for (distance, angle), records in sorted(by_case_records.items())
    }

    _print_summary("[OVERALL]", overall_summary)
    print("[BY_DISTANCE]")
    for distance_key, summary in distance_summaries.items():
        print(f"  {distance_key}: stop@20cm={_format_metric(summary['stop@20cm'])}, "
              f"precise_stop@10cm={_format_metric(summary['precise_stop@10cm'])}, "
              f"timeout_rate={_format_metric(summary['timeout_rate'])}, "
              f"final_error_mean_m={_format_metric(summary['final_error_m_mean'])}")
    print("[BY_ANGLE]")
    for angle_key, summary in angle_summaries.items():
        print(f"  {angle_key}: stop@20cm={_format_metric(summary['stop@20cm'])}, "
              f"precise_stop@10cm={_format_metric(summary['precise_stop@10cm'])}, "
              f"timeout_rate={_format_metric(summary['timeout_rate'])}, "
              f"final_heading_error_deg_mean={_format_metric(summary['final_heading_error_deg_mean'])}")

    payload = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "episodes_per_case": args_cli.episodes_per_case,
        "layout": args_cli.layout,
        "distances_m": distances,
        "angles_deg": angles_deg,
        "ring_radii_m": _parse_float_list(args_cli.ring_radii),
        "points_per_ring": args_cli.points_per_ring,
        "ring_angle_offset_deg": args_cli.ring_angle_offset_deg,
        "timeout_s": args_cli.timeout_s,
        "num_envs": num_envs,
        "benchmark_progress": args_cli.benchmark_progress,
        "stop_hold_s": args_cli.stop_hold_s,
        "reach_distance_m": args_cli.reach_distance,
        "precise_distance_m": args_cli.precise_distance,
        "stop_lin_vel_threshold": args_cli.stop_lin_vel_threshold,
        "stop_yaw_rate_threshold": args_cli.stop_yaw_rate_threshold,
        "heading_align_deg": args_cli.heading_align_deg,
        "save_trajectory_plots": args_cli.save_trajectory_plots,
        "trajectory_plot_format": args_cli.trajectory_plot_format,
        "trajectory_plots_dir": os.path.join(output_dir, "trajectory_plots") if args_cli.save_trajectory_plots else None,
        "applied_schedule": applied_schedule,
        "overall_summary": overall_summary,
        "distance_summaries": distance_summaries,
        "angle_summaries": angle_summaries,
        "case_summaries": case_summaries,
    }
    with open(os.path.join(output_dir, "benchmark_summary.json"), "w", encoding="utf-8") as summary_file:
        json.dump(payload, summary_file, indent=2, sort_keys=True)

    episode_fieldnames = [
        "case_index",
        "point_index",
        "distance_m",
        "angle_deg",
        "start_distance_m",
        "reach_20cm",
        "stop_20cm",
        "precise_stop_10cm",
        "heading_align_15deg",
        "fall",
        "timeout",
        "overshoot",
        "min_error_m",
        "final_error_m",
        "final_stop_error_m",
        "final_radius_error_m",
        "final_stop_radius_error_m",
        "final_target_bearing_error_deg",
        "final_stop_target_bearing_error_deg",
        "time_to_reach_20cm_s",
        "time_to_stop_20cm_s",
        "time_to_precise_stop_10cm_s",
        "time_to_heading_align_15deg_s",
        "path_length_m",
        "path_efficiency",
        "min_heading_error_deg",
        "final_heading_error_deg",
        "final_stop_heading_error_deg",
        "final_along_track_error_m",
        "final_cross_track_error_m",
        "final_stop_along_track_error_m",
        "final_stop_cross_track_error_m",
    ]
    with open(os.path.join(output_dir, "episode_records.csv"), "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=episode_fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    summary_rows = []
    for name, summaries in [("distance", distance_summaries), ("angle", angle_summaries), ("case", case_summaries)]:
        for key, summary in summaries.items():
            row = {"group": name, "key": key}
            row.update(summary)
            summary_rows.append(row)
    with open(os.path.join(output_dir, "group_summaries.csv"), "w", encoding="utf-8", newline="") as csv_file:
        fieldnames = sorted({field for row in summary_rows for field in row.keys()})
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[INFO] Saved benchmark outputs to: {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
