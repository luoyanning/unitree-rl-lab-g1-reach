"""Evaluate the G1 point-goal task with fixed benchmark cases."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib
import math
import os
import statistics
import time

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate the G1 point-goal policy on fixed benchmark cases.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-PointGoal-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file to evaluate.")
parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    required=True,
    help="Frozen low-level velocity-policy checkpoint used by the hierarchical point-goal controller.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--episodes_per_case", type=int, default=20, help="Episodes per benchmark case.")
parser.add_argument("--benchmark", type=str, default="medium", choices=["easy", "medium", "hard"], help="Preset.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.point_goal")
HierarchicalPointGoalVecEnv = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.point_goal.hierarchical_wrapper"
).HierarchicalPointGoalVecEnv


BENCHMARK_CASES = {
    "easy": [(1.0, 0.0), (0.8, 0.8), (0.0, 1.0), (-0.8, 0.8)],
    "medium": [(2.0, 0.0), (1.5, 1.5), (0.0, 2.0), (-1.5, 1.5)],
    "hard": [(3.0, 0.0), (2.2, 2.2), (0.0, 3.0), (-2.2, 2.2)],
}

SUCCESS_DISTANCE = 0.10
SUCCESS_HOLD_TIME_S = 0.4
STOP_VELOCITY_THRESHOLD = 0.10
STOP_YAW_RATE_THRESHOLD = 0.15


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[int(q * 100) - 1]


def _summarize(records: list[dict[str, float | bool]]) -> dict[str, float]:
    total = len(records)
    successes_20 = sum(record["min_error"] <= 0.20 for record in records) / max(total, 1)
    successes_10 = sum(record["success"] for record in records) / max(total, 1)
    successes_5 = sum(record["min_error"] <= 0.05 for record in records) / max(total, 1)
    fall_rate = sum(record["fall"] for record in records) / max(total, 1)
    timeout_rate = sum(record["timeout"] for record in records) / max(total, 1)
    ttg_success = [record["ttg"] for record in records if record["success"]]
    path_eff = [
        min(record["start_distance"] / max(record["path_length"], 1.0e-6), 1.0)
        for record in records
        if record["path_length"] > 0.0
    ]
    stop_quality = [record["stop_quality"] for record in records if record["success"]]
    final_errors = [record["final_error"] for record in records]
    min_errors = [record["min_error"] for record in records]

    return {
        "episodes": float(total),
        "success@20cm": successes_20,
        "success@10cm": successes_10,
        "success@5cm": successes_5,
        "fall_rate": fall_rate,
        "timeout_rate": timeout_rate,
        "ttg_median_s": statistics.median(ttg_success) if ttg_success else float("nan"),
        "ttg_p90_s": _quantile(ttg_success, 0.90) if ttg_success else float("nan"),
        "final_error_mean_m": statistics.fmean(final_errors) if final_errors else float("nan"),
        "min_error_mean_m": statistics.fmean(min_errors) if min_errors else float("nan"),
        "path_efficiency_mean": statistics.fmean(path_eff) if path_eff else float("nan"),
        "stop_quality_mean": statistics.fmean(stop_quality) if stop_quality else float("nan"),
    }


def _print_summary(title: str, metrics: dict[str, float]):
    print(title)
    for key, value in metrics.items():
        if math.isnan(value):
            print(f"  {key}: nan")
        else:
            print(f"  {key}: {value:.4f}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.terminations.point_goal_success = None
    env_cfg.terminations.base_height = None
    env_cfg.terminations.bad_orientation = None
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    low_level_env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    vec_env = HierarchicalPointGoalVecEnv(
        low_level_env,
        low_level_checkpoint_path=retrieve_file_path(args_cli.low_level_checkpoint),
        clip_actions=1.0,
    )
    rsl_args = argparse.Namespace(
        task=args_cli.task,
        seed=None,
        resume=False,
        load_run=None,
        checkpoint=None,
        run_name=None,
        logger=None,
        log_project_name=None,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, rsl_args)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=vec_env.device)

    base_env = env.unwrapped
    command_term = base_env.command_manager.get_term("base_velocity")
    if hasattr(command_term, "cfg") and hasattr(command_term.cfg, "resampling_time_range"):
        command_term.cfg.resampling_time_range = (1.0e9, 1.0e9)

    step_budget = int(round(base_env.cfg.episode_length_s / base_env.step_dt))
    hold_steps_required = max(1, int(round(SUCCESS_HOLD_TIME_S / base_env.step_dt)))

    benchmark_offsets = BENCHMARK_CASES[args_cli.benchmark]
    all_records: list[dict[str, float | bool]] = []
    step_dt = base_env.step_dt

    for offset_xy in benchmark_offsets:
        case_records: list[dict[str, float | bool]] = []
        for _ in range(args_cli.episodes_per_case):
            reset_result = vec_env.reset()
            del reset_result

            goal_pos_w = torch.zeros(base_env.num_envs, 3, device=base_env.device)
            goal_pos_w[:, :2] = base_env.scene.env_origins[:, :2]
            goal_pos_w[:, 0] += float(offset_xy[0])
            goal_pos_w[:, 1] += float(offset_xy[1])
            command_term.set_goal_positions(torch.arange(base_env.num_envs, device=base_env.device), goal_pos_w)

            obs = vec_env.get_observations()

            robot = base_env.scene["robot"]
            last_root_pos = robot.data.root_pos_w[:, :2].clone()
            start_distance = torch.linalg.norm(goal_pos_w[:, :2] - last_root_pos, dim=-1)
            path_length = torch.zeros(base_env.num_envs, device=base_env.device)
            min_error = start_distance.clone()
            final_error = start_distance.clone()
            hold_steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
            ttg = torch.full((base_env.num_envs,), float("nan"), device=base_env.device)
            done = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            success = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            fall = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            timeout = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            stop_quality = torch.zeros(base_env.num_envs, device=base_env.device)

            for step in range(step_budget):
                start_time = time.time()
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, _, _, _ = vec_env.step(actions)

                root_pos = robot.data.root_pos_w[:, :2].clone()
                root_lin_vel = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
                yaw_rate = torch.abs(robot.data.root_ang_vel_w[:, 2])
                root_height = robot.data.root_pos_w[:, 2]
                projected_gravity_z = robot.data.projected_gravity_b[:, 2]
                current_error = torch.linalg.norm(goal_pos_w[:, :2] - root_pos, dim=-1)

                path_length += torch.linalg.norm(root_pos - last_root_pos, dim=-1) * (~done).float()
                min_error = torch.minimum(min_error, current_error)
                final_error = current_error
                last_root_pos = root_pos

                in_success_zone = (
                    (current_error < SUCCESS_DISTANCE)
                    & (root_lin_vel < STOP_VELOCITY_THRESHOLD)
                    & (yaw_rate < STOP_YAW_RATE_THRESHOLD)
                )
                hold_steps = torch.where(in_success_zone & ~done, hold_steps + 1, torch.zeros_like(hold_steps))

                newly_success = (~done) & (hold_steps >= hold_steps_required)
                success |= newly_success
                ttg[newly_success] = (step + 1) * step_dt
                stop_quality[newly_success] = torch.exp(-root_lin_vel[newly_success] / STOP_VELOCITY_THRESHOLD) * torch.exp(
                    -yaw_rate[newly_success] / STOP_YAW_RATE_THRESHOLD
                )

                newly_fall = (~done) & ((root_height < 0.2) | (projected_gravity_z > -0.7))
                fall |= newly_fall
                done |= newly_success | newly_fall
                if torch.all(done):
                    break

                sleep_time = step_dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            timeout |= ~done

            for env_id in range(base_env.num_envs):
                case_records.append(
                    {
                        "success": bool(success[env_id].item()),
                        "fall": bool(fall[env_id].item()),
                        "timeout": bool(timeout[env_id].item()),
                        "ttg": float(ttg[env_id].item()) if success[env_id] else float("nan"),
                        "start_distance": float(start_distance[env_id].item()),
                        "final_error": float(final_error[env_id].item()),
                        "min_error": float(min_error[env_id].item()),
                        "path_length": float(path_length[env_id].item()),
                        "stop_quality": float(stop_quality[env_id].item()),
                    }
                )

        case_metrics = _summarize(case_records)
        _print_summary(f"[CASE] offset={offset_xy}", case_metrics)
        all_records.extend(case_records)

    overall_metrics = _summarize(all_records)
    _print_summary("[OVERALL]", overall_metrics)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
