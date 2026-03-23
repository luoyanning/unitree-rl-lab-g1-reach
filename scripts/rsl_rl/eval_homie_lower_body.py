"""Evaluate the HOMIE-inspired G1 lower-body task under fixed checkpoints."""

"""Launch Isaac Sim Simulator first."""

import argparse
import statistics
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate a trained G1 HOMIE lower-body checkpoint.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-HomieLowerBody-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file to evaluate.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of evaluation environments.")
parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
parser.add_argument(
    "--disable_disturbances",
    action="store_true",
    default=False,
    help="Disable random pushes and startup randomization for deterministic evaluation.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time when possible.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.pstdev(values)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.disable_disturbances and hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
        if hasattr(env_cfg.events, "physics_material"):
            env_cfg.events.physics_material = None
        if hasattr(env_cfg.events, "add_base_mass"):
            env_cfg.events.add_base_mass = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped

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

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=vec_env.device)

    obs = vec_env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    records: list[dict[str, float]] = []
    step_dt = base_env.step_dt

    while simulation_app.is_running() and len(records) < args_cli.episodes:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = vec_env.step(actions)

        done_ids = torch.nonzero(dones > 0, as_tuple=False).flatten()
        if len(done_ids) > 0:
            episode_metrics = base_env.get_last_episode_metrics()
            for env_id in done_ids.tolist():
                records.append(
                    {
                        "linear_velocity_tracking_error": float(
                            episode_metrics["linear_velocity_tracking_error"][env_id].item()
                        ),
                        "forward_velocity_tracking_error": float(
                            episode_metrics["forward_velocity_tracking_error"][env_id].item()
                        ),
                        "lateral_velocity_tracking_error": float(
                            episode_metrics["lateral_velocity_tracking_error"][env_id].item()
                        ),
                        "yaw_tracking_error": float(episode_metrics["yaw_tracking_error"][env_id].item()),
                        "height_tracking_error": float(episode_metrics["height_tracking_error"][env_id].item()),
                        "symmetry_joint_error": float(episode_metrics["symmetry_joint_error"][env_id].item()),
                        "survival_time": float(episode_metrics["survival_time"][env_id].item()),
                    }
                )
                if len(records) >= args_cli.episodes:
                    break

        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    lin_err_mean, lin_err_std = _summarize([r["linear_velocity_tracking_error"] for r in records])
    fwd_err_mean, fwd_err_std = _summarize([r["forward_velocity_tracking_error"] for r in records])
    lat_err_mean, lat_err_std = _summarize([r["lateral_velocity_tracking_error"] for r in records])
    yaw_err_mean, yaw_err_std = _summarize([r["yaw_tracking_error"] for r in records])
    height_err_mean, height_err_std = _summarize([r["height_tracking_error"] for r in records])
    survival_mean, survival_std = _summarize([r["survival_time"] for r in records])
    symmetry_mean, symmetry_std = _summarize([r["symmetry_joint_error"] for r in records])

    print("[EVAL] G1 HOMIE lower-body policy")
    print(f"  episodes: {len(records)}")
    print(f"  linear_velocity_tracking_error_mean: {lin_err_mean:.6f}")
    print(f"  linear_velocity_tracking_error_std:  {lin_err_std:.6f}")
    print(f"  forward_velocity_tracking_error_mean:{fwd_err_mean:.6f}")
    print(f"  forward_velocity_tracking_error_std: {fwd_err_std:.6f}")
    print(f"  lateral_velocity_tracking_error_mean:{lat_err_mean:.6f}")
    print(f"  lateral_velocity_tracking_error_std: {lat_err_std:.6f}")
    print(f"  yaw_tracking_error_mean:            {yaw_err_mean:.6f}")
    print(f"  yaw_tracking_error_std:             {yaw_err_std:.6f}")
    print(f"  height_tracking_error_mean:         {height_err_mean:.6f}")
    print(f"  height_tracking_error_std:          {height_err_std:.6f}")
    print(f"  survival_time_mean_s:               {survival_mean:.6f}")
    print(f"  survival_time_std_s:                {survival_std:.6f}")
    print(f"  symmetry_joint_error_mean:          {symmetry_mean:.6f}")
    print(f"  symmetry_joint_error_std:           {symmetry_std:.6f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
