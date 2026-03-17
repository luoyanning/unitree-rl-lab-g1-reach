# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a hierarchical G1 point-goal checkpoint and optionally record a video."""

import argparse
import importlib
import os
import time

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


POINT_GOAL_TASK = "Unitree-G1-29dof-PointGoal-v0"

parser = argparse.ArgumentParser(description="Play a hierarchical G1 point-goal checkpoint with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=POINT_GOAL_TASK, help="Task name.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    required=True,
    help="Frozen low-level velocity-policy checkpoint used by the hierarchical point-goal controller.",
)
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
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.point_goal")
HierarchicalPointGoalVecEnv = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.point_goal.hierarchical_wrapper"
).HierarchicalPointGoalVecEnv


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    low_level_checkpoint_path = retrieve_file_path(args_cli.low_level_checkpoint)
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    low_level_env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    vec_env = HierarchicalPointGoalVecEnv(
        low_level_env,
        low_level_checkpoint_path=low_level_checkpoint_path,
        clip_actions=agent_cfg.clip_actions,
    )

    print(f"[INFO]: Loading high-level checkpoint from: {resume_path}")
    print(f"[INFO]: Loading low-level checkpoint from: {low_level_checkpoint_path}")
    agent_cfg_dict = agent_cfg.to_dict()
    if not agent_cfg_dict.get("obs_groups"):
        agent_cfg_dict["obs_groups"] = {"policy": ["policy"], "critic": ["critic"]}
    runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=vec_env.device)

    dt = vec_env.unwrapped.step_dt
    reset_result = vec_env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = vec_env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
