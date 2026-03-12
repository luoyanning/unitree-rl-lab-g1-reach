# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone benchmark scene player with zero actions and optional video recording."""

import argparse
import os
import time
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a benchmark scene without loading an RL checkpoint.")
parser.add_argument(
    "--task",
    type=str,
    default="Unitree-G1-29dof-Benchmark-v1",
    help="Name of the registered benchmark task.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record a short video.")
parser.add_argument("--video_length", type=int, default=240, help="Recorded video length in simulation steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=0, help="Optional hard stop after N steps. Zero means no limit.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True
if getattr(args_cli, "headless", False) and not args_cli.video and args_cli.num_steps == 0:
    args_cli.num_steps = 300

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _get_num_envs(env) -> int:
    return getattr(env.unwrapped, "num_envs", env.unwrapped.scene.num_envs)


def _get_action_dim(env) -> int:
    return sum(term.action_dim for term in env.unwrapped.action_manager._terms.values())


def _zeros_action(env) -> torch.Tensor:
    return torch.zeros(
        (_get_num_envs(env), _get_action_dim(env)),
        device=env.unwrapped.device,
        dtype=torch.float32,
    )


def _done(flag) -> bool:
    if isinstance(flag, bool):
        return flag
    if hasattr(flag, "any"):
        return bool(flag.any())
    return bool(flag)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    video_dir = None
    if args_cli.video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = os.path.abspath(
            os.path.join("logs", "benchmarks", args_cli.task.lower().replace("-", "_"), "videos", timestamp)
        )
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording benchmark video to: {video_dir}")

    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    del obs

    zero_action = _zeros_action(env)
    step_dt = env.unwrapped.step_dt
    step_count = 0

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            step_result = env.step(zero_action)

        if len(step_result) == 5:
            _, _, terminated, truncated, _ = step_result
            if _done(terminated) or _done(truncated):
                env.reset()
        else:
            _, _, terminated, _ = step_result
            if _done(terminated):
                env.reset()

        step_count += 1
        if args_cli.video and step_count >= args_cli.video_length:
            break
        if args_cli.num_steps > 0 and step_count >= args_cli.num_steps:
            break

        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
