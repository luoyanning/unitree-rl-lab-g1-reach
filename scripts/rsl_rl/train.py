# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def resolve_resume_path(
    log_root_path: str,
    load_run: str | None,
    load_checkpoint: str | None,
) -> str:
    """Resolve resume checkpoint from either an explicit path or a task log directory."""
    if load_checkpoint is not None:
        has_path_hint = any(sep in load_checkpoint for sep in [os.path.sep, os.path.altsep] if sep is not None)
        if has_path_hint or os.path.exists(os.path.expanduser(load_checkpoint)):
            return retrieve_file_path(load_checkpoint)
    return get_checkpoint_path(log_root_path, load_run, load_checkpoint)


def _left_hand_loco_reach_policy_command_obs_indices() -> list[int]:
    """Indices of the 3 command features in each stacked policy frame for the loco-reach task."""
    history_length = 5
    frame_dim = 96
    command_start = 6
    command_dim = 3
    indices: list[int] = []
    for history_index in range(history_length):
        frame_offset = history_index * frame_dim
        indices.extend(range(frame_offset + command_start, frame_offset + command_start + command_dim))
    return indices


def load_policy_weights_only(
    runner: OnPolicyRunner,
    checkpoint_path: str,
    task_name: str | None = None,
    init_noise_std: float | None = None,
):
    """Initialize policy weights from a checkpoint without restoring optimizer/iteration state."""
    checkpoint = torch.load(checkpoint_path, map_location=runner.device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at '{checkpoint_path}' is not a dict and cannot be used for warm start.")

    model_state_dict = None
    for key in ("model_state_dict", "policy_state_dict", "state_dict", "model"):
        if key in checkpoint:
            model_state_dict = checkpoint[key]
            break
    if model_state_dict is None:
        available_keys = ", ".join(sorted(checkpoint.keys()))
        raise KeyError(
            f"Could not find model weights in checkpoint '{checkpoint_path}'. Available top-level keys: {available_keys}"
        )

    policy_nn = getattr(runner.alg, "actor_critic", None)
    if policy_nn is None:
        policy_nn = getattr(runner.alg, "policy", None)
    if policy_nn is None:
        raise AttributeError("Runner does not expose 'actor_critic' or 'policy'; cannot initialize weights only.")

    current_state_dict = policy_nn.state_dict()
    filtered_state_dict = {}
    skipped_keys = []
    command_obs_indices = (
        _left_hand_loco_reach_policy_command_obs_indices()
        if task_name == "Unitree-G1-29dof-LeftHand-LocoReach-v0"
        else []
    )

    for key, value in model_state_dict.items():
        if key not in current_state_dict:
            skipped_keys.append(key)
            continue
        if current_state_dict[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        if "critic" in key or "normalizer" in key or key == "std" or key.endswith(".std"):
            skipped_keys.append(key)
            continue
        if not key.startswith("actor."):
            skipped_keys.append(key)
            continue
        if command_obs_indices and key == "actor.0.weight":
            merged_weight = value.clone()
            merged_weight[:, command_obs_indices] = current_state_dict[key][:, command_obs_indices]
            filtered_state_dict[key] = merged_weight
            continue
        filtered_state_dict[key] = value

    incompatible = policy_nn.load_state_dict(filtered_state_dict, strict=False)
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

    if init_noise_std is not None and hasattr(policy_nn, "std"):
        with torch.no_grad():
            policy_nn.std.fill_(float(init_noise_std))

    if missing_keys or unexpected_keys:
        print("[INFO]: Weights-only warm start loaded with non-strict state-dict matching.")
        if missing_keys:
            print(f"[INFO]: Missing keys ({len(missing_keys)}): {missing_keys}")
        if unexpected_keys:
            print(f"[INFO]: Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
    else:
        print("[INFO]: Weights-only warm start matched the policy state dict exactly.")
    print(
        "[INFO]: Warm-start policy transfer summary: "
        f"loaded_actor_keys={len(filtered_state_dict)} skipped_keys={len(skipped_keys)} "
        f"reset_std_to={init_noise_std if init_noise_std is not None else 'unchanged'}"
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = resolve_resume_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = None
    init_checkpoint_path = retrieve_file_path(args_cli.init_checkpoint) if args_cli.init_checkpoint else None

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if init_checkpoint_path is not None:
            print("[INFO]: Ignoring --init_checkpoint because --resume is active.")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    elif init_checkpoint_path is not None:
        print(f"[INFO]: Initializing model weights from checkpoint: {init_checkpoint_path}")
        load_policy_weights_only(
            runner,
            init_checkpoint_path,
            task_name=args_cli.task,
            init_noise_std=agent_cfg.policy.init_noise_std,
        )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
