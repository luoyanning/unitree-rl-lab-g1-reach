# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train G1 point-goal navigation by warm-starting from the velocity policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import inspect
import os
import pathlib
import platform
import shutil
import sys
import torch
import types
from datetime import datetime

import argcomplete
import gymnasium as gym
from packaging import version

from isaaclab.app import AppLauncher

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

import cli_args  # isort: skip

POINT_GOAL_TASK = "Unitree-G1-29dof-PointGoal-v0"

parser = argparse.ArgumentParser(description="Train the G1 point-goal policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=POINT_GOAL_TASK, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
    raise SystemExit(1)

from rsl_rl.runners import OnPolicyRunner

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


def resolve_resume_path(log_root_path: str, load_run: str | None, load_checkpoint: str | None) -> str:
    if load_checkpoint is not None:
        has_path_hint = any(sep in load_checkpoint for sep in [os.path.sep, os.path.altsep] if sep is not None)
        if has_path_hint or os.path.exists(os.path.expanduser(load_checkpoint)):
            return retrieve_file_path(load_checkpoint)
    return get_checkpoint_path(log_root_path, load_run, load_checkpoint)


def _merge_actor_input_history(current_weight: torch.Tensor, pretrained_weight: torch.Tensor) -> torch.Tensor:
    history_length = 5
    old_frame_dim = 96
    if pretrained_weight.ndim != 2 or current_weight.ndim != 2:
        return pretrained_weight
    if pretrained_weight.shape[1] != history_length * old_frame_dim:
        return pretrained_weight
    if current_weight.shape[1] % history_length != 0:
        return pretrained_weight

    new_frame_dim = current_weight.shape[1] // history_length
    if new_frame_dim <= old_frame_dim:
        return pretrained_weight

    merged_weight = torch.zeros_like(current_weight)
    for history_index in range(history_length):
        old_offset = history_index * old_frame_dim
        new_offset = history_index * new_frame_dim
        merged_weight[:, new_offset : new_offset + old_frame_dim] = pretrained_weight[
            :, old_offset : old_offset + old_frame_dim
        ]
    return merged_weight


def load_policy_weights_only(
    runner: OnPolicyRunner,
    checkpoint_path: str,
    task_name: str | None = None,
    init_noise_std: float | None = None,
):
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
    for key, value in model_state_dict.items():
        if key not in current_state_dict:
            skipped_keys.append(key)
            continue
        if "critic" in key or "normalizer" in key or key == "std" or key.endswith(".std"):
            skipped_keys.append(key)
            continue
        if not key.startswith("actor."):
            skipped_keys.append(key)
            continue
        if task_name == POINT_GOAL_TASK and key == "actor.0.weight":
            filtered_state_dict[key] = _merge_actor_input_history(
                current_weight=current_state_dict[key],
                pretrained_weight=value,
            )
            continue
        if current_state_dict[key].shape != value.shape:
            skipped_keys.append(key)
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


def install_positive_std_guard(
    runner: OnPolicyRunner,
    std_min: float = 1.0e-3,
    std_max: float = 5.0,
    freeze_std: bool = False,
):
    policy_nn = getattr(runner.alg, "actor_critic", None)
    if policy_nn is None:
        policy_nn = getattr(runner.alg, "policy", None)
    if policy_nn is None or not hasattr(policy_nn, "std"):
        return

    with torch.no_grad():
        policy_nn.std.data.nan_to_num_(nan=std_min, posinf=std_max, neginf=std_min)
        policy_nn.std.data.abs_()
        policy_nn.std.data.clamp_(min=std_min, max=std_max)
    if freeze_std:
        policy_nn.std.requires_grad_(False)

    if getattr(policy_nn, "_codex_positive_std_guard_installed", False):
        return

    original_update_distribution = policy_nn.update_distribution

    def guarded_update_distribution(self, observations):
        with torch.no_grad():
            self.std.data.nan_to_num_(nan=std_min, posinf=std_max, neginf=std_min)
            self.std.data.abs_()
            self.std.data.clamp_(min=std_min, max=std_max)
        return original_update_distribution(observations)

    policy_nn.update_distribution = types.MethodType(guarded_update_distribution, policy_nn)
    policy_nn._codex_positive_std_guard_installed = True
    print(
        "[INFO]: Installed positive action-std guard "
        f"(std_min={std_min}, std_max={std_max}, freeze_std={freeze_std})."
    )


def install_actor_grad_scale(runner: OnPolicyRunner, scale: float):
    if scale >= 1.0:
        return

    policy_nn = getattr(runner.alg, "actor_critic", None)
    if policy_nn is None:
        policy_nn = getattr(runner.alg, "policy", None)
    if policy_nn is None:
        return
    if getattr(policy_nn, "_codex_actor_grad_scale_installed", False):
        return

    for name, parameter in policy_nn.named_parameters():
        if not name.startswith("actor.") or not parameter.requires_grad:
            continue
        parameter.register_hook(lambda grad, scale=scale: grad * scale)

    policy_nn._codex_actor_grad_scale_installed = True
    print(f"[INFO]: Installed actor gradient scaling (scale={scale}).")


def maybe_export_deploy_cfg(env, log_dir: str):
    try:
        export_deploy_cfg(env, log_dir)
    except AttributeError as exc:
        print(
            "[INFO]: Skipping deploy.yaml export for point-goal task because the current deploy exporter "
            f"expects velocity command ranges. Original error: {exc}"
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = resolve_resume_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = None
    warm_start_checkpoint_path = retrieve_file_path(args_cli.init_checkpoint) if args_cli.init_checkpoint else None

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

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_cfg_dict = agent_cfg.to_dict()
    if not agent_cfg_dict.get("obs_groups"):
        agent_cfg_dict["obs_groups"] = {"policy": ["policy"], "critic": ["critic"]}
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading point-goal checkpoint from: {resume_path}")
        runner.load(resume_path)
        install_positive_std_guard(runner, std_min=0.05, std_max=0.20, freeze_std=True)
    else:
        if warm_start_checkpoint_path is not None:
            print(f"[INFO]: Initializing model weights from checkpoint: {warm_start_checkpoint_path}")
            load_policy_weights_only(
                runner,
                warm_start_checkpoint_path,
                task_name=args_cli.task,
                init_noise_std=agent_cfg.policy.init_noise_std,
            )
            install_actor_grad_scale(runner, scale=0.02)
        install_positive_std_guard(runner, std_min=0.05, std_max=0.20, freeze_std=True)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    maybe_export_deploy_cfg(env.unwrapped, log_dir)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
