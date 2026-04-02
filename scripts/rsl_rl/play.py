# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from rsl_rl_compat import sanitize_runner_cfg  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_folder", type=str, default=None, help="Optional output folder for recorded play videos.")
parser.add_argument(
    "--debug_steps",
    type=int,
    default=0,
    help="Print action, command, and base-motion diagnostics for the first N simulation steps.",
)
parser.add_argument(
    "--use_env_reset",
    dest="use_env_reset",
    action="store_true",
    default=True,
    help="Reset the environment once before playback to initialize commands and task state.",
)
parser.add_argument(
    "--skip_env_reset",
    dest="use_env_reset",
    action="store_false",
    help="Skip the initial env.reset() and read observations directly. Only use for low-level debugging.",
)
parser.add_argument(
    "--debug_command_name",
    type=str,
    default="left_hand_pose",
    help="Command term name to inspect in debug output.",
)
parser.add_argument(
    "--force_world_camera",
    action="store_true",
    default=False,
    help="Override the task camera with a fixed world camera centered on the robot initial pose.",
)
parser.add_argument(
    "--camera_eye_offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional world-camera eye offset relative to the robot initial position.",
)
parser.add_argument(
    "--camera_lookat_offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional world-camera lookat offset relative to the robot initial position.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def validate_checkpoint_path(checkpoint_path: str, flag_name: str = "--checkpoint") -> str:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"{flag_name} resolved to a non-file path: '{checkpoint_path}'")
    if not checkpoint_path.endswith(".pt"):
        raise ValueError(
            f"{flag_name} must point to a model checkpoint '*.pt', but got: '{checkpoint_path}'. "
            "This often happens when a shell variable accidentally captured a TensorBoard event file or run directory."
        )
    return checkpoint_path


def _extract_policy_obs(obs):
    return obs[0] if isinstance(obs, tuple) else obs


def _parse_agent_cfg(task_name: str, args_cli: argparse.Namespace):
    if args_cli.agent in (None, "rsl_rl_cfg_entry_point"):
        return cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    agent_cfg = load_cfg_from_registry(task_name, args_cli.agent)
    if getattr(agent_cfg, "experiment_name", "") == "":
        agent_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")
    return cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)


def _resolve_world_camera_pose(env_cfg, eye_offset=None, lookat_offset=None):
    if not hasattr(env_cfg, "viewer"):
        return None, None
    if not hasattr(env_cfg, "scene") or not hasattr(env_cfg.scene, "robot") or not hasattr(env_cfg.scene.robot, "init_state"):
        return None, None

    init_pos = getattr(env_cfg.scene.robot.init_state, "pos", None)
    if init_pos is None or len(init_pos) < 3:
        return None, None

    if lookat_offset is None:
        lookat_offset = (0.0, 0.0, 0.65)
    if eye_offset is None:
        eye_offset = (2.2, -1.2, 1.15)

    lookat = (
        float(init_pos[0]) + float(lookat_offset[0]),
        float(init_pos[1]) + float(lookat_offset[1]),
        float(init_pos[2]) + float(lookat_offset[2]),
    )
    eye = (
        float(init_pos[0]) + float(eye_offset[0]),
        float(init_pos[1]) + float(eye_offset[1]),
        float(init_pos[2]) + float(eye_offset[2]),
    )

    return eye, lookat


def _set_video_camera(env_cfg, eye_offset=None, lookat_offset=None):
    eye, lookat = _resolve_world_camera_pose(env_cfg, eye_offset=eye_offset, lookat_offset=lookat_offset)
    if eye is None or lookat is None:
        return None, None

    env_cfg.viewer.origin_type = "world"
    if hasattr(env_cfg.viewer, "asset_name"):
        env_cfg.viewer.asset_name = None
    env_cfg.viewer.eye = eye
    env_cfg.viewer.lookat = lookat
    return eye, lookat


def _iter_env_targets(env):
    seen = set()
    stack = [env]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr_name in ("env", "unwrapped"):
            next_env = getattr(current, attr_name, None)
            if next_env is not None and id(next_env) not in seen:
                stack.append(next_env)


def _get_runtime_root_pos(env):
    for target in _iter_env_targets(env):
        scene = getattr(target, "scene", None)
        if scene is None:
            continue
        try:
            robot = scene["robot"]
        except Exception:
            continue
        root_pos_w = getattr(getattr(robot, "data", None), "root_pos_w", None)
        if not isinstance(root_pos_w, torch.Tensor) or root_pos_w.ndim != 2 or root_pos_w.shape[0] == 0:
            continue
        root = root_pos_w[0].detach().cpu().tolist()
        if len(root) < 3:
            continue
        return float(root[0]), float(root[1]), float(root[2])
    return None


def _get_runtime_sim(env):
    for target in _iter_env_targets(env):
        sim = getattr(target, "sim", None)
        if sim is not None and hasattr(sim, "set_camera_view"):
            return sim
    return None


def _apply_runtime_camera(env, env_cfg, eye_offset=None, lookat_offset=None):
    runtime_root = _get_runtime_root_pos(env)
    if runtime_root is None:
        eye, lookat = _resolve_world_camera_pose(env_cfg, eye_offset=eye_offset, lookat_offset=lookat_offset)
        if eye is None or lookat is None:
            return False
    else:
        if lookat_offset is None:
            lookat_offset = (0.0, 0.0, 0.65)
        if eye_offset is None:
            eye_offset = (2.2, -1.2, 1.15)
        eye = (
            runtime_root[0] + float(eye_offset[0]),
            runtime_root[1] + float(eye_offset[1]),
            runtime_root[2] + float(eye_offset[2]),
        )
        lookat = (
            runtime_root[0] + float(lookat_offset[0]),
            runtime_root[1] + float(lookat_offset[1]),
            runtime_root[2] + float(lookat_offset[2]),
        )

    if eye is None or lookat is None:
        return False

    sim = _get_runtime_sim(env)
    if sim is None:
        return False

    try:
        sim.set_camera_view(eye, lookat)
    except Exception:
        return False

    print(f"[PLAY_CAMERA] runtime_world_camera root={runtime_root} eye={eye} lookat={lookat}", flush=True)
    return True


def _get_command_tensor(env, command_name: str):
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return None
    try:
        command_term = command_manager.get_term(command_name)
    except Exception:
        return None
    for attr_name in ("_command", "command"):
        value = getattr(command_term, attr_name, None)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value
    return None


def _format_tensor_sample(value: torch.Tensor | None, max_dim: int = 6) -> str:
    if value is None:
        return "None"
    if value.numel() == 0:
        return "[]"
    sample = value[0, : min(max_dim, value.shape[1])].detach().cpu().tolist()
    rounded = [round(float(x), 4) for x in sample]
    return str(rounded)


def _print_debug_step(
    env,
    actions: torch.Tensor,
    step_index: int,
    command_name: str,
    frame_delta_mean: float | None = None,
    frame_source: str | None = None,
):
    robot = None
    scene = getattr(env, "scene", None)
    if scene is not None:
        try:
            robot = scene["robot"]
        except Exception:
            robot = None
    action_abs_mean = float(actions.detach().abs().mean().item())
    action_l2 = float(torch.linalg.vector_norm(actions[0]).item()) if actions.ndim == 2 and actions.shape[0] > 0 else 0.0

    root_lin_vel = None
    root_pos = None
    if robot is not None and hasattr(robot, "data") and hasattr(robot.data, "root_lin_vel_w"):
        root_lin_vel = robot.data.root_lin_vel_w[0].detach().cpu().tolist()
        root_lin_vel = [round(float(x), 4) for x in root_lin_vel]
    if robot is not None and hasattr(robot, "data") and hasattr(robot.data, "root_pos_w"):
        root_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
        root_pos = [round(float(x), 4) for x in root_pos]

    adapter_command = getattr(env, "_left_hand_adapter_command", None)
    command_tensor = _get_command_tensor(env, command_name)
    has_active_target = getattr(env, "_left_hand_has_active_target", None)
    success_hold = getattr(env, "_left_hand_success_hold_counter", None)

    print(
        "[PLAY_DEBUG] "
        f"step={step_index} "
        f"common_step={int(getattr(env, 'common_step_counter', -1))} "
        f"action_abs_mean={action_abs_mean:.5f} "
        f"action_l2={action_l2:.5f} "
        f"root_pos={root_pos} "
        f"root_lin_vel={root_lin_vel} "
        f"adapter_cmd={_format_tensor_sample(adapter_command)} "
        f"command_tensor={_format_tensor_sample(command_tensor)} "
        f"has_active_target={None if has_active_target is None else bool(has_active_target[0].item())} "
        f"success_hold={None if success_hold is None else int(success_hold[0].item())} "
        f"frame_delta_mean={None if frame_delta_mean is None else round(frame_delta_mean, 4)} "
        f"frame_source={frame_source}",
        flush=True,
    )


def _iter_render_targets(env):
    seen = set()
    current = env
    depth = 0
    while current is not None and id(current) not in seen and depth < 8:
        seen.add(id(current))
        yield f"unwrap_depth_{depth}:{type(current).__name__}", current
        current = getattr(current, "env", None)
        depth += 1

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and id(unwrapped) not in seen:
        seen.add(id(unwrapped))
        yield f"unwrapped:{type(unwrapped).__name__}", unwrapped


def _render_frame(env):
    for source_name, candidate in _iter_render_targets(env):
        render_fn = getattr(candidate, "render", None)
        if not callable(render_fn):
            continue
        try:
            frame = render_fn()
        except Exception:
            continue
        if frame is None:
            continue
        if isinstance(frame, torch.Tensor):
            return frame.detach().cpu(), source_name
        return frame, source_name
    return None, None


def _frame_delta_mean(prev_frame, frame) -> float | None:
    if prev_frame is None or frame is None:
        return None
    try:
        prev_tensor = torch.as_tensor(prev_frame)
        frame_tensor = torch.as_tensor(frame)
        if prev_tensor.shape != frame_tensor.shape:
            return None
        return float((frame_tensor.float() - prev_tensor.float()).abs().mean().item())
    except Exception:
        return None


def main():
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = _parse_agent_cfg(args_cli.task, args_cli)

    # Certain randomizations occur during environment construction, so seed the env config before gym.make.
    env_cfg.seed = agent_cfg.seed
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = validate_checkpoint_path(retrieve_file_path(args_cli.checkpoint))
    else:
        resume_path = validate_checkpoint_path(get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint))

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": args_cli.video_folder or os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    agent_cfg_dict = sanitize_runner_cfg(agent_cfg.to_dict())
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    if not args_cli.use_env_reset:
        print("[INFO] Skipping initial env.reset(); playback may not initialize commands or task state.", flush=True)
    obs = env.reset() if args_cli.use_env_reset else env.get_observations()
    obs = _extract_policy_obs(obs)
    if args_cli.video and args_cli.force_world_camera:
        applied_camera = _apply_runtime_camera(
            env,
            env_cfg,
            eye_offset=args_cli.camera_eye_offset,
            lookat_offset=args_cli.camera_lookat_offset,
        )
        if not applied_camera:
            print("[PLAY_CAMERA] runtime_world_camera unavailable; falling back to task play camera.", flush=True)
    prev_frame, prev_frame_source = _render_frame(env) if args_cli.debug_steps > 0 and args_cli.video else (None, None)

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            obs = _extract_policy_obs(obs)
        frame_delta_mean = None
        frame_source = prev_frame_source
        if timestep < args_cli.debug_steps and args_cli.video:
            frame, frame_source = _render_frame(env)
            frame_delta_mean = _frame_delta_mean(prev_frame, frame)
            prev_frame = frame
            prev_frame_source = frame_source
        if timestep < args_cli.debug_steps:
            _print_debug_step(env.unwrapped, actions, timestep, args_cli.debug_command_name, frame_delta_mean, frame_source)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
