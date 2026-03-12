# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    # ===== scene test overrides =====
    env_cfg.scene.num_envs = 1

    # use scanned USD scene
    env_cfg.scene.terrain.terrain_type = "usd"
    env_cfg.scene.terrain.usd_path = "/mlp_vepfs/share/lyn/try0310/scene_usd/only_N3_nohole_cleaned.usd"
    env_cfg.scene.terrain.env_spacing = 2.5
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.use_terrain_origins = False

    # disable terrain ray-caster style sensors for custom scene preview
    for _name in ("height_scanner", "terrain_scanner", "ray_caster", "raycast_sensor"):
        if hasattr(env_cfg.scene, _name):
            setattr(env_cfg.scene, _name, None)

    # command ranges frozen; actual commands will be injected manually in the play loop
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    # disable perturbations / curriculum
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base = None
    if hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None
    if hasattr(env_cfg.curriculum, "lin_vel_cmd_levels"):
        env_cfg.curriculum.lin_vel_cmd_levels = None

    # fixed world camera, not robot-follow camera
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (8.0, -8.0, 12.0)
    env_cfg.viewer.lookat = (8.0, 6.0, 1.0)

    # try a simple spawn above the scene center
    if hasattr(env_cfg.scene, "robot") and hasattr(env_cfg.scene.robot, "init_state"):
        env_cfg.scene.robot.init_state.pos = (7.0, 0.0, 7.0)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
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
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
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

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0

    # freeze command resampling and manually inject route commands
    base_env = env.unwrapped
    cmd_term = base_env.command_manager._terms["base_velocity"]
    if hasattr(cmd_term, "cfg") and hasattr(cmd_term.cfg, "resampling_time_range"):
        cmd_term.cfg.resampling_time_range = (1.0e9, 1.0e9)

    def set_base_command(vx, vy, wz):
        if hasattr(cmd_term, "_command"):
            cmd = cmd_term._command
        elif hasattr(cmd_term, "command"):
            cmd = cmd_term.command
        else:
            raise RuntimeError("Could not find command tensor on base_velocity term.")
        cmd[:, 0] = vx
        cmd[:, 1] = vy
        cmd[:, 2] = wz

    # Route (approximate):
    # segment 1: forward 1.5m      -> 0.5 m/s * 3.0s  => 150 steps
    # turn 1: left about 90 deg    -> 0.8 rad/s * 2.0s => ~1.6 rad
    # segment 2: forward 1.5m
    # turn 2: left about 90 deg
    # segment 3: forward 1.5m
    #
    # step_dt is 0.02s, so:
    # 3.0s  = 150 steps
    # 2.0s  = 100 steps
    route = [
        (150, (0.5, 0.0, 0.0)),  # +x about 1.5m
        (450, (0.0, 0.0, 0.8)),  # turn left
        (600, (0.5, 0.0, 0.0)),  # then +y about 1.5m in world frame approximately
        (900, (0.0, 0.0, 0.8)),  # turn left
        (1050, (0.5, 0.0, 0.0)),  # then -x about 1.5m in world frame approximately
        (1200, (0.0, 0.0, 0.0)),  # stop
    ]

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        if timestep < route[0][0]:
            vx, vy, wz = route[0][1]
        elif timestep < route[1][0]:
            vx, vy, wz = route[1][1]
        elif timestep < route[2][0]:
            vx, vy, wz = route[2][1]
        elif timestep < route[3][0]:
            vx, vy, wz = route[3][1]
        elif timestep < route[4][0]:
            vx, vy, wz = route[4][1]
        else:
            vx, vy, wz = route[5][1]

        set_base_command(vx, vy, wz)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

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
