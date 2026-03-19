#!/usr/bin/env python3

"""Benchmark the two-model G1 local pick-place pipeline in one fixed scene."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from collections import deque
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark the two-model local pick-place pipeline.")
parser.add_argument(
    "--task",
    type=str,
    default="Unitree-G1-29dof-LeftHand-PickPlace-Local-v0",
    help="Physical scene task used for the benchmark.",
)
parser.add_argument("--nav_checkpoint", type=str, required=True, help="Navigation policy checkpoint.")
parser.add_argument(
    "--nav_low_level_checkpoint",
    type=str,
    default=None,
    help="Required only when the navigation checkpoint is a PointGoal-style high-level policy.",
)
parser.add_argument("--manip_checkpoint", type=str, required=True, help="Manipulation policy checkpoint.")
parser.add_argument("--num_episodes", type=int, default=20, help="Number of benchmark episodes.")
parser.add_argument("--seed", type=int, default=0, help="Torch seed.")
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
parser.add_argument("--ball_x_range", type=str, default="0.78,1.00", help="Ball x-range on the table.")
parser.add_argument("--ball_y_range", type=str, default="0.06,0.22", help="Ball y-range on the table.")
parser.add_argument("--goal_x_range", type=str, default="0.84,1.10", help="Goal x-range on the table.")
parser.add_argument("--goal_y_range", type=str, default="0.02,0.24", help="Goal y-range on the table.")
parser.add_argument("--max_nav_steps", type=int, default=500, help="Max steps for each navigation phase.")
parser.add_argument("--max_manip_steps", type=int, default=280, help="Max steps for each manipulation phase.")
parser.add_argument("--nav_pos_tol", type=float, default=0.08, help="Navigation xy success tolerance in meters.")
parser.add_argument("--nav_yaw_tol_deg", type=float, default=10.0, help="Navigation yaw success tolerance in degrees.")
parser.add_argument("--nav_speed_tol", type=float, default=0.08, help="Navigation linear speed tolerance.")
parser.add_argument("--release_goal_tol", type=float, default=0.08, help="Final ball-goal tolerance after release.")
parser.add_argument("--output_dir", type=str, default=None, help="Benchmark output directory.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

pp_mdp = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.left_hand_pick_place_local.left_hand_pick_place_local_mdp"
)


POINT_GOAL_MIN_LIN_VEL_X = -0.12
POINT_GOAL_MAX_LIN_VEL_X = 0.45
POINT_GOAL_MAX_LIN_VEL_Y = 0.18
POINT_GOAL_MAX_ANG_VEL_Z = 0.30
POINT_GOAL_TERMINAL_SLOW_DISTANCE = 0.30
POINT_GOAL_TERMINAL_LATCH_DISTANCE = 0.16
POINT_GOAL_TERMINAL_MAX_LIN_VEL_X = 0.10
POINT_GOAL_TERMINAL_MAX_LIN_VEL_Y = 0.06
POINT_GOAL_TERMINAL_MAX_ANG_VEL_Z = 0.25
POINT_GOAL_TERMINAL_SETTLE_LIN_VEL_X = 0.05
POINT_GOAL_TERMINAL_SETTLE_REVERSE_LIN_VEL_X = 0.04


def _parse_range(raw: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 2:
        raise ValueError(f"Expected exactly two comma-separated values, got: {raw}")
    return values[0], values[1]


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _get_action_dim(env) -> int:
    return sum(term.action_dim for term in env.unwrapped.action_manager._terms.values())


class FrozenActor(nn.Module):
    def __init__(self, actor_state_dict: dict[str, torch.Tensor]):
        super().__init__()
        linear_ids = sorted(
            int(key.split(".")[0])
            for key in actor_state_dict
            if key.endswith(".weight") and key.split(".")[0].isdigit()
        )
        if not linear_ids:
            raise ValueError("Checkpoint does not contain actor linear layers.")

        layers = []
        for index, layer_id in enumerate(linear_ids):
            weight = actor_state_dict[f"{layer_id}.weight"]
            out_dim, in_dim = weight.shape
            layers.append(nn.Linear(in_dim, out_dim))
            if index < len(linear_ids) - 1:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)
        self.actor.load_state_dict(actor_state_dict, strict=True)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str | torch.device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint at '{checkpoint_path}' is not a dict.")

        model_state_dict = None
        for key in ("model_state_dict", "policy_state_dict", "state_dict", "model"):
            if key in checkpoint:
                model_state_dict = checkpoint[key]
                break
        if model_state_dict is None:
            raise KeyError(f"Could not find model weights in checkpoint '{checkpoint_path}'.")

        actor_state_dict = {
            key[len("actor.") :]: value for key, value in model_state_dict.items() if key.startswith("actor.")
        }
        if not actor_state_dict:
            raise KeyError(f"Checkpoint '{checkpoint_path}' does not contain actor weights.")

        module = cls(actor_state_dict)
        module.to(device)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return module

    @property
    def input_dim(self) -> int:
        return self.actor[0].in_features

    @property
    def output_dim(self) -> int:
        return self.actor[-1].out_features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)


class HistoryStack:
    def __init__(self, history_length: int):
        self.history_length = history_length
        self._frames: deque[torch.Tensor] = deque(maxlen=history_length)

    def reset(self, frame: torch.Tensor):
        self._frames.clear()
        for _ in range(self.history_length):
            self._frames.append(frame.clone())

    def append(self, frame: torch.Tensor):
        if not self._frames:
            self.reset(frame)
            return
        self._frames.append(frame.clone())

    def stacked(self) -> torch.Tensor:
        return torch.cat(list(self._frames), dim=-1)


def _build_policy_frame(env, command: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
    robot = env.scene["robot"]
    base_ang_vel = robot.data.root_ang_vel_b * 0.2
    projected_gravity = robot.data.projected_gravity_b
    joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
    joint_vel_rel = robot.data.joint_vel * 0.05
    return torch.cat((base_ang_vel, projected_gravity, command, joint_pos_rel, joint_vel_rel, last_action), dim=-1)


def _sample_position(range_x: tuple[float, float], range_y: tuple[float, float], device: torch.device) -> torch.Tensor:
    pos = torch.zeros(1, 3, device=device)
    pos[:, 0] = range_x[0] + (range_x[1] - range_x[0]) * torch.rand(1, device=device)
    pos[:, 1] = range_y[0] + (range_y[1] - range_y[0]) * torch.rand(1, device=device)
    pos[:, 2] = pp_mdp.BALL_CENTER_Z
    return pos


def _compute_stance_goal(target_pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base_goal = target_pos_w.clone()
    base_goal[:, 0] = target_pos_w[:, 0] - 0.46
    base_goal[:, 1] = target_pos_w[:, 1] - 0.18
    base_goal[:, 2] = 0.0
    base_yaw = torch.zeros(target_pos_w.shape[0], device=target_pos_w.device)
    return base_goal, base_yaw


def _goal_rel_body_xy(env, goal_pos_w: torch.Tensor) -> torch.Tensor:
    robot = env.scene["robot"]
    goal_delta_w = torch.zeros(env.num_envs, 3, device=env.device)
    goal_delta_w[:, :2] = goal_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), goal_delta_w)[:, :2]


def _base_lin_vel_body_xy(env) -> torch.Tensor:
    robot = env.scene["robot"]
    vel_w = torch.zeros(env.num_envs, 3, device=env.device)
    vel_w[:, :2] = robot.data.root_lin_vel_w[:, :2]
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), vel_w)[:, :2]


def _build_point_goal_actor_obs(
    env,
    goal_pos_w: torch.Tensor,
    policy_command: torch.Tensor,
    min_goal_distance: torch.Tensor,
) -> torch.Tensor:
    del min_goal_distance
    robot = env.scene["robot"]
    goal_rel_body = _goal_rel_body_xy(env, goal_pos_w)
    goal_distance = torch.linalg.norm(goal_rel_body, dim=-1, keepdim=True)
    goal_heading = torch.atan2(goal_rel_body[:, 1], goal_rel_body[:, 0]).unsqueeze(-1)
    base_lin_vel_body_xy = _base_lin_vel_body_xy(env)
    base_ang_vel = robot.data.root_ang_vel_b
    projected_gravity = robot.data.projected_gravity_b
    remaining_time_fraction = torch.ones(env.num_envs, 1, device=env.device)
    return torch.cat(
        (
            goal_rel_body,
            goal_distance,
            goal_heading,
            base_lin_vel_body_xy,
            base_ang_vel,
            projected_gravity,
            policy_command,
            remaining_time_fraction,
        ),
        dim=-1,
    )


def _compute_nav_command(env, goal_pos_w: torch.Tensor) -> torch.Tensor:
    goal_delta_body = _goal_rel_body_xy(env, goal_pos_w)
    goal_delta_w = goal_pos_w[:, :2] - env.scene["robot"].data.root_pos_w[:, :2]
    goal_distance = torch.linalg.norm(goal_delta_w, dim=-1)
    heading_error = _wrap_to_pi(torch.atan2(goal_delta_body[:, 1], goal_delta_body[:, 0]))

    lin_x = torch.clamp(1.1 * goal_delta_body[:, 0], min=-0.12, max=0.45)
    lin_y = torch.clamp(0.4 * goal_delta_body[:, 1], min=-0.18, max=0.18)
    ang_z = torch.clamp(1.0 * heading_error, min=-0.30, max=0.30)

    distance_scale = torch.clamp(goal_distance / 0.55, min=0.0, max=1.0)
    stop_scale = torch.clamp((goal_distance - 0.08) / 0.12, min=0.0, max=1.0)
    lin_x = lin_x * distance_scale * stop_scale
    lin_y = lin_y * distance_scale * stop_scale
    ang_z = ang_z * torch.clamp(goal_distance / 0.60, min=0.0, max=1.0)

    hold_mask = goal_distance < 0.08
    lin_x = torch.where(hold_mask, torch.zeros_like(lin_x), lin_x)
    lin_y = torch.where(hold_mask, torch.zeros_like(lin_y), lin_y)
    ang_z = torch.where(hold_mask, torch.zeros_like(ang_z), ang_z)
    return torch.stack((lin_x, lin_y, ang_z), dim=-1)


def _scale_nav_policy_command(actions: torch.Tensor) -> torch.Tensor:
    clipped_actions = torch.clamp(actions, -1.0, 1.0)
    policy_command = torch.zeros(actions.shape[0], 3, device=actions.device)
    forward_branch = torch.clamp(clipped_actions[:, 0], min=0.0) * POINT_GOAL_MAX_LIN_VEL_X
    reverse_branch = torch.clamp(clipped_actions[:, 0], max=0.0) * abs(POINT_GOAL_MIN_LIN_VEL_X)
    policy_command[:, 0] = forward_branch + reverse_branch
    policy_command[:, 1] = clipped_actions[:, 1] * POINT_GOAL_MAX_LIN_VEL_Y
    policy_command[:, 2] = clipped_actions[:, 2] * POINT_GOAL_MAX_ANG_VEL_Z
    return policy_command


def _apply_terminal_command_cap(
    env,
    goal_pos_w: torch.Tensor,
    policy_command: torch.Tensor,
    min_goal_distance: torch.Tensor,
) -> torch.Tensor:
    goal_rel_body = _goal_rel_body_xy(env, goal_pos_w)
    goal_distance = torch.linalg.norm(goal_rel_body, dim=-1)
    min_goal_distance[:] = torch.minimum(min_goal_distance, goal_distance)
    settle_latched = min_goal_distance < POINT_GOAL_TERMINAL_LATCH_DISTANCE
    terminal_gate = torch.clamp(goal_distance / max(POINT_GOAL_TERMINAL_SLOW_DISTANCE, 1.0e-6), min=0.0, max=1.0)

    lin_x_cap = POINT_GOAL_TERMINAL_MAX_LIN_VEL_X + (POINT_GOAL_MAX_LIN_VEL_X - POINT_GOAL_TERMINAL_MAX_LIN_VEL_X) * terminal_gate
    lin_y_cap = POINT_GOAL_TERMINAL_MAX_LIN_VEL_Y + (POINT_GOAL_MAX_LIN_VEL_Y - POINT_GOAL_TERMINAL_MAX_LIN_VEL_Y) * terminal_gate
    ang_z_cap = POINT_GOAL_TERMINAL_MAX_ANG_VEL_Z + (POINT_GOAL_MAX_ANG_VEL_Z - POINT_GOAL_TERMINAL_MAX_ANG_VEL_Z) * terminal_gate
    reverse_cap = torch.clamp(lin_x_cap, max=abs(POINT_GOAL_MIN_LIN_VEL_X))

    capped_command = policy_command.clone()
    capped_command[:, 0] = torch.maximum(torch.minimum(capped_command[:, 0], lin_x_cap), -reverse_cap)
    capped_command[:, 1] = torch.clamp(capped_command[:, 1], min=-lin_y_cap, max=lin_y_cap)
    capped_command[:, 2] = torch.clamp(capped_command[:, 2], min=-ang_z_cap, max=ang_z_cap)

    if torch.any(settle_latched):
        settle_forward = torch.clamp(
            goal_rel_body[:, 0] * 0.8,
            min=-POINT_GOAL_TERMINAL_SETTLE_REVERSE_LIN_VEL_X,
            max=POINT_GOAL_TERMINAL_SETTLE_LIN_VEL_X,
        )
        settle_forward = torch.where(torch.abs(goal_rel_body[:, 0]) < 0.02, torch.zeros_like(settle_forward), settle_forward)
        capped_command[settle_latched, 0] = settle_forward[settle_latched]
        capped_command[settle_latched, 1] = 0.0
        capped_command[settle_latched, 2] = 0.0
    return capped_command


def _nav_success(
    env,
    goal_pos_w: torch.Tensor,
    goal_yaw: torch.Tensor,
    pos_tol: float,
    yaw_tol: float,
    speed_tol: float,
):
    robot = env.scene["robot"]
    pos_error = torch.linalg.norm(goal_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    root_yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w)
    yaw_error = torch.abs(_wrap_to_pi(goal_yaw - root_yaw))
    base_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    yaw_rate = torch.abs(robot.data.root_ang_vel_w[:, 2])
    return (pos_error <= pos_tol) & (yaw_error <= yaw_tol) & (base_speed <= speed_tol) & (yaw_rate <= 0.25)


def _build_left_arm_override(env, scale: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    joint_ids = torch.tensor(
        robot.find_joints(
            [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
            preserve_order=True,
        )[0],
        device=env.device,
        dtype=torch.long,
    )
    desired_joint_pos = torch.tensor([[0.42, 0.34, 0.00, 1.00, 0.10, 0.0, 0.0]], device=env.device)
    default_joint_pos = robot.data.default_joint_pos[:, joint_ids]
    normalized_action = torch.clamp((desired_joint_pos - default_joint_pos) / scale, min=-1.0, max=1.0)
    return joint_ids, normalized_action


def _apply_left_arm_override(action: torch.Tensor, joint_ids: torch.Tensor, normalized_action: torch.Tensor):
    action[:, joint_ids] = normalized_action
    return action


class NavigationController:
    def __init__(
        self,
        env,
        nav_checkpoint: str,
        nav_low_level_checkpoint: str | None,
    ):
        self.env = env
        self.device = env.device
        self.action_dim = _get_action_dim(env)
        self.nav_actor = FrozenActor.from_checkpoint(nav_checkpoint, device=self.device)
        self.mode = ""

        if self.nav_actor.output_dim == 3:
            if nav_low_level_checkpoint is None:
                raise ValueError(
                    "The navigation checkpoint outputs 3 actions, so it looks like a PointGoal high-level policy. "
                    "Please also pass --nav_low_level_checkpoint."
                )
            self.mode = "point_goal_high_level"
            self.nav_low_level_actor = FrozenActor.from_checkpoint(nav_low_level_checkpoint, device=self.device)
            if self.nav_low_level_actor.output_dim != self.action_dim:
                raise ValueError(
                    "The navigation low-level checkpoint does not match the robot action dimension: "
                    f"{self.nav_low_level_actor.output_dim} vs {self.action_dim}."
                )
            self.low_level_last_action = torch.zeros(env.num_envs, self.action_dim, device=self.device)
            self.policy_command = torch.zeros(env.num_envs, 3, device=self.device)
            self.min_goal_distance = torch.full((env.num_envs,), float("inf"), device=self.device)
            self.low_level_history = HistoryStack(history_length=5)
        elif self.nav_actor.output_dim == self.action_dim:
            self.mode = "direct_joint_actor"
            self.low_level_last_action = torch.zeros(env.num_envs, self.action_dim, device=self.device)
            self.low_level_history = HistoryStack(history_length=5)
        else:
            raise ValueError(
                "Unsupported navigation checkpoint output dim. Expected either 3 "
                f"(PointGoal high-level) or {self.action_dim} (direct joint actor), got {self.nav_actor.output_dim}."
            )

    def reset(self, goal_pos_w: torch.Tensor):
        self.low_level_last_action.zero_()
        if self.mode == "point_goal_high_level":
            self.policy_command.zero_()
            self.min_goal_distance.fill_(float("inf"))
            frame = _build_policy_frame(self.env, self.policy_command, self.low_level_last_action)
            self.low_level_history.reset(frame)
            return

        nav_command = _compute_nav_command(self.env, goal_pos_w)
        frame = _build_policy_frame(self.env, nav_command, self.low_level_last_action)
        self.low_level_history.reset(frame)

    def act(self, goal_pos_w: torch.Tensor) -> torch.Tensor:
        if self.mode == "point_goal_high_level":
            actor_obs = _build_point_goal_actor_obs(
                self.env,
                goal_pos_w=goal_pos_w,
                policy_command=self.policy_command,
                min_goal_distance=self.min_goal_distance,
            )
            with torch.inference_mode():
                nav_actions = self.nav_actor(actor_obs)
            self.policy_command = _scale_nav_policy_command(nav_actions)
            self.policy_command = _apply_terminal_command_cap(
                self.env,
                goal_pos_w=goal_pos_w,
                policy_command=self.policy_command,
                min_goal_distance=self.min_goal_distance,
            )
            self.low_level_history.append(
                _build_policy_frame(self.env, self.policy_command, self.low_level_last_action)
            )
            with torch.inference_mode():
                action = self.nav_low_level_actor(self.low_level_history.stacked())
            self.low_level_last_action = action
            return action

        nav_command = _compute_nav_command(self.env, goal_pos_w)
        self.low_level_history.append(_build_policy_frame(self.env, nav_command, self.low_level_last_action))
        with torch.inference_mode():
            action = self.nav_actor(self.low_level_history.stacked())
        self.low_level_last_action = action
        return action


def _ensure_output_dir(path: str | None) -> str:
    if path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join("logs", "benchmark_pick_place", timestamp)
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def main():
    torch.manual_seed(args_cli.seed)

    ball_x_range = _parse_range(args_cli.ball_x_range)
    ball_y_range = _parse_range(args_cli.ball_y_range)
    goal_x_range = _parse_range(args_cli.goal_x_range)
    goal_y_range = _parse_range(args_cli.goal_y_range)
    yaw_tol = math.radians(args_cli.nav_yaw_tol_deg)
    output_dir = _ensure_output_dir(args_cli.output_dir)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 60.0
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.terminations.task_success = None
    env_cfg.terminations.task_timeout = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    device = base_env.device

    nav_controller = NavigationController(
        base_env,
        nav_checkpoint=retrieve_file_path(args_cli.nav_checkpoint),
        nav_low_level_checkpoint=(
            retrieve_file_path(args_cli.nav_low_level_checkpoint) if args_cli.nav_low_level_checkpoint else None
        ),
    )
    manip_actor = FrozenActor.from_checkpoint(retrieve_file_path(args_cli.manip_checkpoint), device=device)
    action_dim = _get_action_dim(base_env)
    if manip_actor.output_dim != action_dim:
        raise ValueError(
            "The manipulation checkpoint does not match the robot action dimension: "
            f"{manip_actor.output_dim} vs {action_dim}."
        )
    manip_last_action = torch.zeros(base_env.num_envs, action_dim, device=device)
    manip_history = HistoryStack(history_length=5)
    left_arm_joint_ids, carry_override_action = _build_left_arm_override(base_env)

    records: list[dict] = []

    for episode_index in range(args_cli.num_episodes):
        env.reset()
        manip_last_action.zero_()

        ball_pos_w = _sample_position(ball_x_range, ball_y_range, device=device)
        goal_pos_w = _sample_position(goal_x_range, goal_y_range, device=device)
        if torch.linalg.norm(goal_pos_w[:, :2] - ball_pos_w[:, :2], dim=-1).item() < 0.12:
            goal_pos_w[:, 1] = torch.clamp(goal_pos_w[:, 1] + 0.14, min=goal_y_range[0], max=goal_y_range[1])

        pp_mdp.set_pick_place_benchmark_state(
            base_env,
            ball_pos_w=ball_pos_w,
            goal_pos_w=goal_pos_w,
            mode="acquire",
            attach=False,
        )
        pp_mdp.refresh_pick_place_state(base_env)

        pick_goal_pos_w, pick_goal_yaw = _compute_stance_goal(ball_pos_w)
        place_goal_pos_w, place_goal_yaw = _compute_stance_goal(goal_pos_w)

        stage = "nav_to_pick"
        stage_step = 0
        pick_steps = 0
        place_steps = 0
        success = False
        failure_stage = ""

        nav_controller.reset(pick_goal_pos_w)
        manip_command = pp_mdp.target_pos_command_obs(base_env)
        manip_history.reset(_build_policy_frame(base_env, manip_command, manip_last_action))

        total_budget = 2 * args_cli.max_nav_steps + 2 * args_cli.max_manip_steps + 60
        for _ in range(total_budget):
            if stage == "nav_to_pick":
                action = nav_controller.act(pick_goal_pos_w)
                action = _apply_left_arm_override(action, left_arm_joint_ids, carry_override_action)
                next_stage = _nav_success(
                    base_env,
                    pick_goal_pos_w,
                    pick_goal_yaw,
                    pos_tol=args_cli.nav_pos_tol,
                    yaw_tol=yaw_tol,
                    speed_tol=args_cli.nav_speed_tol,
                )[0].item()
                if next_stage:
                    stage = "acquire"
                    stage_step = 0
                    pp_mdp.set_pick_place_benchmark_state(base_env, ball_pos_w, goal_pos_w, mode="acquire", attach=False)
                    manip_command = pp_mdp.target_pos_command_obs(base_env)
                    manip_history.reset(_build_policy_frame(base_env, manip_command, manip_last_action))
                elif stage_step >= args_cli.max_nav_steps:
                    failure_stage = "nav_to_pick"
                    break

            elif stage == "acquire":
                manip_command = pp_mdp.target_pos_command_obs(base_env)
                manip_history.append(_build_policy_frame(base_env, manip_command, manip_last_action))
                with torch.inference_mode():
                    action = manip_actor(manip_history.stacked())
                if pp_mdp.task_success_mask(base_env)[0].item():
                    pp_mdp.attach_ball_to_hand(base_env)
                    pp_mdp.set_pick_place_benchmark_state(base_env, ball_pos_w, goal_pos_w, mode="place", attach=True)
                    pick_steps = stage_step
                    stage = "nav_to_place"
                    stage_step = 0
                    nav_controller.reset(place_goal_pos_w)
                elif stage_step >= args_cli.max_manip_steps:
                    failure_stage = "acquire"
                    break

            elif stage == "nav_to_place":
                pp_mdp.refresh_pick_place_state(base_env)
                action = nav_controller.act(place_goal_pos_w)
                action = _apply_left_arm_override(action, left_arm_joint_ids, carry_override_action)
                next_stage = _nav_success(
                    base_env,
                    place_goal_pos_w,
                    place_goal_yaw,
                    pos_tol=args_cli.nav_pos_tol,
                    yaw_tol=yaw_tol,
                    speed_tol=args_cli.nav_speed_tol,
                )[0].item()
                if next_stage:
                    stage = "place"
                    stage_step = 0
                    pp_mdp.set_pick_place_benchmark_state(base_env, ball_pos_w, goal_pos_w, mode="place", attach=True)
                    manip_command = pp_mdp.target_pos_command_obs(base_env)
                    manip_history.reset(_build_policy_frame(base_env, manip_command, manip_last_action))
                elif stage_step >= args_cli.max_nav_steps:
                    failure_stage = "nav_to_place"
                    break

            else:
                manip_command = pp_mdp.target_pos_command_obs(base_env)
                manip_history.append(_build_policy_frame(base_env, manip_command, manip_last_action))
                with torch.inference_mode():
                    action = manip_actor(manip_history.stacked())
                if pp_mdp.task_success_mask(base_env)[0].item():
                    pp_mdp.release_ball(base_env)
                    pp_mdp.refresh_pick_place_state(base_env)
                    success = True
                    place_steps = stage_step
                    break
                elif stage_step >= args_cli.max_manip_steps:
                    failure_stage = "place"
                    break

            env.step(action)
            if stage.startswith("nav_"):
                nav_controller.low_level_last_action = action
            else:
                manip_last_action = action
            stage_step += 1

            if base_env.termination_manager.terminated[0].item():
                failure_stage = "terminated"
                break

        pp_mdp.refresh_pick_place_state(base_env)
        final_ball_error = float(torch.linalg.norm(base_env._pp_ball_pos_w[0] - goal_pos_w[0]).item())
        success = success and final_ball_error <= args_cli.release_goal_tol
        records.append(
            {
                "episode": episode_index,
                "success": int(success),
                "failure_stage": failure_stage if not success else "",
                "nav_mode": nav_controller.mode,
                "ball_x": float(ball_pos_w[0, 0].item()),
                "ball_y": float(ball_pos_w[0, 1].item()),
                "goal_x": float(goal_pos_w[0, 0].item()),
                "goal_y": float(goal_pos_w[0, 1].item()),
                "pick_steps": pick_steps,
                "place_steps": place_steps,
                "final_ball_error_m": final_ball_error,
            }
        )
        print(
            "[PICK_PLACE_BENCH] "
            f"episode={episode_index} success={success} nav_mode={nav_controller.mode} "
            f"failure_stage={failure_stage or 'none'} final_ball_error_m={final_ball_error:.4f}"
        )

    summary = {
        "episodes": len(records),
        "nav_mode": nav_controller.mode,
        "success_rate": sum(record["success"] for record in records) / max(len(records), 1),
        "mean_final_ball_error_m": sum(record["final_ball_error_m"] for record in records) / max(len(records), 1),
        "records": records,
    }

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with open(os.path.join(output_dir, "episodes.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)

    print(f"[PICK_PLACE_BENCH] Summary written to: {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
