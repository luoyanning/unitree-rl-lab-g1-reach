from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from tensordict import TensorDict

from isaaclab.assets import Articulation
from isaaclab.utils.math import yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from . import point_goal_mdp


class FrozenVelocityActor(nn.Module):
    def __init__(self, actor_state_dict: dict[str, torch.Tensor]):
        super().__init__()

        linear_ids = sorted(
            int(key.split(".")[0])
            for key in actor_state_dict
            if key.endswith(".weight") and key.split(".")[0].isdigit()
        )
        if not linear_ids:
            raise ValueError("Velocity checkpoint does not contain actor linear layers.")

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
            key[len("actor.") :]: value
            for key, value in model_state_dict.items()
            if key.startswith("actor.")
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


class HierarchicalPointGoalVecEnv:
    def __init__(
        self,
        low_level_env,
        low_level_checkpoint_path: str,
        command_name: str = "base_velocity",
        clip_actions: float = 1.0,
    ):
        self.low_level_env = low_level_env
        self.base_env = getattr(low_level_env, "unwrapped", low_level_env)
        self.unwrapped = self.base_env
        self.command_name = command_name
        self.clip_actions = float(clip_actions)

        self.device = low_level_env.device
        self.num_envs = low_level_env.num_envs
        self.max_episode_length = getattr(low_level_env, "max_episode_length", self.base_env.max_episode_length)
        self.episode_length_buf = getattr(low_level_env, "episode_length_buf", self.base_env.episode_length_buf)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.extras = getattr(low_level_env, "extras", {})

        self.low_level_actor = FrozenVelocityActor.from_checkpoint(low_level_checkpoint_path, device=self.device)
        self.low_level_obs_dim = int(self.low_level_actor.input_dim)
        self.low_level_action_dim = int(self.low_level_actor.output_dim)
        self.low_level_history_len = 5
        self.low_level_frame_dim = self.low_level_obs_dim // self.low_level_history_len
        self.command_frame_slice = slice(6, 9)
        if self.low_level_frame_dim * self.low_level_history_len != self.low_level_obs_dim:
            raise ValueError(
                f"Unexpected velocity actor input dim {self.low_level_obs_dim}; "
                "expected a 5-step stacked observation history."
            )

        self.num_actions = 3
        self.num_obs = 17
        self.num_privileged_obs = 23
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
        self._low_level_obs = None

        self.single_action_space = gym.spaces.Box(low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.observation_space = None
        self.render_mode = getattr(low_level_env, "render_mode", None)

        command_cfg = self.base_env.command_manager.get_term(self.command_name).cfg
        self._lin_x_min = float(command_cfg.min_lin_vel_x)
        self._lin_x_max = float(command_cfg.max_lin_vel_x)
        self._lin_y_max = float(command_cfg.max_lin_vel_y)
        self._ang_z_max = float(command_cfg.max_ang_vel_z)

    def __getattr__(self, name: str):
        return getattr(self.low_level_env, name)

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def _scale_high_level_actions(self, actions: torch.Tensor) -> torch.Tensor:
        clipped_actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        normalized_actions = clipped_actions / max(self.clip_actions, 1.0e-6)
        policy_command = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        policy_command[:, 0] = 0.5 * (normalized_actions[:, 0] + 1.0) * (self._lin_x_max - self._lin_x_min) + self._lin_x_min
        policy_command[:, 1] = normalized_actions[:, 1] * self._lin_y_max
        policy_command[:, 2] = normalized_actions[:, 2] * self._ang_z_max
        return policy_command

    def _inject_policy_command(self, observations: torch.Tensor, policy_command: torch.Tensor) -> torch.Tensor:
        if observations.shape[1] != self.low_level_obs_dim:
            raise ValueError(
                f"Expected low-level observations with dim {self.low_level_obs_dim}, got {observations.shape[1]}."
            )
        patched_obs = observations.clone()
        patched_obs = patched_obs.view(self.num_envs, self.low_level_history_len, self.low_level_frame_dim)
        patched_obs[:, :, self.command_frame_slice] = policy_command.unsqueeze(1)
        return patched_obs.view(self.num_envs, self.low_level_obs_dim)

    def _base_lin_vel_body_xy(self, robot: Articulation) -> torch.Tensor:
        vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        vel_w[:, :2] = robot.data.root_lin_vel_w[:, :2]
        return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), vel_w)[:, :2]

    def _actor_observations(self) -> torch.Tensor:
        robot: Articulation = self.base_env.scene["robot"]
        goal_rel_body = point_goal_mdp.point_goal_rel_body_xy(self.base_env, command_name=self.command_name)
        goal_distance = point_goal_mdp.point_goal_distance_obs(self.base_env, command_name=self.command_name)
        goal_heading = point_goal_mdp.point_goal_heading_error_obs(self.base_env, command_name=self.command_name)
        base_lin_vel_body_xy = self._base_lin_vel_body_xy(robot)
        base_ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        policy_command = point_goal_mdp.point_goal_policy_command_obs(self.base_env)
        remaining_time_fraction = getattr(
            self.base_env,
            "_point_goal_remaining_time_fraction",
            torch.ones(self.num_envs, device=self.device),
        ).unsqueeze(-1)
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

    def _critic_observations(self) -> torch.Tensor:
        actor_obs = self._actor_observations()
        target_world = point_goal_mdp.point_goal_target_pos_env(self.base_env, command_name=self.command_name)
        root_world = point_goal_mdp.point_goal_root_pos_env(self.base_env)
        min_goal_distance = getattr(
            self.base_env,
            "_point_goal_min_distance",
            point_goal_mdp.point_goal_distance_obs(self.base_env, command_name=self.command_name).squeeze(-1),
        ).unsqueeze(-1)
        stop_quality = getattr(
            self.base_env,
            "_point_goal_stop_quality",
            torch.zeros(self.num_envs, device=self.device),
        ).unsqueeze(-1)
        return torch.cat((actor_obs, target_world, root_world, min_goal_distance, stop_quality), dim=-1)

    def _refresh_observation_buffers(self):
        self.obs_buf = self._actor_observations()
        self.privileged_obs_buf = self._critic_observations()

    def _obs_tensordict(self) -> TensorDict:
        return TensorDict(
            {"policy": self.obs_buf, "critic": self.privileged_obs_buf},
            batch_size=[self.num_envs],
        )

    def _split_obs_and_extras(self, result):
        if isinstance(result, tuple):
            if len(result) == 2:
                return result
            if len(result) > 2:
                return result[0], result[-1]
        return result, {}

    def _policy_obs_from_result(self, result) -> torch.Tensor:
        observations, extras = self._split_obs_and_extras(result)
        if isinstance(observations, TensorDict):
            return observations["policy"], extras
        if isinstance(observations, dict):
            return observations["policy"], extras
        return observations, extras

    def get_observations(self):
        self._refresh_observation_buffers()
        return self._obs_tensordict(), self.extras

    def get_privileged_observations(self):
        self._refresh_observation_buffers()
        return self.privileged_obs_buf

    def reset(self, env_ids=None):
        del env_ids
        low_level_obs, extras = self._policy_obs_from_result(self.low_level_env.reset())
        self._low_level_obs = low_level_obs
        zero_command = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        point_goal_mdp.set_point_goal_policy_command(self.base_env, zero_command)
        self._refresh_observation_buffers()
        self.extras = extras
        return self._obs_tensordict(), extras

    def step(self, actions: torch.Tensor):
        policy_command = self._scale_high_level_actions(actions)
        point_goal_mdp.set_point_goal_policy_command(self.base_env, policy_command)
        if self._low_level_obs is None:
            low_level_obs, _ = self._policy_obs_from_result(self.low_level_env.get_observations())
        else:
            low_level_obs = self._low_level_obs
        low_level_obs = self._inject_policy_command(low_level_obs, policy_command)
        with torch.inference_mode():
            low_level_actions = self.low_level_actor(low_level_obs)

        low_level_result = self.low_level_env.step(low_level_actions)
        if len(low_level_result) != 4:
            raise ValueError(f"Expected low-level env step to return 4 items, got {len(low_level_result)}.")
        low_level_obs, rewards, dones, extras = low_level_result
        self._low_level_obs, _ = self._policy_obs_from_result((low_level_obs, extras))
        self.rew_buf = rewards
        self.reset_buf = dones
        self.extras = extras
        self._refresh_observation_buffers()
        return self._obs_tensordict(), rewards, dones, extras

    def close(self):
        return self.low_level_env.close()
