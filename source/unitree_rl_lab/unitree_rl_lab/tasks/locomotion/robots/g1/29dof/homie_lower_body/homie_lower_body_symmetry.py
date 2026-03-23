from __future__ import annotations

from typing import Any

import torch


def homie_lower_body_symmetry_augmentation(env, obs, action, obs_type):
    """Return original + mirrored batches for Isaac Lab's built-in RSL-RL symmetry support."""

    base_env = getattr(env, "unwrapped", env)

    obs_aug = None
    if obs is not None:
        if obs_type == "policy":
            mirrored_obs = base_env.mirror_policy_obs(obs)
        elif obs_type == "critic":
            mirrored_obs = base_env.mirror_critic_obs(obs)
        else:
            raise ValueError(f"Unsupported symmetry observation type: {obs_type}")
        obs_aug = torch.cat((obs, mirrored_obs), dim=0)

    action_aug = None
    if action is not None:
        mirrored_action = base_env.mirror_lower_actions(action)
        action_aug = torch.cat((action, mirrored_action), dim=0)

    return obs_aug, action_aug


def maybe_get_symmetry_cfg_kwargs() -> dict[str, Any]:
    return {
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "data_augmentation_func": homie_lower_body_symmetry_augmentation,
        "mirror_loss_coeff": 5.0e-4,
    }
