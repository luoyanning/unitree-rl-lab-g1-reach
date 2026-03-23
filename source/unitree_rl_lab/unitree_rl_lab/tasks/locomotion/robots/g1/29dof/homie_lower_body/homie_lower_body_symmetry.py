from __future__ import annotations

from typing import Any

import torch


def _get_obs_keys(obs) -> list:
    if not hasattr(obs, "keys"):
        return []
    try:
        return list(obs.keys())
    except TypeError:
        return list(obs.keys(include_nested=False))


def _get_obs_value(obs, key):
    if hasattr(obs, "get"):
        try:
            return obs.get(key)
        except Exception:
            pass
    try:
        return obs[key]
    except Exception:
        return None


def _set_obs_value(obs, key, value):
    if hasattr(obs, "set"):
        obs.set(key, value)
        return obs
    obs[key] = value
    return obs


def _concat_observations(obs, mirrored_obs):
    try:
        return torch.cat((obs, mirrored_obs), dim=0)
    except Exception:
        return {key: torch.cat((obs[key], mirrored_obs[key]), dim=0) for key in obs.keys()}


def homie_lower_body_symmetry_augmentation(env, obs=None, actions=None, obs_type=None, **kwargs):
    """Return original + mirrored batches for Isaac Lab's built-in RSL-RL symmetry support."""

    base_env = getattr(env, "unwrapped", env)

    obs_aug = None
    if obs is not None:
        if hasattr(obs, "keys"):
            mirrored_obs = obs.clone()
            keys = set(_get_obs_keys(obs))
            processed = False

            for key in ("policy", "actor", "obs"):
                if key in keys:
                    _set_obs_value(mirrored_obs, key, base_env.mirror_policy_obs(_get_obs_value(obs, key)))
                    processed = True
                    break

            for key in ("critic", "critic_obs", "state", "states"):
                if key in keys:
                    _set_obs_value(mirrored_obs, key, base_env.mirror_critic_obs(_get_obs_value(obs, key)))
                    processed = True
                    break

            if not processed:
                raise ValueError(f"Unsupported symmetry observation keys: {sorted(keys)}")

            obs_aug = _concat_observations(obs, mirrored_obs)
        elif obs_type in (None, "policy", "actor", "obs"):
            mirrored_obs = base_env.mirror_policy_obs(obs)
            obs_aug = torch.cat((obs, mirrored_obs), dim=0)
        elif obs_type in ("critic", "critic_obs", "state", "states"):
            mirrored_obs = base_env.mirror_critic_obs(obs)
            obs_aug = torch.cat((obs, mirrored_obs), dim=0)
        else:
            raise ValueError(f"Unsupported symmetry observation type: {obs_type}")

    action_aug = None
    if actions is not None:
        mirrored_action = base_env.mirror_lower_actions(actions)
        action_aug = torch.cat((actions, mirrored_action), dim=0)

    return obs_aug, action_aug


def maybe_get_symmetry_cfg_kwargs() -> dict[str, Any]:
    return {
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "data_augmentation_func": homie_lower_body_symmetry_augmentation,
        "mirror_loss_coeff": 5.0e-4,
    }
