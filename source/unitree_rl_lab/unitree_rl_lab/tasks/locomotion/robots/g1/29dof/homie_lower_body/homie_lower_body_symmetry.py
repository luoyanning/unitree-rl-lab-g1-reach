from __future__ import annotations

from typing import Any


def homie_lower_body_symmetry_augmentation(env, obs, action):
    """Mirror policy/state/action tensors for Isaac Lab's built-in RSL-RL symmetry support."""

    base_env = getattr(env, "unwrapped", env)

    mirrored_obs = None
    if obs is not None:
        mirrored_obs = obs.clone()
        if "policy" in mirrored_obs:
            mirrored_obs["policy"] = base_env.mirror_policy_obs(obs["policy"])
        if "critic" in mirrored_obs:
            mirrored_obs["critic"] = base_env.mirror_critic_obs(obs["critic"])

    mirrored_action = None
    if action is not None:
        mirrored_action = base_env.mirror_lower_actions(action)

    return mirrored_obs, mirrored_action


def maybe_get_symmetry_cfg_kwargs() -> dict[str, Any]:
    return {
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "data_augmentation_func": homie_lower_body_symmetry_augmentation,
        "mirror_loss_coeff": 5.0e-4,
    }
