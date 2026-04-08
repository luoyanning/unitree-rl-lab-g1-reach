"""
Clean V1 tabletop-touch MDP.

Design principles:
  - No bootstrap blending, no observation phase-switching.
  - Hand-to-target reward is ALWAYS active (no workspace gate).
  - Three reward layers: dense exponential approach, per-step progress, sparse completion.
  - Single fixed block (target_block_0); V2 adds position randomisation.

State tracked on env (lazy-init):
  _v1_hand_body_id      : int
  _v1_target_w          : (N, 3) world-frame touch target (block top)
  _v1_hand_dist         : (N,)   current hand-to-target L2 distance
  _v1_prev_hand_dist    : (N,)   previous step distance (for progress)
  _v1_touch_counter     : (N,)   consecutive steps hand is within touch_radius
  _v1_prev_touch_qual   : (N,)   whether hold was qualified last step
  _v1_recent_success    : (N,)   rising-edge flag when hold first achieved
  _v1_ep_succeeded      : (N,)   latched True once success fired this episode
  _v1_state_synced_step : int    last env.common_step_counter synced
  _v1_prev_ep_len       : (N,)   episode_length_buf snapshot for reset detection
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


V1_LEFT_HAND_BODY_NAME = "left_wrist_yaw_link"
V1_DEFAULT_TOUCH_Z_OFFSET = 0.025  # above block centre → near top surface


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _v1_robot(env):
    return env.scene["robot"]


def _v1_hand_pos_w(env) -> torch.Tensor:
    return _v1_robot(env).data.body_pos_w[:, env._v1_hand_body_id]


def _ensure_v1_state(env) -> None:
    if hasattr(env, "_v1_init"):
        return
    robot = _v1_robot(env)
    n = env.num_envs
    d = env.device
    env._v1_hand_body_id = int(
        robot.find_bodies([V1_LEFT_HAND_BODY_NAME], preserve_order=True)[0][0]
    )
    env._v1_target_w = torch.zeros(n, 3, device=d)
    env._v1_hand_dist = torch.ones(n, device=d)           # init to 1m
    env._v1_prev_hand_dist = torch.ones(n, device=d)
    env._v1_touch_counter = torch.zeros(n, dtype=torch.long, device=d)
    env._v1_prev_touch_qual = torch.zeros(n, dtype=torch.bool, device=d)
    env._v1_recent_success = torch.zeros(n, dtype=torch.bool, device=d)
    env._v1_ep_succeeded = torch.zeros(n, dtype=torch.bool, device=d)
    env._v1_state_synced_step = -1
    env._v1_prev_ep_len = torch.full((n,), -1, dtype=torch.long, device=d)
    env._v1_init = True


def _v1_refresh_target(env, touch_z_offset: float) -> None:
    """Read block position from scene and compute touch target (block top)."""
    target_names = tuple(getattr(env.cfg, "left_hand_scene_target_names", ("target_block_0",)))
    target_name = target_names[0]
    pos_w = env.scene[target_name].data.root_pos_w
    if pos_w.ndim == 3:
        pos_w = pos_w[:, 0]
    env._v1_target_w[:] = pos_w
    env._v1_target_w[:, 2] += touch_z_offset


def _sync_v1_state(
    env,
    touch_radius: float,
    touch_hold_steps: int,
    touch_z_offset: float,
) -> None:
    _ensure_v1_state(env)
    if env._v1_state_synced_step == env.common_step_counter:
        return

    # ---------- detect episode resets ----------
    current_ep: torch.Tensor
    if hasattr(env, "episode_length_buf"):
        current_ep = env.episode_length_buf.clone()
    else:
        current_ep = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    prev_ep = env._v1_prev_ep_len
    just_reset = (prev_ep < 0) | (current_ep == 0) | (current_ep < prev_ep)

    if torch.any(just_reset):
        env._v1_touch_counter[just_reset] = 0
        env._v1_prev_touch_qual[just_reset] = False
        env._v1_ep_succeeded[just_reset] = False
        env._v1_hand_dist[just_reset] = 1.0
        env._v1_prev_hand_dist[just_reset] = 1.0

    # ---------- refresh target ----------
    _v1_refresh_target(env, touch_z_offset=touch_z_offset)

    # ---------- hand distance ----------
    hand_pos = _v1_hand_pos_w(env)
    dist = torch.linalg.norm(env._v1_target_w - hand_pos, dim=-1)
    env._v1_prev_hand_dist[:] = env._v1_hand_dist
    env._v1_hand_dist[:] = dist

    # ---------- touch state machine ----------
    in_touch = dist <= touch_radius
    env._v1_touch_counter = torch.where(
        in_touch,
        env._v1_touch_counter + 1,
        torch.zeros_like(env._v1_touch_counter),
    )
    hold_qualified = env._v1_touch_counter >= touch_hold_steps
    # Rising edge: fires on the step hold is first qualified (can re-fire after leaving zone)
    env._v1_recent_success[:] = hold_qualified & ~env._v1_prev_touch_qual & ~just_reset
    env._v1_prev_touch_qual[:] = hold_qualified
    env._v1_ep_succeeded[:] |= env._v1_recent_success

    env._v1_prev_ep_len[:] = current_ep
    env._v1_state_synced_step = env.common_step_counter


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def target_pos_v1_obs(
    env,
    touch_radius: float = 0.06,
    touch_hold_steps: int = 5,
    touch_z_offset: float = V1_DEFAULT_TOUCH_Z_OFFSET,
) -> torch.Tensor:
    """
    3-D target vector (touch target minus robot base) expressed in the robot's
    yaw frame.  Always points to the block top — no phase switching.
    """
    _sync_v1_state(env, touch_radius=touch_radius,
                   touch_hold_steps=touch_hold_steps, touch_z_offset=touch_z_offset)
    robot = _v1_robot(env)
    delta_w = env._v1_target_w - robot.data.root_pos_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), delta_w)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def hand_approach_reward(
    env,
    touch_radius: float = 0.06,
    touch_hold_steps: int = 5,
    touch_z_offset: float = V1_DEFAULT_TOUCH_Z_OFFSET,
    reach_std: float = 0.12,
) -> torch.Tensor:
    """
    Exponential reward: exp(-dist / reach_std).
    Always active — no workspace gate.  Provides continuous gradient from any
    starting configuration.  Peaks at 1.0 when hand is on target.
    """
    _sync_v1_state(env, touch_radius=touch_radius,
                   touch_hold_steps=touch_hold_steps, touch_z_offset=touch_z_offset)
    return torch.exp(-env._v1_hand_dist / max(reach_std, 1.0e-6))


def approach_progress_reward(
    env,
    touch_radius: float = 0.06,
    touch_hold_steps: int = 5,
    touch_z_offset: float = V1_DEFAULT_TOUCH_Z_OFFSET,
    progress_scale: float = 0.04,
) -> torch.Tensor:
    """
    tanh reward for reducing hand-to-target distance THIS step.
    Only positive progress is rewarded (clamp min=0) so oscillation is not
    reinforced.  Helps early exploration discover "move toward target".
    """
    _sync_v1_state(env, touch_radius=touch_radius,
                   touch_hold_steps=touch_hold_steps, touch_z_offset=touch_z_offset)
    positive_progress = torch.clamp(
        env._v1_prev_hand_dist - env._v1_hand_dist, min=0.0, max=0.10
    )
    return torch.tanh(positive_progress / max(progress_scale, 1.0e-6))


def touch_in_zone_reward(
    env,
    touch_radius: float = 0.06,
    touch_hold_steps: int = 5,
    touch_z_offset: float = V1_DEFAULT_TOUCH_Z_OFFSET,
) -> torch.Tensor:
    """
    Binary +1 while hand is inside the touch zone.  Extra incentive to stay
    in contact rather than just graze the surface.
    """
    _sync_v1_state(env, touch_radius=touch_radius,
                   touch_hold_steps=touch_hold_steps, touch_z_offset=touch_z_offset)
    return (env._v1_hand_dist <= touch_radius).float()


def touch_completion_bonus(
    env,
    touch_radius: float = 0.06,
    touch_hold_steps: int = 5,
    touch_z_offset: float = V1_DEFAULT_TOUCH_Z_OFFSET,
) -> torch.Tensor:
    """
    Sparse one-shot bonus on the step the touch-hold is first achieved.
    Rising-edge: fires once per touch sequence (can fire again after hand
    leaves zone and returns).
    """
    _sync_v1_state(env, touch_radius=touch_radius,
                   touch_hold_steps=touch_hold_steps, touch_z_offset=touch_z_offset)
    return env._v1_recent_success.float()


# ---------------------------------------------------------------------------
# Termination functions
# ---------------------------------------------------------------------------

def v1_episode_timeout(
    env,
    touch_radius: float = 0.06,
    touch_hold_steps: int = 5,
    touch_z_offset: float = V1_DEFAULT_TOUCH_Z_OFFSET,
    episode_timeout_s: float = 12.0,
) -> torch.Tensor:
    """Terminate when episode exceeds timeout."""
    _sync_v1_state(env, touch_radius=touch_radius,
                   touch_hold_steps=touch_hold_steps, touch_z_offset=touch_z_offset)
    timeout_steps = max(1, int(round(episode_timeout_s / env.step_dt)))
    ep_len = getattr(
        env, "episode_length_buf",
        torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
    )
    return ep_len >= timeout_steps
