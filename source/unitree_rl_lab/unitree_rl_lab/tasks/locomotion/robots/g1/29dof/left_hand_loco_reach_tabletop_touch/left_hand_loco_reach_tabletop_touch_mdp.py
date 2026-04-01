from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from ..left_hand_loco_reach import left_hand_loco_reach_mdp as fixed_mdp


LEFT_HAND_BODY_NAME = "left_wrist_yaw_link"


def _bootstrap_enabled(env) -> bool:
    return bool(getattr(env.cfg, "tabletop_bootstrap_enabled", False))


def _bootstrap_alpha_scalar(env) -> float:
    if not _bootstrap_enabled(env):
        return 1.0
    warmup_steps = max(0, int(getattr(env.cfg, "tabletop_bootstrap_warmup_steps", 0)))
    anneal_steps = max(1, int(getattr(env.cfg, "tabletop_bootstrap_anneal_steps", 1)))
    if env.common_step_counter <= warmup_steps:
        return 0.0
    return float(min(1.0, (env.common_step_counter - warmup_steps) / anneal_steps))


def _bootstrap_active_mask(env) -> torch.Tensor:
    return env._tt_bootstrap_alpha < (1.0 - 1.0e-6)


def _blended_target_error(env) -> torch.Tensor:
    return torch.lerp(env._tt_pre_target_error, env._tt_hand_object_error, env._tt_bootstrap_alpha)


def _blended_completion_signal(env) -> torch.Tensor:
    return torch.lerp(
        env._tt_recent_bootstrap_success.float(),
        env._tt_recent_success.float(),
        env._tt_bootstrap_alpha,
    )


def _episode_length_buf(env) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        return env.episode_length_buf.clone()
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _compute_just_reset_mask(env):
    current_episode_length = _episode_length_buf(env)
    if not hasattr(env, "_tt_prev_episode_length_buf"):
        env._tt_prev_episode_length_buf = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    prev_episode_length = env._tt_prev_episode_length_buf
    just_reset = prev_episode_length < 0
    just_reset |= current_episode_length == 0
    just_reset |= current_episode_length < prev_episode_length
    return just_reset, current_episode_length


def _robot(env):
    return env.scene["robot"]


def _hand_pos_w(env) -> torch.Tensor:
    robot = _robot(env)
    return robot.data.body_pos_w[:, env._tt_hand_body_id]


def _hand_vel_w(env) -> torch.Tensor:
    robot = _robot(env)
    return robot.data.body_lin_vel_w[:, env._tt_hand_body_id]


def _active_target_pos_base_yaw(env) -> torch.Tensor:
    robot = _robot(env)
    target_delta_w = env._tt_target_w - robot.data.root_pos_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), target_delta_w)


def _workspace_error_l2(
    target_pos_base: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> torch.Tensor:
    x_error = torch.where(
        target_pos_base[:, 0] < x_range[0],
        x_range[0] - target_pos_base[:, 0],
        torch.where(target_pos_base[:, 0] > x_range[1], target_pos_base[:, 0] - x_range[1], 0.0),
    )
    y_error = torch.where(
        target_pos_base[:, 1] < y_range[0],
        y_range[0] - target_pos_base[:, 1],
        torch.where(target_pos_base[:, 1] > y_range[1], target_pos_base[:, 1] - y_range[1], 0.0),
    )
    return torch.sqrt(x_error * x_error + y_error * y_error)


def _workspace_ready_gate(
    target_pos_base: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    gate_std: float,
) -> torch.Tensor:
    workspace_error = _workspace_error_l2(target_pos_base, x_range=x_range, y_range=y_range)
    return torch.exp(-workspace_error / max(gate_std, 1.0e-6))


def _ensure_tabletop_touch_state(env):
    if hasattr(env, "_tt_target_w"):
        return

    robot = _robot(env)
    num_envs = env.num_envs
    device = env.device

    env._tt_selected_target_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._tt_object_w = torch.zeros(num_envs, 3, device=device)
    env._tt_target_w = torch.zeros(num_envs, 3, device=device)
    env._tt_pre_target_w = torch.zeros(num_envs, 3, device=device)
    env._tt_final_target_w = torch.zeros(num_envs, 3, device=device)
    env._tt_prev_workspace_error = torch.zeros(num_envs, device=device)
    env._tt_workspace_error = torch.zeros(num_envs, device=device)
    env._tt_workspace_progress = torch.zeros(num_envs, device=device)
    env._tt_workspace_ready = torch.zeros(num_envs, device=device)
    env._tt_pre_target_error = torch.zeros(num_envs, device=device)
    env._tt_hand_target_error = torch.zeros(num_envs, device=device)
    env._tt_prev_hand_target_error = torch.zeros(num_envs, device=device)
    env._tt_hand_progress = torch.zeros(num_envs, device=device)
    env._tt_hand_object_error = torch.zeros(num_envs, device=device)
    env._tt_base_speed = torch.zeros(num_envs, device=device)
    env._tt_hand_speed = torch.zeros(num_envs, device=device)
    env._tt_base_ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._tt_pretouch_hold_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._tt_hold_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._tt_success_zone_time = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._tt_prev_bootstrap_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._tt_recent_bootstrap_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._tt_prev_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._tt_recent_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._tt_recent_pretouch = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._tt_timed_out = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._tt_target_age_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._tt_completed = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._tt_use_final_target = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._tt_prev_use_final_target = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._tt_bootstrap_alpha = torch.ones(num_envs, device=device)
    env._tt_state_synced_step = -1

    env._tt_hand_body_id = int(robot.find_bodies([LEFT_HAND_BODY_NAME], preserve_order=True)[0][0])
    env._tt_arm_joint_ids = torch.tensor(
        robot.find_joints(
            [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
            ],
            preserve_order=True,
        )[0],
        dtype=torch.long,
        device=device,
    )


def _sample_tabletop_targets(env, env_ids: torch.Tensor, scene_target_names):
    if len(env_ids) == 0:
        return
    target_positions_w = fixed_mdp._scene_target_positions_w(env, scene_target_names)
    num_targets = len(scene_target_names)
    selected_idx = torch.randint(num_targets, (len(env_ids),), device=env.device)
    env._tt_selected_target_idx[env_ids] = selected_idx
    env._tt_object_w[env_ids] = target_positions_w[env_ids, selected_idx]


def _refresh_selected_target_positions(env, scene_target_names):
    target_positions_w = fixed_mdp._scene_target_positions_w(env, scene_target_names)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env._tt_object_w[:] = target_positions_w[env_ids, env._tt_selected_target_idx]


def _sync_tabletop_touch_state(
    env,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _ensure_tabletop_touch_state(env)
    if env._tt_state_synced_step == env.common_step_counter:
        return

    scene_target_names = tuple(getattr(env.cfg, "left_hand_scene_target_names", ()))
    if len(scene_target_names) == 0:
        raise RuntimeError("TableTopTouch requires env.cfg.left_hand_scene_target_names to be set.")

    reset_ids, current_episode_length = _compute_just_reset_mask(env)
    bootstrap_alpha_scalar = _bootstrap_alpha_scalar(env)
    env._tt_bootstrap_alpha.fill_(bootstrap_alpha_scalar)
    bootstrap_success_radius = float(getattr(env.cfg, "tabletop_bootstrap_pretouch_success_radius", pre_target_switch_radius))
    bootstrap_hold_steps = max(1, int(getattr(env.cfg, "tabletop_bootstrap_hold_steps", success_hold_steps)))
    bootstrap_timeout_s = float(getattr(env.cfg, "tabletop_bootstrap_timeout_s", per_target_timeout_s))
    env._tt_recent_success.zero_()
    env._tt_recent_bootstrap_success.zero_()
    env._tt_recent_pretouch.zero_()

    if torch.any(reset_ids):
        reset_env_ids = torch.where(reset_ids)[0]
        _sample_tabletop_targets(env, reset_env_ids, scene_target_names)
        env._tt_pretouch_hold_counter[reset_ids] = 0
        env._tt_hold_counter[reset_ids] = 0
        env._tt_success_zone_time[reset_ids] = 0
        env._tt_prev_bootstrap_success[reset_ids] = False
        env._tt_prev_success[reset_ids] = False
        env._tt_timed_out[reset_ids] = False
        env._tt_target_age_steps[reset_ids] = 0
        env._tt_completed[reset_ids] = 0
        env._tt_use_final_target[reset_ids] = False
        env._tt_prev_use_final_target[reset_ids] = False

    _refresh_selected_target_positions(env, scene_target_names)

    hand_pos_w = _hand_pos_w(env)
    hand_vel_w = _hand_vel_w(env)
    robot = _robot(env)
    base_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    hand_speed = torch.linalg.norm(hand_vel_w, dim=-1)

    env._tt_pre_target_w[:] = env._tt_object_w
    env._tt_pre_target_w[:, 0] -= pretouch_backoff_x
    env._tt_pre_target_w[:, 2] += pretouch_height

    env._tt_final_target_w[:] = env._tt_object_w
    env._tt_final_target_w[:, 2] += touch_height_offset

    pre_target_error = torch.linalg.norm(env._tt_pre_target_w - hand_pos_w, dim=-1)
    env._tt_pre_target_error[:] = pre_target_error
    target_pos_base = quat_apply_inverse(
        yaw_quat(robot.data.root_quat_w),
        env._tt_final_target_w - robot.data.root_pos_w,
    )
    workspace_error = _workspace_error_l2(target_pos_base, x_range=x_range, y_range=y_range)
    workspace_ready = _workspace_ready_gate(target_pos_base, x_range=x_range, y_range=y_range, gate_std=0.02)
    env._tt_workspace_error[:] = workspace_error
    env._tt_workspace_progress[:] = env._tt_prev_workspace_error - workspace_error
    env._tt_prev_workspace_error[:] = workspace_error
    env._tt_workspace_ready[:] = workspace_ready
    env._tt_base_ready[:] = (workspace_error <= 0.03) & (base_speed <= base_speed_threshold)

    env._tt_use_final_target[:] = env._tt_base_ready & (pre_target_error <= pre_target_switch_radius)
    env._tt_recent_pretouch[:] = env._tt_use_final_target & ~env._tt_prev_use_final_target & ~reset_ids
    env._tt_prev_use_final_target[:] = env._tt_use_final_target
    env._tt_target_w[:] = torch.where(
        env._tt_use_final_target.unsqueeze(-1),
        env._tt_final_target_w,
        env._tt_pre_target_w,
    )
    env._tt_hand_target_error[:] = torch.linalg.norm(env._tt_target_w - hand_pos_w, dim=-1)
    env._tt_hand_object_error[:] = torch.linalg.norm(env._tt_final_target_w - hand_pos_w, dim=-1)
    env._tt_hand_progress[:] = torch.clamp(
        env._tt_prev_hand_target_error - env._tt_hand_target_error,
        min=-0.10,
        max=0.10,
    )
    env._tt_hand_progress[reset_ids] = 0.0
    env._tt_prev_hand_target_error[:] = env._tt_hand_target_error
    env._tt_base_speed[:] = base_speed
    env._tt_hand_speed[:] = hand_speed

    bootstrap_success_zone = (
        env._tt_base_ready
        & (pre_target_error <= bootstrap_success_radius)
        & (hand_speed <= hand_speed_threshold * 1.5)
    )
    env._tt_pretouch_hold_counter = torch.where(
        bootstrap_success_zone,
        env._tt_pretouch_hold_counter + 1,
        torch.zeros_like(env._tt_pretouch_hold_counter),
    )
    bootstrap_hold_qualified = env._tt_pretouch_hold_counter >= bootstrap_hold_steps
    env._tt_recent_bootstrap_success[:] = bootstrap_hold_qualified & ~env._tt_prev_bootstrap_success & ~reset_ids
    env._tt_prev_bootstrap_success[:] = bootstrap_hold_qualified

    success_zone = (
        env._tt_use_final_target
        & (env._tt_hand_object_error <= success_threshold)
        & (base_speed <= base_speed_threshold)
        & (hand_speed <= hand_speed_threshold)
    )
    env._tt_success_zone_time = torch.where(
        success_zone,
        env._tt_success_zone_time + 1,
        torch.zeros_like(env._tt_success_zone_time),
    )
    env._tt_hold_counter = env._tt_success_zone_time
    hold_qualified = env._tt_hold_counter >= max(1, int(success_hold_steps))
    env._tt_recent_success[:] = hold_qualified & ~env._tt_prev_success & ~reset_ids
    env._tt_prev_success[:] = hold_qualified
    env._tt_completed[:] = torch.where(
        env._tt_recent_success | (_bootstrap_active_mask(env) & env._tt_recent_bootstrap_success),
        torch.ones_like(env._tt_completed),
        env._tt_completed,
    )

    env._tt_target_age_steps += 1
    env._tt_target_age_steps[reset_ids] = 0
    effective_timeout_s = (1.0 - bootstrap_alpha_scalar) * bootstrap_timeout_s + bootstrap_alpha_scalar * float(
        per_target_timeout_s
    )
    per_target_timeout_steps = max(1, int(round(effective_timeout_s / env.step_dt)))
    env._tt_timed_out[:] = env._tt_target_age_steps >= per_target_timeout_steps

    command_term = env.command_manager.get_term("left_hand_pose")
    if hasattr(command_term, "metrics"):
        command_term.metrics.setdefault("success_zone_flag", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("success_hold_counter", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("success_zone_time", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("held_success_count", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("completion_distance", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("completion_after_hold", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("post_success_dwell_flag", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("post_success_dwell_counter", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("completion_after_dwell", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("pretouch_switch_flag", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("bootstrap_alpha", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics.setdefault("bootstrap_success_flag", torch.zeros(env.num_envs, device=env.device))
        completion_flag = (env._tt_recent_success | env._tt_recent_bootstrap_success).float()
        completion_distance = torch.where(
            env._tt_recent_success,
            env._tt_hand_object_error,
            torch.where(
                env._tt_recent_bootstrap_success,
                env._tt_pre_target_error,
                torch.zeros_like(env._tt_hand_object_error),
            ),
        )
        command_term.metrics["success_zone_flag"][:] = success_zone.float()
        command_term.metrics["success_hold_counter"][:] = env._tt_hold_counter.float()
        command_term.metrics["success_zone_time"][:] = env._tt_success_zone_time.float()
        command_term.metrics["held_success_count"][:] = env._tt_completed.float()
        command_term.metrics["completion_distance"][:] = completion_distance
        command_term.metrics["completion_after_hold"][:] = completion_flag
        command_term.metrics["post_success_dwell_flag"][:] = torch.zeros(env.num_envs, device=env.device)
        command_term.metrics["post_success_dwell_counter"][:] = torch.zeros(env.num_envs, device=env.device)
        command_term.metrics["completion_after_dwell"][:] = completion_flag
        command_term.metrics["pretouch_switch_flag"][:] = env._tt_recent_pretouch.float()
        command_term.metrics["bootstrap_alpha"][:] = env._tt_bootstrap_alpha
        command_term.metrics["bootstrap_success_flag"][:] = env._tt_recent_bootstrap_success.float()

    env._tt_prev_episode_length_buf = current_episode_length
    env._tt_state_synced_step = env.common_step_counter


def target_pos_command_obs(
    env,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return _active_target_pos_base_yaw(env)


def target_relative_base_stance_l2(
    env,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return env._tt_workspace_error


def target_relative_base_stance_ready(
    env,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
    gate_std: float = 0.01,
):
    del gate_std
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return env._tt_workspace_ready


def target_relative_base_stance_progress(
    env,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return env._tt_workspace_progress


def static_target_position_error(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    del asset_cfg
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return env._tt_hand_target_error


def gated_position_command_error_tanh(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
    std: float = 0.14,
    gate_std: float = 0.01,
):
    del asset_cfg
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    gate = env._tt_workspace_ready
    return (1.0 - torch.tanh(env._tt_hand_target_error / std)) * gate


def target_hold_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
    hold_reward_std: float = 0.02,
):
    del asset_cfg
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    bootstrap_hold_gate = (env._tt_pretouch_hold_counter > 0).float()
    final_hold_gate = (env._tt_hold_counter > 0).float()
    hold_gate = torch.lerp(bootstrap_hold_gate, final_hold_gate, env._tt_bootstrap_alpha)
    return hold_gate * torch.exp(-_blended_target_error(env) / max(hold_reward_std, 1.0e-6))


def hand_phase_progress_reward(
    env,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
    progress_scale: float = 0.04,
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    positive_progress = torch.clamp(env._tt_hand_progress, min=0.0)
    phase_gate = 0.25 + 0.75 * env._tt_workspace_ready
    return phase_gate * torch.tanh(positive_progress / max(progress_scale, 1.0e-6))


def pretouch_ready_bonus(
    env,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    bootstrap_ready = env._tt_recent_bootstrap_success.float() * (1.0 - env._tt_bootstrap_alpha)
    return torch.maximum(env._tt_recent_pretouch.float(), bootstrap_ready)


def near_target_left_hand_stillness_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
    near_target_radius: float = 0.10,
    hand_speed_scale: float = 0.08,
):
    del asset_cfg
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    near_target = (_blended_target_error(env) <= near_target_radius).float()
    return near_target * torch.exp(-env._tt_hand_speed / max(hand_speed_scale, 1.0e-6))


def target_completion_bonus(
    env,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return _blended_completion_signal(env)


def success_posture_bonus(
    env,
    arm_joint_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    del asset_cfg
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    robot = env.scene[arm_joint_cfg.name]
    deviation = torch.sum(
        torch.abs(robot.data.joint_pos[:, arm_joint_cfg.joint_ids] - robot.data.default_joint_pos[:, arm_joint_cfg.joint_ids]),
        dim=1,
    )
    return _blended_completion_signal(env) * torch.exp(-0.5 * deviation)


def pre_stance_torso_lean_penalty(
    env,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
    gate_std: float = 0.02,
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    gate = env._tt_workspace_ready
    torso_lean = torch.linalg.norm(_robot(env).data.projected_gravity_b[:, :2], dim=-1)
    return (1.0 - gate) * torso_lean


def pre_stance_joint_deviation_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
    gate_std: float = 0.02,
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    gate = env._tt_workspace_ready
    robot = env.scene[asset_cfg.name]
    deviation = torch.sum(
        torch.abs(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return (1.0 - gate) * deviation


def pre_stance_joint_limit_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    margin_threshold: float = 0.18,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
    gate_std: float = 0.02,
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    gate = env._tt_workspace_ready
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    joint_range = torch.clamp(joint_limits[..., 1] - joint_limits[..., 0], min=1.0e-6)
    normalized_margin = torch.minimum(joint_pos - joint_limits[..., 0], joint_limits[..., 1] - joint_pos) / joint_range
    limit_pressure = torch.clamp(margin_threshold - normalized_margin, min=0.0)
    return (1.0 - gate) * torch.sum(limit_pressure, dim=1)


def pre_stance_foot_motion_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
    gate_std: float = 0.02,
    speed_scale: float = 0.35,
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    gate = env._tt_workspace_ready
    robot = env.scene[asset_cfg.name]
    foot_vel_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2] - robot.data.root_lin_vel_w[:, None, :2]
    foot_speed = torch.linalg.norm(foot_vel_xy, dim=-1).mean(dim=1)
    return (1.0 - gate) * torch.tanh(foot_speed / max(speed_scale, 1.0e-6))


def target_success_reached(
    env,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return env._tt_recent_success | (_bootstrap_active_mask(env) & env._tt_recent_bootstrap_success)


def target_timeout_reached(
    env,
    success_threshold: float = 0.05,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pretouch_height: float = 0.10,
    pretouch_backoff_x: float = 0.06,
    touch_height_offset: float = 0.02,
    base_speed_threshold: float = 0.08,
    hand_speed_threshold: float = 0.12,
    pre_target_switch_radius: float = 0.10,
    x_range: tuple[float, float] = (0.40, 0.56),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_tabletop_touch_state(
        env,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pretouch_height=pretouch_height,
        pretouch_backoff_x=pretouch_backoff_x,
        touch_height_offset=touch_height_offset,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return env._tt_timed_out
