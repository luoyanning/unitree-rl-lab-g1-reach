from __future__ import annotations

import os
import torch
from collections.abc import Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_apply, sample_uniform, yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

ENABLE_LONG_HORIZON_DEBUG_METRICS = False
ENABLE_TARGET_TIMEOUT_DEBUG = os.getenv("UTRL_LH_LOCO_REACH_TIMEOUT_DEBUG", "1") == "1"
TARGET_TIMEOUT_DEBUG_MAX_GLOBAL_STEPS = int(os.getenv("UTRL_LH_LOCO_REACH_TIMEOUT_DEBUG_STEPS", "8"))
TARGET_SAMPLING_REGIME_ORDER = ("near", "posture", "far")
DEFAULT_SAMPLE_REGIMES = {
    "near": {"pos_x": (0.25, 0.48), "pos_y": (0.08, 0.28), "pos_z": (0.18, 0.34)},
    "posture": {"pos_x": (0.35, 0.72), "pos_y": (0.02, 0.38), "pos_z": (0.00, 0.20)},
    "far": {"pos_x": (0.50, 1.00), "pos_y": (-0.05, 0.60), "pos_z": (0.08, 0.24)},
}
DEFAULT_SAMPLE_WEIGHTS = {"near": 0.45, "posture": 0.30, "far": 0.25}
TARGET_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/left_hand_loco_reach_target")
TARGET_MARKER_CFG.markers["frame"].scale = (0.12, 0.12, 0.12)


def _clone_sample_regimes(sample_regimes: dict[str, dict[str, tuple[float, float]]] | None):
    base_regimes = DEFAULT_SAMPLE_REGIMES if sample_regimes is None else sample_regimes
    return {
        regime_name: {
            axis_name: tuple(float(v) for v in axis_range)
            for axis_name, axis_range in base_regimes[regime_name].items()
        }
        for regime_name in TARGET_SAMPLING_REGIME_ORDER
    }


def _clone_sample_weights(sample_weights: dict[str, float] | None):
    base_weights = DEFAULT_SAMPLE_WEIGHTS if sample_weights is None else sample_weights
    return {regime_name: float(base_weights[regime_name]) for regime_name in TARGET_SAMPLING_REGIME_ORDER}


def _range_union(sample_regimes: dict[str, dict[str, tuple[float, float]]], axis_name: str):
    axis_ranges = [sample_regimes[regime_name][axis_name] for regime_name in TARGET_SAMPLING_REGIME_ORDER]
    return (
        min(axis_range[0] for axis_range in axis_ranges),
        max(axis_range[1] for axis_range in axis_ranges),
    )


def _lerp_range(
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    alpha: torch.Tensor,
):
    start_tensor = torch.tensor(start_range, device=alpha.device)
    end_tensor = torch.tensor(end_range, device=alpha.device)
    return tuple(float(v) for v in torch.lerp(start_tensor, end_tensor, alpha).tolist())


def _set_sampling_distribution_state(
    env,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None,
    sample_weights: dict[str, float] | None,
):
    env._left_hand_sample_regimes = _clone_sample_regimes(sample_regimes)
    env._left_hand_sample_weights = _clone_sample_weights(sample_weights)


def _get_sampling_distribution_state(
    env,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None,
    sample_weights: dict[str, float] | None,
):
    if not hasattr(env, "_left_hand_sample_regimes") or not hasattr(env, "_left_hand_sample_weights"):
        _set_sampling_distribution_state(env, sample_regimes=sample_regimes, sample_weights=sample_weights)
    return env._left_hand_sample_regimes, env._left_hand_sample_weights


def _target_marker_quat(env):
    if not hasattr(env, "_left_hand_target_marker_quat"):
        env._left_hand_target_marker_quat = torch.zeros(env.num_envs, 4, device=env.device)
        env._left_hand_target_marker_quat[:, 0] = 1.0
    return env._left_hand_target_marker_quat


def _update_target_debug_visualization(env):
    if not hasattr(env, "_left_hand_target_visualizer"):
        env._left_hand_target_visualizer = VisualizationMarkers(TARGET_MARKER_CFG)
    env._left_hand_target_visualizer.visualize(env._left_hand_active_target_w, _target_marker_quat(env))


def _episode_length_buf(env):
    if hasattr(env, "episode_length_buf"):
        return env.episode_length_buf.clone()
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _compute_just_reset_mask(env):
    current_episode_length = _episode_length_buf(env)
    if not hasattr(env, "_left_hand_prev_episode_length_buf"):
        env._left_hand_prev_episode_length_buf = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    prev_episode_length = env._left_hand_prev_episode_length_buf
    just_reset = prev_episode_length < 0
    just_reset |= current_episode_length == 0
    just_reset |= current_episode_length < prev_episode_length
    return just_reset, current_episode_length, prev_episode_length


def _current_reset_mask(env):
    if hasattr(env, "_left_hand_just_reset_this_step"):
        return env._left_hand_just_reset_this_step
    return _episode_length_buf(env) == 0


def _debug_timeout_state(
    env,
    current_episode_length,
    prev_episode_length,
    just_reset,
    just_spawned,
    pre_timeout,
    post_timeout,
    max_target_steps: int,
):
    if not ENABLE_TARGET_TIMEOUT_DEBUG:
        return
    if env.common_step_counter > TARGET_TIMEOUT_DEBUG_MAX_GLOBAL_STEPS:
        return
    env_id = 0
    terminated = getattr(env.termination_manager, "terminated", torch.zeros(env.num_envs, device=env.device))
    print(
        "[LH_LOCO_REACH_TIMEOUT_DEBUG] "
        f"global_step={int(env.common_step_counter)} "
        f"env={env_id} "
        f"episode_len={int(current_episode_length[env_id])} "
        f"prev_episode_len={int(prev_episode_length[env_id])} "
        f"just_reset={bool(just_reset[env_id])} "
        f"just_spawned={bool(just_spawned[env_id])} "
        f"target_age_steps={int(env._left_hand_target_age_steps[env_id])} "
        f"max_target_steps={int(max_target_steps)} "
        f"pre_timeout={bool(pre_timeout[env_id])} "
        f"post_timeout={bool(post_timeout[env_id])} "
        f"terminated={bool(terminated[env_id])}",
        flush=True,
    )


def _active_target_pos_base_yaw(env):
    _ensure_long_horizon_state(env, command_name="left_hand_pose", max_targets_per_episode=1, switch_phase_steps=0)
    robot = env.scene["robot"]
    target_delta_w = env._left_hand_active_target_w - robot.data.root_pos_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), target_delta_w)


def _static_target_position_error(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
):
    _ensure_long_horizon_state(env, command_name="left_hand_pose", max_targets_per_episode=1, switch_phase_steps=0)
    asset = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if ee_pos_w.ndim == 3:
        ee_pos_w = ee_pos_w[:, 0]
    return torch.linalg.norm(env._left_hand_active_target_w - ee_pos_w, dim=-1)


def static_target_position_error(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    return _static_target_position_error(env, asset_cfg=asset_cfg)


def _static_target_position_error_tanh(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    std: float = 0.14,
):
    position_error = _static_target_position_error(env, asset_cfg=asset_cfg)
    return 1.0 - torch.tanh(position_error / std)


def _spawn_new_fixed_targets(
    env,
    env_ids,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None,
    sample_weights: dict[str, float] | None,
):
    if len(env_ids) == 0:
        return

    sample_regimes, sample_weights = _get_sampling_distribution_state(
        env, sample_regimes=sample_regimes, sample_weights=sample_weights
    )
    robot = env.scene["robot"]
    regime_ranges = torch.tensor(
        [
            [
                sample_regimes[regime_name]["pos_x"],
                sample_regimes[regime_name]["pos_y"],
                sample_regimes[regime_name]["pos_z"],
            ]
            for regime_name in TARGET_SAMPLING_REGIME_ORDER
        ],
        dtype=torch.float32,
        device=env.device,
    )
    weight_tensor = torch.tensor(
        [sample_weights[regime_name] for regime_name in TARGET_SAMPLING_REGIME_ORDER],
        dtype=torch.float32,
        device=env.device,
    )
    weight_tensor = torch.clamp(weight_tensor, min=0.0)
    if torch.sum(weight_tensor) <= 0.0:
        weight_tensor = torch.ones_like(weight_tensor)
    weight_tensor = weight_tensor / torch.sum(weight_tensor)

    regime_ids = torch.multinomial(weight_tensor, len(env_ids), replacement=True)
    selected_ranges = regime_ranges[regime_ids]
    local_target_pos = sample_uniform(
        selected_ranges[:, :, 0], selected_ranges[:, :, 1], (len(env_ids), 3), device=env.device
    )
    root_pos_w = robot.data.root_pos_w[env_ids]
    root_yaw_w = yaw_quat(robot.data.root_quat_w[env_ids])
    env._left_hand_active_target_w[env_ids] = root_pos_w + quat_apply(root_yaw_w, local_target_pos)
    env._left_hand_has_active_target[env_ids] = True


def _ensure_long_horizon_state(env, command_name: str, max_targets_per_episode: int, switch_phase_steps: int):
    num_envs = env.num_envs
    if not hasattr(env, "_left_hand_prev_target_w"):
        robot = env.scene["robot"]
        env._left_hand_prev_target_w = torch.zeros(num_envs, 3, device=env.device)
        env._left_hand_active_target_w = torch.zeros(num_envs, 3, device=env.device)
        env._left_hand_has_active_target = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        env._left_hand_prev_success = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        env._left_hand_completed_targets = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        env._left_hand_target_index = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        env._left_hand_post_switch_steps = torch.full(
            (num_envs,), switch_phase_steps, dtype=torch.long, device=env.device
        )
        env._left_hand_target_age_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        env._left_hand_steps_since_switch = env._left_hand_target_age_steps
        env._left_hand_foot_motion_before_contact = torch.zeros(num_envs, device=env.device)
        env._left_hand_workspace_error_at_contact = torch.zeros(num_envs, device=env.device)
        env._left_hand_torso_lean_at_contact = torch.zeros(num_envs, device=env.device)
        env._left_hand_arm_extension_at_contact = torch.zeros(num_envs, device=env.device)
        env._left_hand_recent_success = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        env._left_hand_target_switched_this_step = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        env._left_hand_just_reset_this_step = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        env._left_hand_state_synced_step = -1
        env._left_hand_timeout_cfg_logged = False
        env._left_hand_arm_joint_ids = torch.tensor(
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
            device=env.device,
        )
        env._left_hand_foot_body_ids = torch.tensor(
            robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0],
            dtype=torch.long,
            device=env.device,
        )
        env._left_hand_ee_body_id = robot.find_bodies(["left_wrist_yaw_link"], preserve_order=True)[0][0]
    command_term = env.command_manager.get_term(command_name)
    if ENABLE_LONG_HORIZON_DEBUG_METRICS and hasattr(command_term, "metrics"):
        command_term.metrics.setdefault("targets_completed", torch.zeros(num_envs, device=env.device))
        command_term.metrics.setdefault("target_index", torch.zeros(num_envs, device=env.device))
        command_term.metrics.setdefault("workspace_error", torch.zeros(num_envs, device=env.device))
        command_term.metrics.setdefault("workspace_error_at_contact", torch.zeros(num_envs, device=env.device))
        command_term.metrics.setdefault("torso_lean_at_contact", torch.zeros(num_envs, device=env.device))
        command_term.metrics.setdefault("arm_extension_at_contact", torch.zeros(num_envs, device=env.device))
        command_term.metrics.setdefault("foot_motion_before_contact", torch.zeros(num_envs, device=env.device))
        command_term.metrics.setdefault("post_switch_steps", torch.zeros(num_envs, device=env.device))
        command_term.metrics.setdefault("post_switch_posture_quality", torch.zeros(num_envs, device=env.device))
        command_term.metrics.setdefault("switch_failure_risk", torch.zeros(num_envs, device=env.device))
        for index in range(max_targets_per_episode):
            command_term.metrics.setdefault(f"success_target_{index}", torch.zeros(num_envs, device=env.device))


def _switch_phase_scale(env, switch_phase_steps: int):
    if switch_phase_steps <= 0:
        return torch.zeros(env.num_envs, device=env.device)
    return (env._left_hand_post_switch_steps > 0).float()


def _ee_position_error(env, command_name: str):
    del command_name
    _ensure_long_horizon_state(env, command_name="left_hand_pose", max_targets_per_episode=1, switch_phase_steps=0)
    robot = env.scene["robot"]
    ee_pos_w = robot.data.body_pos_w[:, env._left_hand_ee_body_id]
    return torch.linalg.norm(env._left_hand_active_target_w - ee_pos_w, dim=-1)


def _workspace_error_components(
    env,
    command_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
):
    del command_name
    target_pos = _active_target_pos_base_yaw(env)[:, :2]
    x_error_low = torch.clamp(x_range[0] - target_pos[:, 0], min=0.0)
    x_error_high = torch.clamp(target_pos[:, 0] - x_range[1], min=0.0)
    y_error_low = torch.clamp(y_range[0] - target_pos[:, 1], min=0.0)
    y_error_high = torch.clamp(target_pos[:, 1] - y_range[1], min=0.0)
    return x_error_low, x_error_high, y_error_low, y_error_high


def _workspace_error_l2(
    env,
    command_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
):
    x_error_low, x_error_high, y_error_low, y_error_high = _workspace_error_components(
        env, command_name=command_name, x_range=x_range, y_range=y_range
    )
    return torch.square(x_error_low) + torch.square(x_error_high) + torch.square(y_error_low) + torch.square(y_error_high)


def _workspace_ready_gate(
    env,
    command_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    gate_std: float,
):
    workspace_error = _workspace_error_l2(env, command_name=command_name, x_range=x_range, y_range=y_range)
    return torch.exp(-workspace_error / gate_std)


def _sync_long_horizon_state(
    env,
    command_name: str,
    success_threshold: float,
    max_targets_per_episode: int,
    switch_phase_steps: int,
    static_target_hold_s: float,
    per_target_timeout_s: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None,
    sample_weights: dict[str, float] | None,
):
    _ensure_long_horizon_state(
        env, command_name=command_name, max_targets_per_episode=max_targets_per_episode, switch_phase_steps=switch_phase_steps
    )
    if env._left_hand_state_synced_step == env.common_step_counter:
        return

    command_term = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    reset_ids, current_episode_length, prev_episode_length = _compute_just_reset_mask(env)
    env._left_hand_just_reset_this_step[:] = reset_ids
    env._left_hand_recent_success.zero_()
    env._left_hand_target_switched_this_step.zero_()
    _get_sampling_distribution_state(env, sample_regimes=sample_regimes, sample_weights=sample_weights)
    per_target_timeout_steps = max(1, int(round(per_target_timeout_s / env.step_dt)))

    if ENABLE_TARGET_TIMEOUT_DEBUG and not env._left_hand_timeout_cfg_logged:
        print(
            "[LH_LOCO_REACH_TIMEOUT_CONFIG] "
            f"step_dt={float(env.step_dt):.6f} "
            f"sim_dt={float(env.cfg.sim.dt):.6f} "
            f"decimation={int(env.cfg.decimation)} "
            f"per_target_timeout_s={float(per_target_timeout_s):.3f} "
            f"per_target_timeout_steps={int(per_target_timeout_steps)} "
            f"episode_length_s={float(env.cfg.episode_length_s):.3f} "
            f"max_episode_length={int(env.max_episode_length)}",
            flush=True,
        )
        env._left_hand_timeout_cfg_logged = True

    if torch.any(reset_ids):
        env._left_hand_completed_targets[reset_ids] = 0
        env._left_hand_target_index[reset_ids] = 0
        env._left_hand_post_switch_steps[reset_ids] = switch_phase_steps
        env._left_hand_target_age_steps[reset_ids] = 0
        env._left_hand_prev_success[reset_ids] = False
        env._left_hand_recent_success[reset_ids] = False
        env._left_hand_foot_motion_before_contact[reset_ids] = 0.0
        env._left_hand_workspace_error_at_contact[reset_ids] = 0.0
        env._left_hand_torso_lean_at_contact[reset_ids] = 0.0
        env._left_hand_arm_extension_at_contact[reset_ids] = 0.0
        env._left_hand_has_active_target[reset_ids] = False

    just_spawned = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    inactive_ids = torch.where(~env._left_hand_has_active_target)[0]
    if len(inactive_ids) > 0:
        _spawn_new_fixed_targets(
            env, inactive_ids, sample_regimes=sample_regimes, sample_weights=sample_weights
        )
        just_spawned[inactive_ids] = True

    success = _ee_position_error(env, command_name=command_name) < success_threshold
    success_edge = success & ~env._left_hand_prev_success & ~reset_ids

    workspace_error = _workspace_error_l2(env, command_name=command_name, x_range=x_range, y_range=y_range)
    torso_lean = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    arm_deviation = torch.sum(
        torch.abs(
            robot.data.joint_pos[:, env._left_hand_arm_joint_ids]
            - robot.data.default_joint_pos[:, env._left_hand_arm_joint_ids]
        ),
        dim=1,
    )

    if torch.any(success_edge):
        env._left_hand_workspace_error_at_contact[success_edge] = workspace_error[success_edge]
        env._left_hand_torso_lean_at_contact[success_edge] = torso_lean[success_edge]
        env._left_hand_arm_extension_at_contact[success_edge] = arm_deviation[success_edge]
        env._left_hand_completed_targets[success_edge] += 1
        env._left_hand_recent_success[success_edge] = True
        if ENABLE_LONG_HORIZON_DEBUG_METRICS and hasattr(command_term, "metrics"):
            for index in range(max_targets_per_episode):
                mask = success_edge & (env._left_hand_completed_targets == (index + 1))
                command_term.metrics[f"success_target_{index}"][mask] = 1.0
        active_ids = torch.where(success_edge & (env._left_hand_completed_targets < max_targets_per_episode))[0]
        if len(active_ids) > 0:
            _spawn_new_fixed_targets(
                env, active_ids, sample_regimes=sample_regimes, sample_weights=sample_weights
            )
            just_spawned[active_ids] = True

    switch_detected = torch.norm(env._left_hand_active_target_w - env._left_hand_prev_target_w, dim=-1) > 1.0e-5
    switch_detected |= reset_ids
    pre_timeout = env._left_hand_target_age_steps >= per_target_timeout_steps

    env._left_hand_post_switch_steps = torch.clamp(env._left_hand_post_switch_steps - 1, min=0)
    env._left_hand_target_age_steps += 1

    switched_non_reset = switch_detected & ~reset_ids
    env._left_hand_target_index[switched_non_reset] = torch.clamp(
        env._left_hand_target_index[switched_non_reset] + 1, max=max_targets_per_episode - 1
    )
    env._left_hand_post_switch_steps[switch_detected] = switch_phase_steps
    env._left_hand_target_age_steps[switch_detected] = 0
    env._left_hand_foot_motion_before_contact[switch_detected] = 0.0
    env._left_hand_target_switched_this_step[:] = switch_detected
    post_timeout = env._left_hand_target_age_steps >= per_target_timeout_steps

    foot_vel_xy = (
        robot.data.body_lin_vel_w[:, env._left_hand_foot_body_ids, :2] - robot.data.root_lin_vel_w[:, None, :2]
    )
    foot_speed = torch.linalg.norm(foot_vel_xy, dim=-1).mean(dim=1)
    env._left_hand_foot_motion_before_contact = torch.maximum(
        env._left_hand_foot_motion_before_contact, torch.tanh(foot_speed / 0.35)
    )

    if ENABLE_LONG_HORIZON_DEBUG_METRICS and hasattr(command_term, "metrics"):
        posture_quality = torch.exp(-(1.25 * torso_lean + 0.12 * arm_deviation))
        command_term.metrics["targets_completed"][:] = env._left_hand_completed_targets.float()
        command_term.metrics["target_index"][:] = env._left_hand_target_index.float()
        command_term.metrics["workspace_error"][:] = workspace_error
        command_term.metrics["workspace_error_at_contact"][:] = env._left_hand_workspace_error_at_contact
        command_term.metrics["torso_lean_at_contact"][:] = env._left_hand_torso_lean_at_contact
        command_term.metrics["arm_extension_at_contact"][:] = env._left_hand_arm_extension_at_contact
        command_term.metrics["foot_motion_before_contact"][:] = env._left_hand_foot_motion_before_contact
        command_term.metrics["post_switch_steps"][:] = env._left_hand_post_switch_steps.float()
        command_term.metrics["post_switch_posture_quality"][:] = posture_quality
        command_term.metrics.setdefault("per_target_timeout_steps", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics["per_target_timeout_steps"][:] = float(per_target_timeout_steps)
        command_term.metrics.setdefault("target_age_steps", torch.zeros(env.num_envs, device=env.device))
        command_term.metrics["target_age_steps"][:] = env._left_hand_target_age_steps.float()
        if hasattr(env, "termination_manager"):
            command_term.metrics["switch_failure_risk"][:] = (
                getattr(env.termination_manager, "terminated", torch.zeros(env.num_envs, device=env.device)).float()
                * (env._left_hand_post_switch_steps > 0).float()
            )

    _debug_timeout_state(
        env,
        current_episode_length=current_episode_length,
        prev_episode_length=prev_episode_length,
        just_reset=reset_ids,
        just_spawned=just_spawned,
        pre_timeout=pre_timeout,
        post_timeout=post_timeout,
        max_target_steps=per_target_timeout_steps,
    )
    _update_target_debug_visualization(env)
    env._left_hand_prev_target_w = env._left_hand_active_target_w.clone()
    env._left_hand_prev_success = success & ~switch_detected
    env._left_hand_prev_episode_length_buf = current_episode_length
    env._left_hand_state_synced_step = env.common_step_counter


def target_pos_command_obs(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    """Return the fixed world target expressed in the robot base-yaw frame."""
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    return _active_target_pos_base_yaw(env)


def reach_success(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
    threshold: float = 0.05,
):
    """Check whether the end-effector reaches the fixed world target threshold."""
    del command_name
    position_error = _static_target_position_error(env, asset_cfg=asset_cfg)
    return position_error < threshold


def target_quota_reached(
    env,
    max_targets_per_episode: int = 6,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    return env._left_hand_completed_targets >= max_targets_per_episode


def target_timeout_reached(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    per_target_timeout_steps = max(1, int(round(per_target_timeout_s / env.step_dt)))
    return env._left_hand_target_age_steps >= per_target_timeout_steps


def left_hand_target_pos_levels(
    env,
    env_ids: Sequence[int],
    command_name: str = "left_hand_pose",
    num_curriculum_episodes: int = 40,
    near_pos_x: tuple[float, float] = (0.25, 0.48),
    posture_pos_x: tuple[float, float] = (0.35, 0.72),
    far_pos_x: tuple[float, float] = (0.50, 1.00),
    near_pos_y: tuple[float, float] = (0.08, 0.28),
    posture_pos_y: tuple[float, float] = (0.02, 0.38),
    far_pos_y: tuple[float, float] = (-0.05, 0.60),
    near_pos_z: tuple[float, float] = (0.18, 0.34),
    posture_pos_z: tuple[float, float] = (0.00, 0.20),
    far_pos_z: tuple[float, float] = (0.08, 0.24),
):
    """Expand the sampling distribution from near targets to full local loco-reach targets."""
    del env_ids
    command_term = env.command_manager.get_term(command_name)

    progress = min(env.common_step_counter / (env.max_episode_length * num_curriculum_episodes), 1.0)
    progress_tensor = torch.tensor(progress, device=env.device)
    third_tensor = torch.tensor(1.0 / 3.0, device=env.device)
    two_third_tensor = torch.tensor(2.0 / 3.0, device=env.device)

    if env.common_step_counter % env.max_episode_length == 0:
        if progress <= 1.0 / 3.0:
            phase_progress = progress_tensor / third_tensor
            sample_regimes = {
                "near": {"pos_x": near_pos_x, "pos_y": near_pos_y, "pos_z": near_pos_z},
                "posture": {
                    "pos_x": _lerp_range(near_pos_x, posture_pos_x, phase_progress),
                    "pos_y": _lerp_range(near_pos_y, posture_pos_y, phase_progress),
                    "pos_z": _lerp_range(near_pos_z, posture_pos_z, phase_progress),
                },
                "far": {
                    "pos_x": _lerp_range(near_pos_x, posture_pos_x, phase_progress),
                    "pos_y": _lerp_range(near_pos_y, posture_pos_y, phase_progress),
                    "pos_z": _lerp_range(near_pos_z, posture_pos_z, phase_progress),
                },
            }
            sample_weights = {
                "near": float(torch.lerp(torch.tensor(0.85, device=env.device), torch.tensor(0.60, device=env.device), phase_progress)),
                "posture": float(torch.lerp(torch.tensor(0.15, device=env.device), torch.tensor(0.30, device=env.device), phase_progress)),
                "far": float(torch.lerp(torch.tensor(0.00, device=env.device), torch.tensor(0.10, device=env.device), phase_progress)),
            }
        elif progress <= 2.0 / 3.0:
            phase_progress = (progress_tensor - third_tensor) / third_tensor
            sample_regimes = {
                "near": {"pos_x": near_pos_x, "pos_y": near_pos_y, "pos_z": near_pos_z},
                "posture": {"pos_x": posture_pos_x, "pos_y": posture_pos_y, "pos_z": posture_pos_z},
                "far": {
                    "pos_x": _lerp_range(posture_pos_x, far_pos_x, phase_progress),
                    "pos_y": _lerp_range(posture_pos_y, far_pos_y, phase_progress),
                    "pos_z": _lerp_range(posture_pos_z, far_pos_z, phase_progress),
                },
            }
            sample_weights = {
                "near": float(torch.lerp(torch.tensor(0.60, device=env.device), torch.tensor(0.45, device=env.device), phase_progress)),
                "posture": float(torch.lerp(torch.tensor(0.30, device=env.device), torch.tensor(0.30, device=env.device), phase_progress)),
                "far": float(torch.lerp(torch.tensor(0.10, device=env.device), torch.tensor(0.25, device=env.device), phase_progress)),
            }
        else:
            sample_regimes = {
                "near": {"pos_x": near_pos_x, "pos_y": near_pos_y, "pos_z": near_pos_z},
                "posture": {"pos_x": posture_pos_x, "pos_y": posture_pos_y, "pos_z": posture_pos_z},
                "far": {"pos_x": far_pos_x, "pos_y": far_pos_y, "pos_z": far_pos_z},
            }
            sample_weights = {"near": 0.45, "posture": 0.30, "far": 0.25}

        _set_sampling_distribution_state(env, sample_regimes=sample_regimes, sample_weights=sample_weights)
        command_term.cfg.ranges.pos_x = _range_union(sample_regimes, "pos_x")
        command_term.cfg.ranges.pos_y = _range_union(sample_regimes, "pos_y")
        command_term.cfg.ranges.pos_z = _range_union(sample_regimes, "pos_z")

    return progress_tensor


def target_relative_base_stance_l2(
    env,
    command_name: str = "left_hand_pose",
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    """Penalize target positions that lie outside a favorable left-hand reach corridor in body/base-yaw frame."""
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    return _workspace_error_l2(env, command_name=command_name, x_range=x_range, y_range=y_range)


def target_relative_base_stance_ready(
    env,
    command_name: str = "left_hand_pose",
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    gate_std: float = 0.01,
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    post_switch_bonus_scale: float = 1.75,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    """Positive reward for bringing the target into a comfortable body-frame pre-reach corridor."""
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    gate = _workspace_ready_gate(env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std)
    return gate * (1.0 + post_switch_bonus_scale * _switch_phase_scale(env, switch_phase_steps))


def target_relative_base_stance_progress(
    env,
    command_name: str = "left_hand_pose",
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    post_switch_bonus_scale: float = 1.5,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    """Reward step-to-step reduction of target workspace error."""
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    workspace_error = _workspace_error_l2(env, command_name=command_name, x_range=x_range, y_range=y_range)
    if not hasattr(env, "_left_hand_prev_workspace_error"):
        env._left_hand_prev_workspace_error = workspace_error.clone()
    reset_ids = _current_reset_mask(env)
    env._left_hand_prev_workspace_error[reset_ids] = workspace_error[reset_ids]
    env._left_hand_prev_workspace_error[env._left_hand_target_switched_this_step] = workspace_error[
        env._left_hand_target_switched_this_step
    ]
    progress = env._left_hand_prev_workspace_error - workspace_error
    env._left_hand_prev_workspace_error = workspace_error.clone()
    return progress * (1.0 + post_switch_bonus_scale * _switch_phase_scale(env, switch_phase_steps))


def gated_position_command_error_tanh(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
    std: float = 0.14,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    gate_std: float = 0.01,
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    post_switch_scale: float = 0.25,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    """Allow strong fine reach reward only after the target is brought into a reasonable stance corridor."""
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    reach_reward = _static_target_position_error_tanh(env, asset_cfg=asset_cfg, std=std)
    stance_gate = _workspace_ready_gate(env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std)
    switch_scale = torch.where(
        env._left_hand_post_switch_steps > 0,
        torch.full((env.num_envs,), post_switch_scale, device=env.device),
        torch.ones(env.num_envs, device=env.device),
    )
    return reach_reward * stance_gate * switch_scale


def pre_stance_torso_lean_penalty(
    env,
    command_name: str = "left_hand_pose",
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    gate_std: float = 0.02,
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    post_switch_penalty_scale: float = 1.5,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    """Penalize premature torso leaning while the target is still outside the staging corridor."""
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    robot = env.scene["robot"]
    stance_not_ready = 1.0 - _workspace_ready_gate(
        env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std
    )
    torso_lean = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    return stance_not_ready * torso_lean * (1.0 + post_switch_penalty_scale * _switch_phase_scale(env, switch_phase_steps))


def pre_stance_joint_deviation_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "left_hand_pose",
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    gate_std: float = 0.02,
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    post_switch_penalty_scale: float = 1.5,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    """Penalize premature joint deviation before stance is ready."""
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    robot = env.scene[asset_cfg.name]
    stance_not_ready = 1.0 - _workspace_ready_gate(
        env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std
    )
    deviation = torch.sum(
        torch.abs(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return stance_not_ready * deviation * (1.0 + post_switch_penalty_scale * _switch_phase_scale(env, switch_phase_steps))


def pre_stance_joint_limit_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "left_hand_pose",
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    gate_std: float = 0.02,
    margin_threshold: float = 0.16,
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    post_switch_penalty_scale: float = 1.5,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    """Penalize pushing selected joints close to their soft limits before stance is ready."""
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    robot = env.scene[asset_cfg.name]
    stance_not_ready = 1.0 - _workspace_ready_gate(
        env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std
    )
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    joint_range = torch.clamp(joint_limits[..., 1] - joint_limits[..., 0], min=1e-6)
    normalized_margin = torch.minimum(joint_pos - joint_limits[..., 0], joint_limits[..., 1] - joint_pos) / joint_range
    limit_pressure = torch.clamp(margin_threshold - normalized_margin, min=0.0)
    return stance_not_ready * torch.sum(limit_pressure, dim=1) * (
        1.0 + post_switch_penalty_scale * _switch_phase_scale(env, switch_phase_steps)
    )


def pre_stance_foot_motion_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    command_name: str = "left_hand_pose",
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    gate_std: float = 0.02,
    speed_scale: float = 0.35,
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    """Lightly reward local foot motion while the target is still outside the staging corridor."""
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    robot = env.scene[asset_cfg.name]
    stance_not_ready = 1.0 - _workspace_ready_gate(
        env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std
    )
    foot_vel_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2] - robot.data.root_lin_vel_w[:, None, :2]
    foot_speed = torch.linalg.norm(foot_vel_xy, dim=-1).mean(dim=1)
    return stance_not_ready * torch.tanh(foot_speed / speed_scale)


def target_completion_bonus(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    return env._left_hand_recent_success.float()


def position_command_progress_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
):
    """Reward step-to-step reduction in hand-to-target error."""
    del command_name
    position_error = _static_target_position_error(env, asset_cfg=asset_cfg)
    if not hasattr(env, "_left_hand_prev_error"):
        env._left_hand_prev_error = position_error.clone()
    reset_ids = _current_reset_mask(env)
    env._left_hand_prev_error[reset_ids] = position_error[reset_ids]
    if hasattr(env, "_left_hand_target_switched_this_step"):
        env._left_hand_prev_error[env._left_hand_target_switched_this_step] = position_error[
            env._left_hand_target_switched_this_step
        ]
    progress = env._left_hand_prev_error - position_error
    env._left_hand_prev_error = position_error.clone()
    return progress


def success_posture_bonus(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    arm_joint_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
        ],
    ),
    command_name: str = "left_hand_pose",
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    gate_std: float = 0.01,
    reach_std: float = 0.08,
    success_threshold: float = 0.06,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    """Prefer reaching the target from a recoverable stance instead of a desperate stretched posture."""
    _sync_long_horizon_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    robot = env.scene["robot"]
    position_error = _static_target_position_error(env, asset_cfg=asset_cfg)
    stance_gate = _workspace_ready_gate(env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std)
    torso_lean = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    arm_joint_deviation = torch.sum(
        torch.abs(robot.data.joint_pos[:, arm_joint_cfg.joint_ids] - robot.data.default_joint_pos[:, arm_joint_cfg.joint_ids]),
        dim=1,
    )
    posture_quality = torch.exp(-(1.5 * torso_lean + 0.15 * arm_joint_deviation))
    return torch.exp(-position_error / reach_std) * stance_gate * posture_quality
