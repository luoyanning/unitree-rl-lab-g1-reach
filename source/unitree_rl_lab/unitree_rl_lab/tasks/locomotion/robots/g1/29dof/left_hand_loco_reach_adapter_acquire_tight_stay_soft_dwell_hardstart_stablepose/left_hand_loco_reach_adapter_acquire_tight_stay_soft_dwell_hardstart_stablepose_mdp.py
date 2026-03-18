from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from ..left_hand_loco_reach import left_hand_loco_reach_mdp as fixed_mdp


left_hand_target_pos_levels = fixed_mdp.left_hand_target_pos_levels
SOFT_DWELL_DECAY_STEPS = 2


def _command_tensor(env, command_name: str):
    command_term = env.command_manager.get_term(command_name)
    for attr_name in ("_command", "command"):
        if hasattr(command_term, attr_name):
            command_tensor = getattr(command_term, attr_name)
            if isinstance(command_tensor, torch.Tensor) and command_tensor.ndim == 2:
                return command_tensor
    return None


def _ensure_adapter_state(env):
    if not hasattr(env, "_left_hand_adapter_command"):
        env._left_hand_adapter_command = torch.zeros(env.num_envs, 3, device=env.device)
    if not hasattr(env, "_left_hand_in_post_success_dwell"):
        env._left_hand_in_post_success_dwell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._left_hand_post_success_dwell_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._left_hand_recent_dwell_completion = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _adapter_command_range(sample_regimes, axis_name: str):
    return fixed_mdp._range_union(sample_regimes, axis_name)


def _dwell_gate_and_progress(env, post_success_dwell_steps: int):
    dwell_gate = env._left_hand_in_post_success_dwell.float()
    dwell_steps = max(1, int(post_success_dwell_steps))
    dwell_progress = torch.clamp(
        env._left_hand_post_success_dwell_counter.float() / float(dwell_steps),
        min=0.0,
        max=1.0,
    )
    return dwell_gate, dwell_progress


def _late_dwell_gate(env, post_success_dwell_steps: int, activation_progress: float):
    dwell_gate, dwell_progress = _dwell_gate_and_progress(env, post_success_dwell_steps)
    late_progress = torch.clamp(
        (dwell_progress - float(activation_progress)) / max(1.0e-6, 1.0 - float(activation_progress)),
        min=0.0,
        max=1.0,
    )
    return dwell_gate, dwell_progress, late_progress


def _compute_adapter_command(
    env,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    switch_phase_steps: int,
    sample_regimes,
    sample_weights,
    command_name: str,
    adapter_gate_std: float,
    adapter_post_switch_bias: float,
    adapter_min_z_blend: float,
    adapter_snap_to_target_radius: float,
):
    del sample_weights, command_name
    fixed_mdp._get_sampling_distribution_state(env, sample_regimes=sample_regimes, sample_weights=None)
    target_pos_base = fixed_mdp._active_target_pos_base_yaw(env)
    workspace_gate = fixed_mdp._workspace_ready_gate(
        env,
        command_name="left_hand_pose",
        x_range=x_range,
        y_range=y_range,
        gate_std=adapter_gate_std,
    )
    switch_phase = fixed_mdp._switch_phase_scale(env, switch_phase_steps)
    corridor_center = torch.tensor(
        [0.5 * (x_range[0] + x_range[1]), 0.5 * (y_range[0] + y_range[1])],
        device=env.device,
    )

    blend = torch.clamp(workspace_gate * (1.0 - adapter_post_switch_bias * switch_phase), min=0.0, max=1.0)
    near_success = getattr(env, "_left_hand_in_success_zone", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    in_dwell = getattr(env, "_left_hand_in_post_success_dwell", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    near_target = fixed_mdp._static_target_position_error(env) <= adapter_snap_to_target_radius
    direct_target_mode = near_success | near_target | in_dwell
    blend = torch.where(direct_target_mode, torch.ones_like(blend), blend)

    adapted_xy = corridor_center.unsqueeze(0) + (target_pos_base[:, :2] - corridor_center.unsqueeze(0)) * blend.unsqueeze(-1)

    active_regimes, _ = fixed_mdp._get_sampling_distribution_state(env, sample_regimes=sample_regimes, sample_weights=None)
    x_cmd_range = _adapter_command_range(active_regimes, "pos_x")
    y_cmd_range = _adapter_command_range(active_regimes, "pos_y")
    z_cmd_range = _adapter_command_range(active_regimes, "pos_z")

    z_center = 0.5 * (z_cmd_range[0] + z_cmd_range[1])
    raw_z = torch.clamp(target_pos_base[:, 2], min=z_cmd_range[0], max=z_cmd_range[1])
    z_blend = torch.clamp(adapter_min_z_blend + (1.0 - adapter_min_z_blend) * blend, min=0.0, max=1.0)
    adapted_z = z_center + (raw_z - z_center) * z_blend

    adapter_command = torch.stack(
        (
            torch.clamp(adapted_xy[:, 0], min=x_cmd_range[0], max=x_cmd_range[1]),
            torch.clamp(adapted_xy[:, 1], min=y_cmd_range[0], max=y_cmd_range[1]),
            adapted_z,
        ),
        dim=-1,
    )
    adapter_command[direct_target_mode] = torch.stack(
        (
            torch.clamp(target_pos_base[direct_target_mode, 0], min=x_cmd_range[0], max=x_cmd_range[1]),
            torch.clamp(target_pos_base[direct_target_mode, 1], min=y_cmd_range[0], max=y_cmd_range[1]),
            torch.clamp(target_pos_base[direct_target_mode, 2], min=z_cmd_range[0], max=z_cmd_range[1]),
        ),
        dim=-1,
    )
    return adapter_command


def _sync_adapter_hold_stay_state(
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
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    adapter_gate_std: float = 0.04,
    adapter_post_switch_bias: float = 0.35,
    adapter_min_z_blend: float = 0.35,
    adapter_snap_to_target_radius: float = 0.12,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
):
    fixed_mdp._ensure_long_horizon_state(
        env,
        command_name=command_name,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
    )
    _ensure_adapter_state(env)
    if env._left_hand_state_synced_step == env.common_step_counter:
        return

    command_term = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    num_envs = env.num_envs
    reset_ids, current_episode_length, prev_episode_length = fixed_mdp._compute_just_reset_mask(env)
    env._left_hand_just_reset_this_step[:] = reset_ids
    env._left_hand_recent_success.zero_()
    env._left_hand_recent_dwell_completion.zero_()
    env._left_hand_completion_after_hold.zero_()
    env._left_hand_target_switched_this_step.zero_()
    fixed_mdp._get_sampling_distribution_state(env, sample_regimes=sample_regimes, sample_weights=sample_weights)
    per_target_timeout_steps = max(1, int(round(per_target_timeout_s / env.step_dt)))
    dwell_steps = max(1, int(post_success_dwell_steps))

    if fixed_mdp.ENABLE_TARGET_TIMEOUT_DEBUG and not env._left_hand_timeout_cfg_logged:
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

    if hasattr(command_term, "metrics"):
        command_term.metrics.setdefault("post_success_dwell_flag", torch.zeros(num_envs, device=env.device))
        command_term.metrics.setdefault("post_success_dwell_counter", torch.zeros(num_envs, device=env.device))
        command_term.metrics.setdefault("completion_after_dwell", torch.zeros(num_envs, device=env.device))

    if torch.any(reset_ids):
        env._left_hand_completed_targets[reset_ids] = 0
        env._left_hand_held_success_count[reset_ids] = 0
        env._left_hand_target_index[reset_ids] = 0
        env._left_hand_post_switch_steps[reset_ids] = switch_phase_steps
        env._left_hand_target_age_steps[reset_ids] = 0
        env._left_hand_prev_success[reset_ids] = False
        env._left_hand_recent_success[reset_ids] = False
        env._left_hand_in_success_zone[reset_ids] = False
        env._left_hand_success_hold_counter[reset_ids] = 0
        env._left_hand_success_zone_time[reset_ids] = 0
        env._left_hand_completion_after_hold[reset_ids] = False
        env._left_hand_foot_motion_before_contact[reset_ids] = 0.0
        env._left_hand_workspace_error_at_contact[reset_ids] = 0.0
        env._left_hand_torso_lean_at_contact[reset_ids] = 0.0
        env._left_hand_arm_extension_at_contact[reset_ids] = 0.0
        env._left_hand_distance_at_completion[reset_ids] = 0.0
        env._left_hand_has_active_target[reset_ids] = False
        env._left_hand_in_post_success_dwell[reset_ids] = False
        env._left_hand_post_success_dwell_counter[reset_ids] = 0
        env._left_hand_recent_dwell_completion[reset_ids] = False

    just_spawned = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    inactive_ids = torch.where(~env._left_hand_has_active_target)[0]
    if len(inactive_ids) > 0:
        fixed_mdp._spawn_new_fixed_targets(
            env,
            inactive_ids,
            sample_regimes=sample_regimes,
            sample_weights=sample_weights,
        )
        just_spawned[inactive_ids] = True

    position_error = fixed_mdp._ee_position_error(env, command_name=command_name)
    enter_success_zone = position_error <= success_threshold
    remain_in_success_zone = env._left_hand_in_success_zone & (position_error <= success_exit_radius)
    in_success_zone = enter_success_zone | remain_in_success_zone
    env._left_hand_success_zone_time += in_success_zone.long()
    env._left_hand_success_hold_counter = torch.where(
        in_success_zone,
        env._left_hand_success_hold_counter + 1,
        torch.zeros_like(env._left_hand_success_hold_counter),
    )
    env._left_hand_in_success_zone[:] = in_success_zone

    hold_qualified = env._left_hand_success_hold_counter >= max(1, int(success_hold_steps))
    success_edge = hold_qualified & ~env._left_hand_prev_success & ~reset_ids & ~env._left_hand_in_post_success_dwell

    workspace_error = fixed_mdp._workspace_error_l2(env, command_name=command_name, x_range=x_range, y_range=y_range)
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
        env._left_hand_in_post_success_dwell[success_edge] = True
        env._left_hand_post_success_dwell_counter[success_edge] = 0

    dwell_in_zone = env._left_hand_in_post_success_dwell & (position_error <= post_success_exit_radius)
    decay = torch.full_like(env._left_hand_post_success_dwell_counter, SOFT_DWELL_DECAY_STEPS)
    decayed_counter = torch.clamp(env._left_hand_post_success_dwell_counter - decay, min=0)
    env._left_hand_post_success_dwell_counter = torch.where(
        dwell_in_zone,
        env._left_hand_post_success_dwell_counter + 1,
        decayed_counter,
    )

    # Keep the dwell phase active after first contact and allow short excursions to
    # decay progress instead of resetting the whole phase in one frame.
    dwell_done = env._left_hand_in_post_success_dwell & (env._left_hand_post_success_dwell_counter >= dwell_steps)
    if torch.any(dwell_done):
        env._left_hand_distance_at_completion[dwell_done] = position_error[dwell_done]
        env._left_hand_completed_targets[dwell_done] += 1
        env._left_hand_held_success_count[dwell_done] += 1
        env._left_hand_recent_success[dwell_done] = True
        env._left_hand_recent_dwell_completion[dwell_done] = True
        env._left_hand_completion_after_hold[dwell_done] = True
        if fixed_mdp.ENABLE_LONG_HORIZON_DEBUG_METRICS and hasattr(command_term, "metrics"):
            for index in range(max_targets_per_episode):
                mask = dwell_done & (env._left_hand_completed_targets == (index + 1))
                command_term.metrics[f"success_target_{index}"][mask] = 1.0
        active_ids = torch.where(dwell_done & (env._left_hand_completed_targets < max_targets_per_episode))[0]
        if len(active_ids) > 0:
            fixed_mdp._spawn_new_fixed_targets(
                env,
                active_ids,
                sample_regimes=sample_regimes,
                sample_weights=sample_weights,
            )
            just_spawned[active_ids] = True

    switch_detected = torch.norm(env._left_hand_active_target_w - env._left_hand_prev_target_w, dim=-1) > 1.0e-5
    switch_detected |= reset_ids
    pre_timeout = env._left_hand_target_age_steps >= per_target_timeout_steps

    env._left_hand_post_switch_steps = torch.clamp(env._left_hand_post_switch_steps - 1, min=0)
    env._left_hand_target_age_steps += 1

    switched_non_reset = switch_detected & ~reset_ids
    env._left_hand_target_index[switched_non_reset] = torch.clamp(
        env._left_hand_target_index[switched_non_reset] + 1,
        max=max_targets_per_episode - 1,
    )
    env._left_hand_post_switch_steps[switch_detected] = switch_phase_steps
    env._left_hand_target_age_steps[switch_detected] = 0
    env._left_hand_in_success_zone[switch_detected] = False
    env._left_hand_success_hold_counter[switch_detected] = 0
    env._left_hand_success_zone_time[switch_detected] = 0
    env._left_hand_in_post_success_dwell[switch_detected] = False
    env._left_hand_post_success_dwell_counter[switch_detected] = 0
    env._left_hand_foot_motion_before_contact[switch_detected] = 0.0
    env._left_hand_target_switched_this_step[:] = switch_detected
    post_timeout = env._left_hand_target_age_steps >= per_target_timeout_steps

    foot_vel_xy = robot.data.body_lin_vel_w[:, env._left_hand_foot_body_ids, :2] - robot.data.root_lin_vel_w[:, None, :2]
    foot_speed = torch.linalg.norm(foot_vel_xy, dim=-1).mean(dim=1)
    env._left_hand_foot_motion_before_contact = torch.maximum(
        env._left_hand_foot_motion_before_contact,
        torch.tanh(foot_speed / 0.35),
    )

    if hasattr(command_term, "metrics"):
        command_term.metrics["success_zone_flag"][:] = env._left_hand_in_success_zone.float()
        command_term.metrics["success_hold_counter"][:] = env._left_hand_success_hold_counter.float()
        command_term.metrics["success_zone_time"][:] = env._left_hand_success_zone_time.float()
        command_term.metrics["held_success_count"][:] = env._left_hand_held_success_count.float()
        command_term.metrics["completion_distance"][:] = env._left_hand_distance_at_completion
        command_term.metrics["completion_after_hold"][:] = env._left_hand_completion_after_hold.float()
        command_term.metrics["post_success_dwell_flag"][:] = env._left_hand_in_post_success_dwell.float()
        command_term.metrics["post_success_dwell_counter"][:] = env._left_hand_post_success_dwell_counter.float()
        command_term.metrics["completion_after_dwell"][:] = env._left_hand_recent_dwell_completion.float()

    if fixed_mdp.ENABLE_LONG_HORIZON_DEBUG_METRICS and hasattr(command_term, "metrics"):
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
        command_term.metrics.setdefault("per_target_timeout_steps", torch.zeros(num_envs, device=env.device))
        command_term.metrics["per_target_timeout_steps"][:] = float(per_target_timeout_steps)
        command_term.metrics.setdefault("target_age_steps", torch.zeros(num_envs, device=env.device))
        command_term.metrics["target_age_steps"][:] = env._left_hand_target_age_steps.float()
        if hasattr(env, "termination_manager"):
            command_term.metrics["switch_failure_risk"][:] = (
                getattr(env.termination_manager, "terminated", torch.zeros(num_envs, device=env.device)).float()
                * (env._left_hand_post_switch_steps > 0).float()
            )

    fixed_mdp._set_base_velocity_guidance_command(env, x_range=x_range, y_range=y_range)
    base_velocity_command = _command_tensor(env, "base_velocity")
    if base_velocity_command is not None:
        base_velocity_command[env._left_hand_in_post_success_dwell] = 0.0

    fixed_mdp._debug_timeout_state(
        env,
        current_episode_length=current_episode_length,
        prev_episode_length=prev_episode_length,
        just_reset=reset_ids,
        just_spawned=just_spawned,
        pre_timeout=pre_timeout,
        post_timeout=post_timeout,
        max_target_steps=per_target_timeout_steps,
    )
    fixed_mdp._update_target_debug_visualization(env)
    env._left_hand_prev_target_w = env._left_hand_active_target_w.clone()
    env._left_hand_prev_success = (hold_qualified | env._left_hand_in_post_success_dwell) & ~switch_detected
    env._left_hand_prev_episode_length_buf = current_episode_length
    env._left_hand_state_synced_step = env.common_step_counter

    adapter_command = _compute_adapter_command(
        env,
        x_range=x_range,
        y_range=y_range,
        switch_phase_steps=switch_phase_steps,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
        command_name=command_name,
        adapter_gate_std=adapter_gate_std,
        adapter_post_switch_bias=adapter_post_switch_bias,
        adapter_min_z_blend=adapter_min_z_blend,
        adapter_snap_to_target_radius=adapter_snap_to_target_radius,
    )
    env._left_hand_adapter_command[:] = adapter_command
    pose_command = _command_tensor(env, command_name)
    if pose_command is not None and pose_command.shape[1] >= 3:
        pose_command[:, :3] = adapter_command
        if pose_command.shape[1] >= 6:
            pose_command[:, 3:6] = 0.0


def target_pos_command_obs(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    adapter_gate_std: float = 0.04,
    adapter_post_switch_bias: float = 0.35,
    adapter_min_z_blend: float = 0.35,
    adapter_snap_to_target_radius: float = 0.12,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
        adapter_gate_std=adapter_gate_std,
        adapter_post_switch_bias=adapter_post_switch_bias,
        adapter_min_z_blend=adapter_min_z_blend,
        adapter_snap_to_target_radius=adapter_snap_to_target_radius,
    )
    return env._left_hand_adapter_command


def static_target_position_error(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    return fixed_mdp._static_target_position_error(env, asset_cfg=asset_cfg)


def target_quota_reached(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
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
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
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


def target_relative_base_stance_l2(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    return fixed_mdp._workspace_error_l2(env, command_name=command_name, x_range=x_range, y_range=y_range)


def target_relative_base_stance_ready(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    gate_std: float = 0.01,
    post_switch_bonus_scale: float = 1.75,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    gate = fixed_mdp._workspace_ready_gate(env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std)
    return gate * (1.0 + post_switch_bonus_scale * fixed_mdp._switch_phase_scale(env, switch_phase_steps))


def target_relative_base_stance_progress(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    post_switch_bonus_scale: float = 1.5,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    workspace_error = fixed_mdp._workspace_error_l2(env, command_name=command_name, x_range=x_range, y_range=y_range)
    if not hasattr(env, "_left_hand_prev_workspace_error"):
        env._left_hand_prev_workspace_error = workspace_error.clone()
    reset_ids = fixed_mdp._current_reset_mask(env)
    env._left_hand_prev_workspace_error[reset_ids] = workspace_error[reset_ids]
    env._left_hand_prev_workspace_error[env._left_hand_target_switched_this_step] = workspace_error[
        env._left_hand_target_switched_this_step
    ]
    progress = env._left_hand_prev_workspace_error - workspace_error
    env._left_hand_prev_workspace_error = workspace_error.clone()
    return progress * (1.0 + post_switch_bonus_scale * fixed_mdp._switch_phase_scale(env, switch_phase_steps))


def gated_position_command_error_tanh(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    std: float = 0.14,
    gate_std: float = 0.01,
    post_switch_scale: float = 0.25,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    reach_reward = fixed_mdp._static_target_position_error_tanh(env, asset_cfg=asset_cfg, std=std)
    stance_gate = fixed_mdp._workspace_ready_gate(env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std)
    switch_scale = torch.where(
        env._left_hand_post_switch_steps > 0,
        torch.full((env.num_envs,), post_switch_scale, device=env.device),
        torch.ones(env.num_envs, device=env.device),
    )
    return reach_reward * stance_gate * switch_scale


def pre_stance_torso_lean_penalty(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    gate_std: float = 0.02,
    post_switch_penalty_scale: float = 1.5,
    near_success_penalty_radius: float = 0.10,
    near_success_penalty_scale: float = 0.2,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
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
    stance_not_ready = 1.0 - fixed_mdp._workspace_ready_gate(env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std)
    torso_lean = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    relief_scale = fixed_mdp._near_success_penalty_scale(
        env,
        position_error=fixed_mdp._ee_position_error(env, command_name=command_name),
        near_success_penalty_radius=near_success_penalty_radius,
        near_success_penalty_scale=near_success_penalty_scale,
    )
    return stance_not_ready * torso_lean * (1.0 + post_switch_penalty_scale * fixed_mdp._switch_phase_scale(env, switch_phase_steps)) * relief_scale


def pre_stance_joint_deviation_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    gate_std: float = 0.02,
    post_switch_penalty_scale: float = 1.5,
    near_success_penalty_radius: float = 0.10,
    near_success_penalty_scale: float = 0.2,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
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
    stance_not_ready = 1.0 - fixed_mdp._workspace_ready_gate(env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std)
    deviation = torch.sum(torch.abs(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)
    relief_scale = fixed_mdp._near_success_penalty_scale(
        env,
        position_error=fixed_mdp._ee_position_error(env, command_name=command_name),
        near_success_penalty_radius=near_success_penalty_radius,
        near_success_penalty_scale=near_success_penalty_scale,
    )
    return stance_not_ready * deviation * (1.0 + post_switch_penalty_scale * fixed_mdp._switch_phase_scale(env, switch_phase_steps)) * relief_scale


def pre_stance_joint_limit_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    gate_std: float = 0.02,
    margin_threshold: float = 0.16,
    post_switch_penalty_scale: float = 1.5,
    near_success_penalty_radius: float = 0.10,
    near_success_penalty_scale: float = 0.2,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
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
    stance_not_ready = 1.0 - fixed_mdp._workspace_ready_gate(env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std)
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    joint_range = torch.clamp(joint_limits[..., 1] - joint_limits[..., 0], min=1e-6)
    normalized_margin = torch.minimum(joint_pos - joint_limits[..., 0], joint_limits[..., 1] - joint_pos) / joint_range
    limit_pressure = torch.clamp(margin_threshold - normalized_margin, min=0.0)
    relief_scale = fixed_mdp._near_success_penalty_scale(
        env,
        position_error=fixed_mdp._ee_position_error(env, command_name=command_name),
        near_success_penalty_radius=near_success_penalty_radius,
        near_success_penalty_scale=near_success_penalty_scale,
    )
    return stance_not_ready * torch.sum(limit_pressure, dim=1) * (1.0 + post_switch_penalty_scale * fixed_mdp._switch_phase_scale(env, switch_phase_steps)) * relief_scale


def pre_stance_foot_motion_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    gate_std: float = 0.02,
    speed_scale: float = 0.35,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
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
    stance_not_ready = 1.0 - fixed_mdp._workspace_ready_gate(env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std)
    foot_vel_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2] - robot.data.root_lin_vel_w[:, None, :2]
    foot_speed = torch.linalg.norm(foot_vel_xy, dim=-1).mean(dim=1)
    return stance_not_ready * torch.tanh(foot_speed / speed_scale)


def target_completion_bonus(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
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


def target_hold_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    hold_reward_std: float = 0.03,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    position_error = fixed_mdp._static_target_position_error(env, asset_cfg=asset_cfg)
    hold_gate = (env._left_hand_in_success_zone & ~env._left_hand_in_post_success_dwell).float()
    return hold_gate * torch.exp(-position_error / hold_reward_std) * (0.5 + 0.5 * fixed_mdp._hold_progress(env, success_hold_steps))


def post_success_stay_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    stay_reward_std: float = 0.02,
    hand_speed_scale: float = 0.15,
    base_speed_scale: float = 0.20,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
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
    ee_lin_vel = robot.data.body_lin_vel_w[:, asset_cfg.body_ids]
    if ee_lin_vel.ndim == 3:
        ee_lin_vel = ee_lin_vel[:, 0]
    ee_speed = torch.linalg.norm(ee_lin_vel, dim=-1)
    base_speed = torch.linalg.norm(env.scene["robot"].data.root_lin_vel_w[:, :2], dim=-1)
    position_error = fixed_mdp._static_target_position_error(env, asset_cfg=asset_cfg)
    stay_gate, dwell_progress = _dwell_gate_and_progress(env, post_success_dwell_steps)
    stability = torch.exp(-ee_speed / hand_speed_scale) * torch.exp(-base_speed / base_speed_scale)
    return stay_gate * torch.exp(-position_error / stay_reward_std) * stability * dwell_progress


def dwell_right_arm_neutral_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    joint_dev_scale: float = 0.30,
    activation_progress: float = 0.5,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
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
    joint_dev = torch.sum(
        torch.abs(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    dwell_gate, _dwell_progress, late_progress = _late_dwell_gate(
        env, post_success_dwell_steps, activation_progress
    )
    return dwell_gate * torch.exp(-joint_dev / joint_dev_scale) * late_progress


def dwell_stationary_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    base_lin_speed_scale: float = 0.10,
    base_ang_speed_scale: float = 0.30,
    foot_speed_scale: float = 0.08,
    activation_progress: float = 0.5,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
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
    base_lin_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    base_ang_speed = torch.linalg.norm(robot.data.root_ang_vel_w[:, :3], dim=-1)
    foot_vel_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2] - robot.data.root_lin_vel_w[:, None, :2]
    foot_speed = torch.linalg.norm(foot_vel_xy, dim=-1).mean(dim=1)
    dwell_gate, _dwell_progress, late_progress = _late_dwell_gate(
        env, post_success_dwell_steps, activation_progress
    )
    stability = (
        torch.exp(-base_lin_speed / base_lin_speed_scale)
        * torch.exp(-base_ang_speed / base_ang_speed_scale)
        * torch.exp(-foot_speed / foot_speed_scale)
    )
    return dwell_gate * stability * late_progress


def near_target_action_rate_l2(
    env,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    near_success_penalty_radius: float = 0.10,
    near_success_penalty_scale: float = 0.2,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    position_error = fixed_mdp._ee_position_error(env, command_name=command_name)
    return fixed_mdp.loco_mdp.action_rate_l2(env) * fixed_mdp._near_success_penalty_scale(
        env,
        position_error=position_error,
        near_success_penalty_radius=near_success_penalty_radius,
        near_success_penalty_scale=near_success_penalty_scale,
    )


def near_target_joint_deviation_l1(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "left_hand_pose",
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    near_success_penalty_radius: float = 0.10,
    near_success_penalty_scale: float = 0.2,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
        max_targets_per_episode=max_targets_per_episode,
        switch_phase_steps=switch_phase_steps,
        static_target_hold_s=static_target_hold_s,
        per_target_timeout_s=per_target_timeout_s,
        x_range=x_range,
        y_range=y_range,
        sample_regimes=sample_regimes,
        sample_weights=sample_weights,
    )
    position_error = fixed_mdp._ee_position_error(env, command_name=command_name)
    return fixed_mdp.loco_mdp.joint_deviation_l1(env, asset_cfg=asset_cfg) * fixed_mdp._near_success_penalty_scale(
        env,
        position_error=position_error,
        near_success_penalty_radius=near_success_penalty_radius,
        near_success_penalty_scale=near_success_penalty_scale,
    )


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
    success_threshold: float = 0.06,
    success_exit_radius: float = 0.09,
    success_hold_steps: int = 8,
    post_success_dwell_steps: int = 10,
    post_success_exit_radius: float = 0.12,
    max_targets_per_episode: int = 6,
    switch_phase_steps: int = 30,
    static_target_hold_s: float = 1.0e9,
    per_target_timeout_s: float = 4.0,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (0.08, 0.28),
    sample_regimes: dict[str, dict[str, tuple[float, float]]] | None = None,
    sample_weights: dict[str, float] | None = None,
    gate_std: float = 0.01,
    reach_std: float = 0.08,
):
    _sync_adapter_hold_stay_state(
        env,
        command_name=command_name,
        success_threshold=success_threshold,
        success_exit_radius=success_exit_radius,
        success_hold_steps=success_hold_steps,
        post_success_dwell_steps=post_success_dwell_steps,
        post_success_exit_radius=post_success_exit_radius,
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
    position_error = fixed_mdp._static_target_position_error(env, asset_cfg=asset_cfg)
    stance_gate = fixed_mdp._workspace_ready_gate(env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std)
    torso_lean = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    arm_joint_deviation = torch.sum(
        torch.abs(robot.data.joint_pos[:, arm_joint_cfg.joint_ids] - robot.data.default_joint_pos[:, arm_joint_cfg.joint_ids]),
        dim=1,
    )
    posture_quality = torch.exp(-(1.5 * torso_lean + 0.15 * arm_joint_deviation))
    return torch.exp(-position_error / reach_std) * stance_gate * posture_quality
