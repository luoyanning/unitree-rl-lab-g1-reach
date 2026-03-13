from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp
from isaaclab.managers import SceneEntityCfg

ENABLE_LONG_HORIZON_DEBUG_METRICS = False


def _best_effort_command_resample(command_term, env_ids, static_target_hold_s: float):
    if len(env_ids) == 0:
        return
    if hasattr(command_term, "_resample_command"):
        command_term._resample_command(env_ids)
    for attr_name in ("time_left", "_time_left", "command_time_left"):
        if hasattr(command_term, attr_name):
            timer = getattr(command_term, attr_name)
            if isinstance(timer, torch.Tensor) and timer.ndim > 0:
                timer[env_ids] = static_target_hold_s


def _ensure_long_horizon_state(env, command_name: str, max_targets_per_episode: int, switch_phase_steps: int):
    num_envs = env.num_envs
    if not hasattr(env, "_left_hand_prev_command"):
        command = env.command_manager.get_command(command_name)[:, :3]
        robot = env.scene["robot"]
        env._left_hand_prev_command = command.clone()
        env._left_hand_prev_success = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        env._left_hand_completed_targets = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        env._left_hand_target_index = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        env._left_hand_post_switch_steps = torch.full(
            (num_envs,), switch_phase_steps, dtype=torch.long, device=env.device
        )
        env._left_hand_steps_since_switch = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        env._left_hand_foot_motion_before_contact = torch.zeros(num_envs, device=env.device)
        env._left_hand_workspace_error_at_contact = torch.zeros(num_envs, device=env.device)
        env._left_hand_torso_lean_at_contact = torch.zeros(num_envs, device=env.device)
        env._left_hand_arm_extension_at_contact = torch.zeros(num_envs, device=env.device)
        env._left_hand_recent_success = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        env._left_hand_state_synced_step = -1
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
    _ensure_long_horizon_state(env, command_name=command_name, max_targets_per_episode=1, switch_phase_steps=0)
    robot = env.scene["robot"]
    ee_pos_w = robot.data.body_pos_w[:, env._left_hand_ee_body_id]
    command = env.command_manager.get_command(command_name)[:, :3]
    return torch.linalg.norm(command - ee_pos_w, dim=-1)


def _workspace_error_components(
    env,
    command_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
):
    target_pos = env.command_manager.get_command(command_name)[:, :2]
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
):
    _ensure_long_horizon_state(
        env, command_name=command_name, max_targets_per_episode=max_targets_per_episode, switch_phase_steps=switch_phase_steps
    )
    if env._left_hand_state_synced_step == env.common_step_counter:
        return

    command_term = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    current_command = env.command_manager.get_command(command_name)[:, :3].clone()
    reset_ids = env.episode_length_buf == 0
    env._left_hand_recent_success.zero_()
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
            _best_effort_command_resample(command_term, active_ids, static_target_hold_s=static_target_hold_s)
            current_command = env.command_manager.get_command(command_name)[:, :3].clone()

    switch_detected = torch.norm(current_command - env._left_hand_prev_command, dim=-1) > 1.0e-5
    switch_detected |= reset_ids
    per_target_timeout_steps = max(1, int(round(per_target_timeout_s / env.step_dt)))

    env._left_hand_post_switch_steps = torch.clamp(env._left_hand_post_switch_steps - 1, min=0)
    env._left_hand_steps_since_switch += 1

    switched_non_reset = switch_detected & ~reset_ids
    env._left_hand_target_index[switched_non_reset] = torch.clamp(
        env._left_hand_target_index[switched_non_reset] + 1, max=max_targets_per_episode - 1
    )
    env._left_hand_post_switch_steps[switch_detected] = switch_phase_steps
    env._left_hand_steps_since_switch[switch_detected] = 0
    env._left_hand_foot_motion_before_contact[switch_detected] = 0.0

    if torch.any(reset_ids):
        env._left_hand_completed_targets[reset_ids] = 0
        env._left_hand_target_index[reset_ids] = 0
        env._left_hand_post_switch_steps[reset_ids] = switch_phase_steps
        env._left_hand_steps_since_switch[reset_ids] = 0
        env._left_hand_prev_success[reset_ids] = False
        env._left_hand_recent_success[reset_ids] = False
        env._left_hand_foot_motion_before_contact[reset_ids] = 0.0
        env._left_hand_workspace_error_at_contact[reset_ids] = 0.0
        env._left_hand_torso_lean_at_contact[reset_ids] = 0.0
        env._left_hand_arm_extension_at_contact[reset_ids] = 0.0

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
        if hasattr(env, "termination_manager"):
            command_term.metrics["switch_failure_risk"][:] = (
                getattr(env.termination_manager, "terminated", torch.zeros(env.num_envs, device=env.device)).float()
                * (env._left_hand_post_switch_steps > 0).float()
            )

    env._left_hand_prev_command = current_command
    env._left_hand_prev_success = success & ~switch_detected
    env._left_hand_state_synced_step = env.common_step_counter


def target_pos_command_obs(env, command_name: str = "left_hand_pose"):
    """Return the target position command as a 3D vector for policy/critic compatibility."""
    return env.command_manager.get_command(command_name)[:, :3]


def reach_success(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
    threshold: float = 0.05,
):
    """Terminate an episode when the left hand reaches the target threshold."""
    position_error = reach_mdp.position_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
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
    )
    per_target_timeout_steps = max(1, int(round(per_target_timeout_s / env.step_dt)))
    return env._left_hand_steps_since_switch >= per_target_timeout_steps


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
    """Expand explicit near, posture-heavy, and far-local loco-reach target regimes."""
    del env_ids
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges

    progress = min(env.common_step_counter / (env.max_episode_length * num_curriculum_episodes), 1.0)
    progress_tensor = torch.tensor(progress, device=env.device)
    third_tensor = torch.tensor(1.0 / 3.0, device=env.device)
    two_third_tensor = torch.tensor(2.0 / 3.0, device=env.device)

    if env.common_step_counter % env.max_episode_length == 0:
        if progress <= 1.0 / 3.0:
            phase_progress = progress_tensor / third_tensor
            ranges.pos_x = torch.lerp(
                torch.tensor(near_pos_x, device=env.device),
                torch.tensor(posture_pos_x, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_y = torch.lerp(
                torch.tensor(near_pos_y, device=env.device),
                torch.tensor(posture_pos_y, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_z = torch.lerp(
                torch.tensor(near_pos_z, device=env.device),
                torch.tensor(posture_pos_z, device=env.device),
                phase_progress,
            ).tolist()
        elif progress <= 2.0 / 3.0:
            phase_progress = (progress_tensor - third_tensor) / third_tensor
            ranges.pos_x = torch.lerp(
                torch.tensor(posture_pos_x, device=env.device),
                torch.tensor(far_pos_x, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_y = torch.lerp(
                torch.tensor(posture_pos_y, device=env.device),
                torch.tensor(far_pos_y, device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_z = torch.lerp(
                torch.tensor(posture_pos_z, device=env.device),
                torch.tensor(far_pos_z, device=env.device),
                phase_progress,
            ).tolist()
        else:
            phase_progress = (progress_tensor - two_third_tensor) / third_tensor
            ranges.pos_x = torch.lerp(
                torch.tensor(far_pos_x, device=env.device),
                torch.tensor((near_pos_x[0], far_pos_x[1]), device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_y = torch.lerp(
                torch.tensor(far_pos_y, device=env.device),
                torch.tensor((far_pos_y[0], far_pos_y[1]), device=env.device),
                phase_progress,
            ).tolist()
            ranges.pos_z = torch.lerp(
                torch.tensor(far_pos_z, device=env.device),
                torch.tensor((posture_pos_z[0], far_pos_z[1]), device=env.device),
                phase_progress,
            ).tolist()

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
    )
    workspace_error = _workspace_error_l2(env, command_name=command_name, x_range=x_range, y_range=y_range)
    if not hasattr(env, "_left_hand_prev_workspace_error"):
        env._left_hand_prev_workspace_error = workspace_error.clone()
    if hasattr(env, "episode_length_buf"):
        reset_ids = env.episode_length_buf == 0
        env._left_hand_prev_workspace_error[reset_ids] = workspace_error[reset_ids]
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
    )
    reach_reward = reach_mdp.position_command_error_tanh(env, asset_cfg=asset_cfg, command_name=command_name, std=std)
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
    )
    return env._left_hand_recent_success.float()


def position_command_progress_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
    command_name: str = "left_hand_pose",
):
    """Reward step-to-step reduction in hand-to-target error."""
    position_error = reach_mdp.position_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    if not hasattr(env, "_left_hand_prev_error"):
        env._left_hand_prev_error = position_error.clone()
    if hasattr(env, "episode_length_buf"):
        reset_ids = env.episode_length_buf == 0
        env._left_hand_prev_error[reset_ids] = position_error[reset_ids]
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
    )
    robot = env.scene["robot"]
    position_error = reach_mdp.position_command_error(env, asset_cfg=asset_cfg, command_name=command_name)
    stance_gate = _workspace_ready_gate(env, command_name=command_name, x_range=x_range, y_range=y_range, gate_std=gate_std)
    torso_lean = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    arm_joint_deviation = torch.sum(
        torch.abs(robot.data.joint_pos[:, arm_joint_cfg.joint_ids] - robot.data.default_joint_pos[:, arm_joint_cfg.joint_ids]),
        dim=1,
    )
    posture_quality = torch.exp(-(1.5 * torso_lean + 0.15 * arm_joint_deviation))
    return torch.exp(-position_error / reach_std) * stance_gate * posture_quality
