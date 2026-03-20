from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


LEFT_HAND_BODY_NAME = "left_wrist_yaw_link"
ACQUIRE_MODE = 0
PLACE_MODE = 1
TABLE_TOP_Z = 0.78
BALL_RADIUS = 0.04
BALL_CENTER_Z = TABLE_TOP_Z + BALL_RADIUS


def _episode_length_buf(env) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        return env.episode_length_buf.clone()
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _compute_just_reset_mask(env):
    current_episode_length = _episode_length_buf(env)
    if not hasattr(env, "_pp_prev_episode_length_buf"):
        env._pp_prev_episode_length_buf = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    prev_episode_length = env._pp_prev_episode_length_buf
    just_reset = prev_episode_length < 0
    just_reset |= current_episode_length == 0
    just_reset |= current_episode_length < prev_episode_length
    return just_reset, current_episode_length


def _robot(env) -> Articulation:
    return env.scene["robot"]


def _ball(env) -> RigidObject:
    return env.scene["ball"]


def _place_target(env) -> RigidObject:
    return env.scene["place_target"]


def _hand_pos_w(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    hand_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if hand_pos_w.ndim == 3:
        hand_pos_w = hand_pos_w[:, 0]
    return hand_pos_w


def _hand_vel_w(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    hand_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    if hand_vel_w.ndim == 3:
        hand_vel_w = hand_vel_w[:, 0]
    return hand_vel_w


def _call_first_available(obj, names: tuple[str, ...], *args, **kwargs):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)(*args, **kwargs)
    raise AttributeError(f"{type(obj).__name__} does not expose any of {names}.")


def _set_rigid_object_state(
    asset: RigidObject,
    env_ids: torch.Tensor,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor | None = None,
    lin_vel_w: torch.Tensor | None = None,
    ang_vel_w: torch.Tensor | None = None,
):
    if len(env_ids) == 0:
        return
    if quat_w is None:
        quat_w = torch.zeros(len(env_ids), 4, device=pos_w.device)
        quat_w[:, 0] = 1.0
    if lin_vel_w is None:
        lin_vel_w = torch.zeros(len(env_ids), 3, device=pos_w.device)
    if ang_vel_w is None:
        ang_vel_w = torch.zeros(len(env_ids), 3, device=pos_w.device)

    root_pose = torch.cat((pos_w, quat_w), dim=-1)
    root_vel = torch.cat((lin_vel_w, ang_vel_w), dim=-1)
    root_state = torch.cat((root_pose, root_vel), dim=-1)

    if hasattr(asset, "write_root_state_to_sim"):
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    else:
        _call_first_available(asset, ("write_root_pose_to_sim",), root_pose, env_ids=env_ids)
        _call_first_available(asset, ("write_root_velocity_to_sim",), root_vel, env_ids=env_ids)


def _sample_uniform_range(
    low: float,
    high: float,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(count, device=device)


def _sample_table_layout(env, env_ids: torch.Tensor):
    device = env.device
    count = len(env_ids)
    ball_pos_w = torch.zeros(count, 3, device=device)
    goal_pos_w = torch.zeros(count, 3, device=device)

    ball_pos_w[:, 0] = _sample_uniform_range(0.78, 1.02, count, device)
    ball_pos_w[:, 1] = _sample_uniform_range(0.06, 0.22, count, device)
    ball_pos_w[:, 2] = BALL_CENTER_Z

    goal_pos_w[:, 0] = _sample_uniform_range(0.84, 1.10, count, device)
    goal_pos_w[:, 1] = _sample_uniform_range(0.02, 0.24, count, device)
    goal_pos_w[:, 2] = BALL_CENTER_Z

    close_mask = torch.linalg.norm(goal_pos_w[:, :2] - ball_pos_w[:, :2], dim=-1) < 0.12
    if torch.any(close_mask):
        goal_pos_w[close_mask, 1] = torch.clamp(goal_pos_w[close_mask, 1] + 0.14, min=0.02, max=0.24)

    return ball_pos_w, goal_pos_w


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


def _ensure_pick_place_state(env):
    if hasattr(env, "_pp_target_w"):
        return

    robot = _robot(env)
    num_envs = env.num_envs
    device = env.device

    env._pp_mode = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._pp_attached = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._pp_hold_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._pp_prev_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._pp_recent_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._pp_timed_out = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._pp_target_age_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._pp_target_w = torch.zeros(num_envs, 3, device=device)
    env._pp_ball_pos_w = torch.zeros(num_envs, 3, device=device)
    env._pp_goal_w = torch.zeros(num_envs, 3, device=device)
    env._pp_prev_workspace_error = torch.zeros(num_envs, device=device)
    env._pp_workspace_error = torch.zeros(num_envs, device=device)
    env._pp_workspace_progress = torch.zeros(num_envs, device=device)
    env._pp_hand_target_error = torch.zeros(num_envs, device=device)
    env._pp_ball_goal_error = torch.zeros(num_envs, device=device)
    env._pp_hand_ball_error = torch.zeros(num_envs, device=device)
    env._pp_base_speed = torch.zeros(num_envs, device=device)
    env._pp_hand_speed = torch.zeros(num_envs, device=device)
    env._pp_attach_offset = torch.zeros(num_envs, 3, device=device)
    env._pp_state_synced_step = -1
    env._pp_benchmark_override = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._pp_override_ball_w = torch.zeros(num_envs, 3, device=device)
    env._pp_override_goal_w = torch.zeros(num_envs, 3, device=device)
    env._pp_hand_body_id = int(robot.find_bodies([LEFT_HAND_BODY_NAME], preserve_order=True)[0][0])
    env._pp_arm_joint_ids = torch.tensor(
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
    env._pp_foot_body_ids = torch.tensor(
        robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0],
        dtype=torch.long,
        device=device,
    )


def _update_attached_ball_pose(env, env_ids: torch.Tensor):
    if len(env_ids) == 0:
        return

    hand_pos_w = _hand_pos_w(env)[env_ids]
    ball_pos_w = hand_pos_w + env._pp_attach_offset[env_ids]
    env._pp_ball_pos_w[env_ids] = ball_pos_w
    _set_rigid_object_state(_ball(env), env_ids, ball_pos_w)


def _active_target_pos_base_yaw(env) -> torch.Tensor:
    robot = _robot(env)
    target_delta_w = env._pp_target_w - robot.data.root_pos_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), target_delta_w)


def _sync_pick_place_state(
    env,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _ensure_pick_place_state(env)
    if env._pp_state_synced_step == env.common_step_counter:
        return

    reset_ids, current_episode_length = _compute_just_reset_mask(env)
    env._pp_recent_success.zero_()

    if torch.any(reset_ids):
        reset_env_ids = torch.where(reset_ids)[0]
        override_mask = env._pp_benchmark_override[reset_env_ids]
        sample_env_ids = reset_env_ids[~override_mask]
        override_env_ids = reset_env_ids[override_mask]

        if len(sample_env_ids) > 0:
            place_mask = torch.rand(len(sample_env_ids), device=env.device) < float(place_episode_ratio)
            sampled_modes = torch.where(
                place_mask,
                torch.full_like(place_mask, PLACE_MODE, dtype=torch.long),
                torch.full_like(place_mask, ACQUIRE_MODE, dtype=torch.long),
            )
            env._pp_mode[sample_env_ids] = sampled_modes
            env._pp_attached[sample_env_ids] = sampled_modes == PLACE_MODE
            ball_pos_w, goal_pos_w = _sample_table_layout(env, sample_env_ids)
            env._pp_ball_pos_w[sample_env_ids] = ball_pos_w
            env._pp_goal_w[sample_env_ids] = goal_pos_w

        if len(override_env_ids) > 0:
            env._pp_ball_pos_w[override_env_ids] = env._pp_override_ball_w[override_env_ids]
            env._pp_goal_w[override_env_ids] = env._pp_override_goal_w[override_env_ids]

        env._pp_hold_counter[reset_ids] = 0
        env._pp_prev_success[reset_ids] = False
        env._pp_timed_out[reset_ids] = False
        env._pp_target_age_steps[reset_ids] = 0
        env._pp_attach_offset[reset_ids] = 0.0

        _set_rigid_object_state(_place_target(env), reset_env_ids, env._pp_goal_w[reset_env_ids])
        free_ball_ids = reset_env_ids[~env._pp_attached[reset_env_ids]]
        if len(free_ball_ids) > 0:
            _set_rigid_object_state(_ball(env), free_ball_ids, env._pp_ball_pos_w[free_ball_ids])
        attached_ids = reset_env_ids[env._pp_attached[reset_env_ids]]
        if len(attached_ids) > 0:
            _update_attached_ball_pose(env, attached_ids)

    if torch.any(env._pp_attached):
        _update_attached_ball_pose(env, torch.where(env._pp_attached)[0])

    hand_pos_w = _hand_pos_w(env)
    hand_vel_w = _hand_vel_w(env)
    robot = _robot(env)
    base_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    hand_speed = torch.linalg.norm(hand_vel_w, dim=-1)

    acquire_pre_target = env._pp_ball_pos_w.clone()
    acquire_pre_target[:, 2] += pregrasp_height
    acquire_final_target = env._pp_ball_pos_w.clone()
    acquire_final_target[:, 2] += 0.02

    place_pre_target = env._pp_goal_w.clone()
    place_pre_target[:, 2] += preplace_height
    place_final_target = env._pp_goal_w.clone()

    mode_is_place = env._pp_mode == PLACE_MODE
    pre_target = torch.where(mode_is_place.unsqueeze(-1), place_pre_target, acquire_pre_target)
    final_target = torch.where(mode_is_place.unsqueeze(-1), place_final_target, acquire_final_target)
    pre_target_error = torch.linalg.norm(pre_target - hand_pos_w, dim=-1)
    use_final_target = pre_target_error <= pre_target_switch_radius
    env._pp_target_w[:] = torch.where(use_final_target.unsqueeze(-1), final_target, pre_target)

    target_pos_base = _active_target_pos_base_yaw(env)
    workspace_error = _workspace_error_l2(target_pos_base, x_range=x_range, y_range=y_range)
    env._pp_workspace_progress = env._pp_prev_workspace_error - workspace_error
    env._pp_workspace_error[:] = workspace_error

    hand_target_error = torch.linalg.norm(env._pp_target_w - hand_pos_w, dim=-1)
    hand_ball_error = torch.linalg.norm(env._pp_ball_pos_w - hand_pos_w, dim=-1)
    ball_goal_error = torch.linalg.norm(env._pp_ball_pos_w - env._pp_goal_w, dim=-1)

    env._pp_hand_target_error[:] = hand_target_error
    env._pp_hand_ball_error[:] = hand_ball_error
    env._pp_ball_goal_error[:] = ball_goal_error
    env._pp_base_speed[:] = base_speed
    env._pp_hand_speed[:] = hand_speed

    success_error = torch.where(mode_is_place, ball_goal_error, hand_ball_error)
    success_zone = (
        (success_error <= success_threshold)
        & (base_speed <= base_speed_threshold)
        & (hand_speed <= hand_speed_threshold)
    )
    env._pp_hold_counter = torch.where(
        success_zone,
        env._pp_hold_counter + 1,
        torch.zeros_like(env._pp_hold_counter),
    )
    hold_qualified = env._pp_hold_counter >= max(1, int(success_hold_steps))
    env._pp_recent_success[:] = hold_qualified & ~env._pp_prev_success & ~reset_ids
    env._pp_prev_success[:] = hold_qualified

    env._pp_target_age_steps += 1
    env._pp_target_age_steps[reset_ids] = 0
    per_target_timeout_steps = max(1, int(round(float(per_target_timeout_s) / env.step_dt)))
    env._pp_timed_out[:] = env._pp_target_age_steps >= per_target_timeout_steps

    env._pp_prev_workspace_error[:] = workspace_error
    env._pp_prev_episode_length_buf = current_episode_length
    env._pp_state_synced_step = env.common_step_counter


def refresh_pick_place_state(env):
    _sync_pick_place_state(env)


def set_pick_place_benchmark_state(
    env,
    ball_pos_w: torch.Tensor,
    goal_pos_w: torch.Tensor,
    mode: int | str = ACQUIRE_MODE,
    attach: bool | None = None,
    env_ids: torch.Tensor | None = None,
):
    _ensure_pick_place_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    if isinstance(mode, str):
        mode = ACQUIRE_MODE if mode.lower() == "acquire" else PLACE_MODE
    mode_value = int(mode)
    attach_value = bool(attach) if attach is not None else mode_value == PLACE_MODE

    if ball_pos_w.ndim == 1:
        ball_pos_w = ball_pos_w.unsqueeze(0).repeat(len(env_ids), 1)
    if goal_pos_w.ndim == 1:
        goal_pos_w = goal_pos_w.unsqueeze(0).repeat(len(env_ids), 1)

    env._pp_benchmark_override[env_ids] = True
    env._pp_override_ball_w[env_ids] = ball_pos_w
    env._pp_override_goal_w[env_ids] = goal_pos_w
    env._pp_ball_pos_w[env_ids] = ball_pos_w
    env._pp_goal_w[env_ids] = goal_pos_w
    env._pp_mode[env_ids] = mode_value
    env._pp_attached[env_ids] = attach_value
    env._pp_hold_counter[env_ids] = 0
    env._pp_prev_success[env_ids] = False
    env._pp_recent_success[env_ids] = False
    env._pp_target_age_steps[env_ids] = 0
    env._pp_timed_out[env_ids] = False
    _set_rigid_object_state(_place_target(env), env_ids, goal_pos_w)
    if attach_value:
        _update_attached_ball_pose(env, env_ids)
    else:
        _set_rigid_object_state(_ball(env), env_ids, ball_pos_w)
    env._pp_state_synced_step = -1


def clear_pick_place_benchmark_override(env, env_ids: torch.Tensor | None = None):
    _ensure_pick_place_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    env._pp_benchmark_override[env_ids] = False
    env._pp_state_synced_step = -1


def attach_ball_to_hand(env, env_ids: torch.Tensor | None = None):
    _ensure_pick_place_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    hand_pos_w = _hand_pos_w(env)[env_ids]
    env._pp_attach_offset[env_ids] = env._pp_ball_pos_w[env_ids] - hand_pos_w
    env._pp_attached[env_ids] = True
    _update_attached_ball_pose(env, env_ids)
    env._pp_state_synced_step = -1


def release_ball(env, env_ids: torch.Tensor | None = None):
    _ensure_pick_place_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    env._pp_attached[env_ids] = False
    current_ball_pos = env._pp_ball_pos_w[env_ids]
    _set_rigid_object_state(_ball(env), env_ids, current_ball_pos)
    env._pp_state_synced_step = -1


def pick_place_target_world(env) -> torch.Tensor:
    _sync_pick_place_state(env)
    return env._pp_target_w


def pick_place_mode(env) -> torch.Tensor:
    _sync_pick_place_state(env)
    return env._pp_mode


def task_success_mask(env) -> torch.Tensor:
    _sync_pick_place_state(env)
    return env._pp_recent_success


def _sync_pick_place_term_state(
    env,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_pick_place_state(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )


def target_pos_command_obs(
    env,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_pick_place_term_state(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return _active_target_pos_base_yaw(env)


def target_relative_base_stance_l2(
    env,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    target_pos_base = target_pos_command_obs(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return _workspace_error_l2(target_pos_base, x_range=x_range, y_range=y_range)


def target_relative_base_stance_ready(
    env,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
    gate_std: float = 0.01,
):
    target_pos_base = target_pos_command_obs(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return _workspace_ready_gate(target_pos_base, x_range=x_range, y_range=y_range, gate_std=gate_std)


def target_relative_base_stance_progress(
    env,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_pick_place_term_state(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return env._pp_workspace_progress


def static_target_position_error(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    del asset_cfg
    _sync_pick_place_term_state(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return env._pp_hand_target_error


def gated_position_command_error_tanh(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
    std: float = 0.14,
    gate_std: float = 0.01,
):
    del asset_cfg
    _sync_pick_place_term_state(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    target_pos_base = _active_target_pos_base_yaw(env)
    gate = _workspace_ready_gate(target_pos_base, x_range=x_range, y_range=y_range, gate_std=gate_std)
    return (1.0 - torch.tanh(env._pp_hand_target_error / std)) * gate


def target_hold_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
    hold_reward_std: float = 0.02,
):
    del asset_cfg
    _sync_pick_place_term_state(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    hold_gate = (env._pp_hold_counter > 0).float()
    return hold_gate * torch.exp(-env._pp_hand_target_error / max(hold_reward_std, 1.0e-6))


def near_target_left_hand_stillness_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
    near_target_radius: float = 0.12,
    hand_speed_scale: float = 0.08,
):
    del asset_cfg
    _sync_pick_place_term_state(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    near_target = (env._pp_hand_target_error <= near_target_radius).float()
    return near_target * torch.exp(-env._pp_hand_speed / max(hand_speed_scale, 1.0e-6))


def target_completion_bonus(
    env,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_pick_place_term_state(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return env._pp_recent_success.float()


def success_posture_bonus(
    env,
    arm_joint_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[LEFT_HAND_BODY_NAME]),
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    del asset_cfg
    _sync_pick_place_term_state(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
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
    return env._pp_recent_success.float() * torch.exp(-0.5 * deviation)


def pre_stance_torso_lean_penalty(
    env,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
    gate_std: float = 0.02,
):
    target_pos_base = target_pos_command_obs(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    gate = _workspace_ready_gate(target_pos_base, x_range=x_range, y_range=y_range, gate_std=gate_std)
    torso_lean = torch.linalg.norm(_robot(env).data.projected_gravity_b[:, :2], dim=-1)
    return (1.0 - gate) * torso_lean


def pre_stance_joint_deviation_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
    gate_std: float = 0.02,
):
    target_pos_base = target_pos_command_obs(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    gate = _workspace_ready_gate(target_pos_base, x_range=x_range, y_range=y_range, gate_std=gate_std)
    robot = env.scene[asset_cfg.name]
    deviation = torch.sum(
        torch.abs(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return (1.0 - gate) * deviation


def pre_stance_joint_limit_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    margin_threshold: float = 0.18,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
    gate_std: float = 0.02,
):
    target_pos_base = target_pos_command_obs(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    gate = _workspace_ready_gate(target_pos_base, x_range=x_range, y_range=y_range, gate_std=gate_std)
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
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
    gate_std: float = 0.02,
    speed_scale: float = 0.35,
):
    target_pos_base = target_pos_command_obs(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    gate = _workspace_ready_gate(target_pos_base, x_range=x_range, y_range=y_range, gate_std=gate_std)
    robot = env.scene[asset_cfg.name]
    foot_vel_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2] - robot.data.root_lin_vel_w[:, None, :2]
    foot_speed = torch.linalg.norm(foot_vel_xy, dim=-1).mean(dim=1)
    return (1.0 - gate) * torch.tanh(foot_speed / max(speed_scale, 1.0e-6))


def target_success_reached(
    env,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_pick_place_term_state(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return env._pp_recent_success


def target_timeout_reached(
    env,
    place_episode_ratio: float = 0.5,
    success_threshold: float = 0.06,
    success_hold_steps: int = 10,
    per_target_timeout_s: float = 6.0,
    pregrasp_height: float = 0.08,
    preplace_height: float = 0.08,
    base_speed_threshold: float = 0.10,
    hand_speed_threshold: float = 0.14,
    pre_target_switch_radius: float = 0.12,
    x_range: tuple[float, float] = (0.36, 0.58),
    y_range: tuple[float, float] = (0.10, 0.26),
):
    _sync_pick_place_term_state(
        env,
        place_episode_ratio=place_episode_ratio,
        success_threshold=success_threshold,
        success_hold_steps=success_hold_steps,
        per_target_timeout_s=per_target_timeout_s,
        pregrasp_height=pregrasp_height,
        preplace_height=preplace_height,
        base_speed_threshold=base_speed_threshold,
        hand_speed_threshold=hand_speed_threshold,
        pre_target_switch_radius=pre_target_switch_radius,
        x_range=x_range,
        y_range=y_range,
    )
    return env._pp_timed_out
