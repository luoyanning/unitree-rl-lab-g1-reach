from __future__ import annotations

import os
from collections.abc import Sequence

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from ..left_hand_loco_reach import left_hand_loco_reach_mdp as fixed_mdp


LEFT_HAND_BODY_NAME = "left_wrist_yaw_link"

PHASE_BALANCE = 0
PHASE_PRETOUCH = 1
PHASE_TOUCH = 2
PHASE_RECOVER = 3
NUM_PHASES = 4
ENABLE_TABLETOP_TARGET_VIS = os.getenv("UTRL_TABLETOP_CLEAN_TARGET_VIS", "0") == "1"
ACTIVE_TARGET_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/tabletop_clean_active_target")
ACTIVE_TARGET_MARKER_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
TOUCH_TARGET_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/tabletop_clean_touch_target")
TOUCH_TARGET_MARKER_CFG.markers["frame"].scale = (0.07, 0.07, 0.07)


def _robot(env):
    return env.scene["robot"]


def _command_term(env):
    return env.command_manager.get_term("left_hand_pose")


def _episode_length_buf(env) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        return env.episode_length_buf.clone()
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _compute_just_reset_mask(env):
    current_episode_length = _episode_length_buf(env)
    if not hasattr(env, "_ttc_prev_episode_length_buf"):
        env._ttc_prev_episode_length_buf = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    prev_episode_length = env._ttc_prev_episode_length_buf
    just_reset = prev_episode_length < 0
    just_reset |= current_episode_length == 0
    just_reset |= current_episode_length < prev_episode_length
    return just_reset, current_episode_length


def _hand_pos_w(env) -> torch.Tensor:
    robot = _robot(env)
    return robot.data.body_pos_w[:, env._ttc_hand_body_id]


def _hand_vel_w(env) -> torch.Tensor:
    robot = _robot(env)
    return robot.data.body_lin_vel_w[:, env._ttc_hand_body_id]


def _target_pos_base_yaw(env, target_w: torch.Tensor) -> torch.Tensor:
    robot = _robot(env)
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), target_w - robot.data.root_pos_w)


def _object_pos_base_yaw(env) -> torch.Tensor:
    return _target_pos_base_yaw(env, env._ttc_object_w)


def _active_target_pos_base_yaw(env) -> torch.Tensor:
    return _target_pos_base_yaw(env, env._ttc_active_target_w)


def _ready_pose_w(env, ready_local_pos: tuple[float, float, float]) -> torch.Tensor:
    robot = _robot(env)
    ready_local = torch.tensor(ready_local_pos, dtype=torch.float32, device=env.device).unsqueeze(0)
    ready_local = ready_local.expand(env.num_envs, -1)
    return robot.data.root_pos_w + quat_apply(yaw_quat(robot.data.root_quat_w), ready_local)


def _start_phase(mode: str) -> int:
    return PHASE_BALANCE if mode == "balance" else PHASE_PRETOUCH


def _touch_requires_recover(mode: str) -> bool:
    return mode.startswith("touch_recover")


def _recover_target_uses_anchor(mode: str) -> bool:
    return mode == "touch_recover_anchor"


def _marker_quat(env) -> torch.Tensor:
    if not hasattr(env, "_ttc_marker_quat"):
        env._ttc_marker_quat = torch.zeros(env.num_envs, 4, device=env.device)
        env._ttc_marker_quat[:, 0] = 1.0
    return env._ttc_marker_quat


def _update_target_debug_visualization(env) -> None:
    if not ENABLE_TABLETOP_TARGET_VIS:
        return
    marker_quat = _marker_quat(env)
    if not hasattr(env, "_ttc_active_target_visualizer"):
        env._ttc_active_target_visualizer = VisualizationMarkers(ACTIVE_TARGET_MARKER_CFG)
    env._ttc_active_target_visualizer.visualize(env._ttc_active_target_w, marker_quat)
    if not hasattr(env, "_ttc_touch_target_visualizer"):
        env._ttc_touch_target_visualizer = VisualizationMarkers(TOUCH_TARGET_MARKER_CFG)
    env._ttc_touch_target_visualizer.visualize(env._ttc_touch_w, marker_quat)


def _anchor_ready_pose_w(env, ready_local_pos: tuple[float, float, float]) -> torch.Tensor:
    anchor_root_pos_w = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    anchor_root_pos_w[:, :2] = env.scene.env_origins[:, :2] + env._ttc_stance_anchor_xy
    anchor_root_pos_w[:, 2] = env._ttc_anchor_root_height
    ready_local = torch.tensor(ready_local_pos, dtype=torch.float32, device=env.device).unsqueeze(0)
    ready_local = ready_local.expand(env.num_envs, -1)
    return anchor_root_pos_w + quat_apply(env._ttc_anchor_yaw_quat, ready_local)


def _ensure_state(env, num_targets: int):
    if (
        hasattr(env, "_ttc_phase")
        and hasattr(env, "_ttc_task_complete")
        and getattr(env, "_ttc_num_targets", None) == num_targets
    ):
        return

    robot = _robot(env)
    device = env.device
    num_envs = env.num_envs

    env._ttc_num_targets = num_targets
    env._ttc_target_order = torch.zeros((num_envs, max(1, num_targets)), dtype=torch.long, device=device)
    env._ttc_target_order_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._ttc_target_slot = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._ttc_selected_target_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._ttc_phase = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._ttc_prev_phase = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._ttc_phase_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._ttc_phase_hold_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._ttc_target_age_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._ttc_completed_targets = torch.zeros(num_envs, dtype=torch.long, device=device)
    env._ttc_task_complete = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._ttc_anchor_yaw_quat = torch.zeros(num_envs, 4, device=device)
    env._ttc_anchor_yaw_quat[:, 0] = 1.0
    env._ttc_anchor_root_height = torch.zeros(num_envs, device=device)
    env._ttc_object_w = torch.zeros(num_envs, 3, device=device)
    env._ttc_pretouch_w = torch.zeros(num_envs, 3, device=device)
    env._ttc_touch_w = torch.zeros(num_envs, 3, device=device)
    env._ttc_recover_w = torch.zeros(num_envs, 3, device=device)
    env._ttc_active_target_w = torch.zeros(num_envs, 3, device=device)
    env._ttc_hand_target_error = torch.zeros(num_envs, device=device)
    env._ttc_hand_object_error = torch.zeros(num_envs, device=device)
    env._ttc_prev_hand_target_error = torch.zeros(num_envs, device=device)
    env._ttc_hand_progress = torch.zeros(num_envs, device=device)
    env._ttc_stance_anchor_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._ttc_stance_anchor_xy = torch.zeros(num_envs, 2, device=device)
    env._ttc_rest_hand_w = torch.zeros(num_envs, 3, device=device)
    env._ttc_stance_anchor_error = torch.zeros(num_envs, device=device)
    env._ttc_base_speed = torch.zeros(num_envs, device=device)
    env._ttc_hand_speed = torch.zeros(num_envs, device=device)
    env._ttc_torso_lean = torch.zeros(num_envs, device=device)
    env._ttc_stability_gate = torch.zeros(num_envs, device=device)
    env._ttc_stable = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._ttc_backward_drift = torch.zeros(num_envs, device=device)
    env._ttc_support_force = torch.zeros(num_envs, device=device)
    env._ttc_support_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._ttc_recent_pretouch = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._ttc_recent_touch = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._ttc_recent_recover = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._ttc_recent_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._ttc_timed_out = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._ttc_state_synced_step = -1

    env._ttc_hand_body_id = int(robot.find_bodies([LEFT_HAND_BODY_NAME], preserve_order=True)[0][0])
    env._ttc_left_arm_joint_ids = torch.tensor(
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


def _refresh_target_order(env, env_ids: torch.Tensor, num_targets: int, randomize_order: bool):
    if len(env_ids) == 0:
        return
    if randomize_order and num_targets > 1:
        priority = torch.rand((len(env_ids), num_targets), device=env.device)
        env._ttc_target_order[env_ids] = torch.argsort(priority, dim=-1)
    else:
        base_order = torch.arange(num_targets, device=env.device, dtype=torch.long)
        env._ttc_target_order[env_ids] = base_order.unsqueeze(0).repeat(len(env_ids), 1)
    env._ttc_target_order_valid[env_ids] = True


def _select_current_target(env, scene_target_names: Sequence[str]):
    target_positions_w = fixed_mdp._scene_target_positions_w(env, scene_target_names)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env._ttc_object_w[:] = target_positions_w[env_ids, env._ttc_selected_target_idx]


def _support_force(
    env,
    support_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[support_sensor_cfg.name]
    forces = torch.linalg.norm(contact_sensor.data.net_forces_w[:, support_sensor_cfg.body_ids], dim=-1)
    return torch.amax(forces, dim=1)


def _phase_target_radius(
    phase: torch.Tensor,
    balance_radius: float,
    pretouch_radius: float,
    touch_radius: float,
    recover_radius: float,
) -> torch.Tensor:
    radius = torch.full_like(phase, recover_radius, dtype=torch.float32)
    radius[phase == PHASE_BALANCE] = balance_radius
    radius[phase == PHASE_PRETOUCH] = pretouch_radius
    radius[phase == PHASE_TOUCH] = touch_radius
    radius[phase == PHASE_RECOVER] = recover_radius
    return radius


def _phase_hold_steps(
    phase: torch.Tensor,
    balance_hold_steps: int,
    pretouch_hold_steps: int,
    touch_hold_steps: int,
    recover_hold_steps: int,
) -> torch.Tensor:
    hold_steps = torch.full_like(phase, max(1, recover_hold_steps))
    hold_steps[phase == PHASE_BALANCE] = max(1, balance_hold_steps)
    hold_steps[phase == PHASE_PRETOUCH] = max(1, pretouch_hold_steps)
    hold_steps[phase == PHASE_TOUCH] = max(1, touch_hold_steps)
    hold_steps[phase == PHASE_RECOVER] = max(1, recover_hold_steps)
    return hold_steps


def _phase_gate(
    env,
    phase: torch.Tensor,
    pretouch_stability_gate: float,
    touch_stability_gate: float,
    recover_stability_gate: float,
) -> torch.Tensor:
    gate = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    gate[phase == PHASE_BALANCE] = env._ttc_stable[phase == PHASE_BALANCE]
    gate[phase == PHASE_PRETOUCH] = env._ttc_stability_gate[phase == PHASE_PRETOUCH] >= pretouch_stability_gate
    gate[phase == PHASE_TOUCH] = env._ttc_stability_gate[phase == PHASE_TOUCH] >= touch_stability_gate
    gate[phase == PHASE_RECOVER] = env._ttc_stability_gate[phase == PHASE_RECOVER] >= recover_stability_gate
    return gate


def _advance_to_next_target(
    env,
    success_env_ids: torch.Tensor,
    mode: str,
    max_targets_per_episode: int,
):
    if len(success_env_ids) == 0:
        return

    env._ttc_completed_targets[success_env_ids] += 1
    task_complete = env._ttc_completed_targets[success_env_ids] >= max_targets_per_episode
    completed_env_ids = success_env_ids[task_complete]
    if len(completed_env_ids) > 0:
        env._ttc_task_complete[completed_env_ids] = True

    active_env_ids = success_env_ids[~task_complete]
    if len(active_env_ids) == 0:
        return

    env._ttc_target_slot[active_env_ids] += 1
    env._ttc_target_age_steps[active_env_ids] = 0
    env._ttc_phase_hold_counter[active_env_ids] = 0
    env._ttc_phase_steps[active_env_ids] = 0
    env._ttc_phase[active_env_ids] = _start_phase(mode)
    env._ttc_prev_phase[active_env_ids] = env._ttc_phase[active_env_ids]

    next_slot = env._ttc_target_slot[active_env_ids]
    env._ttc_selected_target_idx[active_env_ids] = env._ttc_target_order[active_env_ids, next_slot]


def _sync_tabletop_clean_state(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    complete_on_final_touch: bool = False,
):
    if len(scene_target_names) == 0:
        raise RuntimeError("TableTop clean tasks require at least one scene target.")

    _ensure_state(env, num_targets=len(scene_target_names))
    if env._ttc_state_synced_step == env.common_step_counter:
        return

    max_targets_per_episode = max(1, min(max_targets_per_episode, len(scene_target_names)))

    reset_ids, current_episode_length = _compute_just_reset_mask(env)
    env._ttc_recent_pretouch.zero_()
    env._ttc_recent_touch.zero_()
    env._ttc_recent_recover.zero_()
    env._ttc_recent_success.zero_()

    if torch.any(reset_ids):
        reset_env_ids = torch.where(reset_ids)[0]
        _refresh_target_order(env, reset_env_ids, num_targets=len(scene_target_names), randomize_order=randomize_order)
        env._ttc_target_slot[reset_env_ids] = 0
        env._ttc_selected_target_idx[reset_env_ids] = env._ttc_target_order[reset_env_ids, 0]
        env._ttc_phase[reset_env_ids] = _start_phase(mode)
        env._ttc_prev_phase[reset_env_ids] = env._ttc_phase[reset_env_ids]
        env._ttc_phase_steps[reset_env_ids] = 0
        env._ttc_phase_hold_counter[reset_env_ids] = 0
        env._ttc_target_age_steps[reset_env_ids] = 0
        env._ttc_completed_targets[reset_env_ids] = 0
        env._ttc_task_complete[reset_env_ids] = False
        env._ttc_prev_hand_target_error[reset_env_ids] = 0.0

    _select_current_target(env, scene_target_names)

    robot = _robot(env)
    root_pos_local_xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    hand_pos_w = _hand_pos_w(env)
    hand_vel_w = _hand_vel_w(env)
    if torch.any(reset_ids):
        reset_env_ids = torch.where(reset_ids)[0]
        env._ttc_stance_anchor_xy[reset_env_ids] = root_pos_local_xy[reset_env_ids]
        env._ttc_stance_anchor_valid[reset_env_ids] = True
        env._ttc_anchor_yaw_quat[reset_env_ids] = yaw_quat(robot.data.root_quat_w[reset_env_ids])
        env._ttc_anchor_root_height[reset_env_ids] = robot.data.root_pos_w[reset_env_ids, 2]
        env._ttc_rest_hand_w[reset_env_ids] = hand_pos_w[reset_env_ids]
    default_anchor_xy = torch.tensor(stance_anchor_xy, dtype=torch.float32, device=env.device).unsqueeze(0)
    anchor_xy = torch.where(
        env._ttc_stance_anchor_valid.unsqueeze(-1),
        env._ttc_stance_anchor_xy,
        default_anchor_xy.expand(env.num_envs, -1),
    )
    base_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    hand_speed = torch.linalg.norm(hand_vel_w, dim=-1)
    torso_lean = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    stance_anchor_error = torch.linalg.norm(root_pos_local_xy - anchor_xy, dim=-1)
    stability_gate = torch.exp(
        -(stance_anchor_error / max(stance_anchor_std, 1.0e-6))
        - (base_speed / max(stability_speed_scale, 1.0e-6))
        - (torso_lean / max(stability_lean_scale, 1.0e-6))
    )
    stable = (
        (stance_anchor_error <= stance_anchor_tolerance)
        & (base_speed <= base_speed_threshold)
        & (torso_lean <= torso_lean_threshold)
    )
    backward_drift = torch.clamp(anchor_xy[:, 0] - root_pos_local_xy[:, 0], min=0.0)
    support_force = _support_force(env, support_sensor_cfg=support_sensor_cfg)
    support_contact = support_force > support_force_threshold

    if _recover_target_uses_anchor(mode):
        ready_target_w = _anchor_ready_pose_w(env, ready_local_pos=ready_local_pos)
    else:
        ready_target_w = _ready_pose_w(env, ready_local_pos=ready_local_pos)
    env._ttc_pretouch_w[:] = env._ttc_object_w
    env._ttc_pretouch_w[:, 0] -= pretouch_backoff_x
    env._ttc_pretouch_w[:, 2] += pretouch_height
    env._ttc_touch_w[:] = env._ttc_object_w
    env._ttc_touch_w[:, 2] += touch_height_offset
    env._ttc_recover_w[:] = ready_target_w

    phase = env._ttc_phase
    active_target_w = ready_target_w.clone()
    active_target_w[phase == PHASE_PRETOUCH] = env._ttc_pretouch_w[phase == PHASE_PRETOUCH]
    active_target_w[phase == PHASE_TOUCH] = env._ttc_touch_w[phase == PHASE_TOUCH]
    active_target_w[phase == PHASE_RECOVER] = env._ttc_recover_w[phase == PHASE_RECOVER]
    env._ttc_active_target_w[:] = active_target_w

    hand_target_error = torch.linalg.norm(active_target_w - hand_pos_w, dim=-1)
    hand_object_error = torch.linalg.norm(env._ttc_touch_w - hand_pos_w, dim=-1)
    hand_progress = torch.clamp(env._ttc_prev_hand_target_error - hand_target_error, min=-0.10, max=0.10)
    phase_switched = phase != env._ttc_prev_phase
    hand_progress[reset_ids | phase_switched] = 0.0
    env._ttc_prev_hand_target_error[:] = hand_target_error
    env._ttc_prev_phase[:] = phase

    phase_radius = _phase_target_radius(
        phase=phase,
        balance_radius=balance_radius,
        pretouch_radius=pretouch_radius,
        touch_radius=touch_radius,
        recover_radius=recover_radius,
    )
    phase_hold_steps = _phase_hold_steps(
        phase=phase,
        balance_hold_steps=balance_hold_steps,
        pretouch_hold_steps=pretouch_hold_steps,
        touch_hold_steps=touch_hold_steps,
        recover_hold_steps=recover_hold_steps,
    )
    phase_gate = _phase_gate(
        env,
        phase=phase,
        pretouch_stability_gate=pretouch_stability_gate,
        touch_stability_gate=touch_stability_gate,
        recover_stability_gate=recover_stability_gate,
    )
    phase_success_zone = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    balance_mask = phase == PHASE_BALANCE
    phase_success_zone[balance_mask] = phase_gate[balance_mask] & stable[balance_mask]
    active_reach_mask = ~balance_mask
    phase_success_zone[active_reach_mask] = (
        (hand_target_error[active_reach_mask] <= phase_radius[active_reach_mask])
        & (hand_speed[active_reach_mask] <= hand_speed_threshold)
        & stable[active_reach_mask]
        & phase_gate[active_reach_mask]
        & ~support_contact[active_reach_mask]
    )
    active_env_mask = ~env._ttc_task_complete
    phase_success_zone &= active_env_mask

    env._ttc_phase_hold_counter = torch.where(
        active_env_mask,
        torch.where(
            phase_success_zone,
            env._ttc_phase_hold_counter + 1,
            torch.zeros_like(env._ttc_phase_hold_counter),
        ),
        env._ttc_phase_hold_counter,
    )

    phase_done = env._ttc_phase_hold_counter >= phase_hold_steps
    phase_done &= ~reset_ids
    phase_done &= active_env_mask

    if torch.any(phase_done & (phase == PHASE_PRETOUCH)):
        done_ids = torch.where(phase_done & (phase == PHASE_PRETOUCH))[0]
        env._ttc_recent_pretouch[done_ids] = True
        if mode == "pretouch":
            env._ttc_recent_success[done_ids] = True
            _advance_to_next_target(env, done_ids, mode=mode, max_targets_per_episode=max_targets_per_episode)
        else:
            env._ttc_phase[done_ids] = PHASE_TOUCH
            env._ttc_phase_hold_counter[done_ids] = 0
            env._ttc_phase_steps[done_ids] = 0

    if torch.any(phase_done & (phase == PHASE_TOUCH)):
        done_ids = torch.where(phase_done & (phase == PHASE_TOUCH))[0]
        env._ttc_recent_touch[done_ids] = True
        if _touch_requires_recover(mode):
            recover_ids = done_ids
            if complete_on_final_touch:
                final_touch_mask = (env._ttc_completed_targets[done_ids] + 1) >= max_targets_per_episode
                final_touch_ids = done_ids[final_touch_mask]
                recover_ids = done_ids[~final_touch_mask]
                if len(final_touch_ids) > 0:
                    env._ttc_recent_success[final_touch_ids] = True
                    _advance_to_next_target(
                        env,
                        final_touch_ids,
                        mode=mode,
                        max_targets_per_episode=max_targets_per_episode,
                    )
            if len(recover_ids) > 0:
                env._ttc_phase[recover_ids] = PHASE_RECOVER
                env._ttc_phase_hold_counter[recover_ids] = 0
                env._ttc_phase_steps[recover_ids] = 0
        else:
            env._ttc_recent_success[done_ids] = True
            _advance_to_next_target(env, done_ids, mode=mode, max_targets_per_episode=max_targets_per_episode)

    if torch.any(phase_done & (phase == PHASE_RECOVER)):
        done_ids = torch.where(phase_done & (phase == PHASE_RECOVER))[0]
        env._ttc_recent_recover[done_ids] = True
        env._ttc_recent_success[done_ids] = True
        _advance_to_next_target(env, done_ids, mode=mode, max_targets_per_episode=max_targets_per_episode)

    if torch.any(phase_done & (phase == PHASE_BALANCE)):
        done_ids = torch.where(phase_done & (phase == PHASE_BALANCE))[0]
        env._ttc_recent_success[done_ids] = True
        _advance_to_next_target(env, done_ids, mode=mode, max_targets_per_episode=max_targets_per_episode)

    _select_current_target(env, scene_target_names)

    env._ttc_target_age_steps[active_env_mask] += 1
    env._ttc_target_age_steps[reset_ids] = 0
    env._ttc_phase_steps[active_env_mask] += 1
    env._ttc_phase_steps[reset_ids | phase_switched] = 0

    per_target_timeout_steps = max(1, int(round(per_target_timeout_s / env.step_dt)))
    env._ttc_timed_out[:] = env._ttc_target_age_steps >= per_target_timeout_steps

    env._ttc_hand_target_error[:] = hand_target_error
    env._ttc_hand_object_error[:] = hand_object_error
    env._ttc_hand_progress[:] = hand_progress
    env._ttc_stance_anchor_error[:] = stance_anchor_error
    env._ttc_base_speed[:] = base_speed
    env._ttc_hand_speed[:] = hand_speed
    env._ttc_torso_lean[:] = torso_lean
    env._ttc_stability_gate[:] = stability_gate
    env._ttc_stable[:] = stable
    env._ttc_backward_drift[:] = backward_drift
    env._ttc_support_force[:] = support_force
    env._ttc_support_contact[:] = support_contact

    command_term = _command_term(env)
    if hasattr(command_term, "metrics"):
        metric_defaults = {
            "phase_id": torch.zeros(env.num_envs, device=env.device),
            "pretouch_success_flag": torch.zeros(env.num_envs, device=env.device),
            "touch_success_flag": torch.zeros(env.num_envs, device=env.device),
            "recover_success_flag": torch.zeros(env.num_envs, device=env.device),
            "target_completion_flag": torch.zeros(env.num_envs, device=env.device),
            "targets_completed": torch.zeros(env.num_envs, device=env.device),
            "target_slot": torch.zeros(env.num_envs, device=env.device),
            "hand_target_error": torch.zeros(env.num_envs, device=env.device),
            "hand_object_error": torch.zeros(env.num_envs, device=env.device),
            "stability_gate": torch.zeros(env.num_envs, device=env.device),
            "stable_flag": torch.zeros(env.num_envs, device=env.device),
            "stance_anchor_error": torch.zeros(env.num_envs, device=env.device),
            "backward_drift": torch.zeros(env.num_envs, device=env.device),
            "support_contact_flag": torch.zeros(env.num_envs, device=env.device),
            "support_force": torch.zeros(env.num_envs, device=env.device),
            "hand_speed": torch.zeros(env.num_envs, device=env.device),
            "active_target_height": torch.zeros(env.num_envs, device=env.device),
            "touch_target_height": torch.zeros(env.num_envs, device=env.device),
            "object_height": torch.zeros(env.num_envs, device=env.device),
            "target_age_steps": torch.zeros(env.num_envs, device=env.device),
            "phase_hold_counter": torch.zeros(env.num_envs, device=env.device),
        }
        for metric_name, metric_default in metric_defaults.items():
            command_term.metrics.setdefault(metric_name, metric_default)
        command_term.metrics["phase_id"][:] = env._ttc_phase.float()
        command_term.metrics["pretouch_success_flag"][:] = env._ttc_recent_pretouch.float()
        command_term.metrics["touch_success_flag"][:] = env._ttc_recent_touch.float()
        command_term.metrics["recover_success_flag"][:] = env._ttc_recent_recover.float()
        command_term.metrics["target_completion_flag"][:] = env._ttc_recent_success.float()
        command_term.metrics["targets_completed"][:] = env._ttc_completed_targets.float()
        command_term.metrics["target_slot"][:] = env._ttc_target_slot.float()
        command_term.metrics["hand_target_error"][:] = env._ttc_hand_target_error
        command_term.metrics["hand_object_error"][:] = env._ttc_hand_object_error
        command_term.metrics["stability_gate"][:] = env._ttc_stability_gate
        command_term.metrics["stable_flag"][:] = env._ttc_stable.float()
        command_term.metrics["stance_anchor_error"][:] = env._ttc_stance_anchor_error
        command_term.metrics["backward_drift"][:] = env._ttc_backward_drift
        command_term.metrics["support_contact_flag"][:] = env._ttc_support_contact.float()
        command_term.metrics["support_force"][:] = env._ttc_support_force
        command_term.metrics["hand_speed"][:] = env._ttc_hand_speed
        command_term.metrics["active_target_height"][:] = env._ttc_active_target_w[:, 2]
        command_term.metrics["touch_target_height"][:] = env._ttc_touch_w[:, 2]
        command_term.metrics["object_height"][:] = env._ttc_object_w[:, 2]
        command_term.metrics["target_age_steps"][:] = env._ttc_target_age_steps.float()
        command_term.metrics["phase_hold_counter"][:] = env._ttc_phase_hold_counter.float()

    env._ttc_prev_episode_length_buf = current_episode_length
    env._ttc_state_synced_step = env.common_step_counter
    _update_target_debug_visualization(env)


def _task_obs(env) -> torch.Tensor:
    return _active_target_pos_base_yaw(env)


def _joint_deviation(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )


def _joint_limit_pressure(env, asset_cfg: SceneEntityCfg, margin_threshold: float) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    joint_range = torch.clamp(joint_limits[..., 1] - joint_limits[..., 0], min=1.0e-6)
    normalized_margin = torch.minimum(joint_pos - joint_limits[..., 0], joint_limits[..., 1] - joint_pos) / joint_range
    return torch.clamp(margin_threshold - normalized_margin, min=0.0).sum(dim=1)


def target_pos_command_obs(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    complete_on_final_touch: bool = False,
):
    _sync_tabletop_clean_state(
        env,
        mode=mode,
        scene_target_names=scene_target_names,
        randomize_order=randomize_order,
        max_targets_per_episode=max_targets_per_episode,
        complete_on_final_touch=complete_on_final_touch,
        per_target_timeout_s=per_target_timeout_s,
        stance_anchor_xy=stance_anchor_xy,
        stance_anchor_std=stance_anchor_std,
        stance_anchor_tolerance=stance_anchor_tolerance,
        base_speed_threshold=base_speed_threshold,
        torso_lean_threshold=torso_lean_threshold,
        stability_speed_scale=stability_speed_scale,
        stability_lean_scale=stability_lean_scale,
        ready_local_pos=ready_local_pos,
        balance_radius=balance_radius,
        balance_hold_steps=balance_hold_steps,
        pretouch_backoff_x=pretouch_backoff_x,
        pretouch_height=pretouch_height,
        pretouch_radius=pretouch_radius,
        pretouch_hold_steps=pretouch_hold_steps,
        pretouch_stability_gate=pretouch_stability_gate,
        touch_height_offset=touch_height_offset,
        touch_radius=touch_radius,
        touch_hold_steps=touch_hold_steps,
        touch_stability_gate=touch_stability_gate,
        recover_radius=recover_radius,
        recover_hold_steps=recover_hold_steps,
        recover_stability_gate=recover_stability_gate,
        hand_speed_threshold=hand_speed_threshold,
        support_sensor_cfg=support_sensor_cfg,
        support_force_threshold=support_force_threshold,
    )
    return _task_obs(env)


_COMMON_TERM_PARAM_NAMES = (
    "mode",
    "scene_target_names",
    "randomize_order",
    "max_targets_per_episode",
    "per_target_timeout_s",
    "stance_anchor_xy",
    "stance_anchor_std",
    "stance_anchor_tolerance",
    "base_speed_threshold",
    "torso_lean_threshold",
    "stability_speed_scale",
    "stability_lean_scale",
    "ready_local_pos",
    "balance_radius",
    "balance_hold_steps",
    "pretouch_backoff_x",
    "pretouch_height",
    "pretouch_radius",
    "pretouch_hold_steps",
    "pretouch_stability_gate",
    "touch_height_offset",
    "touch_radius",
    "touch_hold_steps",
    "touch_stability_gate",
    "recover_radius",
    "recover_hold_steps",
    "recover_stability_gate",
    "hand_speed_threshold",
    "support_sensor_cfg",
    "support_force_threshold",
)


def _sync_from_locals(env, local_vars: dict, **overrides) -> None:
    params = {name: local_vars[name] for name in _COMMON_TERM_PARAM_NAMES if name in local_vars}
    params.update(overrides)
    target_pos_command_obs(env, **params)


def stance_anchor_penalty(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
):
    _sync_from_locals(env, locals())
    return env._ttc_stance_anchor_error


def stance_stability_reward(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
):
    _sync_from_locals(env, locals())
    return env._ttc_stability_gate


def backward_drift_penalty(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
):
    _sync_from_locals(env, locals())
    active_phase = env._ttc_phase != PHASE_BALANCE
    return active_phase.float() * env._ttc_backward_drift


def phase_progress_reward(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    progress_scale: float = 0.04,
):
    _sync_from_locals(env, locals())
    # Use signed progress so moving away from the active target is explicitly penalized.
    return torch.tanh(env._ttc_hand_progress / max(progress_scale, 1.0e-6))


def phase_target_tracking_reward(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg | None = None,
    use_stability_gate: bool = False,
    block_on_support_contact: bool = False,
):
    del asset_cfg
    _sync_from_locals(env, locals())
    reward = torch.exp(-env._ttc_hand_target_error / max(std, 1.0e-6))
    if use_stability_gate:
        reward = reward * env._ttc_stability_gate
    if block_on_support_contact:
        reward = reward * (~env._ttc_support_contact).float()
    return reward


def phase_hold_reward(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    hold_reward_std: float = 0.03,
    hand_speed_scale: float = 0.10,
    use_stability_gate: bool = False,
    block_on_support_contact: bool = False,
):
    _sync_from_locals(env, locals())
    near_target = torch.exp(-env._ttc_hand_target_error / max(hold_reward_std, 1.0e-6))
    hand_still = torch.exp(-env._ttc_hand_speed / max(hand_speed_scale, 1.0e-6))
    reward = near_target * hand_still
    if use_stability_gate:
        reward = reward * env._ttc_stability_gate
    if block_on_support_contact:
        reward = reward * (~env._ttc_support_contact).float()
    return reward


def lift_intent_reward(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    lift_reference_z: float = 0.08,
    lift_scale: float = 0.08,
):
    _sync_from_locals(env, locals())
    hand_pos_base = _target_pos_base_yaw(env, _hand_pos_w(env))
    lift_height = torch.clamp(hand_pos_base[:, 2] - lift_reference_z, min=0.0)
    reaching_mask = env._ttc_phase == PHASE_PRETOUCH
    return reaching_mask.float() * torch.tanh(lift_height / max(lift_scale, 1.0e-6))


def pretouch_bonus(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
):
    _sync_from_locals(env, locals())
    return env._ttc_recent_pretouch.float()


def touch_bonus(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
):
    _sync_from_locals(env, locals())
    return env._ttc_recent_touch.float()


def target_completion_bonus(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
):
    _sync_from_locals(env, locals())
    return env._ttc_recent_success.float()


def target_age_penalty(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    grace_ratio: float = 0.25,
    power: float = 1.5,
):
    _sync_from_locals(env, locals())
    timeout_steps = max(1.0, round(per_target_timeout_s / env.step_dt))
    age_ratio = torch.clamp(env._ttc_target_age_steps.float() / timeout_steps, min=0.0, max=1.0)
    delayed = torch.clamp(age_ratio - grace_ratio, min=0.0) / max(1.0 - grace_ratio, 1.0e-6)
    active_phase = (env._ttc_phase != PHASE_BALANCE).float()
    return active_phase * torch.pow(delayed, power)


def support_contact_penalty(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    force_scale: float = 10.0,
):
    _sync_from_locals(env, locals())
    return torch.clamp(env._ttc_support_force / max(force_scale, 1.0e-6), min=0.0)


def torso_lean_penalty(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
):
    _sync_from_locals(env, locals())
    return env._ttc_torso_lean


def joint_deviation_penalty(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    asset_cfg: SceneEntityCfg,
):
    _sync_from_locals(env, locals())
    return _joint_deviation(env, asset_cfg=asset_cfg)


def joint_limit_penalty(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    asset_cfg: SceneEntityCfg,
    margin_threshold: float = 0.18,
):
    _sync_from_locals(env, locals())
    return _joint_limit_pressure(env, asset_cfg=asset_cfg, margin_threshold=margin_threshold)


def task_success_reached(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
):
    _sync_from_locals(env, locals())
    return env._ttc_completed_targets >= max_targets_per_episode


def task_timeout_reached(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
):
    _sync_from_locals(env, locals())
    return env._ttc_timed_out


def support_contact_termination(
    env,
    mode: str,
    scene_target_names: Sequence[str],
    randomize_order: bool,
    max_targets_per_episode: int,
    per_target_timeout_s: float,
    stance_anchor_xy: tuple[float, float],
    stance_anchor_std: float,
    stance_anchor_tolerance: float,
    base_speed_threshold: float,
    torso_lean_threshold: float,
    stability_speed_scale: float,
    stability_lean_scale: float,
    ready_local_pos: tuple[float, float, float],
    balance_radius: float,
    balance_hold_steps: int,
    pretouch_backoff_x: float,
    pretouch_height: float,
    pretouch_radius: float,
    pretouch_hold_steps: int,
    pretouch_stability_gate: float,
    touch_height_offset: float,
    touch_radius: float,
    touch_hold_steps: int,
    touch_stability_gate: float,
    recover_radius: float,
    recover_hold_steps: int,
    recover_stability_gate: float,
    hand_speed_threshold: float,
    support_sensor_cfg: SceneEntityCfg,
    support_force_threshold: float,
    termination_force_threshold: float = 5.0,
):
    _sync_from_locals(env, locals(), support_force_threshold=termination_force_threshold)
    return env._ttc_support_contact
