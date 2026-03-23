import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, tuple):
        return [format_value(i) for i in x]
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        return format_value(x.tolist())
    else:
        return x


def _get_robot_cfg(env):
    if hasattr(env.cfg, "robot"):
        return env.cfg.robot
    if hasattr(env.cfg, "scene") and hasattr(env.cfg.scene, "robot"):
        return env.cfg.scene.robot
    return None


def _get_joint_sdk_names(asset: Articulation, robot_cfg) -> tuple[list[str], list[int]]:
    joint_sdk_names = getattr(robot_cfg, "joint_sdk_names", None) if robot_cfg is not None else None
    if joint_sdk_names:
        try:
            joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)
            return list(joint_sdk_names), list(joint_ids_map)
        except ValueError:
            pass

    joint_names = list(asset.data.joint_names)
    return joint_names, list(range(len(joint_names)))


def export_deploy_cfg(env, log_dir):
    asset: Articulation = env.scene["robot"]
    robot_cfg = _get_robot_cfg(env)
    joint_sdk_names, joint_ids_map = _get_joint_sdk_names(asset, robot_cfg)

    cfg = {}  # noqa: SIM904
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    cfg["stiffness"] = stiffness.tolist()
    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg, "commands") and hasattr(env.cfg.commands, "base_velocity"):
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges
    elif all(hasattr(env.cfg, name) for name in ["command_vx_range", "command_yaw_rate_range", "stand_height_range"]):
        cfg["commands"]["homie"] = {
            "ranges": {
                "vx": list(env.cfg.command_vx_range),
                "yaw_rate": list(env.cfg.command_yaw_rate_range),
                "target_height_stand": list(env.cfg.stand_height_range),
            },
            "resample_interval_s": getattr(env.cfg, "command_resample_interval_s", None),
            "transition_duration_s": getattr(env.cfg, "command_transition_duration_s", None),
        }
        if hasattr(env.cfg, "squat_height_range"):
            cfg["commands"]["homie"]["ranges"]["target_height_squat"] = list(env.cfg.squat_height_range)

    # --- actions ---
    cfg["actions"] = {}
    if hasattr(env, "action_manager"):
        action_names = env.action_manager.active_terms
        action_terms = zip(action_names, env.action_manager._terms.values())
        for action_name, action_term in action_terms:
            term_cfg = action_term.cfg.copy()
            if isinstance(term_cfg.scale, float):
                term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
            else:  # dict
                term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

            if term_cfg.clip is not None:
                term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

            if action_name in ["JointPositionAction", "JointVelocityAction"]:
                if term_cfg.use_default_offset:
                    term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
                else:
                    term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

            term_cfg = term_cfg.to_dict()

            for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
                del term_cfg[_]
            cfg["actions"][action_name] = term_cfg

            if action_term._joint_ids == slice(None):
                cfg["actions"][action_name]["joint_ids"] = None
            else:
                cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids
    else:
        direct_joint_names = list(getattr(env.cfg, "lower_joint_names", []))
        if direct_joint_names:
            joint_ids, _ = resolve_matching_names(asset.data.joint_names, direct_joint_names, preserve_order=True)
        else:
            action_dim = int(getattr(env.cfg, "action_space", len(joint_sdk_names)))
            direct_joint_names = joint_sdk_names[:action_dim]
            joint_ids = list(range(action_dim))

        action_scale = getattr(env.cfg, "action_scale", 1.0)
        if isinstance(action_scale, (float, int)):
            action_scale = [float(action_scale) for _ in range(len(direct_joint_names))]
        else:
            action_scale = list(action_scale)

        cfg["actions"]["policy"] = {
            "type": "direct_joint_position_offset",
            "joint_names": direct_joint_names,
            "joint_ids": list(joint_ids),
            "action_dim": int(getattr(env.cfg, "action_space", len(direct_joint_names))),
            "scale": action_scale,
            "clip": [-1.0, 1.0],
        }

    # --- observations ---
    cfg["observations"] = {}
    if hasattr(env, "observation_manager"):
        obs_names = env.observation_manager.active_terms["policy"]
        obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
        obs_terms = zip(obs_names, obs_cfgs)
        for obs_name, obs_cfg in obs_terms:
            obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
            term_cfg = obs_cfg.copy()
            if term_cfg.scale is not None:
                scale = term_cfg.scale.detach().cpu().numpy().tolist()
                if isinstance(scale, float):
                    term_cfg.scale = [scale for _ in range(obs_dims[1])]
                else:
                    term_cfg.scale = scale
            else:
                term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
            if term_cfg.clip is not None:
                term_cfg.clip = list(term_cfg.clip)
            if term_cfg.history_length == 0:
                term_cfg.history_length = 1

            term_cfg = term_cfg.to_dict()
            for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
                del term_cfg[_]
            cfg["observations"][obs_name] = term_cfg
    else:
        policy_dim = int(getattr(env.cfg, "observation_space", 0))
        critic_dim = int(getattr(env.cfg, "state_space", 0))
        cfg["observations"]["policy"] = {
            "shape": [policy_dim],
            "history_length": int(getattr(env.cfg, "history_length", 1)),
        }
        if critic_dim > 0:
            cfg["observations"]["critic"] = {
                "shape": [critic_dim],
                "history_length": 1,
            }

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
