from __future__ import annotations

import copy
import importlib
import inspect
from typing import Any


def _resolve_exported_class(module_name: str, class_name: str | None):
    if not class_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, class_name, None)


def _accepted_kwargs(target) -> set[str] | None:
    if target is None:
        return None
    try:
        signature_target = target.__init__ if inspect.isclass(target) else target
        signature = inspect.signature(signature_target)
    except (TypeError, ValueError):
        return None

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return None

    accepted = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted.add(name)
    return accepted


def _filter_kwargs(kwargs: dict[str, Any], accepted: set[str] | None) -> tuple[dict[str, Any], list[str]]:
    if accepted is None:
        return kwargs, []
    filtered = {key: value for key, value in kwargs.items() if key in accepted}
    dropped = [key for key in kwargs.keys() if key not in accepted]
    return filtered, dropped


def sanitize_runner_cfg(agent_cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop runner config kwargs that the installed rsl_rl version does not accept."""

    cfg = copy.deepcopy(agent_cfg_dict)
    dropped: list[str] = []

    policy_cls = _resolve_exported_class("rsl_rl.modules", cfg.get("policy_class_name"))
    if isinstance(cfg.get("policy"), dict):
        for key in ("state_dependent_std",):
            if key in cfg["policy"]:
                cfg["policy"].pop(key, None)
                dropped.append(f"policy.{key}")
        accepted = _accepted_kwargs(policy_cls)
        cfg["policy"], removed = _filter_kwargs(cfg["policy"], accepted)
        dropped.extend(f"policy.{key}" for key in removed)

    algorithm_cls = _resolve_exported_class("rsl_rl.algorithms", cfg.get("algorithm_class_name"))
    if isinstance(cfg.get("algorithm"), dict):
        for key in ("optimizer", "share_cnn_encoders"):
            if key in cfg["algorithm"]:
                cfg["algorithm"].pop(key, None)
                dropped.append(f"algorithm.{key}")
        accepted = _accepted_kwargs(algorithm_cls)
        cfg["algorithm"], removed = _filter_kwargs(cfg["algorithm"], accepted)
        dropped.extend(f"algorithm.{key}" for key in removed)

    if dropped:
        print(
            "[INFO]: Dropped unsupported rsl_rl config keys for installed runtime: "
            + ", ".join(sorted(dropped)),
            flush=True,
        )

    return cfg
