#!/usr/bin/env python3

"""Launch `train.py` with automatic restart from the latest checkpoint on failure."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Watchdog wrapper around scripts/rsl_rl/train.py.")
    parser.add_argument(
        "--max_auto_restarts",
        type=int,
        default=20,
        help="Maximum number of automatic restarts after non-zero exits.",
    )
    parser.add_argument(
        "--restart_delay",
        type=float,
        default=5.0,
        help="Delay in seconds before restarting after a failure.",
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default="scripts/rsl_rl/train.py",
        help="Train script to launch.",
    )
    args, train_args = parser.parse_known_args()
    return args, train_args


def get_arg_value(args: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def remove_arg(args: list[str], name: str) -> list[str]:
    result: list[str] = []
    skip_next = False
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == name:
            if index + 1 < len(args) and not args[index + 1].startswith("--"):
                skip_next = True
            continue
        if arg.startswith(prefix):
            continue
        result.append(arg)
    return result


def upsert_flag(args: list[str], name: str, value: str | None = None) -> list[str]:
    args = remove_arg(args, name)
    args.append(name)
    if value is not None:
        args.append(value)
    return args


def normalize_experiment_name(task_name: str) -> str:
    experiment_name = task_name.lower().replace("-", "_")
    if experiment_name.endswith("_play"):
        experiment_name = experiment_name[: -len("_play")]
    return experiment_name


def extract_checkpoint_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(log_root: Path) -> Path | None:
    checkpoints = [path for path in log_root.rglob("model_*.pt") if path.is_file()]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda path: (extract_checkpoint_iteration(path), path.stat().st_mtime, str(path)))
    return checkpoints[-1]


def build_resume_args(original_args: list[str], checkpoint_path: Path) -> list[str]:
    args = list(original_args)
    args = remove_arg(args, "--load_run")
    args = remove_arg(args, "--init_checkpoint")
    args = upsert_flag(args, "--resume")
    args = upsert_flag(args, "--checkpoint", str(checkpoint_path))
    return args


def main() -> int:
    wrapper_args, train_args = parse_wrapper_args()
    if not train_args:
        raise SystemExit("Missing train.py arguments. Example: train_autoresume.py --task ...")

    task_name = get_arg_value(train_args, "--task")
    if task_name is None:
        raise SystemExit("Missing required --task argument for train.py.")
    experiment_name = get_arg_value(train_args, "--experiment_name") or normalize_experiment_name(task_name)

    repo_root = Path(__file__).resolve().parents[2]
    log_root = repo_root / "logs" / "rsl_rl" / experiment_name
    train_script = repo_root / wrapper_args.train_script
    base_train_args = list(train_args)

    restart_count = 0
    while True:
        latest_checkpoint = find_latest_checkpoint(log_root)
        launch_args = base_train_args
        if restart_count > 0 and latest_checkpoint is not None:
            launch_args = build_resume_args(base_train_args, latest_checkpoint)
            print(f"[AUTO-RESUME] Restart {restart_count}: resuming from {latest_checkpoint}", flush=True)
        elif restart_count > 0:
            print("[AUTO-RESUME] No checkpoint found to resume from after failure. Stopping.", flush=True)
            return 1

        command = [sys.executable, str(train_script), *launch_args]
        print(f"[AUTO-RESUME] Launching: {' '.join(command)}", flush=True)

        try:
            completed = subprocess.run(command, cwd=repo_root)
        except KeyboardInterrupt:
            print("[AUTO-RESUME] Interrupted by user.", flush=True)
            return 130

        if completed.returncode == 0:
            print("[AUTO-RESUME] Training completed successfully.", flush=True)
            return 0

        restart_count += 1
        print(f"[AUTO-RESUME] Train process exited with code {completed.returncode}.", flush=True)
        if restart_count > wrapper_args.max_auto_restarts:
            print("[AUTO-RESUME] Reached restart limit. Stopping.", flush=True)
            return completed.returncode
        time.sleep(wrapper_args.restart_delay)


if __name__ == "__main__":
    raise SystemExit(main())
