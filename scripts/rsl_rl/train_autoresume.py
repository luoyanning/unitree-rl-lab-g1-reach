#!/usr/bin/env python3

"""Launch `train.py` with automatic restart from the latest checkpoint on failure."""

from __future__ import annotations

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


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
    parser.add_argument(
        "--reward_drop_restart_min_iteration",
        type=int,
        default=50,
        help="Enable reward-collapse restart logic only after this many iterations since the current launch/restart.",
    )
    parser.add_argument(
        "--reward_drop_restart_threshold",
        type=float,
        default=10000.0,
        help="Restart when mean reward stays below (best_reward - threshold).",
    )
    parser.add_argument(
        "--reward_drop_restart_consecutive",
        type=int,
        default=10,
        help="Number of consecutive bad mean-reward iterations required before restart.",
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


def find_checkpoints(log_root: Path) -> list[Path]:
    checkpoints = [path for path in log_root.rglob("model_*.pt") if path.is_file()]
    checkpoints.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return checkpoints


def find_latest_checkpoint(log_root: Path) -> Path | None:
    checkpoints = find_checkpoints(log_root)
    if not checkpoints:
        return None
    return checkpoints[-1]


def is_checkpoint_managed_by_run(log_root: Path, checkpoint_path: Path) -> bool:
    try:
        return checkpoint_path.resolve().is_relative_to(log_root.resolve())
    except FileNotFoundError:
        return False


def checkpoint_iteration(checkpoint_path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.search(checkpoint_path.name)
    if match is None:
        return None
    return int(match.group(1))


def delete_managed_checkpoint_iteration(
    log_root: Path,
    iteration: int,
    protected_paths: set[Path] | None = None,
) -> list[Path]:
    protected_paths = protected_paths or set()
    deleted_paths: list[Path] = []
    for checkpoint_path in find_checkpoints(log_root):
        if checkpoint_iteration(checkpoint_path) != iteration:
            continue
        if not is_checkpoint_managed_by_run(log_root, checkpoint_path):
            continue
        if checkpoint_path.resolve() in protected_paths:
            continue
        checkpoint_path.unlink(missing_ok=True)
        deleted_paths.append(checkpoint_path)
    return deleted_paths


def build_resume_args(original_args: list[str], checkpoint_path: Path) -> list[str]:
    args = list(original_args)
    args = remove_arg(args, "--load_run")
    args = remove_arg(args, "--init_checkpoint")
    args = remove_arg(args, "--load_weights_only")
    args = upsert_flag(args, "--resume")
    args = upsert_flag(args, "--checkpoint", str(checkpoint_path))
    return args


ITERATION_PATTERN = re.compile(r"Learning iteration\s+(\d+)(?:/\d+)?")
MEAN_REWARD_PATTERN = re.compile(r"Mean reward:\s+([^\s]+)")


def parse_iteration(line: str) -> int | None:
    match = ITERATION_PATTERN.search(line)
    if match is None:
        return None
    return int(match.group(1))


def parse_mean_reward(line: str) -> float | None:
    match = MEAN_REWARD_PATTERN.search(line)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def stream_train_process(
    command: list[str],
    cwd: Path,
    min_iteration: int,
    reward_drop_threshold: float,
    reward_drop_consecutive: int,
    best_mean_reward: float | None,
) -> tuple[int, float | None, bool]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    current_iteration = -1
    launch_start_iteration: int | None = None
    consecutive_bad_rewards = 0
    restart_requested = False

    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)

            parsed_iteration = parse_iteration(line)
            if parsed_iteration is not None:
                current_iteration = parsed_iteration
                if launch_start_iteration is None:
                    launch_start_iteration = parsed_iteration

            parsed_mean_reward = parse_mean_reward(line)
            if parsed_mean_reward is None:
                continue

            iterations_since_launch = 0
            if launch_start_iteration is not None and current_iteration >= launch_start_iteration:
                iterations_since_launch = current_iteration - launch_start_iteration

            if math.isfinite(parsed_mean_reward):
                if best_mean_reward is None or parsed_mean_reward > best_mean_reward:
                    best_mean_reward = parsed_mean_reward

                if iterations_since_launch >= min_iteration and best_mean_reward is not None:
                    reward_gap = best_mean_reward - parsed_mean_reward
                    if reward_gap > reward_drop_threshold:
                        consecutive_bad_rewards += 1
                        print(
                            "[AUTO-RESUME] Detected reward collapse: "
                            f"iteration={current_iteration} "
                            f"since_launch={iterations_since_launch} "
                            f"mean_reward={parsed_mean_reward:.4f} "
                            f"best_reward={best_mean_reward:.4f} "
                            f"gap={reward_gap:.4f} "
                            f"consecutive={consecutive_bad_rewards}/{reward_drop_consecutive}",
                            flush=True,
                        )
                    else:
                        consecutive_bad_rewards = 0
                else:
                    consecutive_bad_rewards = 0
            else:
                if iterations_since_launch >= min_iteration:
                    consecutive_bad_rewards += 1
                    print(
                        "[AUTO-RESUME] Detected non-finite mean reward after warm-up: "
                        f"iteration={current_iteration} "
                        f"since_launch={iterations_since_launch} "
                        f"value={parsed_mean_reward} "
                        f"consecutive={consecutive_bad_rewards}/{reward_drop_consecutive}",
                        flush=True,
                    )
                else:
                    consecutive_bad_rewards = 0

            if consecutive_bad_rewards >= reward_drop_consecutive:
                restart_requested = True
                print(
                    "[AUTO-RESUME] Triggering restart from the latest checkpoint due to persistent reward collapse.",
                    flush=True,
                )
                process.terminate()
                break
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        raise
    finally:
        returncode = process.wait()

    return returncode, best_mean_reward, restart_requested


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
    initial_checkpoint_arg = get_arg_value(base_train_args, "--checkpoint")
    protected_initial_checkpoint: Path | None = None
    protected_initial_checkpoint_resolved: Path | None = None
    if initial_checkpoint_arg:
        protected_initial_checkpoint = Path(initial_checkpoint_arg).expanduser()
        protected_initial_checkpoint_resolved = protected_initial_checkpoint.resolve()

    restart_count = 0
    best_mean_reward: float | None = None
    stagnant_failures_by_checkpoint: dict[str, int] = {}
    while True:
        checkpoints_before_launch = find_checkpoints(log_root)
        checkpoints_before_launch_set = {path.resolve() for path in checkpoints_before_launch}
        latest_checkpoint = checkpoints_before_launch[-1] if checkpoints_before_launch else None
        launch_args = base_train_args
        launch_checkpoint: Path | None = None
        if restart_count > 0 and latest_checkpoint is not None:
            launch_args = build_resume_args(base_train_args, latest_checkpoint)
            launch_checkpoint = latest_checkpoint
            print(f"[AUTO-RESUME] Restart {restart_count}: resuming from {latest_checkpoint}", flush=True)
        elif restart_count > 0:
            print("[AUTO-RESUME] No checkpoint found to resume from after failure. Stopping.", flush=True)
            return 1
        elif "--resume" in base_train_args and initial_checkpoint_arg:
            launch_checkpoint = Path(initial_checkpoint_arg).expanduser()

        command = [sys.executable, str(train_script), *launch_args]
        print(f"[AUTO-RESUME] Launching: {' '.join(command)}", flush=True)

        try:
            returncode, best_mean_reward, restart_requested = stream_train_process(
                command,
                cwd=repo_root,
                min_iteration=wrapper_args.reward_drop_restart_min_iteration,
                reward_drop_threshold=wrapper_args.reward_drop_restart_threshold,
                reward_drop_consecutive=wrapper_args.reward_drop_restart_consecutive,
                best_mean_reward=best_mean_reward,
            )
        except KeyboardInterrupt:
            print("[AUTO-RESUME] Interrupted by user.", flush=True)
            return 130

        if returncode == 0 and not restart_requested:
            print("[AUTO-RESUME] Training completed successfully.", flush=True)
            return 0

        checkpoints_after_failure = find_checkpoints(log_root)
        new_checkpoints = [path for path in checkpoints_after_failure if path.resolve() not in checkpoints_before_launch_set]
        if launch_checkpoint is not None:
            launch_checkpoint_iteration = checkpoint_iteration(launch_checkpoint)
            checkpoint_advanced = False
            if launch_checkpoint_iteration is None:
                checkpoint_advanced = len(new_checkpoints) > 0
            else:
                checkpoint_advanced = any(
                    (checkpoint_iteration(path) or -1) > launch_checkpoint_iteration for path in new_checkpoints
                )

            logical_checkpoint_key = str(launch_checkpoint.resolve())
            if is_checkpoint_managed_by_run(log_root, launch_checkpoint) and launch_checkpoint_iteration is not None:
                logical_checkpoint_key = f"iter:{launch_checkpoint_iteration}"

            if checkpoint_advanced:
                stagnant_failures_by_checkpoint.pop(logical_checkpoint_key, None)
            else:
                stagnant_failures_by_checkpoint[logical_checkpoint_key] = (
                    stagnant_failures_by_checkpoint.get(logical_checkpoint_key, 0) + 1
                )
                print(
                    "[AUTO-RESUME] Launch failed before producing a higher-iteration checkpoint: "
                    f"start_checkpoint={launch_checkpoint} "
                    f"attempts={stagnant_failures_by_checkpoint[logical_checkpoint_key]}/2",
                    flush=True,
                )
                if stagnant_failures_by_checkpoint[logical_checkpoint_key] >= 2:
                    if is_checkpoint_managed_by_run(log_root, launch_checkpoint) and launch_checkpoint_iteration is not None:
                        protected_paths: set[Path] = set()
                        if protected_initial_checkpoint_resolved is not None:
                            protected_paths.add(protected_initial_checkpoint_resolved)
                        deleted_paths = delete_managed_checkpoint_iteration(
                            log_root,
                            launch_checkpoint_iteration,
                            protected_paths=protected_paths,
                        )
                        stagnant_failures_by_checkpoint.pop(logical_checkpoint_key, None)
                        best_mean_reward = None
                        print(
                            "[AUTO-RESUME] Deleted repeatedly failing checkpoint family and will roll back to the previous iteration: "
                            f"iteration={launch_checkpoint_iteration} deleted={len(deleted_paths)}",
                            flush=True,
                        )
                    elif (
                        is_checkpoint_managed_by_run(log_root, launch_checkpoint)
                        and launch_checkpoint.exists()
                        and launch_checkpoint.resolve() != protected_initial_checkpoint_resolved
                    ):
                        launch_checkpoint.unlink()
                        stagnant_failures_by_checkpoint.pop(logical_checkpoint_key, None)
                        best_mean_reward = None
                        print(
                            "[AUTO-RESUME] Deleted repeatedly failing checkpoint and will roll back to the previous one: "
                            f"{launch_checkpoint}",
                            flush=True,
                        )
                    else:
                        print(
                            "[AUTO-RESUME] Repeated failure came from a protected or unmanaged checkpoint; "
                            "not deleting it automatically.",
                            flush=True,
                        )

        restart_count += 1
        if restart_requested:
            print("[AUTO-RESUME] Train process was restarted proactively.", flush=True)
        else:
            print(f"[AUTO-RESUME] Train process exited with code {returncode}.", flush=True)
        if restart_count > wrapper_args.max_auto_restarts:
            print("[AUTO-RESUME] Reached restart limit. Stopping.", flush=True)
            return returncode
        time.sleep(wrapper_args.restart_delay)


if __name__ == "__main__":
    raise SystemExit(main())
