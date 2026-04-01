#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


ITERATION_PATTERN = re.compile(r"Learning iteration\s+(\d+)(?:/\d+)?")
METRIC_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_./-]+):\s+(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


@dataclass(frozen=True)
class MetricThreshold:
    name: str
    minimum: float | None = None
    maximum: float | None = None

    def is_satisfied(self, metrics: dict[str, float]) -> bool:
        value = metrics.get(self.name)
        if value is None or not math.isfinite(value):
            return False
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True


@dataclass(frozen=True)
class StageConfig:
    name: str
    task: str
    max_iterations: int
    min_success_iteration: int
    success_patience: int
    thresholds: tuple[MetricThreshold, ...]


@dataclass
class StageResult:
    name: str
    task: str
    run_name: str
    success: bool
    success_iteration: int | None
    checkpoint: str | None
    final_iteration: int | None
    final_metrics: dict[str, float] = field(default_factory=dict)


STAGES: tuple[StageConfig, ...] = (
    StageConfig(
        name="pretouch",
        task="Unitree-G1-29dof-LeftHand-LocoReach-TableTopPreTouch-Clean-v0",
        max_iterations=6000,
        min_success_iteration=15,
        success_patience=5,
        thresholds=(
            MetricThreshold("Episode_Termination/target_quota", minimum=0.95),
            MetricThreshold("Episode_Termination/target_timeout", maximum=0.05),
            MetricThreshold("Metrics/left_hand_pose/pretouch_success_flag", minimum=0.95),
            MetricThreshold("Metrics/left_hand_pose/support_contact_flag", maximum=0.05),
            MetricThreshold("Metrics/left_hand_pose/stance_anchor_error", maximum=0.05),
        ),
    ),
    StageConfig(
        name="touch",
        task="Unitree-G1-29dof-LeftHand-LocoReach-TableTopTouch-Clean-v0",
        max_iterations=6000,
        min_success_iteration=20,
        success_patience=5,
        thresholds=(
            MetricThreshold("Episode_Termination/target_quota", minimum=0.85),
            MetricThreshold("Episode_Termination/target_timeout", maximum=0.15),
            MetricThreshold("Metrics/left_hand_pose/touch_success_flag", minimum=0.75),
            MetricThreshold("Metrics/left_hand_pose/hand_object_error", maximum=0.09),
            MetricThreshold("Metrics/left_hand_pose/support_contact_flag", maximum=0.05),
        ),
    ),
    StageConfig(
        name="touch_spread",
        task="Unitree-G1-29dof-LeftHand-LocoReach-TableTopTouchSpread-Clean-v0",
        max_iterations=8000,
        min_success_iteration=25,
        success_patience=5,
        thresholds=(
            MetricThreshold("Episode_Termination/target_quota", minimum=0.75),
            MetricThreshold("Episode_Termination/target_timeout", maximum=0.25),
            MetricThreshold("Metrics/left_hand_pose/touch_success_flag", minimum=0.65),
            MetricThreshold("Metrics/left_hand_pose/support_contact_flag", maximum=0.08),
        ),
    ),
    StageConfig(
        name="multi_touch",
        task="Unitree-G1-29dof-LeftHand-LocoReach-TableTopMultiTouch-Clean-v0",
        max_iterations=12000,
        min_success_iteration=30,
        success_patience=5,
        thresholds=(
            MetricThreshold("Episode_Termination/target_quota", minimum=0.50),
            MetricThreshold("Episode_Termination/target_timeout", maximum=0.40),
            MetricThreshold("Metrics/left_hand_pose/targets_completed", minimum=2.5),
            MetricThreshold("Metrics/left_hand_pose/support_contact_flag", maximum=0.10),
        ),
    ),
)


def normalize_experiment_name(task_name: str) -> str:
    experiment_name = task_name.lower().replace("-", "_")
    if experiment_name.endswith("_play"):
        experiment_name = experiment_name[: -len("_play")]
    return experiment_name


def checkpoint_iteration(checkpoint_path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.search(checkpoint_path.name)
    if match is None:
        return None
    return int(match.group(1))


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
    result = remove_arg(args, name)
    result.append(name)
    if value is not None:
        result.append(value)
    return result


def build_stage_train_args(
    base_args: list[str],
    stage: StageConfig,
    run_name: str,
    init_checkpoint: Path | None,
) -> list[str]:
    train_args = list(base_args)
    for arg_name in (
        "--task",
        "--run_name",
        "--max_iterations",
        "--init_checkpoint",
        "--resume",
        "--checkpoint",
        "--load_run",
        "--load_checkpoint",
        "--load_weights_only",
    ):
        train_args = remove_arg(train_args, arg_name)

    train_args = upsert_flag(train_args, "--task", stage.task)
    train_args = upsert_flag(train_args, "--max_iterations", str(stage.max_iterations))
    train_args = upsert_flag(train_args, "--run_name", run_name)
    if init_checkpoint is not None:
        train_args = upsert_flag(train_args, "--init_checkpoint", str(init_checkpoint))
    return train_args


def find_latest_checkpoint_for_run(log_root: Path, run_name: str) -> Path | None:
    checkpoints: list[Path] = []
    if not log_root.exists():
        return None
    for checkpoint_path in log_root.rglob("model_*.pt"):
        if run_name not in checkpoint_path.parent.name:
            continue
        if checkpoint_path.is_file():
            checkpoints.append(checkpoint_path)
    if not checkpoints:
        return None
    checkpoints.sort(
        key=lambda path: (
            checkpoint_iteration(path) if checkpoint_iteration(path) is not None else -1,
            path.stat().st_mtime_ns,
            str(path),
        )
    )
    return checkpoints[-1]


def resolve_hold_stay_checkpoint(repo_root: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        checkpoint_path = Path(explicit_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--hold_stay_checkpoint does not point to a file: {checkpoint_path}")
        return checkpoint_path

    log_root = repo_root / "logs" / "rsl_rl" / "unitree_g1_29dof_lefthand_locoreach_adapterholdstay_v0"
    checkpoints = sorted(
        (path for path in log_root.rglob("model_*.pt") if path.is_file()),
        key=lambda path: (checkpoint_iteration(path) if checkpoint_iteration(path) is not None else -1, path.stat().st_mtime_ns),
    )
    if not checkpoints:
        raise FileNotFoundError(
            "Could not find any hold_stay checkpoint under logs/rsl_rl/"
            "unitree_g1_29dof_lefthand_locoreach_adapterholdstay_v0"
        )
    return checkpoints[-1]


def signal_process_group(process: subprocess.Popen[str], sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def metrics_satisfy(stage: StageConfig, iteration: int | None, metrics: dict[str, float]) -> bool:
    if iteration is None or iteration < stage.min_success_iteration:
        return False
    return all(threshold.is_satisfied(metrics) for threshold in stage.thresholds)


def wait_for_checkpoint_after_success(
    log_root: Path,
    run_name: str,
    success_iteration: int,
    timeout_s: float,
) -> Path | None:
    deadline = time.time() + timeout_s
    latest_checkpoint = find_latest_checkpoint_for_run(log_root, run_name)
    while time.time() < deadline:
        latest_checkpoint = find_latest_checkpoint_for_run(log_root, run_name)
        if latest_checkpoint is not None:
            latest_iteration = checkpoint_iteration(latest_checkpoint)
            if latest_iteration is not None and latest_iteration >= success_iteration:
                return latest_checkpoint
        time.sleep(5.0)
    return latest_checkpoint


def launch_stage_and_wait(
    repo_root: Path,
    wrapper_script: Path,
    wrapper_args: list[str],
    stage: StageConfig,
    run_name: str,
    checkpoint_wait_timeout_s: float,
) -> StageResult:
    command = [sys.executable, str(wrapper_script), *wrapper_args]
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    log_root = repo_root / "logs" / "rsl_rl" / normalize_experiment_name(stage.task)
    current_iteration: int | None = None
    current_metrics: dict[str, float] = {}
    last_metrics: dict[str, float] = {}
    success_iteration: int | None = None
    success_streak = 0
    stage_completed = False
    checkpoint_requested = False

    def finalize_iteration(iteration: int | None, metrics: dict[str, float]) -> None:
        nonlocal success_streak, success_iteration, stage_completed, checkpoint_requested, last_metrics
        if iteration is None or not metrics:
            return
        last_metrics = dict(metrics)
        if metrics_satisfy(stage, iteration, metrics):
            success_streak += 1
            print(
                f"[CURRICULUM] Stage '{stage.name}' success streak "
                f"{success_streak}/{stage.success_patience} at iteration {iteration}.",
                flush=True,
            )
        else:
            success_streak = 0

        if success_iteration is None and success_streak >= stage.success_patience:
            success_iteration = iteration
            print(
                f"[CURRICULUM] Stage '{stage.name}' reached success criteria at iteration {iteration}. "
                "Waiting for a fresh checkpoint before advancing.",
                flush=True,
            )

        if success_iteration is not None and not checkpoint_requested:
            latest_checkpoint = find_latest_checkpoint_for_run(log_root, run_name)
            latest_iteration = checkpoint_iteration(latest_checkpoint) if latest_checkpoint else None
            if latest_iteration is not None and latest_iteration >= success_iteration:
                checkpoint_requested = True
                stage_completed = True
                signal_process_group(process, signal.SIGINT)

    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)

            parsed_iteration = None
            iteration_match = ITERATION_PATTERN.search(line)
            if iteration_match is not None:
                parsed_iteration = int(iteration_match.group(1))
                finalize_iteration(current_iteration, current_metrics)
                current_iteration = parsed_iteration
                current_metrics = {}
                continue

            metric_match = METRIC_PATTERN.match(line)
            if metric_match is not None and current_iteration is not None:
                metric_name = metric_match.group(1)
                try:
                    current_metrics[metric_name] = float(metric_match.group(2))
                except ValueError:
                    pass
    finally:
        returncode = process.wait()

    finalize_iteration(current_iteration, current_metrics)

    checkpoint_path: Path | None = None
    if success_iteration is not None:
        checkpoint_path = wait_for_checkpoint_after_success(
            log_root=log_root,
            run_name=run_name,
            success_iteration=success_iteration,
            timeout_s=checkpoint_wait_timeout_s,
        )
        if checkpoint_path is not None:
            stage_completed = True

    if not stage_completed and returncode == 0 and metrics_satisfy(stage, current_iteration, current_metrics):
        success_iteration = current_iteration
        checkpoint_path = find_latest_checkpoint_for_run(log_root, run_name)
        stage_completed = checkpoint_path is not None

    return StageResult(
        name=stage.name,
        task=stage.task,
        run_name=run_name,
        success=stage_completed,
        success_iteration=success_iteration,
        checkpoint=str(checkpoint_path) if checkpoint_path is not None else None,
        final_iteration=current_iteration,
        final_metrics=last_metrics or current_metrics,
    )


def build_stage_sequence(start_stage: str, stop_after_stage: str | None) -> list[StageConfig]:
    stage_names = [stage.name for stage in STAGES]
    if start_stage not in stage_names:
        raise ValueError(f"Unknown start stage '{start_stage}'. Expected one of: {stage_names}")
    if stop_after_stage is not None and stop_after_stage not in stage_names:
        raise ValueError(f"Unknown stop stage '{stop_after_stage}'. Expected one of: {stage_names}")

    start_index = stage_names.index(start_stage)
    stop_index = stage_names.index(stop_after_stage) if stop_after_stage is not None else len(STAGES) - 1
    if stop_index < start_index:
        raise ValueError("--stop_after_stage cannot be earlier than --start_stage.")
    return list(STAGES[start_index : stop_index + 1])


def write_summary(summary_path: Path, results: list[StageResult], run_prefix: str) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_prefix": run_prefix,
        "results": [asdict(result) for result in results],
        "final_checkpoint": results[-1].checkpoint if results else None,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Sequential curriculum launcher for tabletop clean tasks.")
    parser.add_argument(
        "--run_prefix",
        type=str,
        required=True,
        help="Shared prefix used to name each stage run and the curriculum summary file.",
    )
    parser.add_argument(
        "--hold_stay_checkpoint",
        type=str,
        default=None,
        help="Optional explicit hold_stay checkpoint for the first pretouch stage.",
    )
    parser.add_argument(
        "--curriculum_init_checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint to seed the selected --start_stage.",
    )
    parser.add_argument(
        "--start_stage",
        type=str,
        default="pretouch",
        choices=[stage.name for stage in STAGES],
        help="First stage to run.",
    )
    parser.add_argument(
        "--stop_after_stage",
        type=str,
        default=None,
        choices=[stage.name for stage in STAGES],
        help="Optional stage at which to stop.",
    )
    parser.add_argument(
        "--checkpoint_wait_timeout_s",
        type=float,
        default=900.0,
        help="How long to wait for a fresh checkpoint after a stage first meets its success criteria.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Optional explicit summary JSON output path.",
    )
    args, wrapper_args = parser.parse_known_args()
    return args, wrapper_args


def main() -> int:
    args, passthrough_wrapper_args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    wrapper_script = repo_root / "scripts" / "rsl_rl" / "train_autoresume.py"
    stage_sequence = build_stage_sequence(args.start_stage, args.stop_after_stage)

    base_wrapper_args = list(passthrough_wrapper_args)
    for arg_name in (
        "--task",
        "--run_name",
        "--max_iterations",
        "--init_checkpoint",
        "--resume",
        "--checkpoint",
        "--load_run",
        "--load_checkpoint",
        "--load_weights_only",
    ):
        base_wrapper_args = remove_arg(base_wrapper_args, arg_name)

    summary_path = (
        Path(args.summary_path).expanduser()
        if args.summary_path is not None
        else repo_root / "logs" / "rsl_rl" / "curriculum" / f"{args.run_prefix}_summary.json"
    )

    if args.curriculum_init_checkpoint is not None:
        next_init_checkpoint = Path(args.curriculum_init_checkpoint).expanduser().resolve()
        if not next_init_checkpoint.is_file():
            raise FileNotFoundError(
                f"--curriculum_init_checkpoint does not point to a file: {next_init_checkpoint}"
            )
    elif stage_sequence[0].name == "pretouch":
        next_init_checkpoint = resolve_hold_stay_checkpoint(repo_root, args.hold_stay_checkpoint)
    else:
        raise ValueError(
            "Starting after pretouch requires --curriculum_init_checkpoint because there is no previous stage to chain from."
        )

    results: list[StageResult] = []

    print(f"[CURRICULUM] Run prefix: {args.run_prefix}", flush=True)
    print(f"[CURRICULUM] Summary path: {summary_path}", flush=True)
    print(f"[CURRICULUM] Initial checkpoint: {next_init_checkpoint}", flush=True)

    for stage in stage_sequence:
        run_name = f"{args.run_prefix}_{stage.name}"
        print(
            f"[CURRICULUM] Starting stage '{stage.name}' with task '{stage.task}' "
            f"from checkpoint '{next_init_checkpoint}'.",
            flush=True,
        )

        stage_wrapper_args = build_stage_train_args(
            base_args=base_wrapper_args,
            stage=stage,
            run_name=run_name,
            init_checkpoint=next_init_checkpoint,
        )
        result = launch_stage_and_wait(
            repo_root=repo_root,
            wrapper_script=wrapper_script,
            wrapper_args=stage_wrapper_args,
            stage=stage,
            run_name=run_name,
            checkpoint_wait_timeout_s=args.checkpoint_wait_timeout_s,
        )
        results.append(result)
        write_summary(summary_path, results, args.run_prefix)

        if not result.success or result.checkpoint is None:
            print(
                f"[CURRICULUM] Stage '{stage.name}' did not meet its promotion criteria. "
                f"Final iteration={result.final_iteration} checkpoint={result.checkpoint}",
                flush=True,
            )
            return 1

        next_init_checkpoint = Path(result.checkpoint)
        print(
            f"[CURRICULUM] Stage '{stage.name}' complete. "
            f"Promoting checkpoint: {next_init_checkpoint}",
            flush=True,
        )

    print(f"[CURRICULUM] Curriculum finished successfully. Final checkpoint: {next_init_checkpoint}", flush=True)
    write_summary(summary_path, results, args.run_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
