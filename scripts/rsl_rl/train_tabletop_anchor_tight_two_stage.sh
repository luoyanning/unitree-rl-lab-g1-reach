#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_INIT_CKPT="logs/rsl_rl/unitree_g1_29dof_lefthand_locoreach_adapterholdstay_v0/2026-03-17_02-12-54/model_73250.pt"
DEFAULT_VENV="/mlp_vepfs/share/lyn/try0310/env_isaaclab"
DEFAULT_ISAACLAB_PATH="/mlp_vepfs/share/lyn/try0310/IsaacLab"

STAGE1_TASK="Unitree-G1-29dof-LeftHand-LocoReach-TableTopMultiTouchPairAnchorTight-Clean-v0"
STAGE2_TASK="Unitree-G1-29dof-LeftHand-LocoReach-TableTopFixedAcquireStayAnchorTight-Clean-v0"

INIT_CKPT="${DEFAULT_INIT_CKPT}"
STAGE1_RUN_PREFIX="tabletop_pair_anchor_tight_r2_ws73250"
STAGE2_RUN_PREFIX="tabletop_fixed_anchor_tight_from_pair_r2"
NUM_ENVS="384"
STAGE1_MAX_ITERATIONS="5000"
STAGE2_MAX_ITERATIONS="9000"
MAX_AUTO_RESTARTS="100"
RESTART_DELAY="15"
VENV_PATH="${ISAACLAB_VENV:-${DEFAULT_VENV}}"
ISAACLAB_ROOT="${ISAACLAB_PATH:-${DEFAULT_ISAACLAB_PATH}}"
AUTO_ACTIVATE="1"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/rsl_rl/train_tabletop_anchor_tight_two_stage.sh [options]

Options:
  --init-checkpoint PATH         Warm-start checkpoint for stage 1.
  --num-envs N                   Number of environments for both stages.
  --stage1-max-iterations N      Max PPO iterations for stage 1.
  --stage2-max-iterations N      Max PPO iterations for stage 2.
  --stage1-run-prefix NAME       Run name suffix for stage 1.
  --stage2-run-prefix NAME       Run name suffix for stage 2.
  --max-auto-restarts N          train_autoresume watchdog restart limit.
  --restart-delay SECONDS        Delay before watchdog restart.
  --venv PATH                    Virtualenv root containing bin/activate.
  --isaaclab-path PATH           IsaacLab root containing isaaclab.sh.
  --no-activate                  Do not auto-source the virtualenv.
  --help                         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --init-checkpoint)
            INIT_CKPT="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --stage1-max-iterations)
            STAGE1_MAX_ITERATIONS="$2"
            shift 2
            ;;
        --stage2-max-iterations)
            STAGE2_MAX_ITERATIONS="$2"
            shift 2
            ;;
        --stage1-run-prefix)
            STAGE1_RUN_PREFIX="$2"
            shift 2
            ;;
        --stage2-run-prefix)
            STAGE2_RUN_PREFIX="$2"
            shift 2
            ;;
        --max-auto-restarts)
            MAX_AUTO_RESTARTS="$2"
            shift 2
            ;;
        --restart-delay)
            RESTART_DELAY="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --isaaclab-path)
            ISAACLAB_ROOT="$2"
            shift 2
            ;;
        --no-activate)
            AUTO_ACTIVATE="0"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "${INIT_CKPT}" && -f "${REPO_ROOT}/${INIT_CKPT}" ]]; then
    INIT_CKPT="${REPO_ROOT}/${INIT_CKPT}"
fi

if [[ ! -f "${INIT_CKPT}" ]]; then
    echo "Initial checkpoint not found: ${INIT_CKPT}" >&2
    exit 1
fi

if [[ "${AUTO_ACTIVATE}" == "1" ]]; then
    ACTIVATE_SCRIPT="${VENV_PATH}/bin/activate"
    if [[ -f "${ACTIVATE_SCRIPT}" ]]; then
        # shellcheck disable=SC1090
        source "${ACTIVATE_SCRIPT}"
    else
        echo "Virtualenv activate script not found: ${ACTIVATE_SCRIPT}" >&2
        exit 1
    fi
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-yes}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"
if [[ ! -d "${ISAACLAB_ROOT}" && -d "${REPO_ROOT}/../IsaacLab" ]]; then
    ISAACLAB_ROOT="${REPO_ROOT}/../IsaacLab"
fi
export ISAACLAB_PATH="${ISAACLAB_ROOT}"

normalize_experiment_name() {
    local task_name="$1"
    task_name="${task_name,,}"
    task_name="${task_name//-/_}"
    task_name="${task_name%_play}"
    printf '%s\n' "${task_name}"
}

find_latest_run_dir() {
    local experiment_root="$1"
    local run_prefix="$2"

    if [[ ! -d "${experiment_root}" ]]; then
        return 0
    fi

    if [[ -n "${run_prefix}" ]]; then
        find "${experiment_root}" -maxdepth 1 -mindepth 1 -type d -name "*_${run_prefix}" | sort | tail -1
    else
        find "${experiment_root}" -maxdepth 1 -mindepth 1 -type d | sort | tail -1
    fi
}

find_latest_checkpoint_in_run_dir() {
    local run_dir="$1"
    if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
        return 0
    fi
    find "${run_dir}" -type f -name 'model_*.pt' | sort -V | tail -1
}

run_stage() {
    local task="$1"
    local run_prefix="$2"
    local max_iterations="$3"
    local init_checkpoint="$4"

    echo "===================================================================="
    echo "Task: ${task}"
    echo "Run prefix: ${run_prefix}"
    echo "Max iterations: ${max_iterations}"
    echo "Init checkpoint: ${init_checkpoint}"
    echo "===================================================================="

    python scripts/rsl_rl/train_autoresume.py \
        --max_auto_restarts "${MAX_AUTO_RESTARTS}" \
        --restart_delay "${RESTART_DELAY}" \
        --headless \
        --task "${task}" \
        --num_envs "${NUM_ENVS}" \
        --max_iterations "${max_iterations}" \
        --init_checkpoint "${init_checkpoint}" \
        --run_name "${run_prefix}"
}

STAGE1_EXPERIMENT_NAME="$(normalize_experiment_name "${STAGE1_TASK}")"
STAGE1_EXPERIMENT_ROOT="logs/rsl_rl/${STAGE1_EXPERIMENT_NAME}"
STAGE2_EXPERIMENT_NAME="$(normalize_experiment_name "${STAGE2_TASK}")"
STAGE2_EXPERIMENT_ROOT="logs/rsl_rl/${STAGE2_EXPERIMENT_NAME}"

run_stage "${STAGE1_TASK}" "${STAGE1_RUN_PREFIX}" "${STAGE1_MAX_ITERATIONS}" "${INIT_CKPT}"

STAGE1_RUN_DIR="$(find_latest_run_dir "${STAGE1_EXPERIMENT_ROOT}" "${STAGE1_RUN_PREFIX}")"
STAGE1_LATEST_CKPT="$(find_latest_checkpoint_in_run_dir "${STAGE1_RUN_DIR}")"

if [[ -z "${STAGE1_LATEST_CKPT}" || ! -f "${STAGE1_LATEST_CKPT}" ]]; then
    echo "Failed to locate the latest stage-1 checkpoint under: ${STAGE1_EXPERIMENT_ROOT}" >&2
    exit 1
fi

echo "Stage 1 latest checkpoint: ${STAGE1_LATEST_CKPT}"

run_stage "${STAGE2_TASK}" "${STAGE2_RUN_PREFIX}" "${STAGE2_MAX_ITERATIONS}" "${STAGE1_LATEST_CKPT}"

STAGE2_RUN_DIR="$(find_latest_run_dir "${STAGE2_EXPERIMENT_ROOT}" "${STAGE2_RUN_PREFIX}")"
STAGE2_LATEST_CKPT="$(find_latest_checkpoint_in_run_dir "${STAGE2_RUN_DIR}")"

if [[ -n "${STAGE2_LATEST_CKPT}" && -f "${STAGE2_LATEST_CKPT}" ]]; then
    echo "Stage 2 latest checkpoint: ${STAGE2_LATEST_CKPT}"
fi
