#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_VENV="/mlp_vepfs/share/lyn/try0310/env_isaaclab"

STAGE1_TASK="Unitree-G1-29dof-LeftHand-LocoReach-TableTopMultiTouchPairAnchorTight-Clean-v0"
STAGE2_TASK="Unitree-G1-29dof-LeftHand-LocoReach-TableTopFixedAcquireStayAnchorTight-Clean-v0"

STAGE1_RUN_PREFIX="tabletop_pair_anchor_tight_r2_ws73250"
STAGE2_RUN_PREFIX="tabletop_fixed_anchor_tight_from_pair_r2"
VIDEO_LENGTH="1000"
AUTO_ACTIVATE="1"
VENV_PATH="${ISAACLAB_VENV:-${DEFAULT_VENV}}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/rsl_rl/record_tabletop_anchor_tight_two_stage_videos.sh [options]

Options:
  --stage1-run-prefix NAME   Stage-1 run name suffix to search.
  --stage2-run-prefix NAME   Stage-2 run name suffix to search.
  --video-length N           Number of play steps to record per video.
  --venv PATH                Virtualenv root containing bin/activate.
  --no-activate              Do not auto-source the virtualenv.
  --help                     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage1-run-prefix)
            STAGE1_RUN_PREFIX="$2"
            shift 2
            ;;
        --stage2-run-prefix)
            STAGE2_RUN_PREFIX="$2"
            shift 2
            ;;
        --video-length)
            VIDEO_LENGTH="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
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

find_latest_video_in_dir() {
    local video_dir="$1"
    if [[ -z "${video_dir}" || ! -d "${video_dir}" ]]; then
        return 0
    fi
    find "${video_dir}" -type f -name '*.mp4' | sort | tail -1
}

record_video() {
    local task="$1"
    local run_prefix="$2"

    local experiment_name
    experiment_name="$(normalize_experiment_name "${task}")"
    local experiment_root="logs/rsl_rl/${experiment_name}"
    local run_dir
    run_dir="$(find_latest_run_dir "${experiment_root}" "${run_prefix}")"
    local checkpoint
    checkpoint="$(find_latest_checkpoint_in_run_dir "${run_dir}")"

    if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
        echo "Failed to find run directory under ${experiment_root}" >&2
        exit 1
    fi
    if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
        echo "Failed to find checkpoint under ${run_dir}" >&2
        exit 1
    fi

    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    local video_dir="${run_dir}/videos/play_anchor_tight_${stamp}"
    mkdir -p "${video_dir}"

    echo "====================================================================" >&2
    echo "Task: ${task}" >&2
    echo "Run dir: ${run_dir}" >&2
    echo "Checkpoint: ${checkpoint}" >&2
    echo "Video dir: ${video_dir}" >&2
    echo "====================================================================" >&2

    python scripts/rsl_rl/play.py \
        --headless \
        --task "${task}" \
        --checkpoint "${checkpoint}" \
        --video \
        --video_folder "${video_dir}" \
        --video_length "${VIDEO_LENGTH}" \
        --num_envs 1 \
        --enable_cameras \
        1>&2

    local latest_video
    latest_video="$(find_latest_video_in_dir "${video_dir}")"
    if [[ -z "${latest_video}" || ! -f "${latest_video}" ]]; then
        echo "Failed to find recorded video under ${video_dir}" >&2
        exit 1
    fi

    printf '%s\n' "${latest_video}"
}

STAGE1_VIDEO="$(record_video "${STAGE1_TASK}" "${STAGE1_RUN_PREFIX}")"
STAGE2_VIDEO="$(record_video "${STAGE2_TASK}" "${STAGE2_RUN_PREFIX}")"

echo "STAGE1_VIDEO=${STAGE1_VIDEO}"
echo "STAGE2_VIDEO=${STAGE2_VIDEO}"
