#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_VENV="/mlp_vepfs/share/lyn/try0310/env_isaaclab"
DEFAULT_ISAACLAB_PATH="/mlp_vepfs/share/lyn/try0310/IsaacLab"

STAGE1_TASK="Unitree-G1-29dof-LeftHand-LocoReach-TableTopMultiTouchPairAnchorTight-Clean-v0"
STAGE2_TASK="Unitree-G1-29dof-LeftHand-LocoReach-TableTopFixedAcquireStayAnchorTight-Clean-v0"

STAGE1_RUN_PREFIX="tabletop_pair_anchor_tight_r2_ws73250"
STAGE2_RUN_PREFIX="tabletop_fixed_anchor_tight_from_pair_r2"
VIDEO_LENGTH="1000"
AUTO_ACTIVATE="1"
VENV_PATH="${ISAACLAB_VENV:-${DEFAULT_VENV}}"
ISAACLAB_ROOT="${ISAACLAB_PATH:-${DEFAULT_ISAACLAB_PATH}}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/rsl_rl/record_tabletop_anchor_tight_two_stage_videos.sh [options]

Options:
  --stage1-run-prefix NAME   Stage-1 run name suffix to search.
  --stage2-run-prefix NAME   Stage-2 run name suffix to search.
  --video-length N           Number of play steps to record per video.
  --venv PATH                Virtualenv root containing bin/activate.
  --isaaclab-path PATH       IsaacLab root containing isaaclab.sh.
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
EXPORT_ROOT="${REPO_ROOT}/logs/video_exports"
mkdir -p "${EXPORT_ROOT}"
MANIFEST_PATH="${EXPORT_ROOT}/latest_two_stage_videos.txt"

normalize_experiment_name() {
    local task_name="$1"
    task_name="${task_name,,}"
    task_name="${task_name//-/_}"
    task_name="${task_name%_play}"
    printf '%s\n' "${task_name}"
}

sanitize_name() {
    local name="$1"
    name="${name//\//_}"
    name="${name// /_}"
    printf '%s\n' "${name}"
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

abspath() {
    local path="$1"
    if [[ -z "${path}" ]]; then
        return 0
    fi
    local dir_path
    dir_path="$(dirname -- "${path}")"
    local base_name
    base_name="$(basename -- "${path}")"
    (
        cd -- "${dir_path}" >/dev/null 2>&1 && printf '%s/%s\n' "$(pwd -P)" "${base_name}"
    )
}

build_play_command() {
    local play_script="${REPO_ROOT}/scripts/rsl_rl/play.py"
    local isaaclab_sh="${ISAACLAB_ROOT}/isaaclab.sh"

    if [[ -f "${isaaclab_sh}" ]]; then
        printf 'bash\n%s\n-p\n%s\n' "${isaaclab_sh}" "${play_script}"
    else
        printf 'python\n%s\n' "${play_script}"
    fi
}

record_video() {
    local stage_label="$1"
    local task="$2"
    local run_prefix="$3"

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

    run_dir="$(abspath "${run_dir}")"
    checkpoint="$(abspath "${checkpoint}")"

    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    local video_dir="${run_dir}/videos/play_anchor_tight_${stamp}"
    mkdir -p "${video_dir}"
    video_dir="$(abspath "${video_dir}")"

    echo "====================================================================" >&2
    echo "Task: ${task}" >&2
    echo "Run dir: ${run_dir}" >&2
    echo "Checkpoint: ${checkpoint}" >&2
    echo "Video dir: ${video_dir}" >&2
    echo "====================================================================" >&2

    mapfile -t play_cmd < <(build_play_command)

    "${play_cmd[@]}" \
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
    latest_video="$(abspath "${latest_video}")"

    local export_name
    export_name="$(sanitize_name "${experiment_name}_${run_prefix}_${stamp}.mp4")"
    local export_video="${EXPORT_ROOT}/${export_name}"
    cp -f "${latest_video}" "${export_video}"
    local stable_video="${EXPORT_ROOT}/${stage_label}_latest.mp4"
    cp -f "${latest_video}" "${stable_video}"
    sync
    export_video="$(abspath "${export_video}")"
    stable_video="$(abspath "${stable_video}")"

    if [[ ! -f "${export_video}" ]]; then
        echo "Failed to copy recorded video to ${export_video}" >&2
        exit 1
    fi
    if [[ ! -f "${stable_video}" ]]; then
        echo "Failed to copy recorded video to ${stable_video}" >&2
        exit 1
    fi

    echo "Exported video: ${export_video}" >&2
    ls -lh "${export_video}" >&2
    echo "Stable video: ${stable_video}" >&2
    ls -lh "${stable_video}" >&2

    printf '%s|%s\n' "${export_video}" "${stable_video}"
}

STAGE1_RESULT="$(record_video "stage1" "${STAGE1_TASK}" "${STAGE1_RUN_PREFIX}")"
STAGE2_RESULT="$(record_video "stage2" "${STAGE2_TASK}" "${STAGE2_RUN_PREFIX}")"

STAGE1_VIDEO="${STAGE1_RESULT%%|*}"
STAGE1_VIDEO_STABLE="${STAGE1_RESULT#*|}"
STAGE2_VIDEO="${STAGE2_RESULT%%|*}"
STAGE2_VIDEO_STABLE="${STAGE2_RESULT#*|}"

cat > "${MANIFEST_PATH}" <<EOF
STAGE1_VIDEO=${STAGE1_VIDEO}
STAGE1_VIDEO_STABLE=${STAGE1_VIDEO_STABLE}
STAGE2_VIDEO=${STAGE2_VIDEO}
STAGE2_VIDEO_STABLE=${STAGE2_VIDEO_STABLE}
EOF

echo "STAGE1_VIDEO=${STAGE1_VIDEO}"
echo "STAGE1_VIDEO_STABLE=${STAGE1_VIDEO_STABLE}"
echo "STAGE2_VIDEO=${STAGE2_VIDEO}"
echo "STAGE2_VIDEO_STABLE=${STAGE2_VIDEO_STABLE}"
echo "VIDEO_MANIFEST=${MANIFEST_PATH}"
