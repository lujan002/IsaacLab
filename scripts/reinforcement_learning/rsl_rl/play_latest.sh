#!/usr/bin/env bash
set -euo pipefail

# Play the most recent checkpoint for an RSL-RL run.
#
# Defaults match the current ArticulatedArmRev2 lift task setup.
# Optional args:
#   --task <task_name>
#   --num_envs <n>
#   --logs_dir <path_to_task_logs>
#
# Example:
#   ./scripts/reinforcement_learning/rsl_rl/play_latest.sh
#   ./scripts/reinforcement_learning/rsl_rl/play_latest.sh --num_envs 4

TASK="Isaac-Lift-Cube-ArticulatedArmRev2-v0"
NUM_ENVS="1"
LOGS_DIR="/home/ubuntu-22/Documents/IsaacLab/logs/rsl_rl/franka_lift"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK="$2"
            shift 2
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --logs_dir)
            LOGS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--task <task_name>] [--num_envs <n>] [--logs_dir <path>]"
            exit 1
            ;;
    esac
done

if [[ ! -d "$LOGS_DIR" ]]; then
    echo "Logs directory not found: $LOGS_DIR"
    exit 1
fi

# Find latest run directory by modification time.
LATEST_RUN_DIR="$(ls -1dt "$LOGS_DIR"/*/ 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_RUN_DIR}" ]]; then
    echo "No run directories found in: $LOGS_DIR"
    exit 1
fi

# Pick the highest numbered model_XXXX.pt in latest run.
LATEST_CKPT="$(ls -1 "$LATEST_RUN_DIR"/model_*.pt 2>/dev/null | sed -E 's|.*model_([0-9]+)\.pt|\1 &|' | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
if [[ -z "${LATEST_CKPT}" ]]; then
    echo "No model_*.pt checkpoints found in: $LATEST_RUN_DIR"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

echo "Using run: $LATEST_RUN_DIR"
echo "Using checkpoint: $LATEST_CKPT"
echo "Task: $TASK | num_envs: $NUM_ENVS"

"$REPO_ROOT/isaaclab.sh" -p "$REPO_ROOT/scripts/reinforcement_learning/rsl_rl/play.py" \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --checkpoint "$LATEST_CKPT"
