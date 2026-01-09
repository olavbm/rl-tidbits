#!/usr/bin/env bash
set -euo pipefail

REMOTE="hoppetusse"
REMOTE_DIR="dev/python/rl-tidbits"

echo "Syncing code to $REMOTE:$REMOTE_DIR..."
rsync -avz --exclude '.git' --exclude '.venv' --exclude 'runs' --exclude 'tuning' --exclude '__pycache__' \
    ./ "$REMOTE:$REMOTE_DIR/"

SESSION_NAME="rl-train"

echo "Starting tuning on $REMOTE in tmux session '$SESSION_NAME'..."
ssh -t "$REMOTE" "cd $REMOTE_DIR && tmux new-session -A -s $SESSION_NAME 'uv run python $*'"
