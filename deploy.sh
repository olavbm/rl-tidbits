#!/usr/bin/env bash
set -euo pipefail

REMOTE="hoppetusse"
REMOTE_DIR="dev/python/rl-tidbits"

echo "Syncing code to $REMOTE:$REMOTE_DIR..."
rsync -avz --exclude '.git' --exclude '.venv' --exclude 'runs' --exclude 'tuning' --exclude '__pycache__' \
    ./ "$REMOTE:$REMOTE_DIR/"

echo "Starting tuning on $REMOTE..."
ssh -t "$REMOTE" "cd $REMOTE_DIR && uv run python tune.py $*"
