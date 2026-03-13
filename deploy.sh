#!/usr/bin/env bash
set -euo pipefail

REMOTE="hoppetusse"
REMOTE_DIR="dev/python/rl-tidbits"

usage() {
    echo "Usage: $0 <script> [args...]"
    echo "       $0 --fetch <results_dir>  # Fetch results from remote"
    echo "       $0 --analyze <results_file>  # Analyze local results"
    echo ""
    echo "Examples:"
    echo "  $0 jax_boids/run_random_sweep.py --n-configs 100"
    echo "  $0 jax_boids/train_single.py"
    echo "  $0 --fetch runs/expanded_random_sweep"
    echo "  $0 --analyze runs/expanded_random_sweep/results.json"
    echo ""
    echo "Check progress with:"
    echo "  ssh $REMOTE \"tail -f $REMOTE_DIR/\$(basename \$1 .py)_output.log\""
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

SCRIPT="$1"
shift
ARGS="$*"
LOG_FILE="$(basename "$SCRIPT" .py)_output.log"

# Handle special commands
if [ "$SCRIPT" = "--fetch" ]; then
    if [ $# -lt 1 ]; then
        echo "Error: --fetch requires a results directory path"
        usage
    fi
    RESULTS_DIR="$1"
    echo "Fetching results from $REMOTE:$REMOTE_DIR/$RESULTS_DIR..."
    rsync -avz "$REMOTE:$REMOTE_DIR/$RESULTS_DIR/" "./$RESULTS_DIR/"
    echo "Results fetched to .$RESULTS_DIR/"
    exit 0
fi

if [ "$SCRIPT" = "--analyze" ]; then
    if [ $# -lt 1 ]; then
        echo "Error: --analyze requires a results file path"
        usage
    fi
    RESULTS_FILE="$1"
    echo "Analyzing $RESULTS_FILE..."
    uv run python -m jax_boids.analyze_sweep "$RESULTS_FILE" "$@"
    exit 0
fi

echo "Ensuring remote directory exists..."
ssh "$REMOTE" "mkdir -p $REMOTE_DIR"

echo "Syncing code to $REMOTE:$REMOTE_DIR..."
rsync -avz --delete --exclude '.git' --exclude '.venv' --exclude 'runs' --exclude 'tuning' --exclude '__pycache__' --exclude '*.pyc' \
    ./ "$REMOTE:$REMOTE_DIR/"

# Kill any existing process running the same script
ssh "$REMOTE" "pkill -9 -f 'python $SCRIPT' 2>/dev/null || true"
sleep 1

echo "Starting '$SCRIPT $ARGS' on $REMOTE in background..."
ssh "$REMOTE" "bash -c 'cd $REMOTE_DIR && nohup env PYTHONPATH=. uv run python $SCRIPT $ARGS > $LOG_FILE 2>&1 & disown'"

echo ""
echo "Training started. Check progress with:"
echo "  ssh $REMOTE \"tail -f $REMOTE_DIR/$LOG_FILE\""
echo ""
PID=$(ssh "$REMOTE" "pgrep -f 'python $SCRIPT' | head -1" 2>/dev/null || echo 'N/A')
echo "PID: $PID"
