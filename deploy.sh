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
    echo "  $0 jax_boids/train.py --mode sweep --n-configs 100"
    echo "  $0 jax_boids/train.py --mode train --config validated_005"
    echo "  $0 jax_boids/train.py --mode validate --config validated_005 --n-seeds 5"
    echo "  $0 --fetch runs/sweep"
    echo "  $0 --analyze runs/sweep/results.json"
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

LOG_FILE="$(basename "$SCRIPT" .py)_output.log"

echo "Ensuring remote directory exists..."
ssh -o LogLevel=ERROR "$REMOTE" "mkdir -p $REMOTE_DIR" 2>/dev/null || true

echo "Syncing code to $REMOTE:$REMOTE_DIR..."
rsync -avz --delete --exclude '.git' --exclude '.venv' --exclude 'runs' --exclude 'tuning' --exclude '__pycache__' --exclude '*.pyc' --exclude '*_output.log' \
    ./ "$REMOTE:$REMOTE_DIR/" 2>/dev/null || true

# Kill any existing process running the same script
ssh -o LogLevel=ERROR "$REMOTE" "pkill -9 -f 'python $SCRIPT' 2>/dev/null || true" 2>/dev/null || true
sleep 1

echo "Running '$SCRIPT $ARGS' on $REMOTE..."

# Run training in foreground, stream output
# XLA_PYTHON_CLIENT_PREALLOCATE=false: don't grab all GPU memory upfront
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.10: limit to 10% of GPU (~2.5GB on 24GB)
ssh -o LogLevel=ERROR "$REMOTE" "bash -c 'cd $REMOTE_DIR && export PYTHONPATH=. && export XLA_PYTHON_CLIENT_PREALLOCATE=false && python $SCRIPT $ARGS 2>&1 | tee $LOG_FILE'" 2>/dev/null
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "Done."
else
    echo "Failed (exit code $EXIT_CODE). See log:"
    echo "  ssh $REMOTE \"cat $REMOTE_DIR/$LOG_FILE\""
fi
