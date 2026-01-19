#!/bin/bash
# Script to forcefully stop any running Clarifai runner processes

echo "Stopping Clarifai runner processes..."

# Kill by PID file if it exists
PID_FILE="/tmp/clarifai_runner_dfine.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Found PID file: $PID"
    pkill -P $PID 2>/dev/null
    kill $PID 2>/dev/null
    rm -f "$PID_FILE"
fi

# Kill any remaining clarifai.runners.server processes
PIDS=$(pgrep -f "clarifai.runners.server")
if [ -n "$PIDS" ]; then
    echo "Killing remaining clarifai runner processes: $PIDS"
    kill -9 $PIDS 2>/dev/null
    echo "Processes killed"
else
    echo "No clarifai runner processes found"
fi

echo "Done"
