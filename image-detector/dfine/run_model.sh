#!/bin/bash
# Wrapper script to run the model with proper signal handling

# Store the PID file location
PID_FILE="/tmp/clarifai_runner_dfine.pid"

# Function to cleanup on exit
cleanup() {
    echo "Shutting down Clarifai runner..."
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        # Kill the process group to ensure all child processes are terminated
        pkill -P $PID 2>/dev/null
        kill $PID 2>/dev/null
        rm -f "$PID_FILE"
        echo "Runner stopped successfully"
    fi
    exit 0
}

# Register signal handlers
trap cleanup SIGINT SIGTERM

# Start the runner and save its PID
echo "Starting Clarifai model local runner..."
echo "Press Ctrl+C to stop"
clarifai model local-runner . &
RUNNER_PID=$!

# Save PID to file
echo $RUNNER_PID > "$PID_FILE"

# Wait for the process
wait $RUNNER_PID

# Cleanup when process exits normally
cleanup
