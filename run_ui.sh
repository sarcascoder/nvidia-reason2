#!/bin/bash
# Script to run the Gradio UI for Cosmos-Reason2-8B

# Set default values
HOST="0.0.0.0"
PORT=7860
SHARE=false
MODEL_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --share)
            SHARE=true
            shift
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --host HOST        Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT        Port to bind to (default: 7860)"
            echo "  --share            Create a public Gradio link"
            echo "  --model-path PATH  Path to model directory"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python app.py --host $HOST --port $PORT"

if [ "$SHARE" = true ]; then
    CMD="$CMD --share"
fi

if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model-path \"$MODEL_PATH\""
fi

# Run UI
echo "Starting Cosmos-Reason2-8B UI..."
echo "Access at: http://$HOST:$PORT"
eval $CMD
