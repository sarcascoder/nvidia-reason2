#!/bin/bash
# Script to run inference with Cosmos-Reason2-8B

# Set default values
MODEL_PATH="${MODEL_PATH:-$(dirname "$0")}"
VIDEO_PATH=""
IMAGE_PATH=""
PROMPT=""
SYSTEM_PROMPT=""
FPS=4
MAX_TOKENS=4096

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --system-prompt)
            SYSTEM_PROMPT="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --video PATH       Path to video file"
            echo "  --image PATH       Path to image file"
            echo "  --prompt TEXT      Question/prompt for the model"
            echo "  --system-prompt    System prompt"
            echo "  --model-path PATH  Path to model directory"
            echo "  --fps NUM          FPS for video (default: 4)"
            echo "  --max-tokens NUM   Max tokens to generate (default: 4096)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ -z "$VIDEO_PATH" ] && [ -z "$IMAGE_PATH" ]; then
    echo "Error: Please provide either --video or --image"
    exit 1
fi

if [ -z "$PROMPT" ]; then
    echo "Error: Please provide a --prompt"
    exit 1
fi

# Build command
CMD="python inference.py"
CMD="$CMD --model-path \"$MODEL_PATH\""
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD --fps $FPS"
CMD="$CMD --max-tokens $MAX_TOKENS"

if [ -n "$VIDEO_PATH" ]; then
    CMD="$CMD --video \"$VIDEO_PATH\""
fi

if [ -n "$IMAGE_PATH" ]; then
    CMD="$CMD --image \"$IMAGE_PATH\""
fi

if [ -n "$SYSTEM_PROMPT" ]; then
    CMD="$CMD --system-prompt \"$SYSTEM_PROMPT\""
fi

# Run inference
echo "Running inference..."
eval $CMD
