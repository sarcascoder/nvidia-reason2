#!/bin/bash
# Setup script for RunPod environment
# Run this script after cloning your repo on RunPod

set -e  # Exit on error

echo "========================================"
echo "Cosmos-Reason2-8B RunPod Setup"
echo "========================================"

# Update system
echo ""
echo "📦 Updating system packages..."
apt-get update -qq

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
apt-get install -y -qq ffmpeg libsm6 libxext6 git-lfs

# Navigate to project directory (adjust if needed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "📂 Working directory: $(pwd)"

# Install Python dependencies
echo ""
echo "🐍 Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠️  HF_TOKEN environment variable not set."
    echo "   You may need to set it for downloading gated models:"
    echo "   export HF_TOKEN=your_token_here"
    echo ""
fi

# Download model weights
echo ""
echo "📥 Downloading model weights from HuggingFace..."
echo "   This may take a while (~16GB)..."
python download_weights.py

# Verify weights
echo ""
echo "🔍 Verifying model files..."
if ls *.safetensors 1> /dev/null 2>&1; then
    echo "✅ Model weights found!"
    ls -lh *.safetensors | head -5
else
    echo "❌ Model weights not found. Please run download_weights.py manually."
    exit 1
fi

echo ""
echo "========================================"
echo "✅ Setup complete!"
echo "========================================"
echo ""
echo "To start the UI, run:"
echo "  ./run_ui.sh"
echo "  OR"
echo "  python app.py --share"
echo ""
echo "To run inference from command line:"
echo "  python inference.py --video /path/to/video.mp4 --prompt 'Your question'"
echo ""
