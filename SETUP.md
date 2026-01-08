# Cosmos-Reason2-8B Setup Guide

This guide explains how to set up and run the Cosmos-Reason2-8B model with the Gradio UI.

## Quick Start on RunPod

1. **Create a RunPod instance** with at least 32GB GPU memory (A100 or H100 recommended)

2. **Clone your repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Cosmos-Reason2-8B.git
   cd Cosmos-Reason2-8B
   ```

3. **Set your HuggingFace token** (required for gated model):
   ```bash
   export HF_TOKEN=your_huggingface_token
   ```
   
   > Note: You need to accept the model license at https://huggingface.co/nvidia/Cosmos-Reason2-8B first

4. **Run the setup script:**
   ```bash
   chmod +x setup_runpod.sh
   ./setup_runpod.sh
   ```

5. **Start the UI:**
   ```bash
   python app.py --share
   ```
   This will give you a public Gradio link you can access from anywhere.

## Manual Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU with 32GB+ VRAM
- NVIDIA drivers with CUDA 11.8+

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download Model Weights

```bash
# Option 1: Using the download script
python download_weights.py --token YOUR_HF_TOKEN

# Option 2: Using huggingface-cli
huggingface-cli login
huggingface-cli download nvidia/Cosmos-Reason2-8B --local-dir .
```

### Step 3: Verify Installation

```bash
# Check that model weights are present
ls -la *.safetensors
```

You should see 4 safetensor files (~4GB each).

## Running the Application

### Gradio UI (Recommended)

```bash
# Basic launch
python app.py

# With public sharing link
python app.py --share

# Custom host and port
python app.py --host 0.0.0.0 --port 7860
```

Access the UI at `http://localhost:7860` or use the public link if `--share` is enabled.

### Command Line Inference

```bash
# Analyze a video
python inference.py \
    --video /path/to/video.mp4 \
    --prompt "Describe what is happening in this video"

# Analyze an image
python inference.py \
    --image /path/to/image.jpg \
    --prompt "What do you see in this image?"

# With custom system prompt
python inference.py \
    --video /path/to/video.mp4 \
    --prompt "Is it safe to proceed?" \
    --system-prompt "You are a driving safety expert."
```

## Project Structure

```
Cosmos-Reason2-8B/
├── app.py                  # Gradio UI application
├── inference.py            # Core inference module
├── system_prompts.py       # Predefined roles and prompts
├── download_weights.py     # Script to download model weights
├── requirements.txt        # Python dependencies
├── setup_runpod.sh        # RunPod setup script
├── run_ui.sh              # UI launch script
├── run_inference.sh       # CLI inference script
├── .gitignore             # Git ignore file (excludes weights)
├── SETUP.md               # This file
├── README.md              # Model documentation
└── config files...        # Model configuration files
```

## Available Roles

The UI includes several predefined roles:

| Role | Description |
|------|-------------|
| 🤖 General Assistant | General-purpose video/image analysis |
| 🏠 Real Estate Quality Inspector | **Property quality inspection** |
| 🚗 Autonomous Vehicle Safety Analyst | Driving scenario analysis |
| 🤖 Robotics Task Planner | Robot manipulation planning |
| 📊 Video Analytics Expert | Video content analysis |
| 🏭 Manufacturing QC Inspector | Manufacturing quality control |
| 🔒 Security Surveillance Analyst | Surveillance footage analysis |
| ⚽ Sports Performance Analyst | Athletic performance analysis |
| 🏥 Medical/Healthcare Assistant | Healthcare visual analysis |
| 📚 Educational Content Analyzer | Educational content analysis |

## Configuration Options

### Video Processing
- **FPS**: Default 4 (recommended). Higher FPS = more detail but slower processing
- **Max Tokens**: Default 4096. Increase for longer responses

### Generation Parameters
- **Temperature**: Controls randomness (0.7 default)
- **Top-p**: Nucleus sampling parameter (0.8 default)

## Troubleshooting

### Out of Memory Error
- Reduce FPS from 4 to 2
- Use a smaller video
- Ensure you have 32GB+ VRAM

### Model Not Loading
- Verify all safetensor files are present
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Slow Inference
- Use FPS=2 or FPS=4 for videos
- Shorter videos process faster
- H100 > A100 > A10 for speed

## GPU Memory Requirements

| GPU | Status |
|-----|--------|
| H100 (80GB) | ✅ Recommended |
| A100 (80GB) | ✅ Recommended |
| A100 (40GB) | ✅ Works |
| A10 (24GB) | ⚠️ May need optimizations |
| RTX 4090 (24GB) | ⚠️ May need optimizations |

## License

This project uses NVIDIA's Cosmos-Reason2-8B model under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license).

---

For more information, see the [official model page](https://huggingface.co/nvidia/Cosmos-Reason2-8B).
