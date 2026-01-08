#!/usr/bin/env python3
"""
Script to download Cosmos-Reason2-8B model weights from Hugging Face.
Run this script on RunPod or any machine where you want to use the model.
"""

import os
import argparse
from huggingface_hub import snapshot_download, login
from pathlib import Path


def download_model(
    model_id: str = "nvidia/Cosmos-Reason2-8B",
    local_dir: str = None,
    token: str = None,
    resume: bool = True
):
    """
    Download model weights from Hugging Face Hub.
    
    Args:
        model_id: Hugging Face model ID
        local_dir: Local directory to save the model (defaults to current directory)
        token: Hugging Face token for gated models
        resume: Whether to resume partial downloads
    """
    
    if local_dir is None:
        local_dir = Path(__file__).parent
    else:
        local_dir = Path(local_dir)
    
    # Login if token provided
    if token:
        login(token=token)
        print("✓ Logged in to Hugging Face")
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])
        print("✓ Logged in using HF_TOKEN environment variable")
    else:
        print("⚠ No token provided. If this is a gated model, you may need to login first.")
        print("  Run: huggingface-cli login")
        print("  Or set HF_TOKEN environment variable")
    
    print(f"\n📥 Downloading {model_id} to {local_dir}")
    print("This may take a while depending on your internet connection...\n")
    
    # Files to download (weights and essential configs)
    # We skip files that are already in the repo
    allow_patterns = [
        "*.safetensors",
        "model.safetensors.index.json",
    ]
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            resume_download=resume,
            local_dir_use_symlinks=False,
        )
        print("\n✓ Download complete!")
        print(f"  Model saved to: {local_dir}")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have accepted the model license on Hugging Face")
        print("2. Check your internet connection")
        print("3. Ensure you have enough disk space (~16GB for weights)")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download Cosmos-Reason2-8B model weights from Hugging Face"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="nvidia/Cosmos-Reason2-8B",
        help="Hugging Face model ID (default: nvidia/Cosmos-Reason2-8B)"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory to save the model (default: current script directory)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for gated models (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume partial downloads, start fresh"
    )
    
    args = parser.parse_args()
    
    download_model(
        model_id=args.model_id,
        local_dir=args.local_dir,
        token=args.token,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
