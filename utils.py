#!/usr/bin/env python3
"""
Utility functions for Cosmos-Reason2-8B model.
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple


def parse_thinking_response(response: str) -> Tuple[Optional[str], str]:
    """
    Parse response to extract thinking and answer sections.
    
    Args:
        response: Raw model response
        
    Returns:
        Tuple of (thinking, answer)
    """
    thinking = None
    answer = response
    
    # Extract thinking section
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        # Get everything after </think>
        answer = response[think_match.end():].strip()
    
    # Extract answer section if present
    answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return thinking, answer


def format_response_markdown(response: str) -> str:
    """
    Format response with markdown styling.
    
    Args:
        response: Raw model response
        
    Returns:
        Formatted markdown string
    """
    thinking, answer = parse_thinking_response(response)
    
    formatted = ""
    
    if thinking:
        formatted += "### 🧠 Reasoning Process\n\n"
        formatted += f"```\n{thinking}\n```\n\n"
    
    formatted += "### ✅ Answer\n\n"
    formatted += answer
    
    return formatted


def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate a video file.
    
    Args:
        file_path: Path to video file
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not file_path:
        return False, "No file path provided"
    
    path = Path(file_path.replace("file://", ""))
    
    if not path.exists():
        return False, f"File not found: {path}"
    
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    if path.suffix.lower() not in valid_extensions:
        return False, f"Invalid video format: {path.suffix}. Supported: {valid_extensions}"
    
    # Check file size (warn if very large)
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 500:
        return True, f"Warning: Large file ({size_mb:.1f}MB). Processing may be slow."
    
    return True, "Valid video file"


def validate_image_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate an image file.
    
    Args:
        file_path: Path to image file
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not file_path:
        return False, "No file path provided"
    
    path = Path(file_path.replace("file://", ""))
    
    if not path.exists():
        return False, f"File not found: {path}"
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    if path.suffix.lower() not in valid_extensions:
        return False, f"Invalid image format: {path.suffix}. Supported: {valid_extensions}"
    
    return True, "Valid image file"


def estimate_tokens_from_video(video_path: str, fps: int = 4) -> int:
    """
    Estimate the number of tokens a video will generate.
    
    Args:
        video_path: Path to video file
        fps: Frames per second for extraction
        
    Returns:
        Estimated token count
    """
    try:
        import cv2
        
        path = video_path.replace("file://", "")
        cap = cv2.VideoCapture(path)
        
        if not cap.isOpened():
            return 0
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        cap.release()
        
        # Estimate frames to be extracted
        extracted_frames = int(duration * fps)
        
        # Each frame typically generates ~256-576 tokens depending on resolution
        # Using conservative estimate
        tokens_per_frame = 400
        
        return extracted_frames * tokens_per_frame
        
    except Exception:
        return 0


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with GPU memory info
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {"available": False, "message": "CUDA not available"}
        
        device_count = torch.cuda.device_count()
        devices = []
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            
            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(memory_total, 2),
                "reserved_memory_gb": round(memory_reserved, 2),
                "allocated_memory_gb": round(memory_allocated, 2),
                "free_memory_gb": round(memory_total - memory_reserved, 2),
            })
        
        return {
            "available": True,
            "device_count": device_count,
            "devices": devices
        }
        
    except Exception as e:
        return {"available": False, "message": str(e)}


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    try:
        import torch
        import gc
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    except Exception:
        pass


if __name__ == "__main__":
    # Test utilities
    print("GPU Memory Info:")
    print(get_gpu_memory_info())
    
    print("\nTest response parsing:")
    test_response = """<think>
Let me analyze this step by step.
First, I see a car approaching.
The traffic light is red.
</think>

Based on my analysis, it is not safe to proceed because the traffic light is red.
"""
    
    thinking, answer = parse_thinking_response(test_response)
    print(f"Thinking: {thinking[:50]}...")
    print(f"Answer: {answer}")
