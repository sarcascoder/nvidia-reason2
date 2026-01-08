#!/usr/bin/env python3
"""
Inference module for Cosmos-Reason2-8B model.
Provides functions for loading the model and running inference on videos/images.
"""

import os
import torch
import transformers
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CosmosReason2:
    """
    Cosmos-Reason2-8B model wrapper for easy inference.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        max_memory: Optional[Dict] = None,
    ):
        """
        Initialize the Cosmos-Reason2 model.
        
        Args:
            model_path: Path to the model directory or HuggingFace model ID
            device_map: Device mapping strategy ("auto", "cuda:0", etc.)
            torch_dtype: Data type for model weights (bfloat16 recommended)
            attn_implementation: Attention implementation ("sdpa", "flash_attention_2", "eager")
            max_memory: Maximum memory per device (e.g., {0: "24GB", "cpu": "32GB"})
        """
        if model_path is None:
            model_path = str(Path(__file__).parent)
        
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        logger.info(f"Loading model from {model_path}")
        
        # Load model
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            max_memory=max_memory,
            trust_remote_code=True,
        )
        
        # Load processor
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        logger.info("Model loaded successfully")
        
        # Get device
        if hasattr(self.model, 'device'):
            self.device = self.model.device
        else:
            self.device = next(self.model.parameters()).device
    
    def create_messages(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        fps: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Create message format for the model.
        
        Args:
            prompt: User question/prompt
            video_path: Path to video file
            image_path: Path to image file
            system_prompt: System prompt for the model
            fps: Frames per second for video processing
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Build user content
        user_content = []

        def _normalize_media_uri(p: str) -> str:
            """Accept local paths, file:// URIs, and http(s) URLs.

            Important: Transformers' video loader accepts either a URL or a *local path*.
            Some backends treat "file://..." as an iterable/URL-like string and end up
            in a recursion path. To be maximally compatible, we pass local paths as
            plain filesystem paths by default.
            """
            if p.startswith("http://") or p.startswith("https://"):
                return p
            if p.startswith("file://"):
                # Convert file:// URIs back to local paths
                return p[len("file://"):]
            return os.path.abspath(p)
        
        # Add video if provided
        if video_path:
            video_path = _normalize_media_uri(video_path)
            
            user_content.append({
                "type": "video",
                "video": video_path,
                "fps": fps,
            })
        
        # Add image if provided
        if image_path:
            image_path = _normalize_media_uri(image_path)
            
            user_content.append({
                "type": "image",
                "image": image_path,
            })
        
        # Add text prompt
        user_content.append({
            "type": "text",
            "text": prompt
        })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    @torch.inference_mode()
    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        fps: int = 4,
    ) -> str:
        """
        Generate response from the model.
        
        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            fps: Frames per second for video processing
            
        Returns:
            Generated text response
        """
        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=fps,
        )
        
        # Move to device
        inputs = inputs.to(self.model.device)
        
        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        
        # Trim input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return output_text
    
    def analyze_video(
        self,
        video_path: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        fps: int = 4,
        max_new_tokens: int = 4096,
        **generation_kwargs
    ) -> str:
        """
        Analyze a video with a given prompt.
        
        Args:
            video_path: Path to video file
            prompt: Question or instruction about the video
            system_prompt: System prompt for context
            fps: Frames per second for video processing
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Model's response
        """
        messages = self.create_messages(
            prompt=prompt,
            video_path=video_path,
            system_prompt=system_prompt,
            fps=fps,
        )
        
        return self.generate(
            messages=messages,
            max_new_tokens=max_new_tokens,
            fps=fps,
            **generation_kwargs
        )
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 4096,
        **generation_kwargs
    ) -> str:
        """
        Analyze an image with a given prompt.
        
        Args:
            image_path: Path to image file
            prompt: Question or instruction about the image
            system_prompt: System prompt for context
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Model's response
        """
        messages = self.create_messages(
            prompt=prompt,
            image_path=image_path,
            system_prompt=system_prompt,
        )
        
        return self.generate(
            messages=messages,
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        )


def load_model(
    model_path: Optional[str] = None,
    **kwargs
) -> CosmosReason2:
    """
    Convenience function to load the model.
    
    Args:
        model_path: Path to model directory
        **kwargs: Additional arguments for CosmosReason2
        
    Returns:
        Loaded CosmosReason2 instance
    """
    return CosmosReason2(model_path=model_path, **kwargs)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with Cosmos-Reason2-8B")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--prompt", type=str, required=True, help="Question/prompt")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt")
    parser.add_argument("--model-path", type=str, default=None, help="Model path")
    parser.add_argument("--fps", type=int, default=4, help="FPS for video")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    if not args.video and not args.image:
        print("Please provide either --video or --image")
        exit(1)
    
    # Load model
    model = load_model(model_path=args.model_path)
    
    # Run inference
    if args.video:
        response = model.analyze_video(
            video_path=args.video,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            fps=args.fps,
            max_new_tokens=args.max_tokens,
        )
    else:
        response = model.analyze_image(
            image_path=args.image,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_tokens,
        )
    
    print("\n" + "="*50)
    print("RESPONSE:")
    print("="*50)
    print(response)
