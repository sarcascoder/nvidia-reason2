#!/usr/bin/env python3
"""
Gradio UI for Cosmos-Reason2-8B model.
Provides a web interface for video/image analysis with predefined roles.
"""

import os
import gradio as gr
import tempfile
import torch
from pathlib import Path
from typing import Optional, Tuple
import logging

from inference import CosmosReason2, load_model
from system_prompts import (
    ROLES,
    get_role_names,
    get_role_by_name,
    format_system_prompt,
    get_roles_for_dropdown,
    parse_dropdown_selection,
    THINKING_FORMAT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model: Optional[CosmosReason2] = None


def get_model() -> CosmosReason2:
    """Get or initialize the model."""
    global model
    if model is None:
        logger.info("Loading model...")
        model_path = os.environ.get("MODEL_PATH", str(Path(__file__).parent))
        model = load_model(model_path=model_path)
        logger.info("Model loaded successfully!")
    return model


def format_response(response: str) -> str:
    """Format the response for display."""
    # Add some styling to think/answer blocks if present
    if "<think>" in response and "</think>" in response:
        response = response.replace("<think>", "### 🧠 Reasoning:\n```\n")
        response = response.replace("</think>", "\n```\n\n### ✅ Answer:\n")
    
    if "<answer>" in response and "</answer>" in response:
        response = response.replace("<answer>", "")
        response = response.replace("</answer>", "")
    
    return response


def analyze_content(
    video_file,
    image_file,
    question: str,
    role_selection: str,
    custom_system_prompt: str,
    use_custom_prompt: bool,
    enable_thinking: bool,
    fps: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    Analyze video or image content with the model.
    
    Returns:
        Tuple of (response, status)
    """
    # Validation
    if not video_file and not image_file:
        return "", "⚠️ Please upload a video or image file."
    
    if not question.strip():
        return "", "⚠️ Please enter a question."
    
    try:
        progress(0.1, desc="Loading model...")
        cosmos_model = get_model()
        
        # Determine system prompt
        role_name = parse_dropdown_selection(role_selection)
        
        if use_custom_prompt and custom_system_prompt.strip():
            system_prompt = custom_system_prompt
            if enable_thinking:
                system_prompt = f"{system_prompt}\n\n{THINKING_FORMAT}"
        else:
            system_prompt = format_system_prompt(role_name, include_thinking=enable_thinking)
        
        progress(0.3, desc="Processing input...")
        
        def _coerce_input_to_path(x):
            """Coerce Gradio media payloads into a path/URL string.

            Observed return formats across Gradio versions:
            - str: "/tmp/gradio/.../video.mp4"
            - dict: {"path": ..., "url": ...}
            - tuple/list: (path, ) or (path, mime)
            - None
            """
            if x is None:
                return None
            if isinstance(x, str):
                return x
            if isinstance(x, dict):
                # gr.Video / gr.Image can return dicts depending on version/config
                return x.get("path") or x.get("name") or x.get("file") or x.get("url")
            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    return None
                # Prefer first element if it is a string or dict
                return _coerce_input_to_path(x[0])
            return x

        # Get file path
        if video_file:
            file_path = _coerce_input_to_path(video_file)
            input_type = "video"
        else:
            file_path = _coerce_input_to_path(image_file)
            input_type = "image"

        if not file_path:
            return "", "⚠️ Could not read uploaded file path. Please re-upload and try again."
        
        progress(0.5, desc="Running inference...")
        
        # Run inference
        if input_type == "video":
            response = cosmos_model.analyze_video(
                video_path=file_path,
                prompt=question,
                system_prompt=system_prompt,
                fps=fps,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            response = cosmos_model.analyze_image(
                image_path=file_path,
                prompt=question,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        
        progress(1.0, desc="Done!")
        
        formatted_response = format_response(response)
        status = f"✅ Analysis complete using role: {role_name}"
        
        return formatted_response, status
        
    except Exception as e:
        logger.exception("Error during analysis")
        return "", f"❌ Error: {str(e)}"


def update_system_prompt_preview(role_selection: str, enable_thinking: bool) -> str:
    """Update the system prompt preview based on role selection."""
    role_name = parse_dropdown_selection(role_selection)
    return format_system_prompt(role_name, include_thinking=enable_thinking)


def create_example_questions(role_selection: str) -> str:
    """Get example questions for the selected role."""
    role_name = parse_dropdown_selection(role_selection)
    
    examples = {
        "General Assistant": "Describe what is happening in this video and explain the sequence of events.",
        "Real Estate Quality Inspector": """Please inspect this property video/image and provide a detailed quality assessment report including:
1. Overall condition rating (Excellent/Good/Fair/Poor)
2. List of any defects or issues found with severity levels
3. Specific areas requiring attention
4. Recommendations for repairs or improvements""",
        "Autonomous Vehicle Safety Analyst": "Analyze this driving scenario. Is it safe to proceed? What potential hazards do you identify?",
        "Robotics Task Planner": "Analyze the scene and describe step-by-step how a robot should manipulate the objects to complete the task.",
        "Video Analytics Expert": "Provide a detailed timeline analysis of this video, identifying key events and their timestamps.",
        "Manufacturing QC Inspector": "Inspect this manufacturing process/product and identify any quality issues or defects.",
        "Security Surveillance Analyst": "Analyze this surveillance footage and describe any notable activities or potential security concerns.",
        "Sports Performance Analyst": "Analyze the athletic performance in this video. What techniques are being used and how could they be improved?",
        "Medical/Healthcare Assistant": "Describe what is shown in this medical/healthcare related visual content. (Note: This is for educational purposes only, not diagnosis.)",
        "Educational Content Analyzer": "Analyze this educational content and provide a summary with key learning points."
    }
    
    return examples.get(role_name, examples["General Assistant"])


# Custom CSS
custom_css = """
.main-title {
    text-align: center;
    margin-bottom: 10px;
}

.role-description {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}

.status-box {
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}

.thinking-block {
    background-color: #e8f4f8;
    padding: 10px;
    border-radius: 5px;
}
"""


def create_ui():
    """Create the Gradio UI."""
    
    # Gradio 6.0 moved theme/css from Blocks() to launch(). Keep Blocks() minimal for compatibility.
    with gr.Blocks(title="Cosmos-Reason2-8B") as demo:
        
        gr.Markdown(
            """
            # 🌌 Cosmos-Reason2-8B
            ### NVIDIA's Physical AI Reasoning Model
            
            Upload a video or image and ask questions. The model will analyze the content and provide detailed reasoning.
            
            **Features:**
            - 🎬 Video understanding with temporal reasoning
            - 🖼️ Image analysis
            - 🧠 Chain-of-thought reasoning
            - 🎭 Multiple specialized roles
            """,
            elem_classes=["main-title"]
        )
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                
                with gr.Tabs():
                    with gr.TabItem("🎬 Video"):
                        video_input = gr.Video(
                            label="Upload Video",
                            sources=["upload"],
                        )
                    
                    with gr.TabItem("🖼️ Image"):
                        image_input = gr.Image(
                            label="Upload Image",
                            type="filepath",
                            sources=["upload"],
                        )
                
                # Role selection
                gr.Markdown("### 🎭 Role Selection")
                role_dropdown = gr.Dropdown(
                    choices=get_roles_for_dropdown(),
                    value=get_roles_for_dropdown()[0],
                    label="Select Role",
                    interactive=True
                )
                
                role_description = gr.Markdown(
                    value=f"**Description:** {ROLES['General Assistant']['description']}",
                    elem_classes=["role-description"]
                )
                
                # Question input
                question_input = gr.Textbox(
                    label="Question / Instruction",
                    placeholder="Enter your question about the video/image...",
                    lines=3,
                )
                
                example_btn = gr.Button("📝 Load Example Question", size="sm")
                
                # Advanced options
                with gr.Accordion("⚙️ Advanced Options", open=False):
                    enable_thinking = gr.Checkbox(
                        label="Enable Chain-of-Thought Reasoning",
                        value=True,
                        info="Model will show its reasoning process"
                    )
                    
                    use_custom_prompt = gr.Checkbox(
                        label="Use Custom System Prompt",
                        value=False,
                    )
                    
                    custom_system_prompt = gr.Textbox(
                        label="Custom System Prompt",
                        placeholder="Enter your custom system prompt...",
                        lines=4,
                        visible=False
                    )
                    
                    fps_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Video FPS",
                        info="Frames per second to extract from video"
                    )
                    
                    max_tokens_slider = gr.Slider(
                        minimum=256,
                        maximum=8192,
                        value=4096,
                        step=256,
                        label="Max Output Tokens",
                    )
                    
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    
                    top_p_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                        label="Top-p",
                    )
                
                # System prompt preview
                with gr.Accordion("👁️ System Prompt Preview", open=False):
                    system_prompt_preview = gr.Textbox(
                        label="Current System Prompt",
                        value=format_system_prompt("General Assistant", True),
                        lines=10,
                        interactive=False
                    )
                
                # Submit button
                submit_btn = gr.Button(
                    "🚀 Analyze",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Output
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Result")
                
                status_output = gr.Markdown(
                    value="Ready to analyze...",
                    elem_classes=["status-box"]
                )
                
                response_output = gr.Markdown(
                    label="Response",
                    value="",
                )
                
                # Copy button
                copy_btn = gr.Button("📋 Copy Response", size="sm")
        
        # Footer
        gr.Markdown(
            """
            ---
            **Note:** This model requires significant GPU memory (~32GB). 
            For best results, use FPS=4 for videos.
            
            Built with NVIDIA Cosmos-Reason2-8B | [Model Card](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
            """
        )
        
        # Event handlers
        def update_role_description(selection):
            role_name = parse_dropdown_selection(selection)
            role = get_role_by_name(role_name)
            return f"**{role['icon']} Description:** {role['description']}"
        
        role_dropdown.change(
            fn=update_role_description,
            inputs=[role_dropdown],
            outputs=[role_description]
        )
        
        role_dropdown.change(
            fn=update_system_prompt_preview,
            inputs=[role_dropdown, enable_thinking],
            outputs=[system_prompt_preview]
        )
        
        enable_thinking.change(
            fn=update_system_prompt_preview,
            inputs=[role_dropdown, enable_thinking],
            outputs=[system_prompt_preview]
        )
        
        use_custom_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_custom_prompt],
            outputs=[custom_system_prompt]
        )
        
        example_btn.click(
            fn=create_example_questions,
            inputs=[role_dropdown],
            outputs=[question_input]
        )
        
        submit_btn.click(
            fn=analyze_content,
            inputs=[
                video_input,
                image_input,
                question_input,
                role_dropdown,
                custom_system_prompt,
                use_custom_prompt,
                enable_thinking,
                fps_slider,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
            ],
            outputs=[response_output, status_output]
        )
        
        # Copy functionality (JavaScript)
        copy_btn.click(
            fn=None,
            inputs=[response_output],
            outputs=[],
            js="(text) => { navigator.clipboard.writeText(text); alert('Copied to clipboard!'); }"
        )
    
    return demo


def main():
    """Launch the Gradio app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Cosmos-Reason2-8B UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model")
    
    args = parser.parse_args()
    
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    
    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=custom_css,
    )


if __name__ == "__main__":
    main()
