#!/usr/bin/env python3
"""
Example script demonstrating usage of Cosmos-Reason2-8B model.
This shows various use cases and how to use different roles.
"""

from inference import load_model
from system_prompts import format_system_prompt, ROLES, get_role_names

def example_basic_usage():
    """Basic example of video analysis."""
    
    print("=" * 60)
    print("Example: Basic Video Analysis")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Analyze a video with a simple question
    response = model.analyze_video(
        video_path="/path/to/your/video.mp4",
        prompt="What is happening in this video? Describe the scene and any actions taking place.",
        fps=4,
        max_new_tokens=4096,
    )
    
    print(response)


def example_with_role():
    """Example using a predefined role."""
    
    print("=" * 60)
    print("Example: Real Estate Quality Inspection")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Get the system prompt for Real Estate Inspector role
    system_prompt = format_system_prompt(
        "Real Estate Quality Inspector",
        include_thinking=True
    )
    
    # Analyze a property video
    response = model.analyze_video(
        video_path="/path/to/property_walkthrough.mp4",
        prompt="""Please conduct a thorough quality inspection of this property and provide:
1. Overall condition assessment
2. List of any defects found with severity ratings
3. Specific areas requiring immediate attention
4. Recommendations for repairs or improvements
5. Estimated quality score (1-10)""",
        system_prompt=system_prompt,
        fps=4,
        max_new_tokens=4096,
    )
    
    print(response)


def example_driving_analysis():
    """Example for autonomous vehicle analysis."""
    
    print("=" * 60)
    print("Example: Driving Safety Analysis")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Get the AV safety analyst system prompt
    system_prompt = format_system_prompt(
        "Autonomous Vehicle Safety Analyst",
        include_thinking=True
    )
    
    response = model.analyze_video(
        video_path="/path/to/driving_video.mp4",
        prompt="Is it safe to turn right at this intersection? Identify all potential hazards and road users.",
        system_prompt=system_prompt,
        fps=4,
        max_new_tokens=4096,
    )
    
    print(response)


def example_image_analysis():
    """Example of image analysis."""
    
    print("=" * 60)
    print("Example: Image Analysis")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    response = model.analyze_image(
        image_path="/path/to/your/image.jpg",
        prompt="Describe this image in detail. What objects do you see and what is their spatial arrangement?",
        max_new_tokens=2048,
    )
    
    print(response)


def example_custom_system_prompt():
    """Example with a custom system prompt."""
    
    print("=" * 60)
    print("Example: Custom System Prompt")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    custom_prompt = """You are a detailed construction inspector. Focus on:
- Structural elements (walls, floors, ceilings)
- Finishing quality (paint, tiles, fixtures)
- Safety compliance
- Workmanship quality

Provide ratings on a 1-5 scale for each area inspected."""
    
    response = model.analyze_video(
        video_path="/path/to/construction_site.mp4",
        prompt="Inspect this construction work and rate the quality of each visible element.",
        system_prompt=custom_prompt,
        fps=4,
        max_new_tokens=4096,
    )
    
    print(response)


def list_available_roles():
    """List all available roles and their descriptions."""
    
    print("=" * 60)
    print("Available Roles")
    print("=" * 60)
    
    for role_name in get_role_names():
        role = ROLES[role_name]
        print(f"\n{role['icon']} {role_name}")
        print(f"   ID: {role['id']}")
        print(f"   Description: {role['description']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cosmos-Reason2 Examples")
    parser.add_argument(
        "--example",
        type=str,
        choices=["basic", "role", "driving", "image", "custom", "list-roles"],
        default="list-roles",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        example_basic_usage()
    elif args.example == "role":
        example_with_role()
    elif args.example == "driving":
        example_driving_analysis()
    elif args.example == "image":
        example_image_analysis()
    elif args.example == "custom":
        example_custom_system_prompt()
    elif args.example == "list-roles":
        list_available_roles()
