import argparse
import os
import sys
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.vlm_tool import VLMTool
from src.pipeline import ObjectDetectionTool
from src.config import (
    MODEL_TYPES,
    DEFAULT_DETECTOR,
    DEVICE,
    CONFIDENCE_THRESHOLD,
    CONCEPT_EXTRACTION_VLM,
    CRITIQUE_VLM,
    VALIDATION_VLM
)

def main():
    """
    Main function to run the object detection pipeline from the command line.
    """
    parser = argparse.ArgumentParser(description="Run the agentic object detection pipeline.")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The user's request or prompt for object detection."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "output", "final"),
        help="Directory to save the output image."
    )
    parser.add_argument(
        "--no-critique",
        action="store_true",
        help="Disable the critique and refinement step."
    )
    args = parser.parse_args()

    # --- Setup ---
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
        sys.exit(1)

    if not os.path.exists(args.image_path):
        print(f"Error: Image path not found at '{args.image_path}'")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Initialize Tools ---
    vlm_tool = VLMTool(api_key=api_key)
    detector = ObjectDetectionTool(
        model_id=MODEL_TYPES[DEFAULT_DETECTOR],
        device=DEVICE,
        vlm_tool=vlm_tool,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        concept_detection_model=CONCEPT_EXTRACTION_VLM,
        initial_critique_model=CRITIQUE_VLM,
        final_critique_model=VALIDATION_VLM,
    )

    print(f"🖼️  Image: {os.path.basename(args.image_path)}")
    print(f"🗣️  User Request: '{args.prompt}'")
    print(f"⚙️  Using model '{DEFAULT_DETECTOR}' on '{DEVICE}'")
    print("-" * 30)

    # --- Run Pipeline ---
    final_image, result_text = detector.run(
        image_path=args.image_path,
        user_request=args.prompt,
        do_critique=not args.no_critique,
    )

    # --- Process and Save Results ---
    print("\n" + "=" * 30)
    print("✅ FINAL RESULT ✅")
    print("=" * 30)

    if final_image:
        print(result_text)
        
        # Save the final image
        base_name, ext = os.path.splitext(os.path.basename(args.image_path))
        output_filename = f"{base_name}_final{ext}"
        output_path = os.path.join(args.output_dir, output_filename)
        
        final_image.save(output_path)
        print(f"\n💾 Final image saved to: {output_path}")
    else:
        print(result_text)
        print("\nCould not find any objects matching the request after the full process.")

if __name__ == "__main__":
    main()
