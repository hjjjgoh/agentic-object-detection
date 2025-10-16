# gradio player of agentic-object-detection

import gradio as gr
import os
import sys
from dotenv import load_dotenv
from PIL import Image

# 프로젝트 루트 추가
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

load_dotenv()

def run_detect_pipeline(input_image, user_request):  # input_image: PIL.Image
    # 임시 파일로 저장
    temp_path = "temp_input_for_gradio.jpg"
    input_image.save(temp_path)

    # pipeline setup
    vlm_tool = VLMTool(api_key=os.getenv("OPENAI_API_KEY"))
    object_detection_tool = ObjectDetectionTool(
        model_id=MODEL_TYPES[DEFAULT_DETECTOR],
        device=DEVICE,
        vlm_tool=vlm_tool,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        concept_detection_model=CONCEPT_EXTRACTION_VLM,
        initial_critique_model=CRITIQUE_VLM,
        final_critique_model=VALIDATION_VLM,
    )
    # run pipeline
    final_img, _ = object_detection_tool.run(temp_path, user_request)

    # image file delete after execution of pipeline
    try:
        os.remove(temp_path)
    except: pass

    return final_img  

# gradio app interfae setting 
app = gr.Interface(
    fn=run_detect_pipeline,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="User Request")],
    outputs=[gr.Image(type="pil", label="Detection Result")],
    title="Agentic Object Detection",

    theme=gr.themes.Monochrome(),
    # 예시 이미지 및 예시 User Request 문구 추가
)
app.launch()