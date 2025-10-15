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

    # 파이프라인 셋업
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
    # 파이프라인 실행
    final_img, _ = object_detection_tool.run(temp_path, user_request)

    # (선택) 임시 파일 삭제
    try:
        os.remove(temp_path)
    except: pass

    # Gradio는 PIL.Image를 반환하는 것이 가장 직관적임
    return final_img  

app = gr.Interface(
    fn=run_detect_pipeline,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="User Request")],
    outputs=gr.Image(type="pil", label="Detection Result"),
    title="Agentic Object Detection",
    theme=gr.themes.Monochrome(),
)
app.launch()