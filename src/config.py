import torch

# ==== Object detector configuration ====
# object detector model types
MODEL_TYPES = {"owlvit": "google/owlvit-base-patch32",
    "grounding_dino": "IDEA-Research/grounding-dino-tiny",}

DEFAULT_DETECTOR = "grounding_dino"
INV_MODEL_TYPES = {v:k for k,v in MODEL_TYPES.items()} # key, value 뒤집은 딕셔너리

# ==== VLM configuration ====
CONCEPT_EXTRACTION_VLM = "gpt-4.1"  # 초기 개념 추출
CRITIQUE_VLM = "gpt-4o"            # 쿼리 비평 및 개선
VALIDATION_VLM = "gpt-4.1"         # 최종 검증

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.2


# Visualization
COLOR_PALETTE = ["red", "blue", "green", "purple", "orange", 
    "cyan", "magenta", "yellow", "brown", "pink"]
FONT_PATH = "fonts/arial.ttf"