# Agentic Object Detection

An AI-powered object detection system that understands natural language requests and accurately finds desired objects in images using Visual Language Models (VLMs).

## Features

- 🔥 Complete agentic object detection pipeline from scratch
- 🚀 Multi-stage validation: Detection → Critique → Re-detection → Validation
- 📊 Built-in VLM integration for concept extraction and validation
- 📊 Interactive Gradio interface for real-time testing
- 🎨 Customizable themes and model configurations

## Installation

### Prerequisites

- Python >= 3.8
- CUDA-compatible GPU (recommended)
- OpenAI API key

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd agentic-object-detection

# Install dependencies
pip install -e .

# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

### 1. Web Interface

Launch the interactive Gradio interface:

```bash
python sctipts/app.py
```

Open your browser to `http://127.0.0.1:7860` and start detecting objects!

### 2. Command Line Interface

Run object detection from command line:

```bash
python sctipts/run.py --image data/image1.jpg --request "Detect the green tomatoes"
```

### 3. Programmatic Usage

```python
from src.pipeline import ObjectDetectionTool
from src.vlm_tool import VLMTool

# Initialize tools
vlm_tool = VLMTool(api_key="your_api_key")
detector = ObjectDetectionTool(
    model_id="owlvit",
    device="cuda",
    vlm_tool=vlm_tool
)

# Run detection
final_img, processed_text = detector.run(
    image_path="path/to/image.jpg",
    user_request="Detect all cars in the image"
)
```

### 4. Jupyter Notebooks

Explore the notebooks for experimentation:

```bash
# Start Jupyter Lab
jupyter lab

# Open notebooks in the notebook/ directory
```

## Project Structure

```
agentic-object-detection/
├── data/                    # Test images
├── output/                  # Detection results (gitignored)
├── src/                     # Source code
│   ├── config.py          # Configuration settings
│   ├── pipeline.py        # Main detection pipeline
│   ├── vlm_tool.py        # VLM integration
│   ├── utils.py           # Utility functions
│   └── theme.py           # Custom Gradio themes
├── sctipts/                # Execution scripts
│   ├── app.py             # Gradio web interface
│   └── run.py             # CLI interface
├── notebook/               # Jupyter notebooks
│   ├── overview.ipynb     # Project overview
│   ├── scratch.ipynb      # Experiments and development
│   └── piepline.ipynb    # Pipeline testing
└── pyproject.toml         # Project dependencies
```