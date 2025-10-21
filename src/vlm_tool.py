import base64
import json

from openai import OpenAI
from src.utils import encode_image_to_base64

class VLMTool:
    """
    Handles LLM calls 
    """
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def model_response(self, messages, model="gpt-4.1", temperature=0.1, max_tokens=1000, response_format=None):
        try: 
            # === 텍스트 생성 요청 ===
            if model in ["gpt-4.1", "gpt-4o"]:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    response_format=response_format if response_format else {"type": "text"},
                    temperature=temperature
                ) # open ai class method
            else: 
                raise NotImplementedError("This model is not supported")
            
            # === 응답 추출 ===
            return response.choices[0].message.content # open ai 응답 메시지 리스트 중 첫 번쨰 답변만

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None

    def extract_objects_from_request(self, image_path, user_text, model="gpt-4.1"):
        try:
            # === image encoding ===
            base64_image = encode_image_to_base64(image_path)
            if not base64_image:
                return None

            # === prompt ===
            prompt = (
                "You are an AI assistant that identifies the primary subject for an object detection task from a user's request. "
                "Your goal is to extract ONLY the main object(s) the user wants to find, including their specific attributes like color or state. "
                "Pay close attention to the grammar. If an object is mentioned as part of a location or relationship (e.g., 'the cat on the chair', 'tomatoes hidden by leaves'), do NOT extract the contextual object (like 'chair' or 'leaves'). Extract only the primary subject of the request."
                "If the user asks to find 'all objects' or makes a similarly broad request, you must carefully analyze the provided image and return a comma-separated list of all distinct objects you can identify. For example, if the image shows tomatoes on a vine, a good response would be 'tomatoes, tomato vine, leaves'."
                "Respond ONLY with a comma-separated list of the primary objects and NOTHING ELSE."
                "For example, for 'Find the red tomatoes hidden by leaves', you should respond with 'red tomatoes'."
                "For 'Show me the dog on the bed', you should respond with 'dog'."
            )

            # === messages: multi modal ===
            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # === VLM 응답 결과 ===
            result = self.model_response(messages, model=model)

            # === 리스트 생성 ===
            if result:
                detected_objects = [obj.strip().lower() for obj in result.split(",") if obj.strip()]
                return detected_objects
        
        except Exception as e:
            print(f"Error extracting objects from request: {e}")
            return None