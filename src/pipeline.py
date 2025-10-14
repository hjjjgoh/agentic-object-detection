import json
import torch

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from src.utils import encode_image, draw_arrows_and_numbers, draw_bounding_boxes
from src.config import INV_MODEL_TYPES

class ObjectDetectionTool:
    """ Agentic object detection pipeline"""
    """
    Performs object detection using GroundingDINO or OWL-ViT,
    plus an optional 'critique' (refinement) step with a VLM
    to yield a refined set of objects to detect.
    """
    def __init__(self, model_id, device, vlm_tool, confidence_threshold=0.2, concept_detection_model="gpt-4.1", initial_critique_model="gpt-4o", final_critique_model="gpt-4.1"):
        self.model_id = model_id # owlvit or groungding dino
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device
        self.vlm_tool = vlm_tool  # The LLMTool that can handle vision (GPT-4V) or similar
        self.confidence_threshold = confidence_threshold
        self.concept_detection_model = concept_detection_model
        self.initial_critique_model = initial_critique_model
        self.final_critique_model = final_critique_model
        
        # We store bounding boxes for potential usage later (e.g., for SAM).
        self.last_detection_bboxes = []
        self.last_filtered_objects = []

    def _run_detector(self, image_path, query_list):
        """
        Low-level routine to run the detection model on `query_list`.
        Returns: (detected_objects_final, labeled_image_path)
        Where `detected_objects_final` = [(num, label, [x1,y1,x2,y2]), ...].
        """
        from PIL import ImageFont
        
        # Format queries for the model
        if INV_MODEL_TYPES[self.model_id] == "owlvit":
            formatted_queries = [f"An image of {q}" for q in query_list]
        elif INV_MODEL_TYPES[self.model_id] == "grounding_dino":
            formatted_queries = " ".join([f"{q}." for q in list(set(query_list))])
        else:
            raise NotImplementedError("Model not supported")

        # Load image
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=formatted_queries, 
            images=img, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)

        # PyTorch gradient 계산 비활성화
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process bounding boxes
        if INV_MODEL_TYPES[self.model_id] == "grounding_dino":
            results = self.processor.post_process_grounded_object_detection(
                outputs, 
                inputs.input_ids,
                target_sizes=[img.size[::-1]]
            )
            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["text_labels"]
        elif INV_MODEL_TYPES[self.model_id] == "owlvit":
            logits = torch.max(outputs["logits"][0], dim=-1)
            scores = torch.sigmoid(logits.values).cpu().numpy()
            labels = logits.indices.cpu().numpy()
            boxes = outputs["pred_boxes"][0].cpu().numpy()
        else:
            raise NotImplementedError("Model not supported")            

        detected_objects_final = []
        idx = 1
        for score, box, label in zip(scores, boxes, labels):
            if score < self.confidence_threshold:
                continue
            detected_objects_final.append((idx, label, box.tolist()))
            idx += 1

        # Draw numbers
        labeled_image_path = draw_arrows_and_numbers(image_path, detected_objects_final)
        return detected_objects_final, labeled_image_path
    
    def _critique_and_refine_query(self, user_request, original_concepts, labeled_image_path, objects_detected, model="gpt-4o"):
        """
        Asks the VLM/LLM: "We tried to detect <objects_detected> for the user request, 
        but maybe we need a refined set of objects. 
        Return a new list of objects or concepts to detect."
        """
        base64_labeled_image = encode_image(labeled_image_path)

        # For clarity, let's pass the original user request and 
        # the currently detected object list to the LLM. 
        # The LLM can propose a refined set of objects to detect.
        refine_messages = [
    {
    "role": "system",
    "content": """
        You are an AI system that refines detection queries. 
        You are provided with the outputs from an object detection model, along with the user's request and the objects from the user's request that were extracted and provided to the object detection model.
        Your task is to analyze whether the object detector has extracted the results properly to the user's request and, if not, refine the queries by generalizing concepts where possible.
        
        Important guidelines:
        1. If the detection results are already good, no need to refine. 
           - In that case, provide reasoning indicating no refinement was necessary and return the same list.
        2. If the detection results are poor or null, propose synonyms or more generic categories and explain why. Wherever possible, retain the singular version of the concept.
        3. Return your final answer as a JSON object with exactly two fields: "reasoning" and "refined_list".
           - "reasoning" is a short explanation of why you refined or didn't refine.
           - "refined_list" is a comma-separated list of object names that should be re-tried in detection.
        4. Output ONLY the JSON, and no other text.

        Below are some examples:

        EXAMPLE 1
        User's Request: Detect the teacup poodle
        Original concept: "Teacup poodle"

        Final output:
        {
          "reasoning": "The provided image does not have any detections for the concept of teacup poodle. The concept "teacup poodle" might be a very specific concept for the model to detect. This could  be refined to a more higher-level and generic concept like 'Dog',
          "refined_list": "dog"
        }

        EXAMPLE 2
        User's Request: Detect the sparkly stiletto shoe
        Original concept: "Sparkly stiletto shoe"

        Final output:
        {
          "reasoning": "The provided image does not specific detections that correspond for 'Sparkly stiletto shoe'. 'Sparkly stiletto shoe' might be too specific for the model. Refining to 'shoe', a more generic concept might increase the likelihood of detection.",
          "refined_list": "shoe"
        }

        EXAMPLE 3
        User's Request: Detect the hydrangea
        Original concept: "Hydrangea"

        Final output:
        {
          "reasoning": "No detections found for 'hydrangea'. The model might struggle with specific flower types. Refining to the more general concept 'flower' could yield better results.",
          "refined_list": "flower"
        }

        EXAMPLE 4
        User's Request: Detect the gourmet cheeseburger
        Original concept: "Gourmet cheeseburger"

        Final output:
        {
          "reasoning": "No detections observed for 'Gourmet cheeseburger'. 'Gourmet cheeseburger' might be too specific. Refining to 'hamburger' as it aligns with the detected object.",
          "refined_list": "hamburger"
        }

        EXAMPLE 5
        User's Request: Detect the red sports car
        Original concept: "Red sports car"

        Final output:
        {
          "reasoning": "The provided image does not have any reliable detections for 'red sports car'. Color-based detection might be challenging. Refining to the more general concept 'car' could improve detection.",
          "refined_list": "car"
        }

        EXAMPLE 6
        User's Request: Find all the tomatoes on the vine
        Original concept: "Tomatoes on the vine"

        Final output:
        {
          "reasoning": "The model might not understand the composite concept 'tomatoes on the vine'. It's better to detect the primary object. Refining to 'tomato' will likely yield better results.",
          "refined_list": "tomato"
        }

        Remember: 
        • If no refinement is needed (the concept is recognized well), explain that in the reasoning and return the same concept. 
        • When refinement is necessary, prioritize more generic or abstract categories that may be more reliably detected by the model.
        • Provide only the JSON. 
        • No extra commentary.
        """
        },

        {"role": "user", "content": [
            {"type": "text", "text": f"User's request: {user_request}\n Original Concepts for Detection: {original_concepts}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_labeled_image}", "detail": "high"}}
        ]}
        ]
        refine_response = self.vlm_tool.chat_completion(refine_messages, model=model, response_format={"type": "json_object"})
        
        refined_response_objects = json.loads(refine_response)["refined_list"].split(",")

        if not refined_response_objects:
            return []
        
        refined_list = [r.strip().lower() for r in refined_response_objects if r.strip()]
        return refined_list
   

    
    def _validate_bboxes_with_llm(self, user_request, labeled_image_path, model="gpt-4o"):
        """
        Pass the labeled image to the LLM to filter bounding boxes 
        based on user request. Returns 'valid_numbers' list.
        """
        base64_labeled_image = encode_image(labeled_image_path) # VLM API에 이미지 직접 전송을 위함 
        
        messages = [
            {"role": "system", "content": "You are an AI reviewing an object detection output.\n"
                                          "All detected objects have been marked with an arrow mapping to a corresponding number.\n"
                                          "The image contains arrows labeled with numbers pointing to specific objects.\n"
                                          "Your task is to identify the objects indicated by these arrows and determine whether each detected object is relevant to the user's query.\n"
                                          "For each numbered arrow:\n"
                                          "1. Identify the object being pointed to.\n"
                                          "2. Provide a brief description of the object (e.g., 'top-left cup with blue leaves', 'bottom-right cup with watermelon pattern', or 'background birdcage').\n"
                                          "3. Analyze whether the object is valid based on the context and the user's instructions.\n"
                                          "4. Provide a clear, step-by-step explanation for each object's validity decision.\n"                                          
                                          "Return a JSON object with the reasoning and list of valid numbers matching the user's request.\n"
                                          "Example output:\n"
                                          "{ \"reasoning\": <reasoning> , \"valid_numbers\": {object_num :\"object_name\"} }"
            },
            {"role": "user", "content": [
                {"type": "text", "text": f"The user's original request was: {user_request}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_labeled_image}", "detail": "high"}}
            ]
            }
        ] # 각 번호가 가리키는 객체가 사용차의 요청과 관련 있는지 판단 

        valid_numbers_json = self.vlm_tool.chat_completion(
            messages, 
            model=model,
            response_format={"type": "json_object"}
        ) # VLM api 전송, 

        try:
            valid_numbers_data = json.loads(valid_numbers_json)
            return valid_numbers_data.get("valid_numbers", {})
        except json.JSONDecodeError:
            return []
        # VLM으로부터 받은 JSON 형식의 문자열 응답 > 파이썬 딕셔너리 


    def run(self, image_path, user_request, do_critique=True):
        """
        Full pipeline:
        1. Extract objects from user request (LLM).
        2. Detect bounding boxes with that query.
        3. LLM-based validation step => filter bounding boxes.
        4. (Optional) Critique Step => refine the query if needed.
        5. Re-run detection with refined queries.
        6. Final LLM validation => final bounding boxes and annotation.
        """
        # do_critique: critique 여부 선택 

        # ---------------------------------------------------
        # Step 1: initial user queries from request
        # ---------------------------------------------------
        objects_to_detect = self.vlm_tool.extract_objects_from_request(image_path, user_request, model=self.concept_detection_model)
        if not objects_to_detect:
            return None, "⚠️ No objects to detect or invalid request."

        # ---------------------------------------------------
        # Step 2: run detection with the initial user queries
        # ---------------------------------------------------
        detected_objects_final, labeled_image_path = self._run_detector(image_path, objects_to_detect)
        
        # ------------------------------------------------------
        # Step 3: Initial Critique and Object Concept Refinement
        # ------------------------------------------------------
        if do_critique:
            current_labels = ",".join(set([str(lbl) for _, lbl, _ in detected_objects_final]))

            refined_query_list = self._critique_and_refine_query(
                user_request=user_request,
                original_concepts=current_labels,
                labeled_image_path=labeled_image_path,
                objects_detected=current_labels,
                model=self.initial_critique_model
            )
            
            # If the refined list is empty or identical, we might skip re-running
            # But let's suppose we only re-run if we actually get a new set.
            if refined_query_list and set(refined_query_list) != set(objects_to_detect):
                # Re-run detection with refined query
                detected_objects_final, labeled_image_path = self._run_detector(image_path, refined_query_list)
                if not detected_objects_final:
                    return None, "No objects found for the initial query."
        
        # ---------------------------------------------------
        # Step 4: LLM-based critique
        # ---------------------------------------------------
        valid_numbers = self._validate_bboxes_with_llm(user_request, labeled_image_path, model=self.final_critique_model)
        # dictionary 형태의 유효한 객체 번호 목록

        # filter bounding boxes
        if valid_numbers: 
            filtered_objects = [(n, valid_numbers[str(n)], box) for (n, lbl, box) in detected_objects_final if str(n) in valid_numbers]
            # detection에서 유효한 객체 번호 목록에 따른 detection 박스에 대한 튜플만 걸러내기 
        else:
            filtered_objects = detected_objects_final
        

        # store them
        self.last_detection_bboxes = [x[-1] for x in filtered_objects] # 튜플의 마지막 항목 - bbox 좌표 리스트
        self.last_filtered_objects = filtered_objects


        # ---------------------------------------------------
        # Step 5: Produce final annotated image
        # ---------------------------------------------------
        final_img = draw_bounding_boxes(image_path, filtered_objects)

        final_text = (
            f"🔍 Validated objects: {', '.join(set(str(lbl) for _, lbl, _ in filtered_objects))}"
        )
        return final_img, final_text