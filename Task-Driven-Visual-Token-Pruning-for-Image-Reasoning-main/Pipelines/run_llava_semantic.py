import os
import torch

from PIL import Image
from ultralytics import YOLO



from semantic.question_encoder import QuestionEncoder
from semantic.bbox_to_mask import bbox_to_mask

LLAVA_PATH = "/root/autodl-tmp/llava-v1.5-7b"
YOLO_PATH = "/root/autodl-tmp/LLaVA/yolov8s-world.pt"
CACHE_DIR = "semantic/cache"

    


os.makedirs(CACHE_DIR, exist_ok=True)

class SemanticEngine:
    _instance = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.yolo = None
        self.question_encoder = None

        self.initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SemanticEngine()
        return cls._instance

   
    

    
    def get_yolo(self):
        if self.yolo is None:
            print("[Lazy] Loading YOLO...")

            self.yolo = YOLO(YOLO_PATH)
            self.yolo.to(self.device)

        return self.yolo

  
    def get_encoder(self):
        if self.question_encoder is None:
            self.question_encoder = QuestionEncoder()
        return self.question_encoder

    
    
def extract_concepts(question):
    stop_words = {
        "what", "where", "when", "who", "why", "how",
        "is", "are", "was", "were",
        "the", "a", "an",
        "of", "in", "on", "at", "to", "for",
        "does", "do", "did",
        "many", "much", "color"
    }

    words = question.lower().replace("?", "").split()
    return [w for w in words if w not in stop_words]

def build_semantic_cache(question, image_path):

    engine = SemanticEngine.get_instance()    

   
    encoder = engine.get_encoder()
    semantic_feature = encoder.encode(question)

    torch.save(semantic_feature, f"{CACHE_DIR}/semantic.pt")

    
    yolo = engine.get_yolo()

    concepts = extract_concepts(question)
    yolo.set_classes(concepts)

    results = yolo.predict(image_path, verbose=False)

    final_mask = torch.zeros(576)

    for r in results:
        h, w = r.orig_shape

        for box in r.boxes.xyxy:
            bbox = box.tolist()

            mask = bbox_to_mask(
                bbox,
                w,
                h,
                grid_size=24
            )

            final_mask = torch.maximum(final_mask, mask.float())

    torch.save(final_mask.unsqueeze(0), f"{CACHE_DIR}/mask.pt")

   
    return (
        semantic_feature,
        final_mask,
        None
    )



if __name__ == "__main__":

    semantic, mask, visual = build_semantic_cache(
        "What color is the cat?",
        "/root/autodl-tmp/LLaVA/images/cat.png"
    )

    print("semantic shape:", semantic.shape)
    print("mask shape:", mask.shape)
    print("visual shape:", visual.shape)

    print("cache saved successfully")