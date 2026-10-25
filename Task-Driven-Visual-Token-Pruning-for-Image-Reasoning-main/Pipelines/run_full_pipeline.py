import torch

import semantic.global_mask as gm
import semantic.global_semantic as gs

from semantic.question_encoder import QuestionEncoder
from semantic.mask_encoder import bbox_to_mask

from ultralytics import YOLO




question = "What color is the cat?"



concept = "cat"

print("QUESTION:")
print(question)

print()
print("CONCEPT:")
print(concept)




encoder = QuestionEncoder()

semantic_feature = encoder.encode(
    question
)

gs.SEMANTIC_FEATURE = semantic_feature

print()
print("SEMANTIC:")
print(semantic_feature.shape)




model = YOLO(
    "/root/autodl-tmp/LLaVA/yolov8s-world.pt"
)

model.set_classes(
    [concept]
)

results = model.predict(
    "/root/autodl-tmp/LLaVA/images/cat.png",
    verbose=False
)

bbox = results[0].boxes.xyxy[0]

bbox = bbox.tolist()

print()
print("BBOX:")
print(bbox)




mask = bbox_to_mask(
    bbox,
    image_w=1024,
    image_h=1024,
    grid_size=24
)

mask = mask.unsqueeze(0)

gm.TOKEN_MASK = mask

print()
print("MASK:")
print(mask.shape)

print(
    "selected:",
    int(mask.sum())
)




print()
print("GLOBAL SEMANTIC:")

print(
    gs.SEMANTIC_FEATURE.shape
)

print()

print("GLOBAL MASK:")

print(
    gm.TOKEN_MASK.shape
)
