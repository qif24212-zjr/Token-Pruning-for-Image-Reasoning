import torch

from semantic.semantic_predictor import SemanticPredictor

model = SemanticPredictor()

model.load_state_dict(
    torch.load(
        "semantic/semantic_predictor.pth",
        map_location="cpu"
    )
)

print("load ok")