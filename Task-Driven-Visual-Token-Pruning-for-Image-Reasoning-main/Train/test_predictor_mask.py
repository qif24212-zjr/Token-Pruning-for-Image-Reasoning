from semantic.semantic_predictor \
import SemanticPredictor

import torch

model=SemanticPredictor()

visual=torch.randn(
    1,
    576,
    4096
)

semantic=torch.randn(
    1,
    384
)

mask=torch.randint(
    0,
    2,
    (1,576)
).float()

y=model(
    visual,
    semantic,
    mask
)

print(y.shape)