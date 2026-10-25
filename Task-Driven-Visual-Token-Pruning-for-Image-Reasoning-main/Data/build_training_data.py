import torch

data = []

sample = {
    "question":
    "What color is the cat?",

    "concept":
    "cat",

    "bbox":
    [
        357.7,
        164.6,
        816.6,
        700.3
    ]
}

data.append(sample)

torch.save(
    data,
    "semantic/train_data.pt"
)

print("saved")