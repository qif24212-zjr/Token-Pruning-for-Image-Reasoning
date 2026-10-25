import torch
import torch.nn as nn

from semantic.semantic_predictor import SemanticPredictor

dataset = torch.load(
    "semantic/predictor_dataset.pt"
)

model = SemanticPredictor().cuda()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

criterion = nn.BCEWithLogitsLoss()

for epoch in range(20):

    total_loss = 0

    for sample in dataset:

        visual = sample["visual"].unsqueeze(0).cuda()

        semantic = sample["semantic"].unsqueeze(0).cuda()

        mask = sample["mask"].unsqueeze(0).cuda()

        score = model(
            visual,
            semantic,
            mask
        )

        loss = criterion(
            score,
            mask.float()
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        "epoch",
        epoch,
        "loss",
        total_loss / len(dataset)
    )

torch.save(
    model.state_dict(),
    "semantic/semantic_predictor.pth"
)

print("saved predictor")