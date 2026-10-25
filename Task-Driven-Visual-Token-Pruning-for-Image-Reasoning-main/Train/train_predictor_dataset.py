import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


FEATURE_DIR = "/root/autodl-tmp/datasets/mini_coco/features"
LABEL_DIR   = "/root/autodl-tmp/datasets/mini_coco/labels"


class TokenDataset(Dataset):

    def __init__(self):

        feat_names = {
            f.replace(".pt","")
            for f in os.listdir(FEATURE_DIR)
        }

        label_names = {
            f.replace(".pt","")
            for f in os.listdir(LABEL_DIR)
        }

        self.names = sorted(
            list(feat_names & label_names)
        )

        print("dataset size =", len(self.names))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        name = self.names[idx]

        feat = torch.load(
            os.path.join(
                FEATURE_DIR,
                name + ".pt"
            )
        )

        feat = feat.squeeze(0).float()

        label = torch.load(
            os.path.join(
                LABEL_DIR,
                name + ".pt"
            )
        )

        return feat, label


class ImportancePredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(4096,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )

    def forward(self,x):

        score = self.mlp(x)

        return score.squeeze(-1)



dataset = TokenDataset()

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)


predictor = ImportancePredictor().cuda()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    predictor.parameters(),
    lr=1e-4
)


for epoch in range(20):

    total_loss = 0

    for feats, labels in loader:

        feats = feats.cuda()
        labels = labels.cuda()

        scores = predictor(feats)

        loss = criterion(
            scores,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        epoch,
        total_loss / len(loader)
    )


torch.save(
    predictor.state_dict(),
    "predictor.pth"
)

print("saved predictor")