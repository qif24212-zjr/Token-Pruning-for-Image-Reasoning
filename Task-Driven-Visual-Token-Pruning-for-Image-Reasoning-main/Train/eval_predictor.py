import os
import torch
import torch.nn as nn
from tqdm import tqdm




class ImportancePredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):

        score = self.mlp(x)

        return score.squeeze(-1)




FEATURE_DIR = "/root/autodl-tmp/datasets/mini_coco/features"

LABEL_DIR = "/root/autodl-tmp/datasets/mini_coco/labels"

MODEL_PATH = "/root/autodl-tmp/LLaVA/experiments/predictor.pth"



device = "cuda" if torch.cuda.is_available() else "cpu"

predictor = ImportancePredictor().to(device)

predictor.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

predictor.eval()

print("predictor loaded")




total_recall = 0.0
total_precision = 0.0
total_iou = 0.0

count = 0

label_files = [
    f for f in os.listdir(LABEL_DIR)
    if f.endswith(".pt")
]

for fname in tqdm(label_files):

    feature_path = os.path.join(
        FEATURE_DIR,
        fname
    )

    label_path = os.path.join(
        LABEL_DIR,
        fname
    )

    if not os.path.exists(feature_path):
        continue

    feature = torch.load(
        feature_path,
        map_location=device
    ).float()

    label = torch.load(
        label_path,
        map_location=device
    ).float()

    with torch.no_grad():

        score = predictor(feature)

    # -------------------------
    # TopK = 288
    # -------------------------

    idx = torch.topk(
        score,
        k=288,
        dim=1
    ).indices

    pred = torch.zeros(
        576,
        device=device
    )

    pred[idx[0]] = 1.0

    # -------------------------
    # Metrics
    # -------------------------

    tp = ((pred == 1) & (label == 1)).sum().item()

    fp = ((pred == 1) & (label == 0)).sum().item()

    fn = ((pred == 0) & (label == 1)).sum().item()

    recall = tp / (tp + fn + 1e-8)

    precision = tp / (tp + fp + 1e-8)

    iou = tp / (tp + fp + fn + 1e-8)

    total_recall += recall
    total_precision += precision
    total_iou += iou

    count += 1




print()

print("images =", count)

print("Recall    =", total_recall / count)

print("Precision =", total_precision / count)

print("IoU       =", total_iou / count)
