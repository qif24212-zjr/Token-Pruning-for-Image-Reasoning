import os
import pandas as pd
import torch
from tqdm import tqdm

from semantic.run_llava_semantic import build_semantic_cache

df = pd.read_parquet(
    "/root/autodl-tmp/datasets/scienceqa/data/train-00000-of-00001-1028f23e353fbe3e.parquet"
)

train_set = []

tmp_dir = "semantic/tmp_images"
os.makedirs(tmp_dir, exist_ok=True)

for idx, row in enumerate(tqdm(df.itertuples(), total=len(df))):

    if row.image is None:
        continue

    question = row.question

    img_path = f"{tmp_dir}/{idx}.png"

    with open(img_path, "wb") as f:
        f.write(row.image["bytes"])

    semantic, mask, visual = build_semantic_cache(
        question,
        img_path
    )

    train_set.append(
        {
            "visual": visual,
            "semantic": semantic.squeeze(0),
            "mask": mask
        }
    )

    if len(train_set) >= 1200:
        break

torch.save(
    train_set,
    "semantic/predictor_dataset.pt"
)

print("saved =", len(train_set))
