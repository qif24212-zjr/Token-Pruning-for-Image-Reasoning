import pandas as pd
import torch

df = pd.read_parquet(
"/root/autodl-tmp/datasets/scienceqa/data/train-00000-of-00001-1028f23e353fbe3e.parquet"
)

data = []

for _, row in df.iterrows():

    if row["image"] is None:
        continue

    sample = {
        "question": row["question"],
        "image": row["image"]
    }

    data.append(sample)

print("samples =", len(data))

torch.save(
    data,
    "semantic/train_data.pt"
)

print("saved")