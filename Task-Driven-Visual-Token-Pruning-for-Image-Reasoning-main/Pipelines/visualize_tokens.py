import torch
import numpy as np
import matplotlib.pyplot as plt

idx = torch.load("topk_idx.pt")

idx = idx[0].numpy()

mask = np.zeros(576)

mask[idx] = 1

mask = mask.reshape(24,24)

plt.imshow(mask)

plt.colorbar()

plt.savefig("token_map.png")

print("saved token_map.png")