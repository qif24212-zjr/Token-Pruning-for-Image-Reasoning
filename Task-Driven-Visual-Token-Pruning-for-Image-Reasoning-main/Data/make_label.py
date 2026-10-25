import torch

bbox = torch.tensor([
    356.7382,
    165.1826,
    816.1559,
    699.4863
])

orig_w = 1024
orig_h = 768

x1,y1,x2,y2 = bbox

x1 = x1 / orig_w * 336
x2 = x2 / orig_w * 336

y1 = y1 / orig_h * 336
y2 = y2 / orig_h * 336

label = torch.zeros(576)

for idx in range(576):

    row = idx // 24
    col = idx % 24

    cx = col*14 + 7
    cy = row*14 + 7

    if x1 <= cx <= x2 and y1 <= cy <= y2:
        label[idx] = 1

print(label.sum())

torch.save(
    label,
    "token_label.pt"
)

print("saved")