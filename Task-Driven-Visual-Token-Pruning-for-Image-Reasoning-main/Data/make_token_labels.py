import os
import json
import torch
from PIL import Image
from tqdm import tqdm


IMG_DIR = "/root/autodl-tmp/datasets/mini_coco/train2017"
ANN_FILE = "/root/autodl-tmp/datasets/mini_coco/annotations/instances_train2017.json"
SAVE_DIR = "/root/autodl-tmp/datasets/mini_coco/labels"

os.makedirs(SAVE_DIR, exist_ok=True)

GRID = 24  # 24x24 = 576 tokens



def bbox_to_label(bbox, img_w, img_h):
    x, y, w, h = bbox
    x2, y2 = x + w, y + h

    gx1 = int(x / img_w * GRID)
    gx2 = int(x2 / img_w * GRID)
    gy1 = int(y / img_h * GRID)
    gy2 = int(y2 / img_h * GRID)

    label = torch.zeros(GRID * GRID)

    for i in range(gy1, gy2):
        for j in range(gx1, gx2):
            idx = i * GRID + j
            if 0 <= idx < 576:
                label[idx] = 1

    return label



with open(ANN_FILE, "r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
annotations = coco["annotations"]

# group bbox by image_id
ann_dict = {}
for ann in annotations:
    img_id = ann["image_id"]
    if img_id not in ann_dict:
        ann_dict[img_id] = []
    ann_dict[img_id].append(ann["bbox"])



count = 0

for img_id, img_info in tqdm(images.items()):

    file_name = img_info["file_name"]
    img_path = os.path.join(IMG_DIR, file_name)

    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    w, h = img.size

    bboxes = ann_dict.get(img_id, [])
    if len(bboxes) == 0:
        continue

    # multi-object merge → one label
    label = torch.zeros(GRID * GRID)

    for bbox in bboxes:
        label += bbox_to_label(bbox, w, h)

    label = (label > 0).float()

    save_path = os.path.join(
        SAVE_DIR,
        file_name.replace(".jpg", ".pt")
    )

    torch.save(label, save_path)

    count += 1
    if count % 100 == 0:
        print(count)

print("done:", count)
