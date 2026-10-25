import os
import torch
from PIL import Image
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images

IMG_DIR = "/root/autodl-tmp/datasets/mini_coco/train2017"

SAVE_DIR = "/root/autodl-tmp/datasets/mini_coco/features"

os.makedirs(SAVE_DIR, exist_ok=True)

print("loading llava...")

tokenizer, model, image_processor, context_len = \
    load_pretrained_model(
        model_path="/root/autodl-tmp/llava-v1.5-7b",
        model_base=None,
        model_name="llava-v1.5-7b"
    )

vision_tower = model.get_vision_tower()

files = [
    f for f in os.listdir(IMG_DIR)
    if f.endswith(".jpg")
]

print("total images =", len(files))

for i, f in enumerate(tqdm(files)):

    save_path = os.path.join(
        SAVE_DIR,
        f.replace(".jpg", ".pt")
    )

    if os.path.exists(save_path):
        continue

    img_path = os.path.join(
        IMG_DIR,
        f
    )

    image = Image.open(img_path).convert("RGB")

    image_tensor = process_images(
        [image],
        image_processor,
        model.config
    ).half().cuda()

    with torch.no_grad():

        feats = vision_tower(image_tensor)

        feats = model.model.mm_projector(feats)

    torch.save(
        feats.cpu(),
        save_path
    )

    if i % 50 == 0:
        print(i)

print("done")