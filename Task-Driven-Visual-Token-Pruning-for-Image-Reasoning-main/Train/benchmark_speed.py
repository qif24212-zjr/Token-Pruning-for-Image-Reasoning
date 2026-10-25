import os
import time
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images

MODEL_PATH = "/root/autodl-tmp/llava-v1.5-7b"
IMG_DIR = "/root/autodl-tmp/datasets/mini_coco/train2017"

NUM_IMAGES = 100

print("loading model...")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    model_base=None,
    model_name="llava-v1.5-7b"
)

files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.endswith(".jpg")
])[:NUM_IMAGES]

times = []

for i, f in enumerate(files):

    image = Image.open(
        os.path.join(IMG_DIR, f)
    ).convert("RGB")

    image_tensor = process_images(
        [image],
        image_processor,
        model.config
    ).half().cuda()

    torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():

        feats = model.encode_images(
            image_tensor
        )

    torch.cuda.synchronize()

    elapsed = time.time() - start

    times.append(elapsed)

    if i % 10 == 0:
        print(
            f"{i:03d}: "
            f"{elapsed:.4f}s "
            f"{tuple(feats.shape)}"
        )

print()
print("images =", len(times))
print("avg =", sum(times) / len(times))
print("min =", min(times))
print("max =", max(times))