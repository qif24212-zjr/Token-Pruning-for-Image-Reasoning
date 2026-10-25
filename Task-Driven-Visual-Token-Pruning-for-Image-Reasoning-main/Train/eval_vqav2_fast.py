import os
import re
import tempfile
import torch

from datasets import load_dataset
from tqdm import tqdm

from semantic.run_llava_semantic import build_semantic_cache

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN
)

from llava.conversation import conv_templates

from llava.model.builder import load_pretrained_model

from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path
)

from llava.utils import disable_torch_init



MODEL_PATH = "/root/autodl-tmp/llava-v1.5-7b"

disable_torch_init()

print("Loading LLaVA...")

model_name = get_model_name_from_path(
    MODEL_PATH
)

tokenizer, model, image_processor, context_len = (
    load_pretrained_model(
        MODEL_PATH,
        None,
        model_name
    )
)

model.eval()

print("Model loaded.")




def build_prompt(question):
    return question



def inference(image, question):

    prompt_text = build_prompt(question)

    qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

    conv = conv_templates["llava_v1"].copy()

    conv.append_message(
        conv.roles[0],
        qs
    )

    conv.append_message(
        conv.roles[1],
        None
    )

    prompt = conv.get_prompt()

    image_tensor = process_images(
        [image],
        image_processor,
        model.config
    ).to(
        model.device,
        dtype=torch.float16
    )

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).cuda()
    import time

    

    torch.cuda.synchronize()
    start = time.time()

    with torch.inference_mode():

        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0,
            max_new_tokens=16,
            use_cache=True
        )

    torch.cuda.synchronize()

    latency = time.time() - start


    output = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    return output.strip(), latency




import json

questions = json.load(
    open(
        "/root/autodl-tmp/datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
    )
)["questions"]

annotations = json.load(
    open(
        "/root/autodl-tmp/datasets/vqav2/v2_mscoco_val2014_annotations.json"
    )
)["annotations"]

testset = zip(
    questions,
    annotations
)
correct = 0.0
total = 0

latency_sum = 0.0
latency_count = 0




from PIL import Image

for q, a in tqdm(testset):

    question = q["question"]

    image_name = (
        f"COCO_val2014_{q['image_id']:012d}.jpg"
    )

    image_path = os.path.join(
        "/root/autodl-tmp/project/datasets/vqav2/val2014",
        image_name
    )

    if not os.path.exists(image_path):
        continue

    image = Image.open(
        image_path
    ).convert("RGB")

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    ) as tmp:

        image.save(tmp.name)

        tmp_path = tmp.name

    try:
       # build_semantic_cache(
       # question,
       # tmp_path
   # )

        

        pred_text, latency = inference(
            image,
            question
        )

        latency_sum += latency
        latency_count += 1

        pred = pred_text.lower().strip()

        gt_answers = [
            x["answer"].lower().strip()
            for x in a["answers"]
        ]
        print("Q:", question)
        print("PRED:", pred)
        print("GT:", gt_answers[:3])
        print("-"*50)
        matches = sum(
            gt in pred
            for gt in gt_answers
        )

        score = min(
            matches / 3.0,
            1.0
        )

        correct += score

        total += 1

        if total % 10 == 0:

            print(
                f"ACC={correct/total:.4f}"
            )

    except Exception as e:

        print("ERROR:", e)

    finally:

        os.remove(tmp_path)

  
    if total >= 1500:
        break


print()

avg_latency = latency_sum / latency_count

print("=" * 50)
print("TOTAL :", total)
print("VQAv2 SCORE :", correct / total)

print("AVG LATENCY (s):", avg_latency)
print("AVG LATENCY (ms):", avg_latency * 1000)

print("=" * 50)
