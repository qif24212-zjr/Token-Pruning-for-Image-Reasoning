print("CURRENT FILE =", __file__)
import sys

sys.path.append(
    "/root/autodl-tmp/LLaVA/visual_case"
)

from save_tokens import save_token_map
latency_sum = 0
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




LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_prompt(question, choices):

    prompt = question + "\n\n"

    for i, c in enumerate(choices):

        prompt += f"{LETTERS[i]}. {c}\n"

    prompt += (
        "\nAnswer with only the option letter "
        "(A, B, C, D, E)."
    )

    return prompt




def inference(image, question, choices):

    prompt_text = build_prompt(
        question,
        choices
    )

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

    tqdm.write("BEFORE GENERATE")

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

    tqdm.write("AFTER GENERATE")

    output = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    return output.strip(), latency




def parse_answer(text):

    text = text.upper()

    m = re.search(
        r"\b([A-E])\b",
        text
    )

    if m:
        return m.group(1)

    return None




ds = load_dataset(
    "/root/autodl-tmp/datasets/scienceqa"
)

testset = ds["test"]

correct = 0
total = 0

latency_sum = 0.0
latency_count = 0




for sample in tqdm(testset):

    if sample["image"] is None:
        continue

    image = sample["image"]

    question = sample["question"]

    choices = sample["choices"]

    gt_idx = sample["answer"]

    gt_letter = LETTERS[gt_idx]

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    ) as tmp:

        image.save(tmp.name)

        tmp_path = tmp.name

    try:
        semantic_feature, mask, _ = build_semantic_cache(
            question,
            tmp_path
        )

        

        pred_text, latency = inference(
            image,
            question,
            choices
        )
        latency_sum += latency
        latency_count += 1

        pred_letter = parse_answer(pred_text)

        if pred_letter == gt_letter:
            correct += 1

        

        idx = torch.load(
            "semantic/cache/token_idx.pt"
        )

        save_token_map(
            image=image,
            idx=idx,
            save_path=f"case_study/case_{total}.png"
        )

        
    except Exception as e:

        print("ERROR:", e)

    finally:

        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    total += 1
    if total >= 2000:
        print("STOP")
        break



avg_latency = latency_sum / latency_count

print("=" * 50)
print("TOTAL :", total)
print("CORRECT :", correct)
print("ACC :", correct / total)

print("AVG LATENCY (s):", avg_latency)
print("AVG LATENCY (ms):", avg_latency * 1000)

print("=" * 50)
