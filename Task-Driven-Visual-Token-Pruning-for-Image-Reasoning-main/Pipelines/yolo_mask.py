# semantic/yolo_mask.py

import torch

TOKEN_NUM = 24
PATCH_NUM = TOKEN_NUM * TOKEN_NUM


def bbox_to_mask(
    bbox,
    image_w,
    image_h
):

    x1,y1,x2,y2 = bbox

    mask = torch.zeros(PATCH_NUM)

    patch_w = image_w / TOKEN_NUM
    patch_h = image_h / TOKEN_NUM

    for r in range(TOKEN_NUM):

        for c in range(TOKEN_NUM):

            px1 = c * patch_w
            py1 = r * patch_h

            px2 = px1 + patch_w
            py2 = py1 + patch_h

            if not (
                px2 < x1 or
                px1 > x2 or
                py2 < y1 or
                py1 > y2
            ):
                mask[r*24+c] = 1

    return mask
def build_token_mask(
    bbox_list,
    image_w,
    image_h
):

    final_mask = torch.zeros(576)

    for bbox in bbox_list:

        final_mask |= bbox_to_mask(
            bbox,
            image_w,
            image_h
        )

    return final_mask.float()