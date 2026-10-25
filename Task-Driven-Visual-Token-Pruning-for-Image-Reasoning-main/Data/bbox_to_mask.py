import torch

def bbox_to_mask(
    bbox,
    image_w,
    image_h,
    grid_size=24
):

    x1,y1,x2,y2 = bbox

    gx1 = int(x1 / image_w * grid_size)
    gx2 = int(x2 / image_w * grid_size)

    gy1 = int(y1 / image_h * grid_size)
    gy2 = int(y2 / image_h * grid_size)

    gx1 = max(0,min(gx1,grid_size-1))
    gx2 = max(0,min(gx2,grid_size))

    gy1 = max(0,min(gy1,grid_size-1))
    gy2 = max(0,min(gy2,grid_size))

    mask = torch.zeros(
        grid_size,
        grid_size
    )

    mask[
        gy1:gy2,
        gx1:gx2
    ] = 1

    return mask.reshape(-1)


if __name__ == "__main__":

    bbox = [
        357.7,
        164.6,
        816.6,
        700.3
    ]

    image_w = 1024
    image_h = 768

    mask = bbox_to_mask(
        bbox,
        image_w,
        image_h
    )

    print(mask.shape)

    print(
        "selected tokens:",
        int(mask.sum())
    )