import torch

def bbox_to_mask(
    bbox,
    image_w,
    image_h,
    grid_size=24
):

    x1,y1,x2,y2 = bbox

    mask = torch.zeros(
        grid_size,
        grid_size
    )

    gx1=int(x1/image_w*grid_size)
    gy1=int(y1/image_h*grid_size)

    gx2=int(x2/image_w*grid_size)
    gy2=int(y2/image_h*grid_size)

    gx1=max(gx1,0)
    gy1=max(gy1,0)

    gx2=min(gx2,grid_size-1)
    gy2=min(gy2,grid_size-1)

    mask[
        gy1:gy2+1,
        gx1:gx2+1
    ]=1

    return mask.flatten()


if __name__=="__main__":

    bbox=[
        357.7,
        164.6,
        816.6,
        700.3
    ]

    m=bbox_to_mask(
        bbox,
        1024,
        1024
    )

    print(m.shape)

    print(m.sum())