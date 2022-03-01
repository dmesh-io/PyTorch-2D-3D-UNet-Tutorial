import numpy as np
import torch
from PIL import Image
from transformations import normalize_01, re_normalize


def preprocess_function(image: bytes) -> torch.tensor:
    # preprocessing
    image_np = np.array(Image.open(image), dtype="uint8")
    img = np.moveaxis(image_np, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32

    # transform to pytorch tensor
    img_torch = torch.from_numpy(img)

    return img_torch


def postprocess_function(image_torch: torch.tensor) -> torch.tensor:
    # postprocessing
    out = torch.argmax(image_torch, dim=1)  # perform argmax to generate 1 channel
    out = out.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    out = np.squeeze(out)  # remove batch dim and channel dim -> [H, W]
    out = re_normalize(out)  # scale it to the range [0-255]

    return out
