# This file instantiates the model
from unet import UNet
import torch


def get_model(weights_path, model_initialization_parameters) -> torch.nn.Module:
    model = UNet(**model_initialization_parameters)
    weights = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(weights)  # load weights into model
    return model
