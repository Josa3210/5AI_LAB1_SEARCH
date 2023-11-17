import torch

import utilities.utils as utils


def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mseTensor = (target - input_tensor) ** 2
    return torch.Tensor(torch.mean(mseTensor))
