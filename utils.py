"""
Adapted from https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/utils.py
Action format derived from VPT https://github.com/openai/Video-Pre-Training
"""
import math
from tinygrad import nn, Tensor, dtypes
from einops import rearrange, parse_shape
from typing import Mapping, Sequence, List, Tuple
import numpy as np

"""
We really need a .apply function.
This lets you apply inits recursively.
"""

class Module:
    def apply(self, fn):
        for key, value in self.__dict__.items():
            if isinstance(value, Module):
                value.apply(fn)
            elif isinstance(value, Tensor):
                fn(value)
        fn(self)
        return self

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            value.requires_grad = True
        super().__setattr__(name, value)

def linspace(start, stop, steps):
    if steps < 2:
        return Tensor([start])
    
    step = (stop - start) / (steps - 1)
    return Tensor.arange(steps) * step + start

def xavier_uniform_(tensor, gain=1.0):
    fan_in, fan_out = tensor.shape[0], tensor.shape[-1]
    
    # Calculate the range for the uniform distribution
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate boundary of uniform distribution
    
    # Create a new tensor with values drawn from a uniform distribution
    xavier_tensor = (Tensor.uniform(tensor.shape) * 2 - 1) * a
    
    # In-place update of the input tensor
    tensor.assign(xavier_tensor)
    
    return tensor

def constant_(tensor: Tensor, val: float) -> Tensor:
    """
    Fills the input Tensor with the value `val`.
    
    Args:
    tensor (Tensor): an n-dimensional `Tensor`
    val (float): the value to fill the tensor with
    
    Returns:
    Tensor: the modified input tensor
    """
    # Create a new tensor filled with the constant value
    constant_tensor = Tensor.full(tensor.shape, val, dtype=tensor.dtype, device=tensor.device)
    
    # In-place update of the input tensor
    tensor.assign(constant_tensor)
    
    return tensor

def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """
    Fills the input Tensor with values drawn from the normal distribution N(mean, std^2).
    
    Args:
    tensor (Tensor): an n-dimensional `Tensor`
    mean (float): the mean of the normal distribution
    std (float): the standard deviation of the normal distribution
    
    Returns:
    Tensor: the modified input tensor
    """
    # Create a new tensor with values drawn from a normal distribution
    normal_tensor = Tensor.randn(*tensor.shape) * std + mean
    
    # In-place update of the input tensor
    tensor.assign(normal_tensor)
    
    return tensor

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = Tensor(np.linspace(0, timesteps, steps, dtype='f')) / timesteps
    v_start = Tensor(start / tau).sigmoid()
    v_end = Tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clip(0, 0.999)

def broadcast_tensors(*tensors: Tensor) -> List[Tensor]:
    shapes = [t.shape for t in tensors]
    max_dims = max(len(shape) for shape in shapes)
    
    # Pad shapes with 1s to match the max dimensions
    padded_shapes = [(1,) * (max_dims - len(shape)) + shape for shape in shapes]
    
    # Find the maximum size for each dimension
    output_shape = tuple(max(shape[i] for shape in padded_shapes) for i in range(max_dims))
    
    broadcasted_tensors = []
    for t in tensors:
        new_shape = list(output_shape)
        for i, dim in enumerate(reversed(t.shape)):
            if dim != 1:
                new_shape[-(i+1)] = dim
        
        # Reshape and expand the tensor
        broadcasted = t.reshape(*(1,) * (len(output_shape) - len(t.shape)) + t.shape)
        for i, (s, o) in enumerate(zip(broadcasted.shape, output_shape)):
            if s != o:
                expand_shape = list(broadcasted.shape)
                expand_shape[i] = o
                broadcasted = broadcasted.expand(expand_shape)
        
        broadcasted_tensors.append(broadcasted)
    
    return broadcasted_tensors

ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]

def one_hot_actions(actions: Sequence[Mapping[str, int]]) -> Tensor:
    actions_one_hot = np.zeros((len(actions), len(ACTION_KEYS)), dtype='f')

    for i, current_actions in enumerate(actions):
        for j, action_key in enumerate(ACTION_KEYS):
            if action_key.startswith("camera"):
                if action_key == "cameraX":
                    value = current_actions["camera"][0]
                elif action_key == "cameraY":
                    value = current_actions["camera"][1]
                else:
                    raise ValueError(f"Unknown camera action key: {action_key}")
                # NOTE these numbers specific to the camera quantization used in
                # https://github.com/etched-ai/dreamcraft/blob/216e952f795bb3da598639a109bcdba4d2067b69/spark/preprocess_vpt_to_videos_actions.py#L312
                # see method `compress_mouse`
                max_val = 20
                bin_size = 0.5
                num_buckets = int(max_val / bin_size)
                value = (value - num_buckets) / num_buckets
                assert -1 - 1e-3 <= value <= 1 + 1e-3, f"Camera action value must be in [-1, 1], got {value}"
            else:
                value = current_actions[action_key]
                assert 0 <= value <= 1, f"Action value must be in [0, 1] got {value}"
            print(f'settings actions_one_hot[{i}, {j}] = {value}')
            actions_one_hot[i, j] = value
    
    actions_one_hot = Tensor(actions_one_hot, dtype=dtypes.float).contiguous()
    return actions_one_hot
