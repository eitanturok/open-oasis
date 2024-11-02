"""
Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
"""

from typing import Optional, Callable
import math
from tinygrad import Tensor, nn

class Timesteps:
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def __call__(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb

class Positions2d:
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def __call__(self, grid):
        h_emb = get_timestep_embedding(
            grid[0],
            self.num_channels // 2,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        w_emb = get_timestep_embedding(
            grid[1],
            self.num_channels // 2,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        emb = torch.cat((h_emb, w_emb), dim=-1)
        return emb

class TimestepEmbedding:
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: Callable = Tensor.silu,
        out_dim: int = None,
        post_act_fn: Optional[Callable] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None
        self.act = act_fn
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)
        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = post_act_fn

    def __call__(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D or 2-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] or [N x M x dim] Tensor of positional embeddings.
    """
    if len(timesteps.shape) not in [1, 2]:
        raise ValueError("Timesteps should be a 1D or 2D tensor")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = exponent.exp()
    emb = timesteps[..., None].float() * emb

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = Tensor.cat(*[emb.sin(), emb.cos()], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = Tensor.cat(*[emb[..., half_dim:], emb[..., :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = emb.pad((0, 1, 0, 0))
    return emb
