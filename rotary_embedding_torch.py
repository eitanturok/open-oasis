"""
Adapted from https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
"""

from __future__ import annotations
from math import pi, log

import tinygrad
from tinygrad import nn, Tensor, dtypes
import numpy as np

from einops import rearrange, repeat, einsum

from typing import Literal

from utils import broadcast_tensors, linspace

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# broadcat, as tortoise-tts was using it

def unbind(tensor: Tensor, dim: int = 0) -> List[Tensor]:
    if dim < 0:
        dim += tensor.ndim

    if dim < 0 or dim >= tensor.ndim:
        raise ValueError(f"Dimension out of range (expected to be in range of [{-tensor.ndim}, {tensor.ndim-1}], but got {dim})")

    slices = []
    for i in range(tensor.shape[dim]):
        # Create a slice for each index along the specified dimension
        slice_indices = [slice(None)] * tensor.ndim
        slice_indices[dim] = i
        slices.append(tensor[tuple(slice_indices)])

    return slices

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return Tensor.cat(broadcasted_tensors, dim = dim)

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = unbind(x, dim = -1)
    x = Tensor.stack(-x2, x1, dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_emb(freqs, t, start_index = 0, scale = 1., seq_dim = -2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place    
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
        
    out = Tensor.cat(*(t_left, t_transformed, t_right), dim=-1)

    return out.cast(dtype)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = Tensor.einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

# classes

class RotaryEmbedding:
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for:  Literal['lang', 'pixel', 'constant'] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = Tensor(np.linspace(1., max_freq / 2, dim // 2, dtype='f')).float() * pi
        elif freqs_for == 'spacetime':
            time_freqs = 1. / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)].float() / dim))
            freqs = Tensor(np.linspace(1., max_freq / 2, dim // 2, dtype='f')).float() * pi
        elif freqs_for == 'constant':
            freqs = Tensor.ones(num_freqs).float()

        if freqs_for == 'spacetime':
            self.time_freqs = Tensor(time_freqs.numpy(), requires_grad = learned_freq, dtype=dtypes.float32)
        self.freqs = Tensor(freqs.numpy(), requires_grad = learned_freq, dtype=dtypes.float32)

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.cached_freqs = Tensor.zeros(cache_max_seq_len, dim, requires_grad = False).contiguous()
        self.cached_freqs_seq_len = Tensor(0, requires_grad = False)

        self.learned_freq = learned_freq

        # dummy for device

        self.dummy = Tensor(0, requires_grad = False)

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (Tensor.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.scale = Tensor(scale, requires_grad = False)
        self.cached_scales = Tensor.zeros(cache_max_seq_len, dim, requires_grad = False)
        self.cached_scales_seq_len = Tensor(0, requires_grad = False)

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        return (Tensor.arange(seq_len, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, freqs, seq_dim = None, offset = 0, scale = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

        seq_freqs = self(seq, freqs, seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            seq_freqs = seq_freqs.unsqueeze(1)

        return apply_rotary_emb(seq_freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype = dtype, device = device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, scale = q_scale, offset = k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim, scale = k_scale ** -1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, freqs, seq_dim = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device)

        seq_freqs = self(seq, freqs, seq_len = seq_len)
        scale = self.get_scale(seq, seq_len = seq_len).to(dtype)

        if seq_dim == -3:
            seq_freqs = rearrange(seq_freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = apply_rotary_emb(seq_freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(seq_freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        rotated_q = rotated_q.cast(q.dtype)
        rotated_k = rotated_k.cast(k.dtype)

        return rotated_q, rotated_k

    def get_scale(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        assert self.use_xpos

        should_cache = (
            self.cache_if_possible and
            exists(seq_len) and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            exists(self.cached_scales) and \
            (seq_len + offset) <= self.cached_scales_seq_len.item()
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = repeat(scale, 'n d -> n (d r)', r = 2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len.copy_(seq_len)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            #print(f'get_axial_freqs ind: {ind}, dim: {dim}')
            # only allow pixel freqs for last two dimensions
            use_pixel = (self.freqs_for == 'pixel' or self.freqs_for == 'spacetime') and ind >= len(dims) - 2
            if use_pixel:
                pos = linspace(-1, 1, dim).float()
            else:
                pos = Tensor.arange(dim)

            if self.freqs_for == 'spacetime' and not use_pixel:
                seq_freqs = self(pos, self.time_freqs, seq_len = dim)
            else:
                seq_freqs = self(pos, self.freqs, seq_len = dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            #print(f'new_axis_slice: {new_axis_slice}')
            #print(f'seq_freqs: {seq_freqs}')
            #print(f'seq_freqs[new_axis_slice]: {seq_freqs[new_axis_slice]}')
            all_freqs.append(seq_freqs[new_axis_slice])

        #print(f'all_freqs: {all_freqs}')
        all_freqs = broadcast_tensors(*all_freqs)
        return Tensor.cat(*all_freqs, dim = -1)

    def __call__(
        self,
        t: Tensor,
        freqs: Tensor,
        seq_len = None,
        offset = 0
    ):
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs_seq_len.item()
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        #print(f'freqs: {freqs.numpy()}, t casted: {t.cast(freqs.dtype).numpy()}')
        #print(f't shape: {t.shape}, freqs.shape: {freqs.shape}')
        t_casted = t.cast(freqs.dtype)
        # Perform einsum
        try:
            result = Tensor.einsum('..., f -> ... f', t_casted, freqs)
        except Exception as e:
            print("Einsum failed:", str(e))
            # If einsum fails, try a manual implementation
            t_expanded = t_casted.reshape(*t_casted.shape, 1)
            freqs = t_expanded * freqs
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len.assign(seq_len)

        return freqs
