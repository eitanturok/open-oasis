"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import tinygrad
from tinygrad import Tensor, nn, dtypes, TinyJit
from tinygrad.nn.state import safe_load, load_state_dict, torch_load, get_state_dict
from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video, write_video
from utils import one_hot_actions, sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
import numpy as np
import imageio

# load DiT checkpoint
ckpt = safe_load("oasis500m.safetensors")
model = DiT_models["DiT-S/2"]()
load_state_dict(model, ckpt, strict=False)

# load VAE checkpoint
vae_ckpt = safe_load("vit-l-20.safetensors")
vae = VAE_models["vit-l-20-shallow-encoder"]()
load_state_dict(vae, vae_ckpt, strict=False)

# sampling params
B = 1
total_frames = 32
max_noise_level = 1000
ddim_noise_steps = 100
noise_range = Tensor(np.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1))
noise_abs_max = 20
ctx_max_noise_idx = ddim_noise_steps // 10 * 3

# get input video 
video_id = "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001"
mp4_path = f"sample_data/{video_id}.mp4"
actions_path = f"sample_data/{video_id}.actions.pt"
video = Tensor((read_video(mp4_path, pts_unit="sec")[0].float() / 255.0).numpy(), dtype=dtypes.float)
actions = one_hot_actions(torch_load(actions_path))
offset = 100
video = video[offset:offset+total_frames].unsqueeze(0)
actions = actions[offset:offset+total_frames].unsqueeze(0)

# sampling inputs
n_prompt_frames = 1
x = video[:, :n_prompt_frames]

# vae encoding
scaling_factor = 0.07843137255
x = rearrange(x, "b t h w c -> (b t) c h w")
H, W = x.shape[-2:]
Tensor.no_grad = True
Tensor.training = False
x = vae.encode(x * 2 - 1).mean * scaling_factor
x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H//vae.patch_size, w=W//vae.patch_size)

# get alphas
betas = sigmoid_beta_schedule(max_noise_level)
alphas = 1.0 - betas

def cumprod(x: Tensor, axis=0) -> Tensor:
    shape = x.shape
    if axis < 0:
        axis += len(shape)
    
    # Reshape to 2D for simplicity
    x = x.reshape(-1, shape[axis])
    
    # Hack to get .copy() implemented
    result = Tensor(x.numpy(), dtype=x.dtype).contiguous()
    for i in range(1, x.shape[1]):
        result[:, i] = (result[:, i] * result[:, i-1]).realize()
    
    # Reshape back to original shape
    return result.reshape(shape)

alphas_cumprod = cumprod(alphas)
alphas_cumprod = alphas_cumprod.unsqueeze(1).unsqueeze(2).unsqueeze(3).realize()

#@TinyJit
def jit(x_curr:Tensor, t:Tensor, t_next:Tensor, actions_chunk: Tensor) -> Tensor:
    v = model(x_curr, t, actions_chunk).realize()
    x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
    x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) \
            / (1 / alphas_cumprod[t] - 1).sqrt()

    # get frame prediction
    x_pred = alphas_cumprod[t_next].sqrt() * x_start + x_noise * (1 - alphas_cumprod[t_next]).sqrt()
    return x_pred.realize()


# sampling loop
for i in tqdm(range(n_prompt_frames, total_frames)):
    chunk = Tensor.randn((B, 1, *x.shape[-3:]))
    chunk = chunk.clamp(-noise_abs_max, +noise_abs_max)
    x = Tensor.cat(x, chunk, dim=1)
    start_frame = max(0, i + 1 - model.max_frames)
    print(f'step: {i}')

    for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
        # set up noise values
        ctx_noise_idx = min(noise_idx, ctx_max_noise_idx)
        t_ctx  = Tensor.full((B, i), noise_range[ctx_noise_idx].item(), dtype=dtypes.long)
        t      = Tensor.full((B, 1), noise_range[noise_idx].item(),     dtype=dtypes.long)
        t_next = Tensor.full((B, 1), noise_range[noise_idx - 1].item(), dtype=dtypes.long)
        t_next = (t_next < 0).where(t, t_next)
        t = Tensor.cat(t_ctx, t, dim=1).realize()
        t_next = Tensor.cat(t_ctx, t_next, dim=1).realize()

        # sliding window
        x_curr = Tensor(x.numpy(), requires_grad=x.requires_grad)
        x_curr = x_curr[:, start_frame:].detach().realize()
        t = t[:, start_frame:].detach()
        t_next = t_next[:, start_frame:].detach()

        # add some noise to the context
        ctx_noise = Tensor.randn(x_curr[:, :-1].shape, dtype=x_curr[:, :-1].dtype)
        ctx_noise = ctx_noise.clamp(-noise_abs_max, +noise_abs_max)
        x_curr[:, :-1] = alphas_cumprod[t[:, :-1]].sqrt() * x_curr[:, :-1] + (1 - alphas_cumprod[t[:, :-1]]).sqrt() * ctx_noise

        # get model predictions
        Tensor.no_grad = True
        print(f'x_curr: {x_curr}, t: {t}, actions_chunk: {actions[:, start_frame : i + 1]}')
        #v = model(x_curr, t, actions[:, start_frame : i + 1])
        x_pred = jit(x_curr, t, t_next, actions[:, start_frame : i + 1]).contiguous().realize()
        x[:, -1:] = x_pred[:, -1:].contiguous().realize()

# vae decoding
x = rearrange(x, "b t c h w -> (b t) (h w) c")
Tensor.no_grad = True
x = (vae.decode(x / scaling_factor) + 1) / 2
x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

# save video
x = x.clamp(0, 1)
x = (x * 255).cast(dtype=dtypes.uchar)
imageio.mimsave("./video.mp4", np.stack(x[0].numpy()), fps=20)

print("generation saved to video.mp4.")

