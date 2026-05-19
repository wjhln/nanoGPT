import torch.nn as nn
import torch


# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return self.weight * x


# RoPE
def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)
    # return [-b, a]


def apply_rotary_pos_embed(x, cos, sin):
    return x * cos + rotate_half(x) * sin
    # return [a cosθ - b sinθ, a sinθ + b cosθ]


def get_cos_sin(T, inv_freq, device):
    t = torch.arange(T, device=device).type_as(inv_freq)  # 0~T
    freqs = torch.outer(t, inv_freq)  # 0~T 和 对应 inv_freq 的组合表 (T, head_dim // 2)
    emb = freqs.repeat_interleave(2, dim=-1)  # (T, head_dim)
    cos = emb.cos()[None, None, :, :]  # (1, 1, T, head_dim)
    sin = emb.sin()[None, None, :, :]
    return cos, sin
