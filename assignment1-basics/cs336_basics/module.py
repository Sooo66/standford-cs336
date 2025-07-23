import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce, repeat

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        _std = (2 / (self.in_features + self.out_features)) ** 0.5
        self.weights = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(self.out_features, self.in_features, device=device, dtype=dtype),
                mean=0,
                std=_std,
                a=_std*-3,
                b=_std*3
            )
        ) # d_out, d_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = einsum(
            self.weights, x,
            "d_out d_in, ... d_in -> ... d_out"
        )

        return o
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddin_dim = embedding_dim

        self.weights = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(self.num_embeddings, self.embeddin_dim, device=device, dtype=dtype),
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        ) # vocab_size, d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain_weights = nn.Parameter(
            torch.ones(self.d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = (reduce(x.square(), "... d -> ... 1", "mean") + self.eps).sqrt()
        x /= rms
        o = einsum(x, self.gain_weights, "... d, d -> ... d")

        return o.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1_weight = Linear(in_features=d_model, out_features=d_ff)
        self.w2_weight = Linear(in_features=d_ff, out_features=d_model)
        self.w3_weight = Linear(in_features=d_model, out_features=d_ff)

        self.SiLU = lambda x: x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.w1_weight.forward(x)
        o = self.SiLU(o)
        o = o * self.w3_weight.forward(x)
        o = self.w2_weight.forward(o)

        return o
        

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv = 1.0 / (self.theta ** (torch.arange(0, d_k, 2).float() / d_k)).to(device) # (d_k / 2, )
        pos = torch.arange(max_seq_len).to(device) # (max_seq_len, )
        freq = einsum(pos, inv, "... i, ... j -> ... i j") # (max_seq_len, d_k / 2)

        cos, sin = freq.cos(), freq.sin()
        
        self.register_buffer(
            'sin',
            sin,
            persistent=False
        )

        self.register_buffer(
            "cos",
            cos,
            persistent=False
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        _cos = self.cos[token_positions]
        _sin = self.sin[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_even = repeat(x_even, '... d -> ... (d 2)')
        x_odd = repeat(x_odd, '... d -> ... (d 2)')

        r1 = torch.stack([_cos, _sin], dim=-1).flatten(-2)
        r2 = torch.stack([_sin * -1, _cos], dim=-1).flatten(-2)

        o = x_even * r1 + x_odd * r2
        return o


import math
def softmax(x: torch.Tensor, dim: int=-1):
    x -= x.max(dim, keepdim=True).values
    x = math.e ** x
    o = x / x.sum(dim, keepdim=True)

    return o

from loguru import logger

from jaxtyping import Float, Int
from torch import Tensor
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    logger.debug(f"Q shape: {Q.shape}")
    logger.debug(f"V shape: {V.shape}")
    attn = einsum(Q, K.transpose(dim0=-2, dim1=-1), "... queries d_k, ... d_k keys -> ... queries keys")
    attn = attn / (Q.shape[-1] ** 0.5)
    if mask is None:
        mask = torch.tril(torch.ones_like(attn))
    attn = attn.masked_fill(mask == 0, float('-inf'))
    attn = softmax(attn, dim=-1)
    o = einsum(attn, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return o


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
    
    def forward(self, x):
        
        pass