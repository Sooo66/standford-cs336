import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce, repeat
import einx
from jaxtyping import Float, Int
from torch import Tensor

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
            x, self.weights,
            "... d_in, d_out d_in -> ... d_out"
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
        x = x / rms
        o = x * self.gain_weights

        return o.to(in_dtype)

class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1_weight = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2_weight = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3_weight = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

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
    x = torch.exp(x)
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
    # logger.debug(f"Q shape: {Q.shape}")
    # logger.debug(f"V shape: {V.shape}")
    attn = einsum(Q, K.transpose(dim0=-2, dim1=-1), "... queries d_k, ... d_k keys -> ... queries keys")
    attn = attn / (Q.shape[-1] ** 0.5)
    if mask is None:
        mask = torch.tril(torch.ones_like(attn))
    attn = attn.masked_fill(mask == 0, float('-inf'))
    attn = softmax(attn, dim=-1)
    o = einsum(attn, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return o


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None=None, theta: float | None=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(d_model, num_heads * self.d_k)
        self.v_proj = Linear(d_model, num_heads * self.d_v)
        self.o_proj = Linear(num_heads * self.d_v, d_model)

        if theta is not None:
            self.pos_encoder = RoPE(theta, self.d_k, max_seq_len)

    def forward(self, x: Float[Tensor, "seq_len d_model"], token_positions: Int[Tensor, " ... seq_len"] | None = None,) -> Float[Tensor, "seq_len d_model"]:
        *b, seq_len, d_model = x.size()
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q, K, V = (
            rearrange(X, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
            for X in (Q, K, V)
        )

        if token_positions is None:
            token_positions = einx.rearrange(
                'seq -> b... seq',
                torch.arange(seq_len, device=x.device),
                b = [1] * len(b)
            )

        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        if self.theta is not None:
            Q = self.pos_encoder.forward(Q, token_positions)
            K = self.pos_encoder.forward(K, token_positions)

        o = scaled_dot_product_attention(Q, K, V) # [... head seq d_k]
        o = rearrange(o, "... heads seq d_k -> ... seq (heads d_k)").contiguous()
        o = self.o_proj(o)

        return o
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU_FFN(d_model=d_model, d_ff=d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x