import jax
import numpy as np
import jax.numpy as jnp

import functools

from flax import (
    linen as nn,
)

from typing import Any, Callable, Iterable, Mapping, Optional, Union, List

from dataclasses import field
from einops import rearrange
from .core import EnergyLayerNorm, HopfieldTransformer, SelfAttention


def normal(key, shape, mean=0.0, std=1.0):
    x = jax.random.normal(key, shape)
    return (x * std) + mean


def check_type(x):
    if type(x) == float or type(x) == int:
        return None
    else:
        try:
            flag = len(x.shape) == 1 and type(x) == jnp.ndarray
        except:
            return None
        else:
            return None if flag else 0


class EnergyTransformer(nn.Module):
    embed_dim: int
    out_dim: int
    nheads: int = 12
    head_dim: int = 64
    multiplier: float = 4.0
    attn_beta_init: Optional[float] = None
    use_biases_attn: bool = False
    use_biases_chn: bool = False
    use_biases_norm: bool = True
    eps: float = 1e-05
    alpha: float = 0.1
    depth: int = 12
    block: int = 2
    dtype: Any = jnp.float32
    kernel_size: List = field(default_factory=lambda: [3, 3])
    kernel_dilation: List = field(default_factory=lambda: [1, 1])
    compute_corr: bool = True
    vary_noise: bool = False
    chn_atype: str = "relu"
    noise_std: float = 0.02

    def setup(self):
        self.encoder = nn.Dense(self.embed_dim, dtype=self.dtype)

        self.encode_block = [
            HopfieldTransformer(
                self.embed_dim,
                self.nheads,
                self.head_dim,
                self.multiplier,
                self.attn_beta_init,
                self.use_biases_attn,
                self.use_biases_chn,
                self.dtype,
                self.chn_atype,
            )
            for _ in range(self.block)
        ]

        self.encode_norm = [
            EnergyLayerNorm(self.embed_dim, dtype=self.dtype, eps=self.eps)
            for _ in range(self.block)
        ]

        self.cls_tkn = self.param(
            "cls_token", nn.initializers.normal(0.02), (1, self.embed_dim), self.dtype
        )

        self.grad_proj = nn.Dense(self.embed_dim, dtype=self.dtype)

        if self.compute_corr:
            self.conv = nn.Conv(
                self.nheads,
                kernel_size=self.kernel_size,
                kernel_dilation=self.kernel_dilation,
                use_bias=False,
                dtype=self.dtype,
            )

        self.adj_pj = nn.Dense(self.nheads, dtype=self.dtype)
        self.decoder = nn.Dense(self.out_dim, dtype=self.dtype)
        self.pos_pj = nn.Dense(self.embed_dim, dtype=self.dtype)

    def gen_noise(self, key, t: int, std: float = 0.02, gamma: float = 0.55, shape=[1]):
        if key is None:
            return key, 0

        key, _ = jax.random.split(key)

        if self.vary_noise:
            std = gamma / pow(1 + t, gamma)

        return key, normal(key, shape, 0.0, std)

    def encode_features(
        self,
        x: jnp.ndarray,
        corr: jnp.ndarray,
        key: jnp.ndarray = None,
        return_stats: bool = False,
    ) -> jnp.ndarray:
        """Pass tokens into the Energy Transformer"""
        embeddings, energies = [], []
        x0 = x
        for i in range(self.block):
            for t in range(self.depth):
                g = self.encode_norm[i](x)
                e, grad = self.encode_block[i](g, corr)

                if return_stats:
                    embeddings.append(x)
                    energies.append(e)

                key, noise = self.gen_noise(
                    key, t, std=self.noise_std, shape=grad.shape
                )

                x = x - self.alpha * grad + (self.alpha**0.5) * noise

            embeddings.append(x)
        return x, grad, (energies, embeddings)

    def correlate(self, x: jnp.ndarray, adj: jnp.ndarray) -> jnp.ndarray:
        adj = self.adj_pj(adj)

        if not (self.compute_corr):
            return adj

        edges = (x @ x.T)[..., None]
        edges = self.conv(edges) * adj
        return edges

    def encoder_prep_tokens(
        self, x: jnp.ndarray, pos_embeddings: jnp.ndarray
    ) -> jnp.ndarray:
        x = jnp.concatenate([self.cls_tkn, x], axis=0)
        return x + self.pos_pj(pos_embeddings) if pos_embeddings is not None else x

    def forward(
        self,
        X: jnp.ndarray,
        A: jnp.ndarray,
        P: jnp.ndarray,
        key: jnp.ndarray,
        return_stats: bool,
        training: bool,
    ):
        x = self.encoder(X)

        x = self.encoder_prep_tokens(x, P)

        ahat = self.correlate(x, A)

        x, g, stats = self.encode_features(x, ahat, key, return_stats)

        x = self.decoder(x)
        cls_tkn, x = x[:1], x[1:]
        ahat = nn.Dense(1, use_bias=False)(ahat)

        if return_stats:
            return cls_tkn, x, ahat, stats
        return cls_tkn, x, ahat, None

    @nn.compact
    def __call__(
        self,
        X: jnp.ndarray,
        A: jnp.ndarray,
        P: jnp.ndarray,
        key: jnp.ndarray = None,
        return_stats: bool = False,
        training: bool = False,
    ):
        fn = jax.vmap(self.forward, in_axes=(0, 0, 0, None, None, None))

        cls_tkns, xs, ahs, stats = fn(X, A, P, key, return_stats, training)
        return {"CLS": cls_tkns, "X": xs, "A": ahs, "stats": stats}
