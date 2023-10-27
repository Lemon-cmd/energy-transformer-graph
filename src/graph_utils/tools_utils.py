import jax
import torch
import functools
import numpy as np
import jax.numpy as jnp
from einops import rearrange
from flax.core.frozen_dict import FrozenDict


def get_nparams(params):
    nparams = 0
    for item in params:
        if isinstance(params[item], FrozenDict):
            nparams += get_nparams(params[item])
        else:
            nparams += params[item].size
    return nparams


def to_device_split(x):
    if x is None:
        return x

    devices = len(jax.devices())
    valid_size = x.shape[0] - (x.shape[0] - (x.shape[0] // devices * devices))

    x = rearrange(
        x[:valid_size], "(d b) ... -> d b ...", d=devices, b=valid_size // devices
    )
    return x


def get_ev(adj: jnp.ndarray, k: int = 10, flip_sign: bool = False):
    """Adapted baased on
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/add_positional_encoding.html
    """

    assert len(adj.shape) == 2

    ndeg = jnp.diag(adj.sum(-1) ** -0.5)
    lap = jnp.eye(adj.shape[0]) - ndeg @ adj @ ndeg

    eu, ev = jnp.linalg.eigh(lap)
    idx = jnp.real(ev[:, eu.argsort()])

    if ev.shape[1] < k + 1:
        ev = jnp.pad(ev, [(0, 0), (0, (k + 1) - ev.shape[1])])

    ev = ev[:, 1 : k + 1]

    if flip_sign:
        flips = -1 + 2 * np.random.randint(0, 2, size=ev.shape)
        return ev * jnp.asarray(flips)
    return ev


def get_uv(adj: jnp.ndarray, k: int = 10, flip_sign: bool = False):
    """Adapted based on https://github.com/shamim-hussain/egt
    Edge-augmented Graph Transformer
    https://arxiv.org/abs/2108.03348
    """
    assert len(adj.shape) == 2

    U, S, V = jnp.linalg.svd(adj)
    UV = jnp.concatenate([U, V], axis=0)

    if UV.shape[1] < k:
        UV = jnp.pad(UV, [(0, 0), (0, k - UV.shape[1])])
        return UV.reshape(-1, 2 * k)

    UV = UV[:, :k].reshape(-1, 2 * k)

    if flip_sign:
        flips = -1 + 2 * np.random.randint(0, 2, size=UV.shape)
        return UV * jnp.asarray(flips)
    return UV


def get_pos(embed_type: str, k: int = 10, flip_sign: bool = True):
    embed_type = embed_type.lower()

    if embed_type == "svd":
        pos_fn = jax.vmap(functools.partial(get_uv, k=k, flip_sign=flip_sign))
    else:
        pos_fn = jax.vmap(functools.partial(get_ev, k=k, flip_sign=flip_sign))

    return pos_fn
