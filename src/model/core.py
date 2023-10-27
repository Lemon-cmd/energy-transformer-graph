import jax
import jax.numpy as jnp

import flax.linen as nn

from typing import Optional, Any, Callable


def exists(val):
    return val is not None


def default(val, def_val):
    return val if exists(val) else def_val


class EnergyLayerNorm(nn.Module):
    """
    Perform layer norm on the last dimension of input
    While an energy could be defined for this, it is easier to just define the forward operation (activation function) since the
    energy calculation is not needed for the dynamics of the network
    """

    in_dim: int
    dtype: Any = jnp.float32
    use_bias: bool = (
        True  # Whether to use a bias in the layer normalization step or not
    )
    eps: float = 1e-05  # Prevent division by 0

    def setup(self):
        if self.use_bias:
            self.bias = self.param(
                "bias", nn.initializers.zeros, (self.in_dim), self.dtype
            )

        self.gamma = self.param("gamma", nn.initializers.ones, (1), self.dtype)

    def forward(self, x: jnp.ndarray):
        xmeaned = x - x.mean(-1, keepdims=True)
        v = (
            self.gamma
            * xmeaned
            / jnp.sqrt((xmeaned**2.0).mean(-1, keepdims=True) + self.eps)
        )

        if self.use_bias:
            return v + self.bias
        return v

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        return self.forward(x)


class HNN(nn.Module):
    """Hopfield ReLU Layer"""

    in_dim: int
    multiplier: float = 4
    use_bias: bool = True
    dtype: Any = jnp.float32
    fn: Callable = jax.nn.relu

    def setup(self):
        hid_dim = int(self.multiplier * self.in_dim)
        self.proj = nn.Dense(hid_dim, use_bias=self.use_bias, dtype=self.dtype)

    def energy(self, g: jnp.ndarray, adj: jnp.ndarray):
        A = self.fn(self.proj(g))
        return -0.5 * jnp.square(A).sum()

    def energy_and_grad(self, g: jnp.ndarray, adj: jnp.ndarray):
        return jax.value_and_grad(self.energy)(g, adj)

    @nn.compact
    def __call__(self, g: jnp.ndarray, adj: jnp.ndarray):
        return self.energy_and_grad(g, adj)


class HNN_LSE(nn.Module):
    """Hopfield Softmax Layer"""

    in_dim: int
    multiplier: float = 4
    use_bias: bool = True
    beta_init: float = 0.01
    dtype: Any = jnp.float32

    def setup(self):
        hid_dim = int(self.multiplier * self.in_dim)
        self.proj = nn.Dense(hid_dim, use_bias=self.use_bias, dtype=self.dtype)

        self.beta = hid_dim**0.5

    def energy(self, g: jnp.ndarray, adj: jnp.ndarray):
        h = self.proj(g)
        A = jax.nn.logsumexp(self.beta * h, axis=-1)
        return (-1.0 / self.beta) * A.sum()

    def energy_and_grad(self, g: jnp.ndarray, adj: jnp.ndarray):
        return jax.value_and_grad(self.energy)(g, adj)

    @nn.compact
    def __call__(self, g: jnp.ndarray, adj: jnp.ndarray):
        return self.energy_and_grad(g, adj)


class Attention(nn.Module):
    """The energy of attention for a single head"""

    in_dim: int
    nheads: int = 12
    head_dim: int = 64
    use_bias: bool = True
    dtype: Any = jnp.float32
    beta_init: Optional[float] = None

    def setup(self):
        self.Wk = self.param(
            "Wk",
            nn.initializers.normal(0.002),
            (self.nheads, self.head_dim, self.in_dim),
            self.dtype,
        )
        self.Wq = self.param(
            "Wq",
            nn.initializers.normal(0.002),
            (self.nheads, self.head_dim, self.in_dim),
            self.dtype,
        )

        self.Hw = self.param(
            "Hw", nn.initializers.normal(0.002), (self.nheads, self.nheads), self.dtype
        )

        if self.use_bias:
            self.Bk = self.param(
                "Bk", nn.initializers.zeros, (self.nheads, self.head_dim), self.dtype
            )
            self.Bq = self.param(
                "Bq", nn.initializers.zeros, (self.nheads, self.head_dim), self.dtype
            )

        # self.betas = jnp.ones(self.nheads, dtype=self.dtype) * default(self.beta_init, 1.0 / jnp.sqrt(self.head_dim))

        self.betas = self.param(
            "betas",
            lambda key, shape, dtype: nn.initializers.ones(key, shape, dtype)
            * default(self.beta_init, 1.0 / jnp.sqrt(self.head_dim)),
            (self.nheads),
            self.dtype,
        )

    def energy(self, g: jnp.ndarray, adj: jnp.ndarray):
        """Return the energy of the block"""
        K = jnp.einsum("kd, hzd -> khz", g, self.Wk)  # kseq, heads, head_dim
        Q = jnp.einsum("qd, hzd -> qhz", g, self.Wq)  # qseq, heads, head_dim

        if self.use_bias:
            K = K + self.Bk
            Q = Q + self.Bq

        A1 = jnp.einsum("h, qhz, khz -> hqk", self.betas, Q, K)  # NHeads, Nseq, Nseq

        # Attention Matrix times Adjacency Matrix
        if adj is not None:
            A11 = (A1.transpose(1, 2, 0) @ self.Hw) * adj

            A11 = jnp.where(
                A11 == 0, -jnp.inf, A11
            )  # Avoid empty edges s.t. the gradient (softmax) does not account empty edges as part of the distribution

            A21 = jax.nn.logsumexp(A11, 1)  # Nseq, Nheads

            A21 = jnp.where(A21 == -jnp.inf, 0, A21)

            A31 = A21.sum(0)  # Nheads

            A4 = ((-1.0 / self.betas) * A31).sum()
            return A4

        A2 = jax.nn.logsumexp(A1.transpose(1, 2, 0) @ self.Hw, 1)  # Nseq, NHeads
        A3 = A2.sum(0)  # Nheads
        A4 = ((-1.0 / self.betas) * A3).sum()
        return A4

    def energy_and_grad(self, g: jnp.ndarray, adj: jnp.ndarray):
        return jax.value_and_grad(self.energy)(g, adj)

    @nn.compact
    def __call__(self, g: jnp.ndarray, adj: jnp.ndarray = None):
        return self.energy_and_grad(g, adj)


class HopfieldTransformer(nn.Module):
    """Full energy transformer"""

    in_dim: int
    nheads: int = 12
    head_dim: int = 64
    multiplier: float = 4.0
    attn_beta_init: Optional[float] = None
    use_biases_attn: bool = False
    use_biases_chn: bool = False
    dtype: Any = jnp.float32
    atype: str = "relu"

    def setup(self):
        self.attn = Attention(
            in_dim=self.in_dim,
            nheads=self.nheads,
            head_dim=self.head_dim,
            use_bias=self.use_biases_attn,
            beta_init=self.attn_beta_init,
            dtype=self.dtype,
        )

        if self.atype == "relu":
            self.chn = HNN(
                in_dim=self.in_dim,
                multiplier=self.multiplier,
                use_bias=self.use_biases_chn,
                dtype=self.dtype,
            )
        elif self.atype == "gelu":
            self.chn = HNN(
                in_dim=self.in_dim,
                multiplier=self.multiplier,
                use_bias=self.use_biases_chn,
                dtype=self.dtype,
                fn=jax.nn.gelu,
            )
        else:
            self.chn = HNN_LSE(
                in_dim=self.in_dim,
                multiplier=self.multiplier,
                use_bias=self.use_biases_chn,
                dtype=self.dtype,
            )

    def energy(self, g: jnp.ndarray, adj: jnp.ndarray, **kwargs):
        energy = self.attn.energy(g, adj) + self.chn.energy(g, adj)
        return energy

    def energy_and_grad(self, g: jnp.ndarray, adj: jnp.ndarray, **kwargs):
        return jax.value_and_grad(self.energy)(g, adj, **kwargs)

    @nn.compact
    def __call__(self, g: jnp.ndarray, adj: jnp.ndarray, **kwargs):
        return self.energy_and_grad(g, adj, **kwargs)
