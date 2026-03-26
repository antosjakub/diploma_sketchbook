"""Neural network architectures for PINNs.

Components: FourierFeatures, RWFLinear, MLP, ModifiedMLP, AdaptiveActivation.
All built on equinox for clean pytree-based JIT-friendly models.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Sequence


# ---------------------------------------------------------------------------
# Fourier feature embedding
# ---------------------------------------------------------------------------

class FourierFeatures(eqx.Module):
    """Random Fourier feature embedding: gamma(x) = [cos(Bx), sin(Bx)]."""
    B: jax.Array  # (num_features, input_dim), frozen

    def __init__(self, input_dim: int, num_features: int, sigma: float, *, key):
        self.B = jax.random.normal(key, (num_features, input_dim)) * sigma

    def __call__(self, x):
        # x: (input_dim,)
        proj = self.B @ x  # (num_features,)
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)])

    @property
    def output_dim(self):
        return 2 * self.B.shape[0]


# ---------------------------------------------------------------------------
# Random Weight Factorization linear layer
# ---------------------------------------------------------------------------

class RWFLinear(eqx.Module):
    """Dense layer with W = diag(s) * V. Drop-in replacement for eqx.nn.Linear."""
    s: jax.Array  # (out_features,)
    V: jax.Array  # (out_features, in_features)
    bias: jax.Array  # (out_features,)

    def __init__(self, in_features: int, out_features: int, *, key):
        k1, k2 = jax.random.split(key)
        # Glorot-like init for V
        std = jnp.sqrt(2.0 / (in_features + out_features))
        self.V = jax.random.normal(k1, (out_features, in_features)) * std
        # s initialized to ones
        self.s = jnp.ones(out_features)
        self.bias = jnp.zeros(out_features)

    def __call__(self, x):
        W = self.s[:, None] * self.V  # diag(s) @ V
        return W @ x + self.bias


# ---------------------------------------------------------------------------
# Standard dense layer (thin wrapper for consistency)
# ---------------------------------------------------------------------------

class DenseLinear(eqx.Module):
    """Standard dense layer."""
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_features: int, out_features: int, *, key):
        std = jnp.sqrt(2.0 / (in_features + out_features))
        self.weight = jax.random.normal(key, (out_features, in_features)) * std
        self.bias = jnp.zeros(out_features)

    def __call__(self, x):
        return self.weight @ x + self.bias


# ---------------------------------------------------------------------------
# Adaptive activation
# ---------------------------------------------------------------------------

class AdaptiveActivation(eqx.Module):
    """sigma(a * x) with trainable per-layer scale a."""
    a: jax.Array
    act_fn: callable = eqx.field(static=True)

    def __init__(self, act_fn=jax.nn.tanh, init_scale: float = 1.0):
        self.a = jnp.array(init_scale)
        self.act_fn = act_fn

    def __call__(self, x):
        return self.act_fn(self.a * x)


# ---------------------------------------------------------------------------
# Activation dispatcher
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "tanh": jax.nn.tanh,
    "sin": jnp.sin,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
}


def _get_activation(name: str):
    if name in _ACTIVATIONS:
        return _ACTIVATIONS[name]
    raise ValueError(f"Unknown activation: {name}")


# ---------------------------------------------------------------------------
# Standard MLP
# ---------------------------------------------------------------------------

class MLP(eqx.Module):
    """Standard fully-connected MLP with configurable activation."""
    layers: list
    activations: list
    output_layer: eqx.Module

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        depth: int,
        activation: str = "tanh",
        use_rwf: bool = False,
        adaptive_act: bool = False,
        *,
        key,
    ):
        Linear = RWFLinear if use_rwf else DenseLinear
        act_fn = _get_activation(activation)

        keys = jax.random.split(key, depth + 1)
        dims = [in_dim] + [width] * depth
        self.layers = []
        self.activations = []
        for i in range(depth):
            self.layers.append(Linear(dims[i], dims[i + 1], key=keys[i]))
            if adaptive_act:
                self.activations.append(AdaptiveActivation(act_fn))
            else:
                self.activations.append(act_fn)
        self.output_layer = Linear(width, out_dim, key=keys[depth])

    def __call__(self, x):
        h = x
        for layer, act in zip(self.layers, self.activations):
            h = act(layer(h))
        return self.output_layer(h)


# ---------------------------------------------------------------------------
# Modified MLP with gating encoders
# ---------------------------------------------------------------------------

class ModifiedMLP(eqx.Module):
    """Modified MLP: input encoded through U and V branches, gated per layer.

    h = sigma(W*h + b)
    h = h * U_enc + (1-h) * V_enc
    """
    encoder_U: eqx.Module
    encoder_V: eqx.Module
    layers: list
    activations: list
    output_layer: eqx.Module

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        depth: int,
        activation: str = "tanh",
        use_rwf: bool = True,
        adaptive_act: bool = False,
        *,
        key,
    ):
        Linear = RWFLinear if use_rwf else DenseLinear
        act_fn = _get_activation(activation)

        keys = jax.random.split(key, depth + 3)
        self.encoder_U = Linear(in_dim, width, key=keys[0])
        self.encoder_V = Linear(in_dim, width, key=keys[1])

        self.layers = []
        self.activations = []
        dims = [in_dim] + [width] * depth
        for i in range(depth):
            self.layers.append(Linear(dims[i], dims[i + 1], key=keys[i + 2]))
            if adaptive_act:
                self.activations.append(AdaptiveActivation(act_fn))
            else:
                self.activations.append(act_fn)
        self.output_layer = Linear(width, out_dim, key=keys[depth + 2])

    def __call__(self, x):
        U = jax.nn.tanh(self.encoder_U(x))
        V = jax.nn.tanh(self.encoder_V(x))
        h = x
        for layer, act in zip(self.layers, self.activations):
            h = act(layer(h))
            h = h * U + (1.0 - h) * V
        return self.output_layer(h)


# ---------------------------------------------------------------------------
# Full network with optional Fourier front-end
# ---------------------------------------------------------------------------

class PINNNet(eqx.Module):
    """Complete PINN network: optional Fourier features + backbone MLP."""
    fourier: FourierFeatures | None
    backbone: eqx.Module

    def __init__(self, fourier, backbone):
        self.fourier = fourier
        self.backbone = backbone

    def __call__(self, tx):
        """tx: (1+d,) array with t=tx[0], x=tx[1:]."""
        if self.fourier is not None:
            embedded = self.fourier(tx)
            return self.backbone(embedded).squeeze()
        return self.backbone(tx).squeeze()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_network(config: dict, *, key) -> PINNNet:
    """Build a PINNNet from a config dict.

    Config keys:
        dim: int - spatial dimension d (input will be 1+d for t,x)
        arch: "mlp" | "modified_mlp"
        use_fourier: bool
        fourier_sigma: float (default 1.0)
        fourier_features: int (default 64)
        use_rwf: bool (default True)
        depth: int (default 4)
        width: int (default 128)
        activation: str (default "tanh")
        adaptive_act: bool (default False)
    """
    dim = config["dim"]
    input_dim = 1 + dim  # t + x
    arch = config.get("arch", "modified_mlp")
    use_fourier = config.get("use_fourier", True)
    fourier_sigma = config.get("fourier_sigma", 1.0)
    fourier_features = config.get("fourier_features", 64)
    use_rwf = config.get("use_rwf", True)
    depth = config.get("depth", 4)
    width = config.get("width", 128)
    activation = config.get("activation", "tanh")
    adaptive_act = config.get("adaptive_act", False)

    k1, k2 = jax.random.split(key)

    fourier = None
    backbone_in = input_dim
    if use_fourier:
        fourier = FourierFeatures(input_dim, fourier_features, fourier_sigma, key=k1)
        backbone_in = fourier.output_dim

    BackboneClass = ModifiedMLP if arch == "modified_mlp" else MLP
    backbone = BackboneClass(
        in_dim=backbone_in,
        out_dim=1,
        width=width,
        depth=depth,
        activation=activation,
        use_rwf=use_rwf,
        adaptive_act=adaptive_act,
        key=k2,
    )

    return PINNNet(fourier, backbone)
