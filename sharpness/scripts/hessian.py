"""Hessian-vector products, per-sequence curvatures and the preconditioners of the write-up
(sharpness/sharpness.ipynb) on the upstream model; everything a Lanczos run or a lower-bound sample needs.

Vectors live in the parameter pytree (the nnx State the upstream loss_fn takes) or flat, through one ravel_pytree.
Hessians are never formed: H_S v for the mean loss over a sample S of sequences is the forward-over-reverse product
d/dt grad L(theta + t v) at t = 0 (jax.jvp of jax.grad), averaged over the sample's microbatches; the per-sequence
quadratic forms w^T H_x w for the lower bound are second directional derivatives d^2/dt^2 L_x(theta + t w), two nested
forward-mode passes over the per-sequence losses (no backward pass).

Preconditioners (P_t^{-1/2} as a parameter pytree), the scaling the optimizer applies at step t read from its live state
and the minibatch gradient g_t it is about to apply (the step is theta_{t+1} = theta_t - eta_t P_t^{-1} (momentum of) g_t):
  Adam       P_t = diag(sqrt(v_hat) + eps), v_hat = (b2 nu + (1-b2) g_t^2) / (1 - b2^(count+1)): optax.scale_by_adam's second
             moment refreshed with g_t and bias-corrected at the step's count, eps = optax.adamw's default 1e-8.
  Adafactor  P_t = diag(v_hat^(1/2) / (rho_b c_b)) with, per parameter block b, the factored second-moment estimate v_hat refreshed
             with g_t by the upstream scale_by_factored_rms (rank-one v_row v_col / mean(v_row) for factored blocks), the block
             clip c_b = 1 / max(1, RMS_b(g_t v_hat^(-1/2)) / threshold) (optax.clip_by_block_rms) and the relative step size
             rho_b = max(RMS_b(theta_t), min_scale) (optax.scale_by_param_block_rms); defaults as in upstream optimizer.adafactor.
Every operator (raw and preconditioned) is restricted to the parameters other than the token embeddings (token_embed_in and
token_embed_out): the input embedding's rarely-updated rows are dominated by rare tokens (and, with second moments near zero,
would dominate the preconditioned operators with a numerical artefact), and the output embedding would otherwise carry almost
the whole top eigenvector of the raw Hessian.
"""
import inspect
from functools import partial
import jax, jax.numpy as jnp, optax
from flax import nnx
from jax.flatten_util import ravel_pytree
from optax import tree_utils as otu
from . import common as _common                    # noqa: F401  (puts pretraining/ on sys.path)
from train import loss_fn                          # pretraining/train.py (upstream, unchanged)
import factorized, optimizer as optimizer_lib      # pretraining/ (upstream, unchanged)

MICRO = 16                                         # sequences per microbatch for HVPs and curvatures
ADAM_EPS = inspect.signature(optax.adamw).parameters['eps'].default                                   # 1e-8
_AF = inspect.signature(optimizer_lib.adafactor).parameters
AF_MIN_DIM = _AF['min_dim_size_to_factor'].default; AF_CLIP = _AF['clipping_threshold'].default     # 128, 1.0
AF_MIN_SCALE = inspect.signature(optax.scale_by_param_block_rms).parameters['min_scale'].default     # 1e-3


# ---------------------------------------------------------------- losses and derivatives
def seq_losses(params, graphdef, x):               # [m, T] -> [m]: upstream loss_fn's per-sequence mean cross-entropy (no mean over sequences)
    model = nnx.merge(graphdef, params)
    y = jnp.roll(x, -1, axis=1)
    losses = optax.softmax_cross_entropy_with_integer_labels(model(x).astype(jnp.float32), y)      # [m, T]
    return losses[:, :-1].mean(axis=1)


def _hvp(params, v, x, graphdef):                  # H(mean loss over x) v, forward-over-reverse
    return jax.jvp(lambda p: jax.grad(loss_fn)(p, graphdef, x), (params,), (v,))[1]


@partial(jax.jit, static_argnames=('graphdef',))
def hvp_sample(params, v, seqs, graphdef):         # seqs [n_mb, MICRO, T] -> H_S v for the mean loss over the sample
    acc, _ = jax.lax.scan(lambda a, x: (otu.tree_add(a, _hvp(params, v, x, graphdef)), None), otu.tree_zeros_like(v), seqs)
    return otu.tree_scale(1.0 / seqs.shape[0], acc)


@partial(jax.jit, static_argnames=('graphdef',))
def curvatures(params, w, seqs, graphdef):         # seqs [n_mb, MICRO, T] -> [n_mb, MICRO]: w^T H_x w per sequence, forward-over-forward
    def second(x):
        f = lambda p: seq_losses(p, graphdef, x)
        df = lambda p: jax.jvp(f, (p,), (w,))[1]
        return jax.jvp(df, (params,), (w,))[1]
    return jax.lax.map(second, seqs)


@partial(jax.jit, static_argnames=('graphdef',))
def batch_gradient(params, graphdef, batch):       # the step's gradient: [B, T] or [grad_acc, MICRO, T] -> mean gradient over its microbatches
    if batch.ndim == 2: return jax.grad(loss_fn)(params, graphdef, batch)
    acc, _ = jax.lax.scan(lambda a, x: (otu.tree_add(a, jax.grad(loss_fn)(params, graphdef, x)), None), otu.tree_zeros_like(params), batch)
    return otu.tree_scale(1.0 / batch.shape[0], acc)


# ---------------------------------------------------------------- pytree helpers
def pure(tree):                                    # array pytree of an nnx State, as ModelAndOptimizer.update feeds optax
    return nnx.to_arrays(nnx.pure(nnx.state(tree)))


def like(tree, template):                          # the leaves of `tree` (same order, same shapes) in `template`'s pytree structure
    leaves = jax.tree.leaves(tree); tl = jax.tree.leaves(template); assert len(leaves) == len(tl)
    for a, b in zip(leaves, tl): assert a.shape == b.shape, (a.shape, b.shape)
    return jax.tree.unflatten(jax.tree.structure(template), leaves)


EXCLUDED = ('token_embed_in', 'token_embed_out')   # parameter blocks outside every operator


def embed_mask(params):                            # 0 on the token embeddings, 1 elsewhere
    return jax.tree.unflatten(jax.tree.structure(params), [jnp.zeros_like(x) if any(e in jax.tree_util.keystr(p) for e in EXCLUDED) else jnp.ones_like(x)
                                                           for p, x in jax.tree_util.tree_flatten_with_path(params)[0]])


class Flat:
    """Flat <-> pytree view of the parameter space (one ravel_pytree of the parameters)."""
    def __init__(self, params):
        flat, self.unravel = ravel_pytree(params); self.d = int(flat.shape[0])
    def ravel(self, tree): return ravel_pytree(tree)[0]


# ---------------------------------------------------------------- preconditioners
def optax_state(opt_state):                        # the live optax chain state (tuple of stage states, array leaves)
    return nnx.to_arrays(nnx.pure(opt_state.opt_state))


def adam_pinv_half(stage0, g, b2):
    """P_t^{-1/2} (array pytree) from optax.scale_by_adam's state (count, mu, nu) and the step's gradient g."""
    nu = jax.tree.map(lambda n, gg: (1.0 - b2) * gg**2 + b2 * n, stage0.nu, g)      # as optax.tree_utils.tree_update_moment_per_elem_norm
    bias_correction = 1.0 - b2 ** (stage0.count + 1)                                   # as optax.tree_utils.tree_bias_correction
    return jax.tree.map(lambda n: 1.0 / jnp.sqrt(jnp.sqrt(n / bias_correction) + ADAM_EPS), nu)


def adafactor_pinv_half(stage0, g, params, b2):
    """P_t^{-1/2} (array pytree) and the per-block clip factors c_b from the upstream factored-RMS state and the step's gradient g."""
    tx0 = factorized.scale_by_factored_rms(decay_rate=b2, min_dim_size_to_factor=AF_MIN_DIM)        # stage 0 of upstream optimizer.adafactor
    u, state = tx0.update(g, stage0, params)                                                        # the scaled gradient and the statistics refreshed with g
    paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(params)[0]]
    out, clip = [], {}
    for path, p, vr, vc, v, uu in zip(paths, jax.tree.leaves(params), jax.tree.leaves(state.v_row), jax.tree.leaves(state.v_col), jax.tree.leaves(state.v),
                                      jax.tree.leaves(u)):
        dims = factorized._factored_dims(p.shape, True, AF_MIN_DIM)
        if dims is not None:                                                                        # rank-one estimate, as factorized._update
            d1, d0 = dims; reduced_d1 = d1 - 1 if d1 > d0 else d1
            row_col_mean = jnp.mean(vr, axis=reduced_d1, keepdims=True)
            vinv_sqrt = jnp.expand_dims((vr / row_col_mean) ** -0.5, axis=d0) * jnp.expand_dims(vc ** -0.5, axis=d1)
        else:
            vinv_sqrt = v ** -0.5
        c_b = float(1.0 / jnp.maximum(1.0, jnp.sqrt(jnp.mean(uu**2)) / AF_CLIP))                          # optax.clip_by_block_rms
        rho_b = float(jnp.maximum(jnp.sqrt(jnp.mean(p**2)), AF_MIN_SCALE))                                   # optax.scale_by_param_block_rms
        clip[path] = c_b
        out.append(jnp.sqrt(rho_b * c_b) * jnp.sqrt(vinv_sqrt))
    return jax.tree.unflatten(jax.tree.structure(params), out), clip
