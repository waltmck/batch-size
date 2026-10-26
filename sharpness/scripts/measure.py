"""The measurements of the write-up (sharpness/sharpness.ipynb), run inside the training process.

At every instant, for each operator A_t (the raw Hessian H_t of the mean loss over the run's training sequences; for Adam
and Adafactor also the preconditioned P_t^{-1/2} H_t P_t^{-1/2}; every operator restricted to the parameters other than the
token embeddings token_embed_in and token_embed_out, hessian.py):
  lower bound  v^T A_t v <= lambda_max(A_t). v is the top Ritz vector of Lanczos (lanczos.direction: random start, residual
               rule) on the sample operator of M_SAMPLE sequences drawn uniformly with
               replacement from the training sequences; v^T A_t v is then the mean over all training sequences of the 1-d
               variable v^T A_x v, estimated by the robust mean [LMSV25] at confidence DELTA from N_LOWER fresh sequences
               drawn the same way, independently of the direction sample.
At the UPPER_CHECKPOINT_TOKENS instants, for each operator:
  upper bound  lambda_max(A_t) <= E[lambda_max(A_m)] for the sample operator of m = M_SAMPLE sequences. K_UPPER independent
               samples; for each, Lanczos from a uniformly random unit start gives the probabilistic upper bound U_k of
               lambda_max(A_m) at confidence DELTA (lanczos.upper_bound); the K_UPPER values of U_k go through the robust
               mean at confidence DELTA.
Both Lanczos runs: at most LANCZOS_MAX steps, early stop at EPS_STAR after at least LANCZOS_MIN steps. The direction and
fresh samples of an instant are shared by its operators; all draws are seeded by (run seed, update, purpose, sample).
Instants: j*2^21 tokens for j = 8, 16, ..., 280 (every 16.8M tokens) and FINAL_INSTANT_TOKENS (j = 284), one B=4096 update before the
end of the budget, so that every measured instant has a next update whose gradient defines P_t; 36 in all.
"""
import time
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from . import hessian as H, lanczos as LZ, robust_mean as RM

M_SAMPLE = 512                       # sequences per sample operator (direction search and upper-bound samples)
N_LOWER = 16_384                     # fresh sequences for the lower bound's mean
K_UPPER = 32                         # sample operators per upper bound
DELTA = 1e-4                         # confidence of every estimator (robust means, Lanczos upper bounds)
EPS_STAR = 1e-3                      # Lanczos early-stop threshold
LANCZOS_MAX, LANCZOS_MIN = 40, 10    # Lanczos step cap and floor
UPPER_CHECKPOINT_TOKENS = tuple(j * 2**21 for j in (56, 112, 168, 224))     # 117.4M, 234.9M, 352.3M, 469.8M tokens: fifths of the run
FINAL_INSTANT_TOKENS = 284 * 2**21                                              # 595.6M tokens: the last instant, one B=4096 update before the end
INSTANT_TOKENS = (*(j * 2**21 for j in range(8, 281, 8)), FINAL_INSTANT_TOKENS)  # 36 measured instants: every 16.8M tokens, then the final one
PRECONDITIONED = ('adamw', 'adafactor')
CONSTANTS = dict(m_sample=M_SAMPLE, n_lower=N_LOWER, k_upper=K_UPPER, delta=DELTA, eps_star=EPS_STAR, lanczos_max=LANCZOS_MAX, lanczos_min=LANCZOS_MIN,
                 upper_checkpoint_tokens=list(UPPER_CHECKPOINT_TOKENS), instant_tokens=list(INSTANT_TOKENS), micro=H.MICRO, adam_eps=H.ADAM_EPS,
                 excluded=list(H.EXCLUDED))


class Measurer:
    def __init__(self, params, graphdef, ds_train, optimizer_name, b2, seed):
        self.graphdef, self.optimizer_name, self.b2, self.seed = graphdef, optimizer_name, b2, int(seed)
        self.T = int(ds_train.shape[-1]); self.pool = ds_train.reshape(-1, self.T); self.n_seqs = int(self.pool.shape[0])   # the run's training sequences
        self.flat = H.Flat(params)
        self.mask = self.flat.ravel(H.embed_mask(params)); self.d = int(self.mask.sum())           # every operator lives on the non-embedding parameters
        self.kinds = ['raw'] + (['pre'] if optimizer_name in PRECONDITIONED else [])
        unravel = self.flat.unravel
        self._hv = jax.jit(lambda p, v, seqs: ravel_pytree(H.hvp_sample(p, unravel(v), seqs, graphdef))[0])
        self._curv = jax.jit(lambda p, w, seqs: H.curvatures(p, unravel(w), seqs, graphdef).reshape(-1))

    # ------------------------------------------------------------ sampling
    def _rng(self, *tags): return np.random.default_rng([self.seed, *tags])

    def _key(self, *tags):
        key = jax.random.key(self.seed)
        for t in tags: key = jax.random.fold_in(key, int(t))
        return key

    def sample(self, rng, n):                      # [n / MICRO, MICRO, T]: n sequences drawn uniformly with replacement from the training sequences
        idx = jnp.asarray(rng.integers(0, self.n_seqs, size=n))
        return jnp.take(self.pool, idx, axis=0).reshape(n // H.MICRO, H.MICRO, self.T)

    def random_unit(self, key):                    # uniformly random unit vector of the operators' space
        q = jax.random.normal(key, (self.flat.d,), dtype=jnp.float32) * self.mask
        return q / jnp.linalg.norm(q)

    # ------------------------------------------------------------ operators at an instant
    def preconditioner(self, opt_state, g):
        """P_t^{-1/2} as a flat vector (zero on the token embeddings) and a record of how it was formed; g is the gradient of the
        update the optimizer is about to take."""
        stage0 = H.optax_state(opt_state)[0]; params = H.pure(opt_state.model); gp = H.pure(g)
        assert hasattr(stage0, 'nu' if self.optimizer_name == 'adamw' else 'v_row'), f'unexpected first optimizer stage {type(stage0).__name__}'
        if self.optimizer_name == 'adamw':
            pinv_half, rec = H.adam_pinv_half(stage0, gp, self.b2), dict(count=int(stage0.count))
        else:
            pinv_half, clip = H.adafactor_pinv_half(stage0, gp, params, self.b2); rec = dict(count=int(stage0.count), clip=clip)
        return self.flat.ravel(H.like(pinv_half, opt_state.model)) * self.mask, rec

    def operators(self, opt_state, g):
        """kind -> (hv(v, seqs) on the sample operator restricted to the non-embedding parameters, the map v -> w with
        v^T A v = w^T H w, record)."""
        params, m = opt_state.model, self.mask
        ops = {'raw': (lambda v, seqs: m * self._hv(params, m * v, seqs), lambda v: m * v, {})}
        if 'pre' in self.kinds:
            p, rec = self.preconditioner(opt_state, g)
            ops['pre'] = (lambda v, seqs: p * self._hv(params, p * v, seqs), lambda v: p * v, rec)
        return ops

    # ------------------------------------------------------------ the two measurements
    def instant(self, k, opt_state, g):
        """Lower bounds at update k (g: the gradient of the update the optimizer takes from here)."""
        t0 = time.time(); ops = self.operators(opt_state, g); params = opt_state.model
        S = self.sample(self._rng(k, 1), M_SAMPLE); Tset = self.sample(self._rng(k, 2), N_LOWER)
        out = {}
        for kind in self.kinds:
            hv, to_w, rec = ops[kind]; t1 = time.time()
            v, info = LZ.direction(lambda x: hv(x, S), self.random_unit(self._key(k, 0, self.kinds.index(kind))), EPS_STAR, LANCZOS_MAX, LANCZOS_MIN)
            t2 = time.time()
            x = np.asarray(self._curv(params, to_w(v), Tset), dtype=np.float64)
            lower = RM.estimate(x, DELTA)
            out[kind] = dict(lower=lower, direction=info, x=dict(min=float(x.min()), median=float(np.median(x)), max=float(x.max())),
                             seconds=dict(direction=t2 - t1, lower=time.time() - t2), **({'precond': rec} if kind == 'pre' else {}))
        return dict(step=k, operators=out, seconds=time.time() - t0)

    def upper(self, k, opt_state, g):
        """Upper bounds at update k (an UPPER_CHECKPOINT_TOKENS instant)."""
        t0 = time.time(); ops = self.operators(opt_state, g); out = {}
        for kind in self.kinds:
            hv, _, rec = ops[kind]; gamma = LZ.gamma_threshold(self.d, DELTA); samples = []
            for j in range(K_UPPER):
                S = self.sample(self._rng(k, 3, j), M_SAMPLE); q1 = self.random_unit(self._key(k, 4, j)); t1 = time.time()
                u = LZ.upper_bound(lambda x: hv(x, S), q1, EPS_STAR, LANCZOS_MAX, LANCZOS_MIN, gamma)
                u['seconds'] = time.time() - t1; samples.append(u)
            out[kind] = dict(upper=RM.estimate([u['U'] for u in samples], DELTA), samples=samples, gamma=gamma, d=self.d,
                             **({'precond': rec} if kind == 'pre' else {}))
        return dict(step=k, operators=out, seconds=time.time() - t0)


def wandb_fields(entry, which):
    """Flat W&B metrics: which = 'lower' (an instant entry) or 'upper' (a checkpoint entry)."""
    f = {}
    for kind, o in entry['operators'].items():
        if which == 'lower':
            f.update({f'sharpness/{kind}_lower': o['lower']['mean'], f'sharpness/{kind}_lower_lo': o['lower']['lo'], f'sharpness/{kind}_lower_hi': o['lower']['hi'],
                      f'sharpness/{kind}_L': o['direction']['L'], f'sharpness/{kind}_steps': o['direction']['steps']})
        else:
            f.update({f'sharpness/{kind}_upper': o['upper']['mean'], f'sharpness/{kind}_upper_lo': o['upper']['lo'], f'sharpness/{kind}_upper_hi': o['upper']['hi'],
                      f'sharpness/{kind}_upper_steps': float(np.mean([s['steps'] for s in o['samples']]))})
    return f
