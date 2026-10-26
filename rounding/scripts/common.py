"""Shared pieces of the rounding experiment: config / model / data / optimizer construction through the upstream pretraining
code, pytree IO, provenance. Self-contained: depends only on `pretraining/`. Run everything as modules from the repo root
(python -m rounding.scripts.train ...): `pretraining/` is put on sys.path here and upstream modules are imported by their
bare names (data, model, optimizer, factorized, train, utils), as upstream expects.

Fidelity to the paper's Figure 1a / Figure 9 Adafactor runs: the config is composed as pretraining/main.py composes it for
the lm30m sweeps (+model=lm11m +dataset=fwedu_gpt2 opt.optimizer=adafactor opt.max_microbatch_size=16, as in
pretraining/runs/lm30m/optimizer_comp/lm30m_adafactor_grid.yaml); the keys are split as train_and_evaluate splits them
(seed 0); the data come from upstream data.load_ds (order, batch composition and validation set depend on B exactly as in
the paper); the budget is the paper's 20 tokens per parameter (598,310,400 on one device) cut to whole updates; every
update is upstream train_step, which also holds the paper's weight rounding (pretraining/optimizer.py apply_updates:
model.param_dtype=bfloat16 stores the weights in bfloat16, opt.stochastic_round picks utils.to_bf16_stochastic over the
nearest rounding of the float32 sum p + u; activations and the optimizer state stay float32). Note that with bfloat16
weights the gradient of a single microbatch is bfloat16 too (jax.grad returns the parameter dtype), so at B <= 16 the
second moment and the update see bfloat16-rounded gradients, while at B >= 32 upstream accumulates microbatch gradients
in float32; this is the paper's code path and is kept.

The one departure from upstream is the second-moment schedule. Upstream get_optimizer converts t2 to b2 = 0.5^(B*T/t2)
and passes b2 to factorized.scale_by_factored_rms as `decay_rate`, which that transform uses as the EXPONENT c of
Adafactor's power-law schedule beta2_t = 1 - (t+1)^(-c) (Shazeer & Stern 2018), not as an EMA coefficient: at B <= 16
the exponent is > 0.99 for every t2 of the paper's grid and the second moment is a near-cumulative average of g^2.
`corrected_adafactor` keeps upstream's chain (factored second moment, update clipping at 1, relative step size, no first
moment) and gives scale_by_factored_rms a `decay_rate_fn` that realises the bias-corrected EMA with the constant
beta2 = 0.5^(B*T/t2) the paper describes, i.e. the second moment the paper's Adam (optax.adamw) uses:
v_t = beta2 v_{t-1} + (1-beta2) g_t^2, v_hat_t = v_t / (1-beta2^t). As a schedule: beta2_t = 1 - (1-beta2)/(1-beta2^t),
t = 1, 2, ... (beta2_1 = 0, so v_hat_1 = g_1^2 exactly as under upstream's power law).
"""
import hashlib, importlib.metadata, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PRETRAINING = REPO / 'pretraining'
if str(PRETRAINING) not in sys.path: sys.path.insert(0, str(PRETRAINING))

import jax, jax.numpy as jnp, optax
from jax.sharding import AxisType
from omegaconf import OmegaConf
import data as data_lib, model as model_lib, optimizer as optimizer_lib, factorized, utils   # pretraining/

MAX_MICROBATCH = 16                        # upstream sweeps: opt.max_microbatch_size=16
PRECISIONS = {'fp32':   dict(param_dtype='float32',  stochastic_round=False),   # the paper's Figure 1a setting
              'bf16':   dict(param_dtype='bfloat16', stochastic_round=False),   # bfloat16 weights, nearest rounding of p + u
              'bf16sr': dict(param_dtype='bfloat16', stochastic_round=True)}    # bfloat16 weights, stochastic rounding of p + u


# ---------------------------------------------------------------- config (as pretraining/main.py)
def load_config(overrides):
    from configs import resolver_setup  # noqa: F401  (registers the floordiv/mul/min/pow resolvers)
    from hydra import compose, initialize_config_dir
    from utils import flatten_dict
    with initialize_config_dir(str(PRETRAINING / 'configs'), version_base=None):
        c = compose(config_name='base', overrides=list(overrides))
    if 'bs_configs' in c:
        c = OmegaConf.merge(c, c.bs_configs[f'bs{c.opt.batch_size}']); del c.bs_configs
    if 'overwrite' in c:
        c = OmegaConf.merge(c, c.overwrite); del c.overwrite
    if 'scaling_1d' in c:
        OmegaConf.update(c, f'scaling.{c.scaling_1d.key}', c.scaling_1d.value, force_add=True); del c.scaling_1d
    if 'scaling' in c:
        for k, scaling in flatten_dict(c.scaling).items():
            OmegaConf.update(c, k, scaling * OmegaConf.select(c, k))
        del c.scaling
    OmegaConf.resolve(c)
    return c


def build_config(batch_size, peak_lr, t2, precision, budget_tokens=None):
    """lm11m on FineWeb-Edu (the paper's 30M model), Adafactor at the given peak learning rate and second-moment half-life t2
    (tokens), weights stored in the given precision (PRECISIONS); max microbatch 16; weight decay and gradient clipping at
    their base.yaml defaults (off). budget_tokens None = the paper's 20 tokens per parameter (resolved by train.py)."""
    p = PRECISIONS[precision]
    ov = ['+model=lm11m', '+dataset=fwedu_gpt2', 'opt.optimizer=adafactor', f'opt.batch_size={batch_size}', f'opt.peak_lr={peak_lr}',
          f'opt.t2={int(t2)}', f'opt.max_microbatch_size={MAX_MICROBATCH}', f'model.param_dtype={p["param_dtype"]}',
          f'opt.stochastic_round={str(p["stochastic_round"]).lower()}']
    if budget_tokens is not None: ov.append(f'num_tokens_train={int(budget_tokens)}')
    c = load_config(ov)
    assert c.opt.optimizer == 'adafactor' and c.model.param_dtype == p['param_dtype'] and bool(c.opt.stochastic_round) == p['stochastic_round']
    assert c.model.activ_dtype == 'float32'
    return c


def paper_budget(c):
    """upstream train_and_evaluate: tokens_params_ratio * (12 L D^2 + D V), V after rounding up to the device count."""
    return int(c.tokens_params_ratio * (12 * c.model.L * c.model.D**2 + c.model.D * c.model.V))


# ---------------------------------------------------------------- model, optimizer, data (upstream)
def keys(seed):
    """Same split as upstream train_and_evaluate: (key, key_model, key_dataset)."""
    return jax.random.split(jax.random.key(seed), 3)


def set_mesh(c):
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ('data', 'model'), axis_types=(AxisType.Auto, AxisType.Auto))
    jax.set_mesh(mesh)
    return mesh


def build_model(c):
    """After set_mesh(c). Rounds c.model.V up to the device count in place, as upstream does (idempotent)."""
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count())
    _, key_model, _ = keys(c.seed)
    return model_lib.create_sharded_model(c.model, key_model)


def ema_decay_rate(step, b2):
    """decay_rate_fn for factorized.scale_by_factored_rms: the bias-corrected EMA with constant beta2 written as a schedule,
    beta2_t = 1 - (1-beta2)/(1-beta2^t) with t = step + 1 (the transform calls it with its count, 0 at the first update)."""
    lb = math.log(b2); t = jnp.asarray(step + 1, jnp.float32)
    return 1.0 - jnp.expm1(lb) / jnp.expm1(t * lb)


def corrected_adafactor(learning_rate, b2):
    """optimizer_lib.adafactor (upstream, pretraining/optimizer.py) with the EMA schedule in place of the power law; the chain is
    otherwise identical: factored float32 second moment (min_dim_size_to_factor 128), update clipping at RMS 1, learning rate,
    relative step size (scale_by_param_block_rms)."""
    return optax.chain(
        factorized.scale_by_factored_rms(decay_rate=b2, min_dim_size_to_factor=128, decay_rate_fn=ema_decay_rate),
        optax.clip_by_block_rms(1.0),
        optax.scale_by_learning_rate(learning_rate),
        optax.scale_by_param_block_rms(),
    )


def build_optimizer(c, model, num_opt_steps):
    """upstream get_optimizer + ModelAndOptimizer for Adafactor, step by step (warmup-cosine schedule, t2 -> b2 resolved in place),
    with corrected_adafactor in place of optimizer_lib.adafactor."""
    o = c.opt
    assert o.optimizer == 'adafactor' and o.b1 is None and o.t1 is None and o.b2 is None and o.t2 is not None and o.peak_lr is not None
    assert o.weight_decay == 0 and o.clip_by_global_norm is None and o.b2_min is None
    warmup_steps = int(o.warmup_frac * num_opt_steps)
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, o.peak_lr, warmup_steps, num_opt_steps)
    o.b2 = float(utils.halflife_to_decay(o.t2, o.batch_size * c.model.T))
    return optimizer_lib.ModelAndOptimizer(model, corrected_adafactor(lr_schedule, o.b2), stochastic_round=o.stochastic_round)


def load_data(c, mesh):
    """upstream data.load_ds with the run's key_dataset: (ds_train, ds_valid) device arrays [n, microbatch, T]; c.num_tokens_train
    must be set (train.py resolves the paper budget when it is None, as upstream does)."""
    _, _, key_dataset = keys(c.seed)
    return data_lib.load_ds(key_dataset, mesh, c.ds_path, c.model.T, c.opt.microbatch_size, c.num_tokens_valid, c.num_tokens_train)


def data_meta(c):
    return dict(ds_path=os.path.expanduser(c.ds_path), ds_bytes=os.path.getsize(os.path.expanduser(c.ds_path)), seed=c.seed,
                n_tokens_train=int(c.num_tokens_train), n_tokens_valid=int(c.num_tokens_valid), microbatch=int(c.opt.microbatch_size),
                order='upstream data.load_ds: the first (train+valid) microbatches of the stream permuted with key_dataset; the last n_valid are the validation set')


# ---------------------------------------------------------------- pytree IO
def save_tree(path, tree, **extra):
    """Atomic write of the tree leaves (keyed by their keystr paths) + extra arrays/scalars under '__<name>' to an .npz.
    bfloat16 leaves are stored as float32 (exact); load_tree casts them back (exact)."""
    d = {jax.tree_util.keystr(p): np.asarray(l.astype(jnp.float32) if getattr(l, 'dtype', None) == jnp.bfloat16 else l)
         for p, l in jax.tree_util.tree_flatten_with_path(tree)[0]}
    d.update({f'__{k}': np.asarray(v) for k, v in extra.items()})
    with open(path + '.tmp', 'wb') as f: np.savez(f, **d)
    os.replace(path + '.tmp', path)


def load_tree(path, like, prefix=''):
    """`like`'s structure with leaves replaced by the arrays stored under prefix + the same paths; also the '__' extras."""
    z = np.load(path)
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(like)
    leaves = []
    for p, l in leaves_with_paths:
        a = z[prefix + jax.tree_util.keystr(p)]
        assert a.shape == tuple(l.shape), (prefix + jax.tree_util.keystr(p), a.shape, l.shape)
        leaves.append(jnp.asarray(a, dtype=l.dtype))
    extra = {k[2:]: z[k] for k in z.files if k.startswith('__')}
    return jax.tree_util.tree_unflatten(treedef, leaves), extra


def write_json(path, obj):
    json.dump(obj, open(path + '.tmp', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(path + '.tmp', path)


# ---------------------------------------------------------------- provenance
PACKAGE_FILES = ('common.py', 'train.py', 'figure9.py')


def code_info():
    """Commit, dirtiness (untracked files included) and a SHA-256 over this package's sources."""
    out = {}
    try:
        out['commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=REPO, capture_output=True, text=True).stdout
        out['dirty'] = status.strip() != ''
    except Exception as e:
        out['error'] = str(e)
    h = hashlib.sha256()
    for f in PACKAGE_FILES:
        p = Path(__file__).with_name(f)
        if p.exists(): h.update(f.encode()); h.update(p.read_bytes())
    out['rounding_code_sha256'] = h.hexdigest()
    return out


def versions():
    import platform
    v = {m: __import__(m).__version__ for m in ('jax', 'jaxlib', 'flax', 'optax', 'numpy', 'wandb', 'hydra', 'omegaconf')}
    v['python'] = platform.python_version()
    for d in ('jax-cuda12-plugin', 'jax-cuda12-pjrt', 'nvidia-cudnn-cu12', 'nvidia-cublas-cu12'):
        try: v[d] = importlib.metadata.version(d)
        except importlib.metadata.PackageNotFoundError: pass
    return v
