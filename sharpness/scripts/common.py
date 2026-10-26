"""Shared pieces of the sharpness-vs-batch-size experiment: config / model / optimizer / data construction through the
upstream pretraining code (unchanged), pytree IO. Self-contained: depends only on `pretraining/`.

Fidelity to the paper's Figure 1a runs (pretraining/main.py + train.py with the grid overrides): the config is composed the
way main.py does it; the keys are split as train_and_evaluate splits them (seed 0); the data are loaded by upstream
data.load_ds, which cuts the first (train + valid) tokens of the FineWeb-Edu stream into microbatches of min(B, 16)
consecutive sequences, permutes the microbatches with the seed's data key and takes the last 20M tokens' worth as the
validation set (so order, batch composition and validation set depend on B exactly as in the paper); the budget is
20 tokens per parameter (598,310,400 on one device: upstream rounds V up to a multiple of the device count before
counting parameters, so a multi-device TPU host would have used a slightly larger budget) cut to whole updates; updates
use upstream train_step. The sharpness measurements (measure.py) draw their sequences from the run's own training set.

Run everything as modules from the repo root (python -m sharpness.scripts.train ...): `pretraining/` is put on sys.path here and
upstream modules are imported by their bare names (data, model, optimizer, train, utils), as upstream expects.
"""
import hashlib, importlib.metadata, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PRETRAINING = REPO / 'pretraining'
if str(PRETRAINING) not in sys.path: sys.path.insert(0, str(PRETRAINING))

import jax, jax.numpy as jnp
from jax.sharding import AxisType
from omegaconf import OmegaConf
import data as data_lib, model as model_lib, optimizer as optimizer_lib   # pretraining/

MAX_MICROBATCH = 16                        # upstream sweeps: opt.max_microbatch_size=16
OPTIMIZERS = ('adamw', 'sgd', 'adafactor')


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


def build_config(optimizer, batch_size, peak_lr=None, b1=None, t1=None, t2=None, budget_tokens=None, hparams=None):
    """lm11m on FineWeb-Edu (the paper's 30M model) with the grid sweeps' settings (max microbatch 16; weight decay,
    clipping and stochastic rounding at their base.yaml defaults: off) and the given optimizer:
    'adamw'     upstream optax.adamw; beta1 from b1 or the token half-life t1; beta2 from the token half-life t2
    'sgd'       upstream SGD (m <- b1*m + g, no dampening); b1 None/0 is vanilla SGD; no second moment
    'adafactor' upstream Adafactor (factored second moment; t2 is converted to 0.5^(B*T/t2) and used by upstream as the exponent of the
                power-law schedule beta2_t = 1 - (t+1)^(-exponent), not as a half-life; update RMS clipped at 1, relative step size; no first moment)
    hparams: a per-batch-size table under configs/hparams (e.g. 'lm11m_fwedu_adam', the paper's tuned Adam values),
    composed exactly as upstream does with +hparams=...; then peak_lr/b1/t1/t2 must be None.
    budget_tokens None = the paper's 20 tokens per parameter (resolved by train.py from the model dims, as upstream does)."""
    assert optimizer in OPTIMIZERS, optimizer
    if hparams is not None:
        assert peak_lr is None and b1 is None and t1 is None and t2 is None, 'hparams tables carry the optimizer hyperparameters'
        ov = [f'+hparams={hparams}', f'opt.batch_size={batch_size}', f'opt.max_microbatch_size={MAX_MICROBATCH}']
    else:
        assert peak_lr is not None
        if optimizer == 'adafactor' or (optimizer == 'sgd' and not b1) or t1 is not None: b1 = None
        ov = ['+model=lm11m', '+dataset=fwedu_gpt2', f'opt.optimizer={optimizer}', f'opt.batch_size={batch_size}', f'opt.peak_lr={peak_lr}',
              f'opt.max_microbatch_size={MAX_MICROBATCH}']
        if b1 is not None: ov.append(f'opt.b1={b1}')
        if t1 is not None: ov.append(f'opt.t1={int(t1)}')
        if t2 is not None: ov.append(f'opt.t2={int(t2)}')
    if budget_tokens is not None: ov.append(f'num_tokens_train={int(budget_tokens)}')
    c = load_config(ov)
    assert c.opt.optimizer == optimizer, (c.opt.optimizer, optimizer)
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


def build_optimizer(c, model, num_opt_steps):
    tx = optimizer_lib.get_optimizer(c.opt, num_opt_steps, c.opt.batch_size * c.model.T)   # resolves b1/b2 from t1/t2 in place
    return optimizer_lib.ModelAndOptimizer(model, tx, stochastic_round=c.opt.stochastic_round)


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
    """Atomic write of the tree leaves (keyed by their keystr paths) + extra arrays/scalars under '__<name>' to an .npz."""
    d = {jax.tree_util.keystr(p): np.asarray(l) for p, l in jax.tree_util.tree_flatten_with_path(tree)[0]}
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
PACKAGE_FILES = ('common.py', 'hessian.py', 'lanczos.py', 'robust_mean.py', 'measure.py', 'train.py', 'bracket.py')


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
    out['sharpness_code_sha256'] = h.hexdigest()
    return out


def versions():
    import platform
    v = {m: __import__(m).__version__ for m in ('jax', 'jaxlib', 'flax', 'optax', 'numpy', 'wandb', 'hydra', 'omegaconf')}
    v['python'] = platform.python_version()
    for d in ('jax-cuda12-plugin', 'jax-cuda12-pjrt', 'nvidia-cudnn-cu12', 'nvidia-cublas-cu12'):
        try: v[d] = importlib.metadata.version(d)
        except importlib.metadata.PackageNotFoundError: pass
    return v
