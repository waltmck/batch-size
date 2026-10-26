"""Train one run of the sharpness-vs-batch-size experiment the way the paper's Figure 1a runs were trained, measuring the
full-batch sharpness along the way as described in sharpness/sharpness.ipynb.

    python -m sharpness.scripts.train --optimizer sgd --batch-size 16 --peak-lr 0.5 --out sharpness/runs/sgd_b16        (from the repo root)
    python -m sharpness.scripts.train --optimizer adamw --batch-size 16 --hparams lm11m_fwedu_adam --out sharpness/runs/adamw_b16

Upstream pretraining code throughout: the config composed as main.py composes it (lm11m on FineWeb-Edu, max microbatch 16,
no weight decay, no clipping, 5% linear warmup + cosine to zero), the keys split as train_and_evaluate splits them
(seed 0), the data cut, permuted and split into train / validation microbatches by data.load_ds with the seed's data key
(so order, batch composition and validation set depend on B exactly as in the paper), the budget 20 tokens per parameter
(598,310,400 on one device, cut to whole updates), each update done by train.train_step (gradient accumulation over
microbatches of 16 when B > 16). Optimizers: 'adamw' (--peak-lr, --b1 or --t1 = beta1 half-life in tokens, --t2 = beta2
half-life in tokens; or --hparams lm11m_fwedu_adam for the paper's tuned per-batch-size values), 'sgd' (--b1 0 / omitted
= vanilla SGD) or 'adafactor' (--t2, which upstream turns into the exponent of its power-law second-moment schedule;
no first moment). Matmuls in TF32 (the A100 default); weights and optimizer state in fp32.

Measurements (measure.py) at the 36 instants j*2^21 tokens, j = 8, 16, ..., 280 and 284 (every 16.8M tokens; the last one
update of B=4096 before the end, so every instant has a next update): the lower bounds of the raw and (Adam, Adafactor)
preconditioned full-batch sharpness; at the four upper-bound checkpoints (56, 112, 168, 224 * 2^21 tokens) also the upper
bounds and a full checkpoint (optimizer state included), which is also written at the end. At those instants, at init and
at the end: validation loss (the run's own validation microbatches, upstream's eval_step) and the weights (snapshots/).
Everything is logged to W&B against tokens and to <run>/{eval.json, sharpness.json, sharpness_upper.json}. A rolling resume checkpoint every 2^25 tokens; per-update loss and lr in steps.f32. Updates run in
chunks of max(2^19, B*T) tokens under one jit/scan.
--sweep (hyperparameter brackets): no snapshots, checkpoints or measurements, validation loss at the end only; a non-finite
chunk writes diverged.json and exits non-zero. Finished runs (done.json / diverged.json) are skipped; interrupted runs
resume from the rolling checkpoint.
"""
import argparse, json, os, random, string, time
from functools import partial
import numpy as np
import jax, jax.numpy as jnp, optax, wandb
from flax import nnx
from omegaconf import OmegaConf
from . import common as C, hessian as H, measure as M
from train import loss_fn, train_step      # pretraining/train.py (upstream, unchanged)
from utils import flatten_dict             # pretraining/utils.py

RESUME_TOKENS = 2**25


def make_run_chunk(opt_graphdef, model_graphdef):
    @partial(jax.jit, donate_argnames=('opt_state',))
    def run_chunk(key, opt_state, batches):                # batches: [n, B, T] (B <= 16) or [n, grad_acc, 16, T] uint16
        def body(carry, batch):
            key, opt_state = carry
            key, opt_state, loss = train_step(key, opt_state, opt_graphdef, model_graphdef, batch)   # upstream update
            return (key, opt_state), loss
        (key, opt_state), losses = jax.lax.scan(body, (key, opt_state), batches)
        return key, opt_state, losses
    return run_chunk


@partial(jax.jit, static_argnames=('graphdef',))
def eval_loss(model_state, graphdef, valid):               # upstream eval_step (mean over microbatches of the microbatch mean) under a scan
    total, _ = jax.lax.scan(lambda acc, x: (acc + loss_fn(model_state, graphdef, x), None), jnp.zeros(()), valid)
    return total / valid.shape[0]


def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--optimizer', required=True, choices=C.OPTIMIZERS); ap.add_argument('--batch-size', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--peak-lr', type=float, default=None); ap.add_argument('--hparams', default=None, help="per-batch-size table under configs/hparams, e.g. 'lm11m_fwedu_adam' (replaces --peak-lr/--b1/--t1/--t2)")
    ap.add_argument('--b1', type=float, default=None, help='beta1 (adamw; sgd momentum, 0/omitted = vanilla)')
    ap.add_argument('--t1', type=int, default=None, help='beta1 half-life in tokens (adamw; replaces --b1)')
    ap.add_argument('--t2', type=int, default=None, help='beta2 half-life in tokens (adamw); for adafactor the number upstream uses as its power-law exponent 0.5^(B*T/t2)')
    ap.add_argument('--budget-tokens', type=int, default=None, help='num_tokens_train; default the paper\'s 20 tokens per parameter')
    ap.add_argument('--chunk-tokens', type=int, default=None, help='default max(2^19, B*T)'); ap.add_argument('--resume-tokens', type=int, default=RESUME_TOKENS)
    ap.add_argument('--sweep', action='store_true', help='bracket mode: no snapshots/checkpoints/measurements, eval at the end only')
    ap.add_argument('--wandb-mode', default='online'); ap.add_argument('--wandb-project', default='batch-size')
    ap.add_argument('--wandb-group', default='sharpness'); ap.add_argument('--wandb-name', default=None)
    ap.add_argument('--eval-batches', type=int, default=None, help='testing only: first N validation microbatches')
    ap.add_argument('--stop-after-chunks', type=int, default=None, help='testing only')
    ap.add_argument('--allow-code-change', action='store_true', help='resume even if the package hash differs from the recorded one')
    args = ap.parse_args(argv)
    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')

    B, out = args.batch_size, args.out
    for marker in ('done', 'diverged'):
        if os.path.exists(f'{out}/{marker}.json'): print(f'{out}: already finished ({marker}.json); nothing to do'); return
    os.makedirs(f'{out}/snapshots', exist_ok=True); os.makedirs(f'{out}/checkpoints', exist_ok=True)

    # --- config, model, data, optimizer: upstream train_and_evaluate step by step ---
    c = C.build_config(args.optimizer, B, peak_lr=args.peak_lr, b1=args.b1, t1=args.t1, t2=args.t2, budget_tokens=args.budget_tokens, hparams=args.hparams)
    mesh = C.set_mesh(c)
    model = C.build_model(c)
    if c.num_tokens_train is None: c.num_tokens_train = C.paper_budget(c)
    ds_train, ds_valid = C.load_data(c, mesh)                               # [n_train, micro, T], [n_valid, micro, T]
    T, micro, acc = int(c.model.T), int(c.opt.microbatch_size), int(c.opt.grad_acc_steps)
    K = len(ds_train) // acc; per_step = B * T; budget = K * per_step
    optimizer = C.build_optimizer(c, model, K)                              # resolves b1 / b2 from t1 / t2 in c.opt
    model_graphdef = nnx.graphdef(model)
    opt_graphdef, opt_state = nnx.split(optimizer)
    warmup = int(c.opt.warmup_frac * K)                                     # mirrors optimizer.get_optimizer's schedule, for logging
    lr_all = np.asarray(jax.vmap(optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup, K))(jnp.arange(K)), dtype=np.float32)
    if args.eval_batches: ds_valid = ds_valid[:args.eval_batches]

    # --- instants ---
    chunk_tokens = args.chunk_tokens or max(2**19, per_step); assert chunk_tokens % per_step == 0; chunk_steps = chunk_tokens // per_step
    resume_steps = args.resume_tokens // per_step
    steps_of = lambda tokens: sorted({t // per_step for t in tokens if 0 < t < budget})
    measure_steps, upper_steps = steps_of(M.INSTANT_TOKENS), steps_of(M.UPPER_CHECKPOINT_TOKENS)      # measured instants (all < K); upper-bound instants
    assert set(upper_steps) <= set(measure_steps)
    ckpt_steps = sorted(set(upper_steps) | {K})                             # full checkpoints (optimizer state): the upper-bound instants and the end
    instants = sorted({0, K} | set(measure_steps))                          # eval + weights at all of these
    assert all(t % per_step == 0 for t in (*M.INSTANT_TOKENS, *M.UPPER_CHECKPOINT_TOKENS, args.resume_tokens)), 'instants must be whole updates'
    for s in (resume_steps, *measure_steps):
        assert s % chunk_steps == 0, f'instant / resume point at {s} updates must be a multiple of the chunk ({chunk_steps} updates)'
    run_params = dict(optimizer=args.optimizer, hparams=args.hparams, batch_size=B, microbatch=micro, grad_acc=acc, K=K, n_tokens_train=int(c.num_tokens_train),
                      budget_tokens=budget, chunk_tokens=chunk_tokens, peak_lr=float(c.opt.peak_lr), b1=None if c.opt.b1 is None else float(c.opt.b1),
                      t1=None if c.opt.t1 is None else float(c.opt.t1), t2=None if c.opt.t2 is None else float(c.opt.t2),
                      b2=None if c.opt.b2 is None else float(c.opt.b2), seed=int(c.seed), matmul_precision='tensorfloat32')

    # --- logs, resume ---
    path = f'{out}/steps.f32'
    if os.path.exists(path): assert os.path.getsize(path) == K * 2 * 4, f'{path} has the wrong size for K={K}'
    steps_mm = np.memmap(path, dtype=np.float32, mode='r+' if os.path.exists(path) else 'w+', shape=(K, 2))   # loss, lr
    key, _, _ = C.keys(c.seed)                              # upstream's training key (only consumed by the disabled stochastic rounding)
    k = 0; resumed = False
    resume_path, manifest_path = f'{out}/checkpoints/resume.npz', f'{out}/run_manifest.json'
    if os.path.exists(resume_path):
        opt_state, extra = C.load_tree(resume_path, opt_state)
        saved = json.loads(str(extra['run_params']))
        assert saved == run_params, f'refusing to resume: checkpoint was written with {saved}, this run has {run_params}'
        manifest = json.load(open(manifest_path))
        recorded = (manifest['resumes'][-1] if manifest.get('resumes') else manifest)['code']['sharpness_code_sha256']
        current = C.code_info()['sharpness_code_sha256']
        if recorded != current and not args.allow_code_change:
            raise SystemExit(f'refusing to resume {out}: package hash {current[:12]} differs from the recorded {recorded[:12]} (--allow-code-change to override)')
        key, k = jax.random.wrap_key_data(jnp.asarray(extra['key'])), int(extra['step']); resumed = True
        assert k % chunk_steps == 0, f'resume point {k} is not a chunk boundary'
        print(f'resumed from update {k} ({k * per_step:,} tokens)')
    wandb_id_path = f'{out}/wandb_id.txt'
    wandb_id = open(wandb_id_path).read().strip() if os.path.exists(wandb_id_path) else ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    open(wandb_id_path, 'w').write(wandb_id)
    if resumed:
        manifest.setdefault('resumes', []).append(dict(time=utc(), step=k, code=C.code_info(), versions=C.versions()))
    else:
        manifest = dict(run_params=run_params, warmup_updates=warmup, tokens_per_update=per_step, n_valid_microbatches=int(len(ds_valid)),
                        instants=[s * per_step for s in instants], checkpoint_tokens=[s * per_step for s in ckpt_steps], resume_tokens=args.resume_tokens,
                        sharpness=None if args.sweep else dict(**M.CONSTANTS, n_train_sequences=int(len(ds_train)) * micro),
                        config=OmegaConf.to_container(c), data=C.data_meta(c), code=C.code_info(), versions=C.versions(),
                        devices=[f'{d} ({d.device_kind})' for d in jax.devices()], wandb=dict(project=args.wandb_project, group=args.wandb_group, id=wandb_id),
                        started=utc(), sweep=args.sweep)
    C.write_json(manifest_path, manifest)
    wandb.init(project=args.wandb_project, name=args.wandb_name or os.path.basename(out.rstrip('/')), group=args.wandb_group, id=wandb_id, resume='allow',
               mode=args.wandb_mode, config={**flatten_dict(OmegaConf.to_container(c)), **{f'run.{a}': getattr(args, a) for a in vars(args)}, 'run.K': K})
    wandb.define_metric('tokens'); wandb.define_metric('*', step_metric='tokens')      # charts against tokens (comparable across B); no explicit step, so re-logging after a resume is accepted
    run_chunk = make_run_chunk(opt_graphdef, model_graphdef)
    meas = None if args.sweep else M.Measurer(opt_state.model, model_graphdef, ds_train, args.optimizer, None if c.opt.b2 is None else float(c.opt.b2), c.seed)

    def save_checkpoint(k, path):
        C.save_tree(path, opt_state, step=k, tokens=k * per_step, key=jax.random.key_data(key), run_params=json.dumps(run_params, sort_keys=True))

    def merge_json(name, entry):
        p = f'{out}/{name}'; rows = json.load(open(p)) if os.path.exists(p) else []
        rows = [r for r in rows if r['step'] != entry['step']] + [entry]
        C.write_json(p, sorted(rows, key=lambda r: r['step']))

    def evaluate(k):
        v = float(eval_loss(opt_state.model, model_graphdef, ds_valid))
        wandb.log({'eval_loss': v, 'tokens': k * per_step})
        merge_json('eval.json', dict(step=k, tokens=k * per_step, eval_loss=v))
        print(f'eval @ {k * per_step:,} tokens: {v:.4f}', flush=True)

    def step_gradient(k):                                  # the minibatch gradient of update k, the one the optimizer takes from this instant
        if args.optimizer not in M.PRECONDITIONED: return None
        batch = ds_train[k * acc:(k + 1) * acc]
        return H.batch_gradient(opt_state.model, model_graphdef, batch.reshape(acc, micro, T) if acc > 1 else batch.reshape(micro, T))

    def fmt(o): return ' '.join(f"{kind} [{v['lower']['lo']:.4g}, {v['lower']['hi']:.4g}] L {v['direction']['L']:.4g} ({v['direction']['steps']} steps, {v['direction']['stop']})" for kind, v in o.items())

    def measure(k):
        g = step_gradient(k); tokens = k * per_step; lr = float(lr_all[k])
        e = meas.instant(k, opt_state, g); e.update(tokens=tokens, lr=lr)
        merge_json('sharpness.json', e); wandb.log({**M.wandb_fields(e, 'lower'), 'tokens': tokens})
        print(f'sharpness @ {tokens:,} tokens: {fmt(e["operators"])}  {e["seconds"]:.0f}s', flush=True)
        if k in upper_steps:
            u = meas.upper(k, opt_state, g); u.update(tokens=tokens, lr=lr)
            merge_json('sharpness_upper.json', u); wandb.log({**M.wandb_fields(u, 'upper'), 'tokens': tokens})
            print(f'upper bounds @ {tokens:,} tokens: ' + ' '.join(f"{kind} [{v['upper']['lo']:.4g}, {v['upper']['hi']:.4g}]" for kind, v in u['operators'].items()) + f'  {u["seconds"]:.0f}s', flush=True)

    def at_instant(k):                                     # everything that happens at a measured instant (not in sweep mode)
        if k in ckpt_steps: save_checkpoint(k, f'{out}/checkpoints/tok{k * per_step:010d}.npz')
        else: C.save_tree(f'{out}/snapshots/tok{k * per_step:010d}.npz', opt_state.model, step=k, tokens=k * per_step)
        evaluate(k)
        if k in measure_steps: measure(k)

    if k == 0 and not args.sweep: at_instant(0)
    chunks_done = 0
    while k < K:
        n = min(chunk_steps, K - k)
        batches = ds_train[k * acc:(k + n) * acc]
        batches = batches.reshape(n, acc, micro, T) if acc > 1 else batches.reshape(n, micro, T)
        t0 = time.time()
        key, opt_state, losses = run_chunk(key, opt_state, batches)
        losses = np.asarray(losses)                                           # blocks until the chunk has finished
        dt = time.time() - t0
        if not np.isfinite(losses).all():
            steps_mm.flush()
            C.write_json(f'{out}/diverged.json', dict(step=k, tokens=k * per_step, last_logged_loss=float(steps_mm[k - 1, 0]) if k else None, time=utc()))
            wandb.finish()
            raise RuntimeError(f'non-finite loss in updates {k}..{k + n - 1}')
        steps_mm[k:k + n, 0] = losses; steps_mm[k:k + n, 1] = lr_all[k:k + n]
        k += n; chunks_done += 1
        wandb.log({'train_loss': float(losses.mean()), 'lr': float(lr_all[k - 1]), 'tokens': k * per_step, 'updates_per_s': n / dt, 'tokens_per_s': n * per_step / dt})
        if args.sweep:
            if k == K: evaluate(k)
        elif k in instants:
            at_instant(k)
        if k % resume_steps == 0 or k in ckpt_steps:
            steps_mm.flush(); save_checkpoint(k, resume_path)
            C.write_json(f'{out}/progress.json', dict(step=k, tokens=k * per_step, updates_per_s=n / dt, time=utc()))
        if args.stop_after_chunks and chunks_done >= args.stop_after_chunks:
            steps_mm.flush(); save_checkpoint(k, resume_path)
            print(f'stopped after {chunks_done} chunks at update {k} (testing)'); wandb.finish(); return
    steps_mm.flush()
    C.write_json(f'{out}/done.json', dict(step=k, tokens=k * per_step, finished=utc()))
    if os.path.exists(resume_path): os.remove(resume_path)              # redundant once done (the end's full checkpoint holds the same state)
    wandb.finish()


if __name__ == '__main__':
    main()
