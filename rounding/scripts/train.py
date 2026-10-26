"""One run of the rounding experiment: Adafactor with the corrected second moment (common.py) on lm11m / FineWeb-Edu at one
batch size, peak learning rate, half-life t2 and weight precision (fp32 | bf16 nearest | bf16 stochastic), every update
through upstream train_step. Outputs in --out: run_manifest.json, steps.f32 (per-update loss and learning rate), curve.json
(loss on the first --curve-batches validation microbatches every --curve-tokens, a cheap training curve), eval.json (the
paper's validation loss at the end), done.json. A non-finite loss, or a curve loss above twice the uniform-prediction loss
ln V (the model has blown up; losses can grow to 1e13 and stay finite), writes diverged.json and stops. Restartable from the
rolling checkpoint (every --resume-tokens); --save-final keeps the final weights + optimizer state.

    python -m rounding.scripts.train --batch-size 16 --peak-lr 0.0332 --t2 1000000 --precision bf16sr --out rounding/runs/figure9/b16_bf16sr   (from the repo root)
"""
import argparse, json, math, os, random, string, time
from functools import partial
import numpy as np
import jax, jax.numpy as jnp, optax, wandb
from flax import nnx
from omegaconf import OmegaConf
from . import common as C
from train import loss_fn, train_step      # pretraining/train.py (upstream, unchanged)
from utils import flatten_dict             # pretraining/utils.py

RESUME_TOKENS = 2**25
CURVE_TOKENS = 2**25
CURVE_BATCHES = 128                         # 128 microbatches of 16 x 512 = 1M validation tokens


def make_run_chunk(opt_graphdef, model_graphdef):
    @partial(jax.jit, donate_argnames=('opt_state',))
    def run_chunk(key, opt_state, batches):                # batches: [n, B, T] (B <= 16) or [n, grad_acc, 16, T] uint16
        def body(carry, batch):
            key, opt_state = carry
            key, opt_state, loss = train_step(key, opt_state, opt_graphdef, model_graphdef, batch)   # upstream update (incl. the rounding)
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
    ap.add_argument('--batch-size', type=int, required=True); ap.add_argument('--peak-lr', type=float, required=True); ap.add_argument('--t2', type=int, required=True)
    ap.add_argument('--precision', required=True, choices=tuple(C.PRECISIONS)); ap.add_argument('--out', required=True)
    ap.add_argument('--budget-tokens', type=int, default=None, help='num_tokens_train; default the paper\'s 20 tokens per parameter')
    ap.add_argument('--chunk-tokens', type=int, default=None, help='default max(2^19, B*T)'); ap.add_argument('--resume-tokens', type=int, default=RESUME_TOKENS)
    ap.add_argument('--curve-tokens', type=int, default=CURVE_TOKENS); ap.add_argument('--curve-batches', type=int, default=CURVE_BATCHES)
    ap.add_argument('--save-final', action='store_true', help='keep the final weights + optimizer state (checkpoints/final.npz)')
    ap.add_argument('--wandb-mode', default='online'); ap.add_argument('--wandb-project', default='batch-size')
    ap.add_argument('--wandb-group', default='rounding'); ap.add_argument('--wandb-name', default=None)
    ap.add_argument('--eval-batches', type=int, default=None, help='testing only: first N validation microbatches')
    ap.add_argument('--stop-after-chunks', type=int, default=None, help='testing only')
    ap.add_argument('--allow-code-change', action='store_true', help='resume even if the package hash differs from the recorded one')
    args = ap.parse_args(argv)
    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')

    B, out = args.batch_size, args.out
    for marker in ('done', 'diverged'):
        if os.path.exists(f'{out}/{marker}.json'): print(f'{out}: already finished ({marker}.json); nothing to do'); return
    os.makedirs(f'{out}/checkpoints', exist_ok=True)

    # --- config, model, data, optimizer: upstream train_and_evaluate step by step ---
    c = C.build_config(B, args.peak_lr, args.t2, args.precision, budget_tokens=args.budget_tokens)
    mesh = C.set_mesh(c)
    model = C.build_model(c)
    if c.num_tokens_train is None: c.num_tokens_train = C.paper_budget(c)
    ds_train, ds_valid = C.load_data(c, mesh)                               # [n_train, micro, T], [n_valid, micro, T]
    T, micro, acc = int(c.model.T), int(c.opt.microbatch_size), int(c.opt.grad_acc_steps)
    K = len(ds_train) // acc; per_step = B * T; budget = K * per_step
    optimizer = C.build_optimizer(c, model, K)                              # resolves b2 from t2 in c.opt
    model_graphdef = nnx.graphdef(model)
    opt_graphdef, opt_state = nnx.split(optimizer)
    param_dtypes = sorted({str(l.dtype) for l in jax.tree.leaves(opt_state.model)})
    assert param_dtypes == [C.PRECISIONS[args.precision]['param_dtype']], param_dtypes
    warmup = int(c.opt.warmup_frac * K)                                     # mirrors get_optimizer's schedule, for logging
    lr_all = np.asarray(jax.vmap(optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup, K))(jnp.arange(K)), dtype=np.float32)
    if args.eval_batches: ds_valid = ds_valid[:args.eval_batches]
    ds_curve = ds_valid[:args.curve_batches]

    # --- chunks ---
    chunk_tokens = args.chunk_tokens or max(2**19, per_step); assert chunk_tokens % per_step == 0; chunk_steps = chunk_tokens // per_step
    resume_steps, curve_steps = args.resume_tokens // per_step, args.curve_tokens // per_step
    assert resume_steps % chunk_steps == 0 and curve_steps % chunk_steps == 0, 'resume / curve points must be chunk boundaries'
    run_params = dict(batch_size=B, microbatch=micro, grad_acc=acc, K=K, n_tokens_train=int(c.num_tokens_train), budget_tokens=budget, chunk_tokens=chunk_tokens,
                      peak_lr=float(c.opt.peak_lr), t2=float(c.opt.t2), b2=float(c.opt.b2), precision=args.precision, param_dtype=param_dtypes[0],
                      stochastic_round=bool(c.opt.stochastic_round), seed=int(c.seed), matmul_precision='tensorfloat32')

    # --- logs, resume ---
    path = f'{out}/steps.f32'
    if os.path.exists(path): assert os.path.getsize(path) == K * 2 * 4, f'{path} has the wrong size for K={K}'
    steps_mm = np.memmap(path, dtype=np.float32, mode='r+' if os.path.exists(path) else 'w+', shape=(K, 2))   # loss, lr
    key, _, _ = C.keys(c.seed)                              # upstream's training key (consumed by the stochastic rounding)
    k = 0; resumed = False
    resume_path, manifest_path = f'{out}/checkpoints/resume.npz', f'{out}/run_manifest.json'
    if os.path.exists(resume_path):
        opt_state, extra = C.load_tree(resume_path, opt_state)
        saved = json.loads(str(extra['run_params']))
        assert saved == run_params, f'refusing to resume: checkpoint was written with {saved}, this run has {run_params}'
        manifest = json.load(open(manifest_path))
        recorded = (manifest['resumes'][-1] if manifest.get('resumes') else manifest)['code']['rounding_code_sha256']
        current = C.code_info()['rounding_code_sha256']
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
        manifest = dict(run_params=run_params, warmup_updates=warmup, tokens_per_update=per_step, n_valid_microbatches=int(len(ds_valid)), curve_batches=int(len(ds_curve)),
                        resume_tokens=args.resume_tokens, curve_tokens=args.curve_tokens, config=OmegaConf.to_container(c), data=C.data_meta(c), code=C.code_info(),
                        versions=C.versions(), devices=[f'{d} ({d.device_kind})' for d in jax.devices()], wandb=dict(project=args.wandb_project, group=args.wandb_group, id=wandb_id),
                        started=utc())
    C.write_json(manifest_path, manifest)
    wandb.init(project=args.wandb_project, name=args.wandb_name or os.path.basename(out.rstrip('/')), group=args.wandb_group, id=wandb_id, resume='allow',
               mode=args.wandb_mode, config={**flatten_dict(OmegaConf.to_container(c)), **{f'run.{a}': getattr(args, a) for a in vars(args)}, 'run.K': K})
    wandb.define_metric('tokens'); wandb.define_metric('*', step_metric='tokens')
    run_chunk = make_run_chunk(opt_graphdef, model_graphdef)

    def save_checkpoint(k, path):
        C.save_tree(path, opt_state, step=k, tokens=k * per_step, key=jax.random.key_data(key), run_params=json.dumps(run_params, sort_keys=True))

    def merge_json(name, entry):
        p = f'{out}/{name}'; rows = json.load(open(p)) if os.path.exists(p) else []
        rows = [r for r in rows if r['step'] != entry['step']] + [entry]
        C.write_json(p, sorted(rows, key=lambda r: r['step']))

    def curve(k):
        v = float(eval_loss(opt_state.model, model_graphdef, ds_curve))
        wandb.log({'curve_loss': v, 'tokens': k * per_step}); merge_json('curve.json', dict(step=k, tokens=k * per_step, curve_loss=v))
        if not v < 2 * math.log(c.model.V):                                  # blown up (or nan): diverged
            steps_mm.flush(); C.write_json(f'{out}/diverged.json', dict(step=k, tokens=k * per_step, curve_loss=v, rule='curve loss > 2 ln V', time=utc()))
            wandb.finish(); raise RuntimeError(f'diverged: curve loss {v:.4g} at {k * per_step:,} tokens')

    if k == 0: curve(0)
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
        if k % curve_steps == 0 and k < K: curve(k)
        if k % resume_steps == 0 and k < K:
            steps_mm.flush(); save_checkpoint(k, resume_path)
            C.write_json(f'{out}/progress.json', dict(step=k, tokens=k * per_step, updates_per_s=n / dt, time=utc()))
        if args.stop_after_chunks and chunks_done >= args.stop_after_chunks:
            steps_mm.flush(); save_checkpoint(k, resume_path)
            print(f'stopped after {chunks_done} chunks at update {k} (testing)'); wandb.finish(); return
    steps_mm.flush()
    curve(k)
    v = float(eval_loss(opt_state.model, model_graphdef, ds_valid))
    wandb.log({'eval_loss': v, 'tokens': k * per_step}); C.write_json(f'{out}/eval.json', dict(step=k, tokens=k * per_step, eval_loss=v))
    print(f'eval @ {k * per_step:,} tokens: {v:.4f}', flush=True)
    if args.save_final: save_checkpoint(k, f'{out}/checkpoints/final.npz')
    C.write_json(f'{out}/done.json', dict(step=k, tokens=k * per_step, eval_loss=v, finished=utc()))
    if os.path.exists(resume_path): os.remove(resume_path)
    wandb.finish()


if __name__ == '__main__':
    main()
