"""Driver of the rounding experiment: the paper's Figure 9 with the corrected Adafactor (common.py). For each batch size of
the sharpness bracket (sharpness/runs/bracket/best.json: B in 16, 256, 4096, 1 with the quarter-budget-tuned peak learning
rate and t2 of the paper's own Adafactor implementation; by the user's decision these are reused as they are, no re-tuning)
the three weight precisions fp32, bf16 (nearest rounding) and bf16sr (stochastic rounding) are run at the paper's full
budget. Batch sizes are run in the given order (fastest first by default, so every curve gets its points early). If the fp32
run diverges, the peak learning rate is stepped down one index of the paper's Adafactor grid (x B^0.3) and all three
precisions are run there (recorded as lr_steps_down); a diverged bf16 run is a result. Diverged = diverged.json (train.py:
non-finite loss or curve loss > 2 ln V) or a final validation loss above the uniform-prediction loss ln V = 10.82.

Outputs: rounding/runs/figure9/b<B>_<precision>_lr<lr>/ (train.py) and rounding/runs/figure9/summary.json. Restartable:
finished runs are skipped and interrupted runs resume; --no-train only collects (missing runs reported as loss = nan).

    python -m rounding.scripts.figure9            (from the repo root; or rounding/scripts/run_bg.sh rounding bash rounding/scripts/run_figure9.sh)
"""
import argparse, json, math, os, subprocess, sys, time

LN_V = math.log(50257)                                                   # the uniform-prediction loss; a run that ends above it has diverged

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.environ.get('FIGURE9_OUT', os.path.join(REPO, 'rounding', 'runs', 'figure9'))
BEST = os.environ.get('FIGURE9_BEST', os.path.join(REPO, 'sharpness', 'runs', 'bracket', 'best.json'))
BATCH_SIZES = (16, 256, 4096, 1)
PRECISIONS = ('fp32', 'bf16', 'bf16sr')
AF_GRID = [0.00010, 0.00035, 0.00120, 0.00416, 0.01443, 0.05000]       # the paper's Adafactor grid, times B^0.3
MAX_STEPS_DOWN = 2


def af_lr(i, B):
    """grid[i] * B^0.3; the grid's geometric ratio continued past either end."""
    ratio = (AF_GRID[-1] / AF_GRID[0]) ** (1 / (len(AF_GRID) - 1))
    base = AF_GRID[i] if 0 <= i < len(AF_GRID) else (AF_GRID[0] * ratio ** i if i < 0 else AF_GRID[-1] * ratio ** (i - len(AF_GRID) + 1))
    return base * B ** 0.3


def grid_index(lr, B):
    i = min(range(-4, len(AF_GRID) + 4), key=lambda i: abs(math.log(af_lr(i, B) / lr)))
    assert abs(math.log(af_lr(i, B) / lr)) < 1e-6, f'lr {lr} at B={B} is not on the Adafactor grid'
    return i


def status(run):
    """(loss, state): finished -> (eval loss, 'done', or 'diverged' if above ln V); diverged.json -> (its curve loss or inf, 'diverged'); otherwise (nan, 'missing')."""
    if os.path.exists(f'{run}/diverged.json'): return float(json.load(open(f'{run}/diverged.json')).get('curve_loss', math.inf)), 'diverged'
    if os.path.exists(f'{run}/done.json'):
        loss = float(json.load(open(f'{run}/eval.json'))['eval_loss']); return loss, ('done' if loss < LN_V else 'diverged')
    return math.nan, 'missing'


def run_one(B, precision, lr, t2, args):
    run = os.path.join(OUT, f'b{B}_{precision}_lr{lr:.4g}')
    loss, state = status(run)
    if state == 'missing' and not args.no_train:
        cmd = [sys.executable, '-m', 'rounding.scripts.train', '--batch-size', str(B), '--peak-lr', repr(lr), '--t2', str(int(t2)), '--precision', precision,
               '--out', run, '--save-final', '--wandb-group', 'rounding']
        if args.extra: cmd += args.extra.split()
        t0 = time.time(); print(f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] {os.path.basename(run)}: {" ".join(cmd[2:])}', flush=True)
        p = subprocess.run(cmd, cwd=REPO, env={**os.environ, 'XLA_PYTHON_CLIENT_PREALLOCATE': 'true', 'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.9'})   # one 90% pool, as run_figure9.sh
        loss, state = status(run)
        if state == 'missing': raise RuntimeError(f'{run}: train exited {p.returncode} without done.json or diverged.json')
        print(f'    -> {state} loss {loss:.4f}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    return dict(dir=os.path.relpath(run, REPO), loss=loss, state=state)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=list(BATCH_SIZES))
    ap.add_argument('--precisions', nargs='+', default=list(PRECISIONS), choices=PRECISIONS)
    ap.add_argument('--extra', default=None, help="extra train.py args (tests), e.g. '--budget-tokens 2097152 --eval-batches 4 --wandb-mode disabled'")
    ap.add_argument('--no-train', action='store_true', help='only collect finished runs (missing ones are reported, never launched); for notebooks')
    args = ap.parse_args(argv)
    best = json.load(open(BEST))['adafactor']
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'summary.json')
    summary = json.load(open(path)) if os.path.exists(path) else {}
    for B in args.batch_sizes:
        lr0, t2 = float(best[str(B)]['peak_lr']), int(best[str(B)]['t2'])
        i0 = grid_index(lr0, B); i = i0; runs = {}; fallbacks = []
        while True:                                        # fp32 first; step the learning rate down the grid while it diverges
            lr = af_lr(i, B)
            r = run_one(B, 'fp32', lr, t2, args)
            if r['state'] == 'diverged' and i0 - i < MAX_STEPS_DOWN: fallbacks.append(r); i -= 1; continue
            runs['fp32'] = r; break
        for precision in args.precisions:
            if precision != 'fp32': runs[precision] = run_one(B, precision, lr, t2, args)
        s = summary[str(B)] = dict(batch_size=B, peak_lr=lr, t2=t2, grid_index=i, lr_steps_down=i0 - i, bracket_peak_lr=lr0, runs={p: runs[p] for p in args.precisions if p in runs},
                                   fp32_diverged_at=fallbacks)
        json.dump(summary, open(path + '.tmp', 'w'), indent=1); os.replace(path + '.tmp', path)
        print(f'B={B}: lr {lr:.4g} (grid index {i}, {i0 - i} steps down) t2 {t2 / 1e6:g}M  ' + '  '.join(f'{p} {r["loss"]:.4f}' if r['state'] == 'done' else f'{p} {r["state"]}' for p, r in s['runs'].items()), flush=True)
    missing = sum(r['state'] == 'missing' for s in summary.values() for r in s['runs'].values())
    print(f'{missing} runs not run yet' if missing else 'all runs finished', flush=True)
    return summary


if __name__ == '__main__':
    main()
