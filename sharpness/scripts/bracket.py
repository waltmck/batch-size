"""Quarter-budget learning-rate brackets for vanilla SGD and Adafactor on the paper's own grids (stage 1 of the
sharpness-vs-batch-size experiment).

The paper tuned SGD and Adafactor per batch size for Figure 1a but published only the grids (git history, commit 57db633):
SGD 20 learning rates 0.05 .. 5 (x1.274) at B <= 256; Adafactor 6 learning rates 1e-4 .. 5e-2 (x3.47) times B^0.3 and
5 values of t2, 1M .. 100M tokens (x3.16), which upstream turns into the exponent 0.5^(B*T/t2) of Adafactor's power-law
second-moment schedule rather than a half-life (so t2 has no effect at B <= 16). This driver re-runs the paper's grids at B in {4096, 256, 16, 1}
on a quarter of the budget (151M tokens, same 5% warmup + cosine shape) in train.py's --sweep mode (TF32, no snapshots /
checkpoints / sharpness, validation loss at the end) and picks the lowest final validation loss, as the paper did:

  SGD        all 20 learning rates; if the best sits at an end of the grid (possible at B = 4096, which the paper never ran
             with SGD) the grid is continued geometrically one index at a time, at most twice per side.
  Adafactor  all 6 x 5 (learning rate, t2) points.

Diverged runs count as loss = inf. Results: sharpness/runs/bracket/{results.json, best.json} (best.json also names
the paper's tuned Adam table, configs/hparams/lm11m_fwedu_adam.yaml, for run_full.sh). Restartable: finished points are
read back, so `main(['--no-train'])` returns (results, best) instantly and never touches the GPU, reporting any points not
run yet (the stage-1 section of sharpness/sharpness.ipynb does exactly that); best.json is only (re)written when every point has run.

    python -m sharpness.scripts.bracket    (from the repo root; or sharpness/scripts/run_bg.sh sharp-bracket bash sharpness/scripts/run_bracket.sh)
"""
import argparse, json, math, os, subprocess, sys, time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.environ.get('BRACKET_OUT', os.path.join(REPO, 'sharpness', 'runs', 'bracket'))
BUDGET = int(os.environ.get('BRACKET_BUDGET', 150_994_944))   # 72 * 2^21 tokens: a whole number of updates at every B here
BATCH_SIZES = (4096, 256, 16, 1)
SGD_GRID = [0.050, 0.064, 0.081, 0.103, 0.132, 0.168, 0.214, 0.273, 0.348, 0.443, 0.564, 0.719, 0.916, 1.168, 1.488, 1.896, 2.416, 3.079, 3.924, 5.000]
AF_GRID = [0.00010, 0.00035, 0.00120, 0.00416, 0.01443, 0.05000]       # times B^0.3 (the grid's opt.peak_lr_scaling)
T2_GRID = [1_000_000, 3_160_000, 10_000_000, 31_600_000, 100_000_000]  # index 2 = 10M
MAX_EXT = 2
ADAM_HPARAMS = 'lm11m_fwedu_adam'


def grid_value(grid, i):
    """grid[i] inside the grid; the grid's geometric ratio continued past either end."""
    ratio = (grid[-1] / grid[0]) ** (1 / (len(grid) - 1))
    if i < 0: return grid[0] * ratio ** i
    if i >= len(grid): return grid[-1] * ratio ** (i - len(grid) + 1)
    return grid[i]


def sgd_lr(i, B): return grid_value(SGD_GRID, i)
def af_lr(i, B): return grid_value(AF_GRID, i) * B ** 0.3


def final_loss(run):
    if os.path.exists(f'{run}/diverged.json'): return math.inf
    if not os.path.exists(f'{run}/done.json'): return None
    return float(max(json.load(open(f'{run}/eval.json')), key=lambda e: e['tokens'])['eval_loss'])


def dump_json(path, obj):
    json.dump(obj, open(path + '.tmp', 'w'), indent=1); os.replace(path + '.tmp', path)


class Runner:
    """Runs (and caches) one bracket point, keyed by (grid index, t2)."""
    def __init__(self, opt, B, lr_of, args): self.opt, self.B, self.lr_of, self.args, self.res = opt, B, lr_of, args, {}

    def __call__(self, i, t2=None):
        if (i, t2) in self.res: return self.res[(i, t2)]
        lr = self.lr_of(i, self.B)
        run = os.path.join(OUT, f'{self.opt}_b{self.B}_lr{lr:.4g}' + (f'_t2{t2 / 1e6:g}M' if self.opt == 'adafactor' else ''))
        loss = final_loss(run)
        if loss is None and self.args.no_train: loss = math.nan                 # not run yet; reported as missing
        if loss is None:
            cmd = [sys.executable, '-m', 'sharpness.scripts.train', '--optimizer', self.opt, '--batch-size', str(self.B), '--peak-lr', repr(lr), '--budget-tokens', str(BUDGET),
                   '--sweep', '--out', run, '--wandb-group', 'sharpness-bracket']
            if self.opt == 'adafactor': cmd += ['--t2', str(t2)]
            if self.args.extra: cmd += self.args.extra.split()
            t0 = time.time(); print(f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] {os.path.basename(run)}', flush=True)
            p = subprocess.run(cmd, cwd=REPO)
            loss = final_loss(run)
            if loss is None: raise RuntimeError(f'{run}: train exited {p.returncode} without done.json or diverged.json')
            print(f'    -> loss {loss:.4f}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
        self.res[(i, t2)] = loss
        return loss

    def best(self, t2=None):
        """(best index, {index: loss}) among the finished points at this t2 (None if none finished)."""
        pts = {i: l for (i, t), l in self.res.items() if t == t2 and not math.isnan(l)}
        return (min(pts, key=lambda i: pts[i]) if pts else None), pts

    def missing(self): return sum(math.isnan(l) for l in self.res.values())

    def runs(self):
        return [dict(grid_index=i, peak_lr=self.lr_of(i, self.B), t2=t2, loss=l) for (i, t2), l in sorted(self.res.items(), key=lambda kv: (kv[0][1] or 0, kv[0][0])) if not math.isnan(l)]


def summary(train, best_i, best_t2):
    return dict(runs=train.runs(), missing=train.missing(), grid_index=best_i, peak_lr=None if best_i is None else train.lr_of(best_i, train.B), t2=best_t2,
                loss=None if best_i is None else train(best_i, best_t2))


def bracket_sgd(B, args):
    train = Runner('sgd', B, sgd_lr, args)
    for i in range(len(SGD_GRID)): train(i)
    ext = {'lo': 0, 'hi': 0}
    while not args.no_train:                                             # continue the grid while the best sits at an end
        best, pts = train.best()
        if best == max(pts) and ext['hi'] < MAX_EXT: ext['hi'] += 1; train(best + 1)
        elif best == min(pts) and ext['lo'] < MAX_EXT: ext['lo'] += 1; train(best - 1)
        else: break
    return summary(train, train.best()[0], None)


def bracket_adafactor(B, args):
    train = Runner('adafactor', B, af_lr, args)
    for j, t2 in enumerate(T2_GRID):
        for i in range(len(AF_GRID)): train(i, t2)
    finished = {(i, t2): l for (i, t2), l in train.res.items() if not math.isnan(l)}
    if not finished: return summary(train, None, None)
    best_i, best_t2 = min(finished, key=finished.get)
    return summary(train, best_i, best_t2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--optimizers', nargs='+', default=['sgd', 'adafactor'], choices=('sgd', 'adafactor'))
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=list(BATCH_SIZES))
    ap.add_argument('--extra', default=None, help="extra train.py args (tests), e.g. '--eval-batches 2 --wandb-mode disabled'")
    ap.add_argument('--no-train', action='store_true', help='only collect finished runs (missing ones are reported, never launched); for notebooks')
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'results.json')
    results = json.load(open(path)) if os.path.exists(path) else {}
    for opt in args.optimizers:
        for B in args.batch_sizes:
            key = f'{opt}_b{B}'                                          # finished points are read back from disk, so this is cheap when complete
            r = results[key] = (bracket_sgd if opt == 'sgd' else bracket_adafactor)(B, args)
            dump_json(path, results)
            print(f'{key}: {len(r["runs"])} points' + (f', {r["missing"]} not run yet' if r['missing'] else '') +
                  ('' if r['peak_lr'] is None else f' -> best lr {r["peak_lr"]:.4g} (grid index {r["grid_index"]}) t2 {r["t2"]} loss {r["loss"]:.4f}'), flush=True)
    best = {opt: {str(B): dict(peak_lr=results[f'{opt}_b{B}']['peak_lr'], t2=results[f'{opt}_b{B}']['t2']) for B in args.batch_sizes} for opt in args.optimizers}
    best['adamw'] = {str(B): dict(hparams=ADAM_HPARAMS) for B in args.batch_sizes}
    if any(r['missing'] for r in results.values()):
        print('bracket incomplete: best.json left unchanged', flush=True)
    else:
        dump_json(os.path.join(OUT, 'best.json'), best); print('wrote', os.path.join(OUT, 'best.json'), json.dumps(best), flush=True)
    return results, best


if __name__ == '__main__':
    main()
