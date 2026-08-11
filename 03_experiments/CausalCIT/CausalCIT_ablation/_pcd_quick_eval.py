import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pcd_eval_from_ckpt import eval_ckpt, DS, PL, VARIANTS, SEEDS

print(f'quick eval of existing ckpts on {DS} pl{PL}')
for v in VARIANTS:
    for s in SEEDS:
        r = eval_ckpt(DS, PL, v, s, 'cpu')
        if r is None:
            print(f'{v:10s} s{s} -> NO CKPT')
        else:
            print(f'{v:10s} s{s} -> mse={r["mse"]:.6f} mae={r["mae"]:.6f}')
