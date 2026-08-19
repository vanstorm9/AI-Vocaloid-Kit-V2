# -*- coding: utf-8 -*-
# python3 scripts/listModels.py [--registry savedModels/registry.json]

import os
import sys
import json
import argparse

parser = argparse.ArgumentParser(description='List and compare registered model experiments')
parser.add_argument('--registry', default='savedModels/registry.json',
                    help='Path to registry.json')
args = parser.parse_args()

if not os.path.exists(args.registry):
    print(f'No registry found at {args.registry}')
    print('Registry is created automatically when you train with --version N')
    sys.exit(0)

with open(args.registry) as f:
    registry = json.load(f)

if not registry:
    print('Registry is empty.')
    sys.exit(0)

# Column widths
COL = {'v': 4, 'note': 24, 'ppl': 8, 'epoch': 7, 'vocab': 7,
       'hid': 5, 'layers': 6, 'penalty': 8, 'rows': 10, 'date': 10}

def _fmt(v, width, align='<'):
    s = str(v) if v is not None else '—'
    return f'{s:{align}{width}}'

def _params(hp):
    """Rough parameter count from hparams dict."""
    v = hp.get('vocab_size', 0)
    h = hp.get('hid_dim', 0)
    l = hp.get('n_layers', 0)
    p = hp.get('pf_dim', 0)
    heads = hp.get('n_heads', 0)
    if not (v and h and l):
        return '—'
    total = (v * h +              # embedding
             l * (4 * h * h +     # attention Q,K,V,O
                  2 * h * p) +    # FFN
             h * v)               # output projection
    if total >= 1e6:
        return f'{total/1e6:.1f}M'
    return f'{total/1e3:.0f}K'

header = (
    f"{'v':<{COL['v']}} "
    f"{'note':<{COL['note']}} "
    f"{'val PPL':>{COL['ppl']}} "
    f"{'epoch':>{COL['epoch']}} "
    f"{'params':>{COL['vocab']}} "
    f"{'hid':>{COL['hid']}} "
    f"{'layers':>{COL['layers']}} "
    f"{'penalty':>{COL['penalty']}} "
    f"{'train rows':>{COL['rows']}} "
    f"{'date':<{COL['date']}}"
)
divider = '-' * len(header)

print(divider)
print(header)
print(divider)

for e in registry:
    v       = e.get('version', '?')
    note    = e.get('note', '') or ''
    ts      = e.get('timestamp', '')[:10]
    hp      = e.get('hparams', {})
    tr      = e.get('training', {})
    ds      = e.get('dataset', {})

    ppl     = tr.get('best_val_ppl', '—')
    epoch   = f"{tr.get('best_epoch', '—')}/{tr.get('total_epochs', '—')}"
    penalty = tr.get('penalty_weight', '—')
    rows    = ds.get('train_rows', '—')
    if isinstance(rows, int) and rows >= 1000:
        rows = f'{rows/1000:.0f}k'

    print(
        f"{_fmt(v, COL['v'])} "
        f"{_fmt(note[:COL['note']], COL['note'])} "
        f"{_fmt(ppl, COL['ppl'], '>')} "
        f"{_fmt(epoch, COL['epoch'], '>')} "
        f"{_fmt(_params(hp), COL['vocab'], '>')} "
        f"{_fmt(hp.get('hid_dim','—'), COL['hid'], '>')} "
        f"{_fmt(hp.get('n_layers','—'), COL['layers'], '>')} "
        f"{_fmt(penalty, COL['penalty'], '>')} "
        f"{_fmt(rows, COL['rows'], '>')} "
        f"{_fmt(ts, COL['date'])}"
    )

print(divider)
print(f'{len(registry)} experiment(s) registered.')
