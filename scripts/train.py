# -*- coding: utf-8 -*-
# python3 train.py --trainCsv dataset/trainNotes.csv --valCsv dataset/valNotes.csv

import os
import sys
import math
import time
import json
import random
import argparse
import subprocess
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support.vocalVocab import VocaloidVocab, initialize_model, PAD_IDX, SOS_IDX, EOS_IDX
from support.vocaloidPenalty import VocaloidRulesPenalty

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def _git_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], text=True
        ).strip()
    except Exception:
        return 'unknown'


def _write_registry(args, epoch, best_val_loss, train_ds, val_ds, vocab):
    registry_dir = os.path.dirname(os.path.abspath(args.modelOutput))
    registry_path = os.path.join(registry_dir, 'registry.json')
    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        registry = []

    entry = {
        'version': args.version,
        'note': args.versionNote,
        'path': args.modelOutput,
        'timestamp': datetime.utcnow().isoformat(timespec='seconds'),
        'git_hash': _git_hash(),
        'hparams': {
            'vocab_size': len(vocab),
            'hid_dim': args.hidDim,
            'n_layers': args.nLayers,
            'n_heads': args.nHeads,
            'pf_dim': args.pfDim,
        },
        'training': {
            'best_epoch': epoch + 1,
            'best_val_loss': round(best_val_loss, 4),
            'best_val_ppl': round(math.exp(best_val_loss), 3),
            'total_epochs': args.epochs,
            'penalty_weight': args.penaltyWeight,
            'seq_len': args.seqLen,
            'batch_size': args.batchSize,
            'lr': args.lr,
        },
        'dataset': {
            'train_csv': args.trainCsv,
            'val_csv': args.valCsv,
            'train_rows': len(train_ds),
            'val_rows': len(val_ds),
        },
    }

    # Upsert: replace existing entry with same version, then re-sort
    registry = [e for e in registry if e.get('version') != args.version]
    registry.append(entry)
    registry.sort(key=lambda e: e.get('version', 0))

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


parser = argparse.ArgumentParser(description='Train MusicTransformerGPT on note sequences')
parser.add_argument('--trainCsv', dest='trainCsv', default='dataset/trainNotes.csv')
parser.add_argument('--valCsv', dest='valCsv', default='dataset/valNotes.csv')
parser.add_argument('--modelOutput', dest='modelOutput', default='savedModels/music-model.pt')
parser.add_argument('--resumeFrom', dest='resumeFrom', default=None,
                    help='Checkpoint path to resume training from')
parser.add_argument('--epochs', dest='epochs', type=int, default=50)
parser.add_argument('--seqLen', dest='seqLen', type=int, default=64)
parser.add_argument('--batchSize', dest='batchSize', type=int, default=64)
parser.add_argument('--lr', dest='lr', type=float, default=3e-4)
parser.add_argument('--hidDim', dest='hidDim', type=int, default=512)
parser.add_argument('--nLayers', dest='nLayers', type=int, default=6)
parser.add_argument('--nHeads', dest='nHeads', type=int, default=8)
parser.add_argument('--pfDim', dest='pfDim', type=int, default=1024)
parser.add_argument('--penaltyWeight', dest='penaltyWeight', type=float, default=0.3,
                    help='Weight for Vocaloid rule penalty added to cross-entropy loss (0 = disabled)')
parser.add_argument('--version', dest='version', type=int, default=None,
                    help='Experiment version number (e.g. 3). Enables registry logging and auto-naming.')
parser.add_argument('--versionNote', dest='versionNote', type=str, default='',
                    help='Short descriptor appended to the model filename (e.g. bar-tokens-12x-aug)')
args = parser.parse_args()

# Auto-name output file when --version is given and --modelOutput is at its default
if args.version is not None and args.modelOutput == 'savedModels/music-model.pt':
    slug = f'-{args.versionNote}' if args.versionNote else ''
    args.modelOutput = f'savedModels/v{args.version}{slug}.pt'

os.makedirs(os.path.dirname(args.modelOutput) or '.', exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')
if args.version is not None:
    note_display = f' ({args.versionNote})' if args.versionNote else ''
    print(f'Experiment v{args.version}{note_display} → {args.modelOutput}')

vocab = VocaloidVocab()


class NoteSequenceDataset(Dataset):
    def __init__(self, csv_path, vocab, seq_len):
        df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.seq_len = seq_len
        self.sequences = df['seq'].tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx].split('|')
        ids = [self.vocab.encode(t) for t in tokens]
        if len(ids) < self.seq_len:
            ids = ids + [PAD_IDX] * (self.seq_len - len(ids))
        else:
            ids = ids[:self.seq_len]
        t = torch.tensor(ids, dtype=torch.long)
        return t[:-1], t[1:]  # input, target — decoder-only shift


train_ds = NoteSequenceDataset(args.trainCsv, vocab, args.seqLen)
val_ds = NoteSequenceDataset(args.valCsv, vocab, args.seqLen)

train_loader = DataLoader(train_ds, batch_size=args.batchSize, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batchSize, shuffle=False, drop_last=False)

model = initialize_model(len(vocab), device,
                        hid_dim=args.hidDim, n_layers=args.nLayers,
                        n_heads=args.nHeads, pf_dim=args.pfDim)
print(f'Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

penalty_fn = None
if args.penaltyWeight > 0:
    penalty_fn = VocaloidRulesPenalty(vocab).to(device)
    print(f'Vocaloid penalty enabled (weight={args.penaltyWeight})')

start_epoch = 0
best_val_loss = float('inf')

if args.resumeFrom and os.path.exists(args.resumeFrom):
    ckpt = torch.load(args.resumeFrom, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    if 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt.get('epoch', 0)
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    print(f'Resumed from epoch {start_epoch}, best val loss so far: {best_val_loss:.3f}')

# Cosine annealing over the remaining epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-5
)
# Advance scheduler to match resumed epoch so the LR curve is consistent
for _ in range(start_epoch):
    scheduler.step()


def run_epoch(model, loader, optimizer, criterion, train):
    model.train() if train else model.eval()
    total_loss = 0
    with torch.set_grad_enabled(train):
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            if train:
                optimizer.zero_grad()
            output = model(src)                       # (B, T-1, vocab)
            ce_loss = criterion(output.reshape(-1, len(vocab)), trg.reshape(-1))
            if train and penalty_fn is not None:
                pen = penalty_fn(output)
                loss = ce_loss + args.penaltyWeight * pen
            else:
                loss = ce_loss
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += ce_loss.item()  # log CE only so val PPL is comparable
    return total_loss / len(loader)


for epoch in range(start_epoch, start_epoch + args.epochs):
    t0 = time.time()
    train_loss = run_epoch(model, train_loader, optimizer, criterion, train=True)
    val_loss = run_epoch(model, val_loader, optimizer, criterion, train=False)
    scheduler.step()
    elapsed = time.time() - t0
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    lr_now = scheduler.get_last_lr()[0]

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'vocab': vocab.tok2idx,
            'version': args.version,
            'version_note': args.versionNote,
            'timestamp': datetime.utcnow().isoformat(timespec='seconds'),
            'git_hash': _git_hash(),
            'hparams': {
                'vocab_size': len(vocab),
                'hid_dim': args.hidDim,
                'n_layers': args.nLayers,
                'n_heads': args.nHeads,
                'pf_dim': args.pfDim,
                'dropout': 0.1,
                'max_seq_len': 512,
            },
            'training_config': {
                'penalty_weight': args.penaltyWeight,
                'seq_len': args.seqLen,
                'batch_size': args.batchSize,
                'lr': args.lr,
                'total_epochs': start_epoch + args.epochs,
            },
            'dataset_info': {
                'train_csv': args.trainCsv,
                'val_csv': args.valCsv,
                'train_rows': len(train_ds),
                'val_rows': len(val_ds),
            },
        }, args.modelOutput)

        if args.version is not None:
            _write_registry(args, epoch, best_val_loss, train_ds, val_ds, vocab)

    print(f'Epoch {epoch+1:02} | {mins}m {secs}s | lr={lr_now:.2e}')
    print(f'  Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'    Val Loss: {val_loss:.3f} |   Val PPL: {math.exp(val_loss):7.3f}')

print(f'Best val loss: {best_val_loss:.3f} | Best val PPL: {math.exp(best_val_loss):7.3f}')
