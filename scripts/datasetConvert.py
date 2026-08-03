# -*- coding: utf-8 -*-
# python3 datasetConvert.py --vsqxDir All-songs/ --seqLen 64 --stride 8

import xml.dom.minidom
import sys
import os
from pathlib import PurePath
import argparse

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support.vocalVocab import VocaloidVocab, tokens_from_notes

DATASET_SEED = 42

parser = argparse.ArgumentParser(description='Convert VSQX/MIDI directory to a note-sequence CSV dataset')
parser.add_argument('--vsqxDir', dest='vsqxDir', default='All-songs/',
                    help='Directory containing VSQX files')
parser.add_argument('--midiDir', dest='midiDir', default=None,
                    help='Optional directory containing MIDI files (uses extractVocalMidi)')
parser.add_argument('--seqLen', dest='seqLen', type=int, default=64,
                    help='Number of tokens per sequence window')
parser.add_argument('--stride', dest='stride', type=int, default=8,
                    help='Window stride over each song')
args = parser.parse_args()

seqLen = args.seqLen
stride = args.stride
rootDir = args.vsqxDir
vocab = VocaloidVocab()

assert os.path.exists(rootDir), f'vsqxDir not found: {rootDir}'


def _get_note_attr(note, key):
    return int(note.getElementsByTagName(key)[0].firstChild.data)


def _parse_vsqx_notes(vsqxPath):
    """Return (pitch, start_tick, duration_ticks) list from all tracks in a VSQX."""
    path = PurePath(vsqxPath)
    vsqx = xml.dom.minidom.parse(str(path))
    try:
        int(vsqx.getElementsByTagName('tempo')[0].childNodes[1].firstChild.data[:-2])
    except (IndexError, ValueError):
        return None

    all_notes = []
    time_offset = None

    for track in vsqx.getElementsByTagName('vsTrack'):
        for note in track.getElementsByTagName('note'):
            try:
                t = _get_note_attr(note, 't')
                n = _get_note_attr(note, 'n')
                dur = _get_note_attr(note, 'dur')
            except (IndexError, AttributeError, ValueError):
                continue
            if time_offset is None:
                time_offset = t - 5
            all_notes.append((n, t - time_offset, dur))

    if not all_notes:
        return None
    all_notes.sort(key=lambda x: x[1])
    return all_notes


def _notes_to_windows(note_list, seqLen, stride):
    """Slide a window of seqLen over token list, return list of '|'-joined strings."""
    token_list = tokens_from_notes(note_list)
    windows = []
    for start in range(0, len(token_list) - seqLen + 1, stride):
        window = token_list[start:start + seqLen]
        if len(window) == seqLen:
            windows.append('|'.join(window))
    return windows


all_windows = []

for i, fileName in enumerate(os.listdir(rootDir)):
    if not fileName.lower().endswith('.vsqx'):
        continue
    print(i, ':', fileName)
    vsqxPath = os.path.join(rootDir, fileName)
    notes = _parse_vsqx_notes(vsqxPath)
    if notes is None:
        print('   Skipping', fileName)
        continue
    windows = _notes_to_windows(notes, seqLen, stride)
    all_windows.extend(windows)

if args.midiDir and os.path.isdir(args.midiDir):
    from scripts.extractVocalMidi import extract_and_tokenize
    midi_files = [f for f in os.listdir(args.midiDir)
                  if f.lower().endswith(('.mid', '.midi'))]
    print(f'Processing {len(midi_files)} MIDI files from {args.midiDir}')
    for j, fname in enumerate(midi_files):
        midi_path = os.path.join(args.midiDir, fname)
        tokens = extract_and_tokenize(midi_path)
        if not tokens:
            continue
        for start in range(0, len(tokens) - seqLen + 1, stride):
            window = tokens[start:start + seqLen]
            if len(window) == seqLen:
                all_windows.append('|'.join(window))
        if j % 100 == 0:
            print(f'  MIDI {j}/{len(midi_files)}')

if not all_windows:
    print('No sequences generated — check your input directories.')
    sys.exit(1)

df = pd.DataFrame({'seq': all_windows})

out_dir = 'dataset/'
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, 'entireNotes.csv'), index=False)

rng = np.random.default_rng(DATASET_SEED)
msk = rng.random(len(df)) < 0.8
df[msk].to_csv(os.path.join(out_dir, 'trainNotes.csv'), index=False)
df[~msk].to_csv(os.path.join(out_dir, 'valNotes.csv'), index=False)

print(f'Dataset written: {len(df)} windows ({msk.sum()} train, {(~msk).sum()} val)')
