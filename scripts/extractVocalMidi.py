# -*- coding: utf-8 -*-
# python3 extractVocalMidi.py --midiDir All-songs/midi/ --outputDir /tmp/midi-tokens/

import re
import os
import sys
import argparse

import mido

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support.vocalVocab import tokens_from_notes

VOCAL_TRACK_RE = re.compile(r'vocal|melody|lead|vox|soprano', re.IGNORECASE)
PITCH_MIN, PITCH_MAX = 48, 84  # C3–C6
MONO_RATIO_THRESH = 0.85


def is_vocal_track(track):
    if not track.name:
        return False
    return bool(VOCAL_TRACK_RE.search(track.name))


def monophonic_ratio(track):
    """Fraction of total active time with exactly 1 note sounding."""
    active = {}  # pitch -> start_time
    intervals = []
    abs_time = 0
    for msg in track:
        abs_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            if msg.channel == 9:
                continue
            active[msg.note] = abs_time
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active:
                intervals.append((active.pop(msg.note), abs_time))
    if not intervals:
        return 0.0
    # measure total ticks and overlapping ticks
    total_end = max(e for _, e in intervals)
    if total_end == 0:
        return 0.0
    # count ticks with exactly 1 note active (sampled at each event boundary)
    events = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()
    mono_ticks = 0
    poly_ticks = 0
    count = 0
    prev_t = 0
    for t, delta in events:
        span = t - prev_t
        if count == 1:
            mono_ticks += span
        elif count > 1:
            poly_ticks += span
        count += delta
        prev_t = t
    denom = mono_ticks + poly_ticks
    return mono_ticks / denom if denom > 0 else 0.0


def extract_notes_from_track(track, ticks_per_beat):
    """Return list of (pitch, start_tick, duration_ticks) sorted by start."""
    active = {}
    notes = []
    abs_time = 0
    for msg in track:
        abs_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0 and msg.channel != 9:
            active[msg.note] = abs_time
        elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
            if msg.note in active:
                start = active.pop(msg.note)
                dur = abs_time - start
                if dur > 0:
                    notes.append((msg.note, start, dur))
    notes.sort(key=lambda n: n[1])
    return notes


def filter_pitch_range(notes):
    return [n for n in notes if PITCH_MIN <= n[0] <= PITCH_MAX]


def extract_and_tokenize(midi_path):
    """Extract vocal track tokens from a MIDI file.

    Returns a list of token strings, or [] if no suitable track found.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return []

    best_track = None
    best_ratio = 0.0

    for track in mid.tracks:
        if is_vocal_track(track):
            ratio = monophonic_ratio(track)
            if ratio >= MONO_RATIO_THRESH and ratio > best_ratio:
                best_ratio = ratio
                best_track = track

    if best_track is None:
        # Fall back to highest monophonic-ratio non-drum track
        for track in mid.tracks:
            ratio = monophonic_ratio(track)
            if ratio > best_ratio:
                best_ratio = ratio
                best_track = track

    if best_track is None or best_ratio < MONO_RATIO_THRESH:
        return []

    notes = extract_notes_from_track(best_track, mid.ticks_per_beat)
    notes = filter_pitch_range(notes)
    if not notes:
        return []
    return tokens_from_notes(notes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract vocal tracks from MIDI files as token sequences')
    parser.add_argument('--midiDir', dest='midiDir', default='All-songs/midi/',
                        help='Directory containing .mid/.midi files')
    parser.add_argument('--outputDir', dest='outputDir', default='/tmp/midi-tokens/',
                        help='Directory to write per-file token text files')
    args = parser.parse_args()

    os.makedirs(args.outputDir, exist_ok=True)
    processed = 0
    skipped = 0

    for dirpath, _dirs, files in os.walk(args.midiDir):
        for fname in files:
            if not fname.lower().endswith(('.mid', '.midi')):
                continue
            path = os.path.join(dirpath, fname)
            tokens = extract_and_tokenize(path)
            if not tokens:
                skipped += 1
                continue
            # Flatten artist/song path into a unique output filename
            rel = os.path.relpath(path, args.midiDir).replace(os.sep, '_')
            out_path = os.path.join(args.outputDir, rel.rsplit('.', 1)[0] + '.txt')
            with open(out_path, 'w') as f:
                f.write('|'.join(tokens))
            processed += 1
            if (processed + skipped) % 500 == 0:
                print(f'  ...{processed} extracted, {skipped} skipped', flush=True)

    print(f'Extracted {processed} files, skipped {skipped}')
