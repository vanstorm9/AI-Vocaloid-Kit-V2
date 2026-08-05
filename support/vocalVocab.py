# -*- coding: utf-8 -*-
import bisect

DURATION_BINS = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300,
                 360, 420, 480, 540, 600, 720, 840, 960, 1080, 1200,
                 1440, 1680, 1920, 2160, 2400, 2880, 3360, 3840,
                 4320, 4800, 5760, 9601]  # 32 bins; last catches >=9601 ticks
INTERVAL_RANGE = 12  # semitones, clamped

# Bar-position tokens B1-B32 (cycle every 32 bars = 4 × 8-bar sections)
BAR_CYCLE = 32
# Default MIDI ticks per beat (480 is universal MIDI standard; VSQX also uses 480)
DEFAULT_TICKS_PER_BEAT = 480

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
_SPECIALS = ['<pad>', '<sos>', '<eos>', '<unk>']


def _quantize_duration(raw_ticks):
    """Map raw tick count to nearest DURATION_BINS entry (floor)."""
    idx = bisect.bisect_right(DURATION_BINS, raw_ticks) - 1
    return DURATION_BINS[max(0, min(idx, len(DURATION_BINS) - 1))]


def _build_vocab_list():
    toks = list(_SPECIALS)
    # a{pitch}/d{bin} — anchor token for first note (absolute pitch + duration)
    for pitch in range(85):
        for dur in DURATION_BINS:
            toks.append(f'a{pitch}/d{dur}')
    # i{+/-N}/d{bin} — interval from previous note + duration
    for interval in range(-INTERVAL_RANGE, INTERVAL_RANGE + 1):
        sign = '+' if interval >= 0 else ''
        for dur in DURATION_BINS:
            toks.append(f'i{sign}{interval}/d{dur}')
    # r/d{bin} — rest of given duration
    for dur in DURATION_BINS:
        toks.append(f'r/d{dur}')
    # B{n} — bar-position tokens (B1 = bar 1, B2 = bar 2, ..., B32 = bar 32, then cycle)
    for n in range(1, BAR_CYCLE + 1):
        toks.append(f'B{n}')
    return toks


class VocaloidVocab:
    def __init__(self):
        toks = _build_vocab_list()
        self.tok2idx = {t: i for i, t in enumerate(toks)}
        self.idx2tok = toks

    def __len__(self):
        return len(self.idx2tok)

    @staticmethod
    def quantize_duration(raw_ticks):
        return _quantize_duration(raw_ticks)

    def encode(self, tok):
        return self.tok2idx.get(tok, UNK_IDX)

    def decode(self, idx):
        if 0 <= idx < len(self.idx2tok):
            return self.idx2tok[idx]
        return '<unk>'


def tokens_from_notes(notes, ticks_per_beat=DEFAULT_TICKS_PER_BEAT):
    """Convert (pitch, start_tick, duration) list to interval-encoded token strings.

    First note -> a{pitch}/d{bin}; subsequent notes -> i{+/-N}/d{bin};
    gaps between notes -> r/d{bin}.
    Bar-position tokens B1-B{BAR_CYCLE} are inserted at each bar boundary.
    """
    if not notes:
        return []
    ticks_per_bar = ticks_per_beat * 4  # assume 4/4 time
    tokens = []
    prev_pitch = None
    prev_end = None
    prev_bar = -1
    for pitch, start, duration in notes:
        dur_bin = _quantize_duration(duration)
        if prev_end is not None and start > prev_end:
            rest_bin = _quantize_duration(start - prev_end)
            tokens.append(f'r/d{rest_bin}')
        current_bar = start // ticks_per_bar
        if current_bar != prev_bar:
            tokens.append(f'B{(current_bar % BAR_CYCLE) + 1}')
            prev_bar = current_bar
        if prev_pitch is None:
            tokens.append(f'a{pitch}/d{dur_bin}')
        else:
            interval = max(-INTERVAL_RANGE, min(INTERVAL_RANGE, pitch - prev_pitch))
            sign = '+' if interval >= 0 else ''
            tokens.append(f'i{sign}{interval}/d{dur_bin}')
        prev_pitch = pitch
        prev_end = start + duration
    return tokens


def notes_from_tokens(tokens, anchor_pitch=60):
    """Reconstruct (pitch, start_tick, duration) from interval-encoded tokens."""
    notes = []
    curr_pitch = anchor_pitch
    curr_tick = 0
    for tok in tokens:
        if tok in _SPECIALS or tok.startswith('B'):
            continue  # skip specials and bar-position markers
        try:
            kind, dur_str = tok.split('/')
            dur = int(dur_str[1:])
        except (ValueError, IndexError):
            continue
        if kind.startswith('r'):
            curr_tick += dur
        elif kind.startswith('a'):
            curr_pitch = int(kind[1:])
            notes.append((curr_pitch, curr_tick, dur))
            curr_tick += dur
        elif kind.startswith('i'):
            interval = int(kind[1:])  # handles both '+N' and '-N'
            curr_pitch = max(0, min(127, curr_pitch + interval))
            notes.append((curr_pitch, curr_tick, dur))
            curr_tick += dur
    return notes


def initialize_model(vocab_size, device, hid_dim=512, n_layers=6, n_heads=8,
                     pf_dim=1024, dropout=0.1, max_seq_len=512):
    from support.model import MusicTransformerGPT
    return MusicTransformerGPT(
        vocab_size=vocab_size,
        hid_dim=hid_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        pf_dim=pf_dim,
        dropout=dropout,
        max_seq_len=max_seq_len,
        device=device,
    ).to(device)
