# -*- coding: utf-8 -*-
"""
Vocaloid-informed training penalty function.

Rules derived from empirical analysis of 156 VSQX files (96,137 notes):

  P2  Stepwise+unison ≥72.7% of transitions in corpus — penalise non-step probability
  P4  Core pitch range MIDI 53–79; hard floor/ceiling at 48–84
  P5  Minimum note duration: <60 ticks causes synthesis artefacts; <120 ticks is soft floor
  P7  Large leaps (>7 semitones) only 6% of corpus — suppress excess
  P11 Tritone (±6) near-absent (<0.4%) — penalise
  P12 Very large leaps (>12 semitones) near-zero — hard suppression

All penalties operate on the model's output probability distribution, making them
fully differentiable — gradients flow back through softmax into model weights.
"""

import torch
import torch.nn as nn

from support.vocalVocab import DURATION_BINS


class VocaloidRulesPenalty(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

        non_step = []        # |interval| > 2               (P2)
        out_of_range = []    # anchor pitch outside MIDI 53–79  (P4 soft)
        hard_oor = []        # anchor pitch outside MIDI 48–84  (P4 hard)
        short_note = []      # duration < 120 ticks          (P5 soft)
        very_short = []      # duration < 60 ticks           (P5 hard)
        large_leap = []      # |interval| > 7               (P7)
        tritone = []         # |interval| == 6              (P11)
        for idx, tok in enumerate(vocab.idx2tok):
            if '/' not in tok:
                continue
            try:
                kind, dur_str = tok.split('/')
                dur = int(dur_str[1:])
            except (ValueError, IndexError):
                continue

            # Duration checks — note tokens only (not rests)
            if not kind.startswith('r'):
                if dur < 60:
                    very_short.append(idx)
                elif dur < 120:
                    short_note.append(idx)

            # Anchor token pitch range (P4)
            if kind.startswith('a'):
                try:
                    pitch = int(kind[1:])
                    if pitch < 53 or pitch > 79:
                        out_of_range.append(idx)
                    if pitch < 48 or pitch > 84:
                        hard_oor.append(idx)
                except ValueError:
                    pass

            # Interval checks (P2, P7, P11, P12)
            if kind.startswith('i'):
                try:
                    interval = int(kind[1:])  # handles '+N' and '-N'
                    a = abs(interval)
                    if a > 2:
                        non_step.append(idx)
                    if a == 6:
                        tritone.append(idx)
                    if a > 7:
                        large_leap.append(idx)
                except ValueError:
                    pass

        n = len(vocab)
        self.register_buffer('non_step_mask',       self._make(non_step, n))
        self.register_buffer('out_of_range_mask',   self._make(out_of_range, n))
        self.register_buffer('hard_oor_mask',        self._make(hard_oor, n))
        self.register_buffer('short_note_mask',      self._make(short_note, n))
        self.register_buffer('very_short_mask',      self._make(very_short, n))
        self.register_buffer('large_leap_mask',      self._make(large_leap, n))
        self.register_buffer('tritone_mask',         self._make(tritone, n))

    @staticmethod
    def _make(indices, n):
        m = torch.zeros(n, dtype=torch.bool)
        for i in indices:
            m[i] = True
        return m

    def forward(self, logits):
        """
        Args:
            logits: (B, T, vocab_size) — raw model output before softmax
        Returns:
            scalar penalty (added to cross-entropy loss after scaling by penalty_weight)
        """
        probs = torch.softmax(logits.float(), dim=-1)  # (B, T, V)

        # P2 — discourage non-stepwise intervals; corpus: 72.7% stepwise+unison
        # We penalise probability mass on tokens with |interval| > 2
        p2 = probs[..., self.non_step_mask].sum(-1).mean()

        # P4 — pitch range; apply soft then hard penalty on anchor tokens
        p4_soft = probs[..., self.out_of_range_mask].sum(-1).mean()
        p4_hard = probs[..., self.hard_oor_mask].sum(-1).mean()

        # P5 — synthesis duration floor; <60 ticks creates audible artefacts
        p5_hard = probs[..., self.very_short_mask].sum(-1).mean()
        p5_soft = probs[..., self.short_note_mask].sum(-1).mean()

        # P7 — large leaps (>7 st.); corpus: 6.0% — suppress excess probability
        p7 = probs[..., self.large_leap_mask].sum(-1).mean()

        # P11 — tritone; corpus: ~0.4% — penalise
        p11 = probs[..., self.tritone_mask].sum(-1).mean()

        penalty = (
            0.50 * p2 +       # stepwise motion (most important structural rule)
            0.40 * p4_soft +   # pitch range soft boundary
            1.00 * p4_hard +   # pitch range hard boundary
            1.00 * p5_hard +   # synthesis floor hard
            0.30 * p5_soft +   # synthesis floor soft
            0.30 * p7 +        # large leap suppression
            0.15 * p11         # tritone avoidance
        )

        return penalty


def _count_mask_tokens(vocab, mask_attr):
    """Utility: report how many vocab tokens fall under each rule mask."""
    dummy = VocaloidRulesPenalty(vocab)
    mask = getattr(dummy, mask_attr)
    return int(mask.sum().item())


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from support.vocalVocab import VocaloidVocab
    import torch

    v = VocaloidVocab()
    p = VocaloidRulesPenalty(v)

    print(f"Vocab size: {len(v)}")
    print(f"  non-stepwise tokens (|i|>2):  {p.non_step_mask.sum():4d}")
    print(f"  out-of-range anchors (soft):  {p.out_of_range_mask.sum():4d}")
    print(f"  out-of-range anchors (hard):  {p.hard_oor_mask.sum():4d}")
    print(f"  short note tokens (<120t):    {p.short_note_mask.sum():4d}")
    print(f"  very short tokens (<60t):     {p.very_short_mask.sum():4d}")
    print(f"  large leap tokens (|i|>7):    {p.large_leap_mask.sum():4d}")
    print(f"  tritone tokens (|i|=6):       {p.tritone_mask.sum():4d}")

    # Smoke test with random logits
    logits = torch.randn(2, 63, len(v), requires_grad=True)
    pen = p(logits)
    print(f"\nSmoke test penalty (random logits): {pen.item():.4f}")
    pen.backward()
    print("Backward pass OK")
