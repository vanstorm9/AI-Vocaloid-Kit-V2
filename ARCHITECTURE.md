# Model Architecture: Decoder-Only GPT Transformer

## Overview

The model is a decoder-only autoregressive transformer — the same architectural family as GPT — trained to predict the next music token given all previous tokens. It replaced an earlier Seq2Seq encoder-decoder design that was adapted from an NLP translation tutorial.

---

## Token Format — Interval Encoding

Raw MIDI note data is converted to a compact token vocabulary before the model ever sees it.

| Token | Meaning |
|-------|---------|
| `a65/d240` | **Anchor** — first note of a sequence; absolute MIDI pitch 65, duration bin 240 ticks |
| `i+2/d240` | **Interval** — pitch moves +2 semitones from previous note; duration bin 240 ticks |
| `i-3/d480` | **Interval** — pitch moves −3 semitones; duration bin 480 ticks |
| `r/d120`   | **Rest** — silence for duration bin 120 ticks |

**Duration quantization:** Raw tick values are mapped to 32 bins, reducing a previously sparse ~480-entry duration space down to a compact, uniform distribution.

**Interval clamping:** Pitch intervals are clamped to ±12 semitones (one octave). Jumps beyond an octave are rare in vocal melodies and clamping keeps the vocab small.

**Transposition invariance:** A melody in C major and the same melody in D major produce identical token sequences. This makes the model key-agnostic, which is critical given that our MIDI training data comes from many different keys. The effective training data is multiplied by ~12× for free.

Total vocabulary: ~3,556 tokens.

---

## Architecture Details

### `MusicTransformerGPT`

```
Token Embedding  (vocab_size → hid_dim=256)
        ↓
  4 × GPTDecoderLayer
        ↓
    LayerNorm
        ↓
  Linear → vocab_size logits
```

- No positional embedding — position is handled entirely by `RelativeAttentionBias`
- ~3.9M trainable parameters
- Causal (upper-triangular) mask prevents attending to future tokens

### `GPTDecoderLayer`

Each layer uses **pre-norm** (LayerNorm before each sub-layer, not after):

```
x → LayerNorm → Self-Attention (+ RelativeAttentionBias) → residual → x
x → LayerNorm → Feedforward (256 → 512 → 256)           → residual → x
```

Self-attention only — no cross-attention, no encoder. 8 heads, head dim = 32.

### `RelativeAttentionBias` (T5-style)

A learned table of shape `(2 × max_dist + 1, n_heads)`. During attention:

```
energy[i, j] += bias_table[clip(i - j, -max_dist, max_dist)]
```

Each attention head independently learns how much to weight notes that are 1, 2, 3, … steps apart. This replaces absolute sinusoidal or learned positional embeddings and generalizes better to sequence lengths not seen during training.

---

## Why This Architecture

**The core problem with Seq2Seq for music:** Music generation is a single continuous stream, not a translation from one language to another. Treating a 7-note input as a "source sentence" and the next 7 notes as a "target sentence" breaks the natural continuity of a melody — the model never learns long-range phrase structure because every generation step restarts from a short, fixed-length context.

**Why decoder-only fits better:**
- The autoregressive next-token objective directly matches how music unfolds: each note conditions on everything that came before it
- The 64-token context window (up from 7) lets the model observe several full musical phrases before predicting the next note
- Relative attention bias lets the model learn the importance of recency without being anchored to absolute positions — a note 3 steps ago matters differently from one 30 steps ago, regardless of where in the sequence they both sit

**Why interval encoding over absolute pitch:**
- The training data combines a small VSQX corpus (~167 files, all in various keys) with a large MIDI corpus spanning all 12 keys
- With absolute pitch tokens, the model must separately learn the same melodic pattern 12 times — once per key — wasting capacity
- With interval encoding, the same pattern looks identical in all keys, giving the model ~12× more data per pattern

---

## Comparison to the Previous Seq2Seq Architecture

| Property | Seq2Seq (old) | Decoder-only GPT (new) |
|---|---|---|
| **Architecture** | Encoder + Decoder (cross-attention) | Decoder only |
| **Training objective** | Predict target sequence from source | Predict next token from all prior tokens |
| **Context window** | 7 tokens | 64 tokens |
| **Position encoding** | Absolute learned (hard cap at 100) | Relative attention bias (no hard cap) |
| **Pitch representation** | Absolute MIDI pitch (`n65`) | Semitone interval from previous note (`i+2`) |
| **Duration representation** | Raw ticks (~480 unique values) | 32 quantized bins |
| **Vocabulary size** | ~1,540 tokens (sparse) | ~3,556 tokens (but interval/rest tokens are dense and regular) |
| **Transposition invariance** | No | Yes (~12× data efficiency on diverse MIDI) |
| **Sampling** | Argmax (deterministic) | Temperature-controlled multinomial |
| **Cross-attention overhead** | Yes (encoder must run each step) | None |
| **Long-range phrase modeling** | Poor (context resets every 7 notes) | Better (64-note sliding window) |
| **Training dependencies** | torchtext (legacy, deprecated) | torch.utils.data.Dataset only |

---

## Pros of the New Architecture

- **Better long-range structure:** 64-token context allows the model to condition on multiple measures of music, enabling it to learn recurring motifs and phrase boundaries
- **Key-agnostic generalization:** Interval encoding means the model trained on a C-major song automatically understands the same melody in F# major
- **Simpler training loop:** No source/target split — just shift the sequence by 1 and compute cross-entropy. No torchtext dependency.
- **Relative position bias is more expressive:** The model learns asymmetric recency preferences per head, rather than a fixed sinusoidal signal
- **Temperature sampling:** Output diversity is tunable at inference time without retraining
- **Checkpoint format:** Saves model state, vocab, and hyperparameters together — no need to reconstruct the vocab separately when loading

## Cons / Tradeoffs

- **Larger vocabulary:** ~3,556 tokens vs ~1,540 in the old model increases the output projection layer size, though the total parameter count (~3.9M) remains modest
- **Interval decoding adds state:** At inference time, the absolute pitch must be tracked across tokens to reconstruct MIDI notes; the old format was self-contained per token
- **No conditioning signal:** The decoder-only model generates unconditionally from a seed. The old Seq2Seq framing at least nominally supported conditioning on a source melody, which could be useful for harmonization or continuation tasks in the future
- **Retraining required:** The new token format is incompatible with the old dataset and model checkpoints — the previous `9-22-music.pt` checkpoint cannot be loaded into the new architecture
- **Anchor token only at sequence start:** The current design places an absolute-pitch anchor only at the first token. Sequences that drift far in pitch over 64 tokens may lose register; a periodic re-anchor every N tokens could help in future iterations
