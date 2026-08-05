# Architecture & Training Decisions

A running log of what has been tried, why, and the outcome. The `/music-arch-advisor` skill uses this to avoid re-recommending approaches already evaluated.

---

## Tried and evaluated

### Seq2Seq Encoder-Decoder (Deprecated)
- **What:** Encoder + Decoder transformer adapted from NLP translation tutorial
- **Encoding:** Absolute MIDI pitch + raw tick durations (~1,540 token vocab)
- **Context window:** 7 tokens
- **Outcome:** Poor. Context too short to learn phrase structure. Absolute pitch forces the model to re-learn every pattern 12 times across keys. Hard cap at positional embedding length 100. Deprecated in favor of decoder-only GPT.

### Decoder-only GPT v1 — 3.9M params (Completed, superseded)
- **What:** 4-layer, 256 hid_dim, 8 heads, 512 FFN, relative attention bias (T5-style)
- **Encoding:** Interval encoding (`a{pitch}/d{bin}`, `i{+/-N}/d{bin}`, `r/d{bin}`, 32 duration bins)
- **Training data:** 167 VSQX + 9,649 MIDI vocal tracks (330k/82k train/val windows)
- **Epochs:** 80 (30 fixed LR + 50 cosine decay)
- **Best val PPL:** 4.683
- **Outcome:** Learned correct vocal pitch range and stepwise motion. Main flaw: phrase-level loops (3-note cells repeated 4× in output). Superseded by larger model.

### Decoder-only GPT v2 — 16M params (In progress, epoch 13)
- **What:** 6-layer, 512 hid_dim, 8 heads, 1024 FFN, relative attention bias
- **Same encoding and dataset as v1**
- **Added:** n-gram blocking at inference (`--noRepeatNgram 4`)
- **Added:** VocaloidRulesPenalty (weight 0.3) — soft prob penalties on logits for non-stepwise motion, pitch range, short notes, large leaps, tritones
- **Training:** 50 epochs cosine decay (3e-4 → 1e-5), MPS, penaltyWeight=0.3
- **Status:** Epoch 13 — val PPL 3.10 (already below v1 best of 4.68)

---

## Implemented (next training run)

The following are implemented and will apply when the current run finishes and we retrain on the new dataset:

- **Bar-position tokens B1–B32** — inserted at every bar boundary during tokenization; gives model explicit phrase-position context. Proven fix for phrase-level repetition (Musical Attention Transformer, 2026).
- **12-key transposition augmentation** — applied to training set only after 80/20 split; turns ~330k raw windows → ~3.9M training windows. Free because interval tokens are already transposition-invariant.
- **Vocabulary updated** — 3,556 → 3,588 tokens (+32 bar-position tokens)
- **MIDI re-extraction** — all 9,647 token files regenerated with bar tokens included

## Considered but not yet tried

- **REMI encoding** — absolute pitch + beat position + duration as separate tokens; better harmonic context at the cost of larger vocab
- **Compound Word Transformer** — predicts multiple token types per timestep; could improve duration/pitch correlation
- **Hierarchical generation** — generate phrase-level structure first, then fill in notes
- **Corruption-refinement secondary objective** — denoising auxiliary loss (ImprovNet/Yin-Yang 2025); adds motif consistency without new data
- **Mora-boundary tokens** — REMI-Aligned style syllable boundary markers for lyric-note alignment (CSL-L2M, 2024)
