# music-arch-advisor

**When to invoke:** When the current architecture or training method is producing unsatisfactory results and we need to decide what to try next. This is a pivot-decision tool, not a general research query.

Trigger: `/music-arch-advisor`, "this architecture isn't working", "what should we switch to", "recommend an alternative architecture", "the model quality is bad, what do we do next"

## What this skill does

Reads the current project state (training logs, architecture, loss curves), diagnoses *why* the current approach is underperforming, then spawns a research agent that returns ranked alternative architectures and training methods — specifically filtered for this project's constraints.

## Instructions

### Step 1 — Diagnose the current approach

Read these files to understand what's failing before recommending anything:
- `DECISIONS.md` — **read this first** to see what has already been tried; never re-recommend an approach listed there
- `train.log`, `train2.log`, `train3.log` (most recent training metrics — look for plateau, overfitting, divergence)
- `ARCHITECTURE.md` (what we're currently running)
- `support/model.py` (implementation details)
- `support/vocalVocab.py` (encoding scheme)
- `outputs/out.mid` analysis if available (qualitative output issues)

Identify the failure mode. Is it:
- **Plateau** — loss stopped improving, model needs more capacity or better encoding
- **Repetition** — model loops; structural modeling is too local
- **Range collapse** — model always generates in a narrow pitch range
- **Rhythm collapse** — all notes same duration
- **Poor phrase structure** — no long-range coherence
- **Overfitting** — train/val gap too large

### Step 2 — Spawn a targeted research agent

Launch a general-purpose agent with a prompt that includes:
1. The diagnosed failure mode from Step 1
2. Current constraints: ~16M param budget, 330k training sequences, interval-encoded MIDI tokens (a/i/r format), MPS training on Apple Silicon, decoder-only GPT baseline
3. Request: survey state-of-the-art symbolic music generation architectures (Music Transformer, REMI/Pop Music Transformer, Compound Word Transformer, FIGARO, MusicBERT, hierarchical models, diffusion in symbolic space) and return ranked alternatives that specifically address the diagnosed failure mode

The agent should return for each recommendation:
- What to adopt and why it fixes the specific failure mode
- Implementation difficulty (Easy = swap a component / Medium = rewrite training loop / Hard = new architecture from scratch)
- Expected quality gain vs current approach
- Whether it requires retraining from scratch or can be layered on top
- A "Why now" field: why this specifically addresses the current failure mode (not generic praise)

After the skill completes, update `DECISIONS.md` with whatever was decided and the reasoning.

### Step 3 — Present a decision

Synthesize the research agent's findings into a clear recommendation:

1. **Root cause** — one sentence on why the current approach is failing
2. **Recommended next step** — the single highest-confidence recommendation with implementation plan
3. **Alternatives** — 2-3 other options ranked by (quality gain × implementation effort)
4. **What NOT to do** — approaches that sound promising but don't fit our constraints

Keep it short and decisive. The goal is to walk away with a clear next action, not an exhaustive survey.
