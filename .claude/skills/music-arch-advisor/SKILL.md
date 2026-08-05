# music-arch-advisor

Trigger: `/music-arch-advisor`, "what architecture should we use", "recommend a better model", "research music generation architectures", "what's the state of the art for music generation"

## What this skill does

Spawns a research agent that surveys the state-of-the-art in neural music generation and returns ranked architectural recommendations tailored to this project's constraints (16M param decoder-only GPT, 330k training sequences, interval-encoded MIDI tokens, MPS training on Apple Silicon).

## Instructions

When triggered, run the following steps:

### Step 1 — Assess current project state

Read these files to understand where the project currently stands before giving recommendations:
- `ARCHITECTURE.md` — current model design
- `support/model.py` — current model implementation
- `support/vocalVocab.py` — vocab and encoding scheme
- `train.log` and `train2.log` and `train3.log` (if they exist) — latest training metrics

### Step 2 — Spawn research agent

Launch a general-purpose agent with this prompt:

> You are an expert ML researcher in neural music generation. Survey the state-of-the-art symbolic and audio-domain music generation architectures (Music Transformer, REMI, MusicGen, DiffSinger, Jukebox, MusicBERT, Compound Word Transformer, FIGARO, MuseNet, SymphonyNet, and others). For each, note: key innovation, quality level (1-5 stars), and feasibility given a ~16M param budget and 330k training sequences.
>
> Then produce a ranked Top 5 list of what this specific project should adopt next if the current approach plateaus, with implementation difficulty (Easy/Medium/Hard) and expected quality gain. Focus especially on: better encoding schemes, hierarchical structure modeling, conditioning mechanisms, and training techniques. Be specific and cite papers.

### Step 3 — Synthesize and present findings

Once the research agent completes, present:

1. **Current architecture assessment** — strengths and known limitations of our decoder-only GPT with interval encoding based on what the research found
2. **Top 5 recommendations** — ranked by (quality gain × feasibility), each with:
   - What to adopt
   - Why it fits our constraints
   - Implementation effort
   - Expected improvement
3. **Quick wins** — things we can add to the existing architecture without retraining from scratch (e.g. conditioning tokens, better encoding, inference tricks)
4. **Longer-term bets** — architectures worth a full rewrite if we want to hit production quality

Keep the output concise and actionable. No need to explain what every paper is about in depth — focus on the recommendation and the reasoning.
