# claude-stem-split

A [Claude skill](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview) that downloads
songs and splits them into an **instrumental** track and an **isolated vocal** track, tuned for getting a
clean backing mix rather than a karaoke novelty.

It works as a plain command-line tool too — the skill is just the instructions that let Claude drive it.

```
collection/
├── inbox/        downloads waiting to be processed
├── original/     sources, moved here after processing
├── no-vocal/     instrumental, same filename as the source
└── vocal/        isolated vocals, same filename
```

## Why two models

Each of two separation models estimates the instrumental on its own, and the results are averaged. The
models make different mistakes, so the average has noticeably less vocal bleed than either one alone.

| Step | Model | Output |
|---|---|---|
| 1 | **Demucs `htdemucs_ft`**, vocal-specialist sub-model only — the fine-tuned bag holds one model per stem, so running just the vocal one costs 1× instead of 4× | instrumental = mix − vocals |
| 2 | **UVR MDX-Net `Inst_HQ_3`**, trained specifically to output instrumentals | instrumental |
| 3 | merge | `no-vocal = 0.5·demucs + 0.5·mdx`, `vocal = mix − no-vocal` |

Details that matter for quality:

- **Instrumental = mix − vocals**, not the sum of drums/bass/other, so no instrument is quietly dropped.
  The subtraction also keeps `vocal + no-vocal` equal to the original mix.
- **Soft limiter** before encoding: peaks above 0.9 are bent smoothly under 1.0 with a tanh knee, so loud
  masters don't hard-clip. Everything below the knee is untouched.
- MDX input is fed at half level and restored afterwards, which stops audio-separator's peak normaliser
  from shifting the level from one chunk to the next.
- `DEMUCS_WEIGHT` trades the two models off: higher is a fuller, warmer instrumental, lower removes vocals
  more aggressively.

### Built for long batches

- **Chunked:** 90 s chunks with 5 s overlap and a linear crossfade. Every unit of work stays under about a
  minute and memory use is bounded. Neither model has long-range context, so chunking is inaudible.
- **Resumable:** intermediates live in `$STEM_WORK/<song>/`, each written to a temp name then renamed, and a
  finished song gets a `DONE` marker. Interrupt it any time; you lose at most one chunk of work.
- **Budgeted:** `--budget N` makes a run stop cleanly before it would exceed N seconds, estimating each step
  from the audio length. This exists because some sandboxes cap how long one shell call may run.

### Considered and rejected

- **BS-RoFormer** (`model_bs_roformer_ep_317_sdr_12.9755`) is the quality ceiling, but needs ~10 s of compute
  per second of audio on a CPU — about 30 minutes per song. Worth it on a GPU; see *Swapping models*.
- **Summing all four `htdemucs_ft` models:** 4× the cost, and drums+bass+other loses content that
  mix − vocals keeps.

## Install

```bash
git clone https://github.com/<you>/claude-stem-split
cd claude-stem-split

python3 -m venv ~/sep
~/sep/bin/pip install torch torchaudio torchvision   # add --index-url https://download.pytorch.org/whl/cpu on Linux CPU
~/sep/bin/pip install "demucs==4.0.1" --no-deps      # newer releases need a Rust toolchain
~/sep/bin/pip install -r scripts/requirements.txt
```

`ffmpeg` must be on PATH. Models download on first run: Demucs weights to `~/.cache/torch/hub` (~320 MB),
the MDX model to `$STEM_MODELS` (~65 MB).

To use it as a skill, put this folder where your Claude client looks for skills (for Claude Code, a
`skills/` directory in your project or `~/.claude/skills/`), keeping `SKILL.md` at its root.

## Usage

```bash
export STEM_ROOT=/path/to/collection   # folder holding inbox/ original/ vocal/ no-vocal/
export STEM_WORK=~/stemwork            # scratch; a few hundred MB per song in flight
export STEM_MODELS=~/models

# 1. download (skip for files you already have — drop those in inbox/)
~/sep/bin/python scripts/stem.py fetch "https://..."
~/sep/bin/python scripts/stem.py fetch --playlist "https://...list=..."

# 2. separate
~/sep/bin/python scripts/stem.py run --budget 100000 "$STEM_ROOT"/inbox/*.mp3
```

Re-run the same `run` command until it prints `ALL DONE`.

- Songs taken from the inbox are **moved** to `original/` once their outputs are written.
- Songs passed from `original/` are re-processed and their outputs overwritten. To redo a song, delete its
  folder in `$STEM_WORK` first, or its `DONE` marker will skip it.

Downloads are named `NN - Title.mp3`, continuing the highest number already present across `inbox/`,
`original/` and `no-vocal/`. A `cookies.txt` in `$STEM_ROOT` (or `$STEM_COOKIES`) is picked up automatically.

### Swapping models

`MDX_MODEL` accepts any model name audio-separator knows, for example
`model_bs_roformer_ep_317_sdr_12.9755.ckpt` or `MDX23C-8KFFT-InstVoc_HQ.ckpt`. Update `COST["mdx"]` to match
so the time budgeting stays honest.

## Performance

About 3 minutes of compute per 3-minute song on four CPU cores (≈0.3 s/s for Demucs, ≈0.65 s/s for MDX).
Much faster on Apple Silicon or a GPU.

## Note on sources

The download step is a thin wrapper around [yt-dlp](https://github.com/yt-dlp/yt-dlp). Downloading from a
site may be against its terms of service, and the separated tracks are derivative works of the recording —
so use this on material you have the right to use, and treat the output accordingly.

## Credits

Built on [Demucs](https://github.com/adefossez/demucs) (Meta), the
[UVR](https://github.com/Anjok07/ultimatevocalremovergui) MDX-Net models,
[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) and
[yt-dlp](https://github.com/yt-dlp/yt-dlp). Those projects do the separation; this repository is the
pipeline, the quality choices and the batch plumbing around them.

## License

MIT — see [LICENSE](LICENSE).
