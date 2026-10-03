---
name: stem-split
description: Download songs with yt-dlp and split them into instrumental (no-vocal) and vocal tracks. Use for vocal removal, karaoke or background-music versions, or stem separation of audio files.
---

# Stem split: download → instrumental + vocal

Turns songs (local files or URLs) into two 320 kbps MP3s each: an **instrumental** (`no-vocal/`) and an
**isolated vocal** (`vocal/`), tuned so the instrumental works as background music. The source file is
moved to `original/` once its outputs exist.

`scripts/stem.py` in this skill folder is the implementation. Read it before changing anything, and reuse
it rather than reaching for a one-shot `demucs` CLI call — that loses both the quality and the
resumability that make long batches workable.

## Folder convention

```
<root>/
├── inbox/        downloads waiting to be processed   ($STEM_INBOX)
├── original/     sources, moved here after processing
├── no-vocal/     instrumental, same filename as the source
└── vocal/        isolated vocals, same filename
```

Ask the user where the collection lives (`$STEM_ROOT`) rather than assuming. Outputs keep the source
filename exactly, so the three folders line up.

## Environment (once per session)

Run the separation **where the files are**: if the folder is on the user's computer, use the shell on that
computer, and do not copy hundreds of megabytes of audio into a cloud sandbox to process it. Install into a
venv in scratch space, never into the user's folders:

```bash
python3 -m venv ~/sep
~/sep/bin/pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
~/sep/bin/pip install "demucs==4.0.1" --no-deps
~/sep/bin/pip install -r scripts/requirements.txt
```

Gotchas that cost real time:

- Newer `demucs` releases pull in a Rust-built dependency (`sphn`) that fails to build; pin `4.0.1 --no-deps`.
- `audio-separator` installs a `torchvision` that may not match `torch` (`operator torchvision::nms does not exist`).
  Reinstall `torchvision` from the same index as `torch`.
- Background installs die when a sandboxed shell call ends (`nohup`/`setsid` do not survive). Install in the
  foreground, in steps that each fit the call timeout; pip's cache makes the retry fast.
- Disk in such a VM is small (~10 GB). `rm -rf ~/.cache/pip ~/.cache/puccinialin` when writes start failing.
- `ffmpeg` must be on PATH.

Models download on first use: Demucs weights to `~/.cache/torch/hub` (~320 MB), the MDX model to
`$STEM_MODELS` (~65 MB).

## Step 1: download

```bash
~/sep/bin/python scripts/stem.py fetch "https://..."              # one track
~/sep/bin/python scripts/stem.py fetch --playlist "https://..."   # whole playlist
```

yt-dlp takes the best audio and converts to MP3 (`-x --audio-format mp3 --audio-quality 0 --embed-metadata`).
Files land in the inbox as `NN - Title.mp3`, where `NN` continues the highest number already in `inbox/`,
`original/` and `no-vocal/`. A `cookies.txt` in `$STEM_ROOT` (or `$STEM_COOKIES`) is used if present.

Downloading is the user's call: process material they have the right to use, and skip this step for files
they already have.

## Step 2: separate

```bash
export STEM_ROOT=/path/to/collection STEM_WORK=~/stemwork STEM_MODELS=~/models
~/sep/bin/python scripts/stem.py run --budget 145 "$STEM_INBOX"/*.mp3
```

Re-run the identical command until it prints `ALL DONE`. Each invocation does as many steps as fit the
budget, prints `PAUSE before <step>`, and exits; finished work is never redone. With no sandbox timeout to
work around, pass a large `--budget` and let it run to completion in one call.

Two models each estimate the instrumental, and the results are averaged — their errors differ, so the
average has less vocal bleed than either alone:

| Step | Model | Output |
|---|---|---|
| 1 | Demucs `htdemucs_ft`, **vocal-specialist sub-model only** (the fine-tuned bag holds one model per stem, so running just the vocal one costs 1× instead of 4×) | instrumental = mix − vocals |
| 2 | UVR MDX-Net `Inst_HQ_3` (trained to output instrumentals) | instrumental |
| 3 | merge | `no-vocal = 0.5·demucs + 0.5·mdx`, `vocal = mix − no-vocal` |

Design points worth keeping if you rework it:

- **Instrumental = mix − vocals**, never the sum of drums+bass+other, which drops content. Subtraction also
  keeps `vocal + no-vocal == mix`.
- **90 s chunks with 5 s overlap and a linear crossfade.** Keeps every unit of work under ~60 s and bounds
  memory; neither model has long-range context, so chunking is inaudible.
- **Resumable, atomic:** intermediates in `$STEM_WORK/<song>/`, written to a temp name then renamed, with a
  `DONE` marker per song. A killed call costs one step. To redo a song, delete its `$STEM_WORK` folder first.
- **Soft limiter** before encoding: peaks above 0.9 bent under 1.0 with a tanh knee, so loud masters do not clip.
- **Feed MDX at half level and double the output**, so audio-separator's peak normaliser cannot change the
  level per chunk.
- **Never write temp files into the user's folders** — encode in scratch, then copy into place. Files inside a
  folder shared with a sandbox often cannot be deleted without the user's permission, so a stray temp file is
  stuck there.
- Time estimates per step come from audio length (`~0.3 s/s` Demucs, `~0.65 s/s` MDX on 4 ARM cores); the
  budget check uses them to avoid starting a step that would be killed mid-way.

Speed: ~3 min of compute per 3-minute song on a 4-core CPU. Say this up front when the user asks for a big
batch, and report progress per batch.

## Tuning

- `DEMUCS_WEIGHT` (0.5): higher = fuller, warmer instrumental; lower = more aggressive vocal removal.
- `shifts=1` in the Demucs call: 2+ averages time-shifted passes, slightly cleaner, proportionally slower.
- `MDX_MODEL`: any model audio-separator knows. **BS-RoFormer** (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`)
  is the quality ceiling but ~10 s of compute per second of audio on CPU (~30 min/song) — GPU or Apple Silicon
  only. Update `COST["mdx"]` when swapping models.

## Checks before reporting done

- `original/`, `vocal/` and `no-vocal/` have equal counts and identical filename lists.
- No suspiciously small MP3s (`find no-vocal vocal -name '*.mp3' -size -100k`).
- Tell the user which folders hold what, and ask them to spot-check one track before a long batch continues.
- Emptied inbox folders and any leftovers need the user's permission to delete; offer, do not assume.
