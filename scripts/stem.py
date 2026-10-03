#!/usr/bin/env python3
"""Instrumental-first stem extraction: Demucs (htdemucs_ft vocal specialist) + MDX-Net Inst HQ3 ensemble.

Pipeline for each song (all steps resumable, each unit of work is short):
  prep    decode the MP3 to 44.1 kHz stereo float WAV              -> work/<song>/mix.wav
  demucs  per chunk: vocals from htdemucs_ft's vocal model;
          instrumental = mix - vocals                               -> work/<song>/demucs_<i>.wav
  mdx     per chunk: instrumental from UVR-MDX-NET-Inst_HQ_3        -> work/<song>/mdx_<i>.wav
  merge   stitch chunks (linear crossfade), average the two instrumentals,
          vocals = mix - instrumental, encode 320 kbps MP3s         -> no-vocal/<song>.mp3, vocal/<song>.mp3
          then move the source from the inbox to original/ (if it came from the inbox).

  fetch   download audio from URLs with yt-dlp into the inbox, numbered to continue
          the existing numbering                                       -> inbox/NN - Title.mp3

Usage:
  stem.py fetch URL [URL ...]
      Downloads each URL as a 320 kbps-source MP3 into the inbox ($STEM_INBOX, or ROOT/inbox).
      Add --playlist to take a whole playlist instead of just the linked track.

  stem.py run --budget 150 SONG.mp3 [SONG.mp3 ...]
      Works through the songs in order, one step at a time, and exits before the time budget
      (seconds) would be exceeded. Run it again to continue; finished work is never redone.
"""
import argparse, glob, math, os, re, shutil, subprocess, sys, time
import numpy as np
import soundfile as sf

INBOX = os.environ.get("STEM_INBOX")                                     # download target / processing inbox
ROOT = os.environ.get("STEM_ROOT", os.path.expanduser("~/mnt/ytdlp"))   # folder holding original/ vocal/ no-vocal/
WORK = os.environ.get("STEM_WORK", os.path.expanduser("~/stemwork"))     # scratch space for intermediate files
MODELS = os.environ.get("STEM_MODELS", os.path.expanduser("~/models"))   # MDX model cache
SR = 44100
CHUNK = 90.0      # seconds of new audio per chunk
OVERLAP = 5.0     # seconds shared with the next chunk (crossfaded)
# rough CPU cost in seconds of compute per second of audio (4-core ARM), used for budgeting
COST = {"prep": 0.02, "demucs": 0.30, "mdx": 0.65, "merge": 0.05}
MDX_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
DEMUCS_WEIGHT = 0.5   # share of the Demucs instrumental in the final average (MDX gets the rest)

_demucs = None
_mdx = None


def log(*a):
    print(*a, flush=True)


def atomic_write_wav(path, audio):
    tmp = path + ".tmp.wav"
    sf.write(tmp, audio.T, SR, subtype="FLOAT")
    os.replace(tmp, path)


def chunk_bounds(n_samples):
    step, ov = int(CHUNK * SR), int(OVERLAP * SR)
    n = max(1, math.ceil(n_samples / step))
    return [(i * step, min(n_samples, (i + 1) * step + ov)) for i in range(n)]


def stitch(parts, bounds, n_samples):
    out = np.zeros((2, n_samples), dtype=np.float32)
    prev_end = 0
    for part, (s, e) in zip(parts, bounds):
        part = part[:, : e - s]
        ov = max(0, prev_end - s)
        if ov:
            fade = np.linspace(0.0, 1.0, ov, dtype=np.float32)
            out[:, s : s + ov] = out[:, s : s + ov] * (1 - fade) + part[:, :ov] * fade
        out[:, s + ov : e] = part[:, ov:]
        prev_end = e
    return out


def load_demucs():
    global _demucs
    if _demucs is None:
        import torch
        from demucs.pretrained import get_model
        torch.set_num_threads(os.cpu_count() or 4)
        bag = get_model("htdemucs_ft")
        vi = bag.sources.index("vocals")
        k = max(range(len(bag.models)), key=lambda i: bag.weights[i][vi])  # the vocal specialist
        _demucs = bag.models[k].eval()
    return _demucs


def run_demucs(mix):
    import torch
    from demucs.apply import apply_model
    m = load_demucs()
    wav = torch.from_numpy(mix)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(m, ((wav - mean) / std)[None], shifts=1, split=True, overlap=0.25, progress=False)[0]
    vocals = (out[m.sources.index("vocals")] * std + mean).numpy()
    return (mix - vocals).astype(np.float32)


def load_mdx(tmpdir):
    global _mdx
    if _mdx is None:
        from audio_separator.separator import Separator
        _mdx = Separator(model_file_dir=MODELS, output_dir=tmpdir, output_format="WAV",
                         output_single_stem="Instrumental", log_level=40)
        _mdx.load_model(model_filename=MDX_MODEL)
    _mdx.output_dir = tmpdir
    if getattr(_mdx, "model_instance", None) is not None:
        _mdx.model_instance.output_dir = tmpdir
    return _mdx


def run_mdx(mix, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    for f in glob.glob(os.path.join(tmpdir, "*")):
        os.remove(f)
    src = os.path.join(tmpdir, "in.wav")
    # feed at half level so the separator's peak normalisation never kicks in; restore afterwards
    sf.write(src, (mix * 0.5).T, SR, subtype="FLOAT")
    load_mdx(tmpdir).separate(src)
    outs = [f for f in glob.glob(os.path.join(tmpdir, "*Instrumental*.wav"))]
    if not outs:
        raise RuntimeError("MDX produced no instrumental")
    inst, sr = sf.read(outs[0], dtype="float32", always_2d=True)
    assert sr == SR, sr
    inst = inst.T * 2.0
    n = mix.shape[1]
    if inst.shape[1] < n:
        inst = np.pad(inst, ((0, 0), (0, n - inst.shape[1])))
    return inst[:, :n]


def soft_limit(x, knee=0.9):
    """Leave everything below the knee untouched; bend peaks above it smoothly under 1.0 (no hard clipping)."""
    a = np.abs(x)
    over = a > knee
    x = x.copy()
    x[over] = np.sign(x[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    return x


def encode_mp3(audio, path):
    """Encode in the scratch folder, then copy into place (no temp files ever land in the output folders)."""
    tmp = os.path.join(WORK, "_encode.wav")
    part = os.path.join(WORK, "_encode.mp3")
    sf.write(tmp, soft_limit(audio).T, SR, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:a", "libmp3lame", "-b:a", "320k", part], check=True)
    shutil.copyfile(part, path)
    os.remove(tmp)
    os.remove(part)


# ---------------------------------------------------------------- download step


def inbox_dir(dest=None):
    return dest or INBOX or os.path.join(ROOT, "inbox")


def next_index(dest):
    """One past the highest NN prefix already used anywhere in the collection."""
    n = 0
    for d in (dest, os.path.join(ROOT, "original"), os.path.join(ROOT, "no-vocal")):
        for f in glob.glob(os.path.join(d, "*.mp3")):
            m = re.match(r"\s*(\d+)", os.path.basename(f))
            if m:
                n = max(n, int(m.group(1)))
    return n + 1


def ytdlp_cmd():
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        return [os.environ.get("STEM_YTDLP", "yt-dlp")]


def fetch(urls, dest=None, playlist=False):
    dest = inbox_dir(dest)
    os.makedirs(dest, exist_ok=True)
    tmp = os.path.join(WORK, "_fetch")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    cmd = ytdlp_cmd() + [
        "-x", "--audio-format", "mp3", "--audio-quality", "0",  # best available, no re-compression loss beyond the encode
        "--embed-metadata", "--no-progress", "--no-overwrites",
        "--yes-playlist" if playlist else "--no-playlist",
        "-o", os.path.join(tmp, "%(autonumber)03d - %(title)s.%(ext)s"),
    ]
    cookies = os.environ.get("STEM_COOKIES", os.path.join(ROOT, "cookies.txt"))
    if os.path.exists(cookies):
        cmd += ["--cookies", cookies]
    subprocess.run(cmd + list(urls), check=True)
    added = []
    i = next_index(dest)
    for f in sorted(glob.glob(os.path.join(tmp, "*.mp3"))):
        title = os.path.basename(f).split(" - ", 1)[1]           # drop yt-dlp's download counter
        out = os.path.join(dest, f"{i:02d} - {title}")
        if os.path.exists(out):
            log("SKIP (exists)", os.path.basename(out))
            continue
        shutil.move(f, out)
        log("FETCHED", os.path.basename(out))
        added.append(out)
        i += 1
    shutil.rmtree(tmp, ignore_errors=True)
    return added


def next_step(song):
    """Return (step, arg, est_seconds) for the next unit of work on this song, or None if done."""
    name = os.path.splitext(os.path.basename(song))[0]
    wd = os.path.join(WORK, name)
    if os.path.exists(os.path.join(wd, "DONE")):
        return None
    mixp = os.path.join(wd, "mix.wav")
    if not os.path.exists(mixp):
        return ("prep", None, 10)
    n = sf.info(mixp).frames
    for kind in ("demucs", "mdx"):
        for i, (s, e) in enumerate(chunk_bounds(n)):
            if not os.path.exists(os.path.join(wd, f"{kind}_{i}.wav")):
                return (kind, i, 8 + COST[kind] * (e - s) / SR)
    return ("merge", None, 5 + COST["merge"] * n / SR)


def do_step(song, step, arg):
    name = os.path.splitext(os.path.basename(song))[0]
    wd = os.path.join(WORK, name)
    os.makedirs(wd, exist_ok=True)
    mixp = os.path.join(wd, "mix.wav")
    if step == "prep":
        tmp = mixp + ".tmp.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", song, "-ar", str(SR), "-ac", "2",
                        "-c:a", "pcm_f32le", tmp], check=True)
        os.replace(tmp, mixp)
        return
    mix = sf.read(mixp, dtype="float32", always_2d=True)[0].T
    bounds = chunk_bounds(mix.shape[1])
    if step in ("demucs", "mdx"):
        s, e = bounds[arg]
        seg = np.ascontiguousarray(mix[:, s:e])
        inst = run_demucs(seg) if step == "demucs" else run_mdx(seg, os.path.join(wd, "mdx_tmp"))
        atomic_write_wav(os.path.join(wd, f"{step}_{arg}.wav"), inst)
        return
    if step == "merge":
        def load(kind):
            parts = [sf.read(os.path.join(wd, f"{kind}_{i}.wav"), dtype="float32", always_2d=True)[0].T
                     for i in range(len(bounds))]
            return stitch(parts, bounds, mix.shape[1])
        inst = DEMUCS_WEIGHT * load("demucs") + (1 - DEMUCS_WEIGHT) * load("mdx")
        vocals = mix - inst
        for d in ("original", "vocal", "no-vocal"):
            os.makedirs(os.path.join(ROOT, d), exist_ok=True)
        encode_mp3(inst, os.path.join(ROOT, "no-vocal", name + ".mp3"))
        encode_mp3(vocals, os.path.join(ROOT, "vocal", name + ".mp3"))
        dest = os.path.join(ROOT, "original", os.path.basename(song))
        if os.path.abspath(os.path.dirname(song)) != os.path.abspath(os.path.join(ROOT, "original")):
            if os.path.exists(dest):
                raise RuntimeError(f"refusing to overwrite {dest}")
            shutil.move(song, dest)
        # drop the big intermediates, keep a DONE marker
        for f in glob.glob(os.path.join(wd, "*.wav")):
            os.remove(f)
        shutil.rmtree(os.path.join(wd, "mdx_tmp"), ignore_errors=True)
        open(os.path.join(wd, "DONE"), "w").close()
        log("DONE", name)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download URLs into the inbox")
    f.add_argument("--dest", help="target folder (default $STEM_INBOX or ROOT/inbox)")
    f.add_argument("--playlist", action="store_true", help="download the whole playlist, not just the linked track")
    f.add_argument("urls", nargs="+")
    r = sub.add_parser("run")
    r.add_argument("--budget", type=float, default=150, help="stop before this many seconds would be exceeded")
    r.add_argument("songs", nargs="+")
    a = ap.parse_args()
    if a.cmd == "fetch":
        got = fetch(a.urls, a.dest, a.playlist)
        log(f"{len(got)} file(s) in {inbox_dir(a.dest)}")
        return 0
    t0 = time.time()
    for song in a.songs:
        while True:
            nxt = next_step(song)
            if nxt is None:
                break
            step, arg, est = nxt
            if time.time() - t0 + est > a.budget:
                log(f"PAUSE before {step}{'' if arg is None else ' ' + str(arg)} of {os.path.basename(song)}")
                return 0
            ts = time.time()
            do_step(song, step, arg)
            log(f"  {step}{'' if arg is None else ' ' + str(arg)} {os.path.basename(song)} {time.time() - ts:.0f}s")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
