"""rehook: a new hook on a published reel. Takes the reel mp4 from the repo, cuts its first seconds (the old hook) at
`cut`, puts a new hook card in front (max 6 words with a number, on screen at frame 0, a count-up moving by 1.5 s over
a blurred, slowly zooming peek of the reel's own reveal), keeps the body untouched, and dissolves the last 1 s into
the new frame 0 (circular, starts before the source's own loop dissolve). New cover from the new hook. No new voice needed (the card runs on music and SFX);
add "hook_say" + the Cartesia secret later if a spoken hook is wanted.

data keys:
  source     repo path of the published reel, e.g. "homicide_race_v4.mp4"
  cut        seconds: where the body starts (pick a pause after the old hook line)
  hook_dur   seconds of the new hook card (default 2.2)
  peek       source time for the card's background (the reveal; no faces), default 60 % of the reel
  count      {from, to, prefix, suffix, dec, label}  the moving number on the card
  hook_y     default 760
"""
import os, sys, math, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "core"))
import numpy as np
from common import *


def read_frames(src, t0, dur):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", src, "-t", f"{dur:.3f}", "-vf", f"scale={W}:{H},fps={FPS}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    n = len(raw) // (W * H * 3); return np.frombuffer(raw[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)


def run(S):
    D = S["data"]; work = workdir(S); src = os.path.join(REPO, D["source"])
    if not os.path.exists(src): raise SystemExit(f"rehook source not found: {D['source']}")
    L = dur_of(src); cut = float(D.get("cut", 3.0)); hd = float(D.get("hook_dur", 2.2)); xf = 0.25
    peek = float(D.get("peek", L * 0.6)); PF = read_frames(src, peek, hd * 0.6 + 0.2)
    hook_img = HK.make(S["_hook"], 130); HY = D.get("hook_y", 760); CT = D.get("count")

    def frame(t):
        k = min(len(PF) - 1, int(t * 0.6 * FPS)); z = 1.0 + 0.08 * (t / hd)
        bg = cv2.GaussianBlur(PF[k], (0, 0), 16).astype(np.float32)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z); bg = cv2.warpAffine(bg, M, (W, H), borderMode=cv2.BORDER_REFLECT) * 0.36
        cv = Canvas(bg)
        if CT:
            u = eout((t - 0.3) / 1.1); v = CT.get("from", 0) + (CT["to"] - CT.get("from", 0)) * u if t > 0.3 else CT.get("from", 0)
            cv.text(WG.fmtn(v, CT.get("dec", 0), CT.get("prefix", ""), CT.get("suffix", "")), 540, 1180, 300, YEL, 1.0, maxw=960)
            if CT.get("label"): cv.text(CT["label"], 540, 1370, 56, WHITE, seg(t, 0.3, 0.8), maxw=960)
        out = HK.paste(cv.out().astype(np.float32), hook_img, 540, HY, 0.94 + 0.06 * eout(t / 0.2), 1.0)
        return np.clip(out, 0, 255).astype(np.uint8)

    if os.environ.get("TPL_SNAPS"): return snaps(frame, work)
    # 1. hook card video + its audio (bed + whoosh + ding)
    card = f"{work}/card.mp4"; encode(frame, hd, None, card, loop=False)
    M = S.get("music") or {}; M = {"bed": M} if isinstance(M, str) else M
    n = int((hd + 0.3) * SR); b = bed(M.get("bed", "doc_piano"), hd + 0.3)[:, :n]; b *= 10 ** ((-18 - lufs_of(b)) / 20)
    import common as CM_; K = CM_._sfx_lib(); sfx = np.zeros(n)
    for t0, kind, db in ((0.0, "whoosh", -16), (1.4, "ding", -18)):
        y = K[kind](); i = int(t0 * SR); sfx[i:i + len(y)] += y[:n - i] * 10 ** (db / 20)
    fade = np.minimum(1, (n - np.arange(n)) / (0.35 * SR)); a = (b + sfx[None]) * fade[None]
    import soundfile as sf
    sf.write(f"{work}/card.wav", a.T.astype(np.float32), SR)
    # 2. body: source from cut, last 0.5 s dissolving into the new frame 0
    f0 = f"{work}/f0.png"; cv2.imwrite(f0, frame(0.0)); BL = L - cut
    fc = (f"[0:v]scale={W}:{H},fps={FPS},setsar=1,format=yuv420p[b];[1:v]format=rgba,fade=in:st={BL - 1.0:.3f}:d=0.9:alpha=1[o];"
          f"[b][o]overlay=shortest=1:format=auto,format=yuv420p[v]")
    body = f"{work}/body.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{cut:.3f}", "-i", src, "-loop", "1", "-framerate", str(FPS), "-t", f"{BL:.3f}", "-i", f0,
                    "-filter_complex", fc, "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", "-t", f"{BL:.3f}", body], check=True)
    # 3. join (0.25 s cross-fade) and set loudness
    joined = f"{work}/joined.mp4"
    fc = (f"[0:v]fps={FPS},format=yuv420p,setsar=1[a];[2:v]fps={FPS},format=yuv420p,setsar=1[c];[a][c]xfade=transition=fade:duration={xf}:offset={hd - xf:.3f}[v];"
          f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo[x];[2:a]aformat=sample_rates=48000:channel_layouts=stereo[y];[x][y]acrossfade=d={xf}[m]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", card, "-i", f"{work}/card.wav", "-i", body, "-filter_complex", fc, "-map", "[v]", "-map", "[m]",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-c:a", "pcm_s16le", f"{work}/joined.mkv"], check=True)
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{work}/joined.mkv", "-vn", "-ar", str(SR), "-ac", "2", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, np.float32).reshape(-1, 2).T.astype(np.float64)
    aud = loudnorm(x, f"{work}/mix.m4a", M.get("lufs", -16))
    final = os.path.join(REPO, "out", f"{S['id']}.mp4"); os.makedirs(os.path.dirname(final), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{work}/joined.mkv", "-i", f"{work}/mix.m4a", "-map", "0:v", "-map", "1:a", "-c", "copy",
                    "-shortest", "-movflags", "+faststart", final], check=True)
    return dict(voice="none (source reel audio kept, hook card on music)", source=D["source"], cut=cut, hook_dur=hd, mix=aud,
                body_kept_s=round(BL, 2))
