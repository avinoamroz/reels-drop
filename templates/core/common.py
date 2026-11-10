"""templates/core/common.py: everything the five templates share.

  spec      load + lint (no dashes, HAMAS in caps, hook max 6 words with a number, max 12 English words per card)
  voice     Cartesia TTS (sonic-3, "Noam Roz - reels", same body as Make TTS1) or a pre-made voice/<id>_*.wav,
            house chain (pitch 0.9715 formant preserved, bass +2 dB at 135 Hz, atempo), lines re-spaced by spec gaps
  timing    T.S(i), T.E(i), T.WT(i, word), T.at("L3:fifteen" | "L3" | "L3e" | 12.5)
  overlays  hook card at frame 0, screen-text titles, source chips, end card (balance line + CTA), subtitles
  audio     music bed at -16 LUFS ducked 9 dB under speech, SFX, final two-pass loudnorm (spec music.lufs, default -16)
  video     parallel frame render, circular ending (last 0.5 s dissolves into frame 0), map opener wrap,
            cover (1080x1920 jpg), 540p preview, frame grabs, report json
"""
import os, sys, re, json, math, subprocess, hashlib, time, shutil, tempfile
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); TPL = os.path.dirname(HERE); REPO = os.path.dirname(TPL)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(TPL, "opener"))
from base import *
import widgets as WG
import hook as HK

SR = 48000
VOICE_ID = "cd06f613-0c92-4347-9b94-b34cb4a0239f"      # Cartesia clone "Noam Roz - reels" (reelkit/tts.py, cloud_voice.py)
TTS_URL = "https://api.cartesia.ai/tts/sse"              # Make TTS1 4998936: https://api.cartesia.ai/tts/{endpoint}, endpoint sse
TTS_VERSION = "2025-04-16"                               # Make TTS1 header Cartesia-Version
HEEBO = os.path.join(TPL, "fonts", "Heebo-800.ttf")
DASH_RE = re.compile("[\u2012\u2013\u2014\u2015]")


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ================================================================== spec
def load_spec(path):
    S = json.load(open(path, encoding="utf-8"))
    S.setdefault("id", os.path.splitext(os.path.basename(path))[0])
    S.setdefault("lines", []); S.setdefault("sources", []); S.setdefault("data", {})
    S["_path"] = os.path.abspath(path)
    S["_hook"] = wrap_hook(S.get("hook", ""))
    return S


def wrap_hook(h):
    """'A b c d e' -> 'A b c|d e' (two balanced lines) unless the spec already breaks it with |"""
    if not h or "|" in h: return h
    w = h.split()
    if len(w) < 4: return h
    best = min(range(1, len(w)), key=lambda k: abs(len(" ".join(w[:k])) - len(" ".join(w[k:]))))
    return " ".join(w[:best]) + "|" + " ".join(w[best:])


def lint(S):
    """hard errors raise; soft problems return as warnings (they go into the report)"""
    errs, warn = [], []
    def walk(x, where):
        if isinstance(x, str):
            if DASH_RE.search(x): errs.append(f"dash in {where}: {x[:60]}")
            if re.search(r"\bhamas\b", x, re.I) and not re.search(r"\bHAMAS\b", x) and not where.startswith("lines") : errs.append(f"HAMAS not in caps in {where}")
        elif isinstance(x, dict):
            for k, v in x.items():
                if not k.startswith("_"): walk(v, f"{where}.{k}" if where else k)
        elif isinstance(x, list):
            for i, v in enumerate(x): walk(v, f"{where}[{i}]")
    walk(S, "")
    if not re.match(r"^[a-z0-9_]+$", S["id"]): errs.append("id must be lower case letters, digits and _")
    if os.path.basename(S["_path"]) != S["id"] + ".json" and "/examples/" not in S["_path"]: errs.append(f"file name must be {S['id']}.json")
    if S.get("account") not in ("noamroz__", "expressenglishcom"): errs.append("account must be noamroz__ or expressenglishcom")
    hook = S.get("hook", "")
    hw = len(re.sub(r"[*|]", " ", hook).split())
    if S.get("template") != "rehook" or hook:
        if not hook: errs.append("hook missing")
        elif hw > 6: errs.append(f"hook has {hw} words (max 6)")
        elif not re.search(r"\d", hook): warn.append("hook has no number")
    for i, L in enumerate(S["lines"]):
        scr = L.get("screen") or ""
        n = len(re.sub(r"[*|]", " ", scr).split())
        if n > 12: errs.append(f"line {i} screen text has {n} words (max 12)")
        if L.get("say") and re.search(r"\bhamas\b", L["say"], re.I) and not re.search(r"\bHAMAS\b", scr or "HAMAS"):
            pass
    for s in S["sources"]:
        if not s.get("year"): warn.append(f"source without year: {s.get('label')}")
    if not S["sources"] and S.get("template") not in ("rehook", "slang_scene"): warn.append("no sources")
    if errs: raise SystemExit("spec lint failed:\n  " + "\n  ".join(errs))
    return warn


# ================================================================== voice
def script_lines(S): return [L["say"].strip() for L in S["lines"] if L.get("say", "").strip()]


def script_hash(lines): return hashlib.sha1("\n".join(lines).encode()).hexdigest()[:12]


def tts_body(text, lang="en"):
    return {"model_id": "sonic-3", "transcript": text, "voice": {"mode": "id", "id": VOICE_ID},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 44100},
            "add_timestamps": True, "language": lang}


def cartesia(text, out_prefix, lang="en"):
    """one SSE call (same body and headers as Make TTS1) -> out_prefix.wav (44.1 kHz mono) + out_prefix_words.json"""
    import requests, base64, wave
    key = os.environ.get("CARTESIA_API_KEY", "").strip()
    if not key: raise RuntimeError("CARTESIA_API_KEY not set")
    r = requests.post(TTS_URL, json=tts_body(text, lang), timeout=300, stream=True,
                      headers={"X-API-Key": key, "Cartesia-Version": TTS_VERSION, "Content-Type": "application/json"})
    if r.status_code != 200: raise RuntimeError(f"Cartesia HTTP {r.status_code}: {r.text[:300]}")
    pcm, words = bytearray(), []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"): continue
        try: ev = json.loads(line[5:].strip())
        except Exception: continue
        if ev.get("type") == "error": raise RuntimeError(f"Cartesia error: {str(ev)[:300]}")
        if ev.get("type") == "chunk" and ev.get("data"): pcm += base64.b64decode(ev["data"])
        wt = ev.get("word_timestamps")
        if wt:
            for w, s, e in zip(wt["words"], wt["start"], wt["end"]): words.append({"word": w, "start": s, "end": e})
    if not pcm: raise RuntimeError("Cartesia returned no audio")
    with wave.open(out_prefix + ".wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(bytes(pcm))
    json.dump(words, open(out_prefix + "_words.json", "w"), indent=0)
    return len(pcm) / 88200, len(words)


def get_voice(S):
    """returns (raw_wav, words_json, how). Order: voice/<id>_vo.wav whose .hash matches the script (no new TTS cost),
    then Cartesia if the secret is set, then any pre-made voice/<id>_*.wav (+ _words.json), else a clear failure."""
    vd = os.path.join(REPO, "voice"); os.makedirs(vd, exist_ok=True)
    lines = script_lines(S); h = script_hash(lines); pre = os.path.join(vd, f"{S['id']}_vo")
    if os.path.exists(pre + ".wav") and os.path.exists(pre + "_words.json") and os.path.exists(pre + ".hash") \
            and open(pre + ".hash").read().strip() == h:
        return pre + ".wav", pre + "_words.json", "cached voice (script unchanged)"
    if os.environ.get("CARTESIA_API_KEY", "").strip() and not S.get("voice", {}).get("file"):
        text = re.sub(r"\s+", " ", " ".join(DASH_RE.sub(",", l) for l in lines)).strip()
        sec, n = cartesia(text, pre, S.get("voice", {}).get("lang", "en"))
        open(pre + ".hash", "w").write(h)
        return pre + ".wav", pre + "_words.json", f"Cartesia sonic-3 ({sec:.1f} s, {n} words)"
    named = S.get("voice", {}).get("file")
    cands = [os.path.join(REPO, named)] if named else sorted(
        p for p in (os.path.join(vd, f) for f in os.listdir(vd)) if os.path.basename(p).startswith(S["id"] + "_") and p.endswith(".wav"))
    for p in cands:
        wj = p[:-4] + "_words.json"
        if os.path.exists(p) and os.path.exists(wj): return p, wj, f"pre-made voice file {os.path.relpath(p, REPO)}"
    raise SystemExit(f"NO VOICE for {S['id']}: CARTESIA_API_KEY is not set and there is no voice/{S['id']}_*.wav "
                     f"with a matching _words.json. Add the repo secret CARTESIA_API_KEY, or commit voice/{S['id']}_vo.wav "
                     f"+ voice/{S['id']}_vo_words.json (Cartesia SSE output, e.g. from Make TTS1 + tools/sse_parse.py).")


def house_chain(raw, tempo):
    """raw wav -> float mono 48 kHz after pitch/bass/atempo (the reels' deeper Noam voice), plus the time factor k"""
    af = f"rubberband=pitch=0.9715:formant=preserved,bass=g=2:f=135,atempo={tempo}"
    def go(f):
        return np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", raw, "-af", f, "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                                            check=True, capture_output=True).stdout, np.float32).copy()
    try: y = go(af)
    except subprocess.CalledProcessError:
        log("rubberband missing, using asetrate pitch fallback")
        y = go(f"asetrate=44100*0.9715,aresample=44100,atempo={tempo / 0.9715:.5f},bass=g=2:f=135")
    dry = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", raw], capture_output=True, text=True).stdout)
    return y, (len(y) / SR) / dry


def build_voice(S, work):
    """-> work/vo.wav (48 kHz mono) + work/timing.json {dur, lines:[{i,text,t0,t1,words:[{w,t0,t1}]}]}"""
    raw, wj, how = get_voice(S); log("voice:", how)
    V = S.get("voice", {}); tempo = float(V.get("tempo", 1.1)); lead = float(V.get("lead", 0.3))
    y, k = house_chain(raw, tempo)
    W_ = json.load(open(wj)); lines = script_lines(S)
    tot = sum(len(l.split()) for l in lines)
    if tot != len(W_): raise SystemExit(f"voice/script mismatch: script has {tot} words, voice has {len(W_)} (re-voice or fix the spec lines)")
    peak = np.abs(y).max() + 1e-9; y = y / peak * 0.89
    segs, out, idx, t = [], [], 0, lead
    gaps = [float(L.get("gap", V.get("gap", 0.3))) for L in S["lines"] if L.get("say", "").strip()]
    for i, text in enumerate(lines):
        n = len(text.split()); ws = W_[idx:idx + n]; idx += n
        a, b = ws[0]["start"] * k - 0.05, ws[-1]["end"] * k + 0.10; a = max(0, a)
        seg_ = y[int(a * SR):int(b * SR)].copy(); f = int(0.01 * SR)
        if len(seg_) > 2 * f: seg_[:f] *= np.linspace(0, 1, f); seg_[-f:] *= np.linspace(1, 0, f)
        segs.append((t, seg_))
        out.append(dict(i=i, text=text, t0=round(t + 0.05, 3), t1=round(t + (b - a) - 0.10, 3),
                        words=[dict(w=sw, t0=round(t + w["start"] * k - a, 3), t1=round(t + max(w["end"], w["start"] + 0.04) * k - a, 3))
                               for w, sw in zip(ws, text.split())]))
        t += len(seg_) / SR + gaps[i]
    vo = np.zeros(int((t + 1) * SR), np.float32)
    for t0, s_ in segs: vo[int(t0 * SR):int(t0 * SR) + len(s_)] += s_
    import soundfile as sf
    sf.write(f"{work}/vo.wav", vo[:int(t * SR)], SR)
    J = dict(dur=round(t, 3), voice=how, lines=out); json.dump(J, open(f"{work}/timing.json", "w"), indent=0)
    return J


# ================================================================== timing
class Timing:
    def __init__(s, tj, tail=0.75):
        s.L = tj["lines"]; s.tail = tail
    def S(s, i): return s.L[i]["t0"]
    def E(s, i): return s.L[i]["t1"]
    def WT(s, i, word, end=False, n=1):
        k = 0; wl = re.sub(r"[^\w]", "", word).lower()
        for w in s.L[i]["words"]:
            if re.sub(r"[^\w]", "", w["w"]).lower().startswith(wl):
                k += 1
                if k == n: return w["t1" if end else "t0"]
        raise SystemExit(f"anchor word '{word}' not found in line {i}: {s.L[i]['text']}")
    @property
    def END(s): return round(s.L[-1]["t1"] + s.tail, 2)
    def at(s, a, default=0.0):
        """anchor -> seconds. 12.5 | "L3" (start of line 3) | "L3e" (end) | "L3:word" | "L3:word#2" | "L3:word+0.2" | "END" | "END-2" """
        if a is None: return default
        if isinstance(a, (int, float)): return float(a)
        a = str(a).strip().replace(" ", "")
        m = re.match(r"^(?:L(\d+)(e?)(?::([^#+\-]+)(?:#(\d+))?)?|(END))([+-]\d+(?:\.\d+)?)?$", a)
        if not m: raise SystemExit(f"bad anchor {a} (use 12.5, L3, L3e, L3:word, L3:word#2, L3+0.4, END-2)")
        off = float(m.group(6) or 0)
        if m.group(5): return s.END + off
        i = int(m.group(1))
        if i >= len(s.L): raise SystemExit(f"anchor {a}: there are only {len(s.L)} lines (L0 to L{len(s.L) - 1})")
        if m.group(3): return s.WT(i, m.group(3), n=int(m.group(4) or 1)) + off
        return (s.E(i) if m.group(2) else s.S(i)) + off


# ================================================================== overlays
def hebrew_img(txt, sz, col=(255, 255, 255)):
    from PIL import Image, ImageDraw, ImageFont
    from bidi.algorithm import get_display
    f = ImageFont.truetype(HEEBO, int(sz), layout_engine=ImageFont.Layout.BASIC); vis = get_display(txt)   # basic layout + bidi = right order everywhere
    w = int(f.getlength(vis)) + 20; h = int(sz * 1.5)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); ImageDraw.Draw(im).text((10, h / 2), vis, font=f, fill=tuple(col) + (255,), anchor="lm")
    return np.asarray(im)


def hebrew(cv, txt, cx, cy, sz=56, col=(255, 255, 255), a=1.0):
    if a > 0.01 and txt: cv.image(hebrew_img(txt, sz, col), cx, cy, 1.0, a)


def screen_cards(S, T):
    """[(text, t0, t1, y, size)] from the lines' "screen" fields: each shows from its anchor until the next screen card"""
    cards = []
    for i, L in enumerate(S["lines"]):
        if L.get("screen"):
            cards.append([L["screen"], T.at(L.get("at"), T.S(i)), L.get("until"), L.get("screen_y"), L.get("screen_size")])
    out = []
    for j, c in enumerate(cards):
        t1 = T.at(c[2]) if c[2] else (cards[j + 1][1] if j + 1 < len(cards) else T.END - 0.6)
        out.append((c[0], c[1], t1, c[3], c[4]))
    return out


def draw_screen(cv, cards, t, y=270, sz=80):
    for txt, t0, t1, yy, ss in cards:
        a = win(t, t0 - 0.1, t1, 0.3, 0.3)
        if a > 0.01: WG.title(cv, txt.replace("|", " "), yy or y, a, ss or sz)


def draw_sources(cv, S, T, t, y=1488):
    for s in S["sources"]:
        t0 = T.at(s.get("at"), T.S(min(1, len(T.L) - 1))); t1 = T.at(s.get("until"), T.END - 0.6)
        a = win(t, t0, t1, 0.3, 0.3)
        if a > 0.01:
            WG.src(cv, f"{s['label']}, {s['year']}" if s.get("year") and str(s["year"]) not in s["label"] else s["label"], a, s.get("y", y))


def draw_end(cv, S, T, t):
    E = S.get("end") or {}
    if not E: return
    t0 = T.at(E.get("at"), T.S(len(T.L) - 2) if len(T.L) > 1 else T.END - 3); t1 = T.END - 0.5
    a = win(t, t0 - 0.1, t1, 0.35, 0.3)
    if a <= 0.01: return
    y = E.get("y", 1010)
    WG.panel(cv, 60, y - 230, 960, 460, a)
    if E.get("balance"): WG.title(cv, E["balance"], y - 90, a, 66)
    if E.get("cta"):
        ca = a * seg(t, t0 + 0.6, t0 + 1.0)
        tw = cv.tw(E["cta"].replace("*", ""), 60) + 90; tw = min(tw, 940)
        cv.rrect(540 - tw / 2, y + 60, tw, 120, 60, SKY, ca)
        WG.title(cv, E["cta"], y + 120, ca, 60)


def sub_chunks(T, repl=None):
    CH = []
    for l in T.L:
        cur = []
        for w in l["words"]:
            cur.append(w)
            if len(cur) >= 6 or re.search(r"[.,?!:;]$", w["w"]) and len(cur) >= 2:
                CH.append([" ".join(x["w"] for x in cur), cur[0]["t0"], cur[-1]["t1"]]); cur = []
        if cur: CH.append([" ".join(x["w"] for x in cur), cur[0]["t0"], cur[-1]["t1"]])
    for c in CH:
        c[0] = re.sub(r"[\"“”]", "", c[0]).rstrip(",;:")
        for a, b in (repl or {}).items(): c[0] = c[0].replace(a, b)
    for j in range(len(CH) - 1):
        CH[j][2] = min(CH[j + 1][1] - 0.02, CH[j][2] + 0.5) if CH[j + 1][1] - CH[j][2] < 0.5 else CH[j][2] + 0.25
    if CH: CH[-1][2] += 0.4
    return CH


def draw_subs(cv, CH, t, y=1612, sz=56):
    for txt, a, b in CH:
        if a - 0.05 <= t <= b:
            al = min(1, (t - (a - 0.05)) / 0.1) * min(1, (b - t) / 0.1 + 1e-4)
            w = min(1000, cv.tw(txt, sz) + 56)
            cv.rrect(540 - w / 2, y - 48, w, 96, 28, "#0B1730", 0.82 * al, stroke=YEL, lw=2, sa=0.5 * al)
            cv.text(txt, 540, y, sz, WHITE, al, maxw=960, shadow=False)
            return


def brand_bg():
    return BG.copy()


# ================================================================== audio
def bed(name, dur):
    import music as MU
    if name.startswith("file:"):
        y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-stream_loop", "-1", "-i", os.path.join(REPO, name[5:]), "-t", f"{dur:.3f}",
                                          "-ar", str(SR), "-ac", "2", "-f", "f32le", "-"], check=True, capture_output=True).stdout, np.float32)
        return y.reshape(-1, 2).T.astype(np.float64)
    if name == "doc_piano": return MU.doc_music(dur)
    if name == "journey": return MU.music(dur)
    if name in ("bounce", "skip", "warm"): return MU.kit_music(dur, name)
    raise SystemExit(f"unknown music bed {name} (doc_piano, journey, bounce, skip, warm, file:<repo path>)")


def lufs_of(x):
    o = subprocess.run(["ffmpeg", "-v", "info", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0", "-af", "ebur128", "-f", "null", "-"],
                       input=x.T.astype(np.float32).tobytes(), capture_output=True).stderr.decode()
    return float(o[o.rfind("I:"):].split()[1])


def _sfx_lib():
    rng = np.random.default_rng(3); tt = lambda d: np.arange(int(d * SR)) / SR
    def tick(): x = tt(0.03); return np.sin(2 * np.pi * 1800 * x) * np.exp(-x * 220)
    def pop(): x = tt(0.14); return np.sin(2 * np.pi * (500 + 900 * x * 6) * x) * np.exp(-x * 30)
    def whoosh(d=0.6, f0=300, f1=2400):
        x = tt(d); ph = 2 * np.pi * np.cumsum(f0 + (f1 - f0) * (x / d) ** 2) / SR
        return (rng.standard_normal(len(x)) * 0.35 + np.sin(ph) * 0.2) * np.sin(np.pi * x / d) ** 1.5
    def thud(): x = tt(0.35); y = np.sin(2 * np.pi * 70 * x) * np.exp(-x * 16); return y / np.abs(y).max()
    def ding(): x = tt(0.6); return (np.sin(2 * np.pi * 1320 * x) + 0.5 * np.sin(2 * np.pi * 1980 * x)) * np.exp(-x * 7)
    def buzz(): x = tt(0.25); return np.sign(np.sin(2 * np.pi * 110 * x)) * 0.5 * np.exp(-x * 9)
    def check(): x = tt(0.18); return np.sin(2 * np.pi * (880 + 600 * x / 0.18) * x) * np.exp(-x * 18)
    return dict(tick=tick, pop=pop, whoosh=whoosh, thud=thud, ding=ding, buzz=buzz, check=check)


def build_mix(S, T, work, events=(), vo_path=None, dur=None, speech=None):
    """vo + bed (-16 LUFS, ducked 9 dB under speech) + SFX -> work/mix.m4a at spec music.lufs (default -16 LUFS, TP -1.5)"""
    import soundfile as sf
    M = S.get("music") or {}; M = {"bed": M} if isinstance(M, str) else M
    T_ = dur or T.END; n = int(T_ * SR)
    if vo_path or T is not None:
        vo, sr = sf.read(vo_path or f"{work}/vo.wav"); vo = np.pad(vo, (0, max(0, n - len(vo))))[:n]
    else: vo = np.zeros(n)
    b = bed(M.get("bed", "doc_piano"), T_ + 0.6)[:, :n]
    b = np.pad(b, ((0, 0), (0, max(0, n - b.shape[1]))))
    b *= 10 ** ((M.get("bed_lufs", -16) - lufs_of(b)) / 20)
    m = np.zeros(n)
    for a0, a1 in (speech if speech is not None else [(l["t0"], l["t1"]) for l in T.L]):
        m[max(0, int((a0 - 0.08) * SR)):int((a1 + 0.12) * SR)] = 1
    k = int(0.12 * SR); m = np.convolve(m, np.ones(k) / k, "same"); b *= (10 ** (-M.get("duck_db", 9) * m / 20))[None]
    sfx = np.zeros(n); K = _sfx_lib()
    for ev in events:
        t0, kind = ev[0], ev[1]; db = ev[2] if len(ev) > 2 else -24
        if kind == "roll":
            for tt_ in np.arange(t0, ev[2], 0.045):
                y = K["tick"](); i = int(tt_ * SR); j = min(n, i + len(y))
                if 0 <= i < n: sfx[i:j] += y[:j - i] * 10 ** (-36 / 20)
            continue
        y = K[kind](); i = int(t0 * SR); j = min(n, i + len(y))
        if 0 <= i < n: sfx[i:j] += y[:j - i] * 10 ** (db / 20)
    x = np.stack([vo, vo]) + b + np.stack([sfx, sfx])
    return loudnorm(x, f"{work}/mix.m4a", M.get("lufs", -16))


def loudnorm(x, out, target=-16, tp=-1.5):
    raw = x.T.astype(np.float32).tobytes()
    meas = subprocess.run(["ffmpeg", "-v", "info", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0", "-af",
                           f"loudnorm=I={target}:TP={tp - 0.7}:LRA=11:print_format=json", "-f", "null", "-"], input=raw, capture_output=True).stderr.decode()
    J = json.loads(meas[meas.rfind("{"):meas.rfind("}") + 1])
    af = (f"loudnorm=I={target}:TP={tp - 0.7}:LRA=11:measured_I={J['input_i']}:measured_TP={J['input_tp']}:measured_LRA={J['input_lra']}:"
          f"measured_thresh={J['input_thresh']}:offset={J['target_offset']}:linear=true,aresample=48000")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0", "-af", af,
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out], input=raw, check=True)
    return measure(out)


def measure(path):
    o = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    I = [l for l in o.splitlines() if l.strip().startswith("I:")]; P = [l for l in o.splitlines() if l.strip().startswith("Peak:")]
    return dict(lufs=float(I[-1].split()[1]) if I else None, true_peak=float(P[-1].split()[1]) if P else None)


# ================================================================== video
_FR = [None]


def _one(i): return _FR[0](i / FPS).tobytes()


def encode(frame, dur, audio, out, loop=True, crf=18):
    """frame(t) -> BGR uint8 (H, W, 3). Rendered in parallel (fork), muxed with audio. loop: last 0.5 s dissolves into frame 0."""
    import multiprocessing as mp
    f0 = frame(0.0).astype(np.float32)
    def fr(t):
        img = frame(t)
        if loop and t > dur - 0.5:
            u = ease((t - (dur - 0.5)) / 0.5); img = (img.astype(np.float32) * (1 - u) + f0 * u).astype(np.uint8)
        return img
    _FR[0] = fr; nfr = int(round(dur * FPS))
    sil = out + ".silent.mp4"
    pr = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", sil], stdin=subprocess.PIPE)
    ctx = mp.get_context("fork")
    with ctx.Pool(os.cpu_count()) as P:
        for i, b in enumerate(P.imap(_one, range(nfr), chunksize=6)):
            pr.stdin.write(b)
            if i % 300 == 0: log(f"frame {i}/{nfr}")
    pr.stdin.close(); assert pr.wait() == 0
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", sil]
    cmd += (["-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "copy"] if audio else [])
    subprocess.run(cmd + ["-c:v", "copy", "-shortest", "-movflags", "+faststart", out], check=True); os.remove(sil)


def dur_of(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)


def wrap_opener(body, S, out, cam, hook_y=930, work="."):
    """map_open.py motion opener (globe or paper map, hook on screen from frame 0) in front of the body; the body's audio
    starts on the zoom (z0), the body's last 0.5 s dissolves into the opener's frame 0 (circular). Logic of fix_1109/rmk/op_wrap_p.py."""
    O = S.get("opener") or {}; style = O.get("style", "globe_spin_whip"); T = float(O.get("dur", 2.4))
    z0 = (0.5 if style == "globe_unpeel" else 0.625) * T; d = T - z0
    MO = os.path.join(TPL, "opener", "map_open.py"); tmp = tempfile.mkdtemp(prefix="opw_", dir=work)
    common = ["--style", style, "--cam", ",".join(str(x) for x in cam) + ",960", "--hook", S["_hook"], "--hook-y", str(hook_y),
              "--hook-s", "1.0", "--dur", str(T)] + (["--plain"] if O.get("plain", True) else [])
    subprocess.run([sys.executable, MO, *common, "--snaps", "0", "--out", f"{tmp}/op.mp4"], check=True); op0 = f"{tmp}/op_0.00.png"
    L = dur_of(body) - d; bod = f"{tmp}/body.mp4"
    fc = (f"[0:v]setpts=PTS-STARTPTS,format=yuv420p[b];[1:v]scale=1080:1920,format=rgba,fade=in:st={L - 0.5:.3f}:d=0.5:alpha=1[o];"
          f"[b][o]overlay=shortest=1:format=auto,format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{d:.4f}", "-i", body, "-loop", "1", "-framerate", "30", "-t", f"{L:.3f}", "-i", op0,
                    "-filter_complex", fc, "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-t", f"{L:.3f}", bod], check=True)
    subprocess.run([sys.executable, MO, *common, "--reel", bod, "--out", f"{tmp}/joined.mp4"], check=True)
    tot = dur_of(f"{tmp}/joined.mp4"); ms = int(round(z0 * 1000))
    fc = (f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo[s];[1:a]aformat=sample_rates=48000:channel_layouts=stereo,adelay={ms}|{ms}[r];"
          f"[s][r]amix=inputs=2:normalize=0:duration=first,atrim=0:{tot:.3f}[a]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{tmp}/joined.mp4", "-i", body, "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(style=style, opener_s=T, narration_starts=round(z0, 3))


def grab(path, t, scale=None):
    vf = ["-vf", f"scale={scale}:-2"] if scale else []
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t):.3f}", "-i", path, "-frames:v", "1", *vf, "-f", "image2pipe", "-vcodec", "png", "-"],
                       capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(r, np.uint8), cv2.IMREAD_COLOR)


def cover(S, final, out_jpg):
    """1080x1920: one strong visual (a reel frame, blurred and darkened, or the brand gradient), the hook in huge text inside
    the centre 1080x1440 grid-safe area (y 240 to 1680), optional big number. No faces: pick a frame without people (cover.t)."""
    C = S.get("cover") or {}; d = dur_of(final)
    if C.get("bg", "frame") == "frame":
        bg = grab(final, float(C.get("t", d * 0.6))).astype(np.float32)
        bg = cv2.GaussianBlur(bg, (0, 0), C.get("blur", 12)) * 0.45
    else: bg = BG.copy()
    y = np.arange(H, dtype=np.float32)[:, None, None] / H
    bg = bg * (1 - 0.35 * np.clip(1 - np.abs(y - 0.5) * 2.2, 0, 1)) + np.array([97, 50, 22], np.float32) * 0.35 * np.clip(1 - np.abs(y - 0.5) * 2.2, 0, 1)
    cv = Canvas(bg)
    cv.rrect(0, 240, 1080, 1440, 0, None)
    cv_out = cv.out().astype(np.float32)
    text = wrap_hook(C.get("text") or S.get("hook", ""))
    img = HK.make(text, size=int(C.get("size", 150)), maxw=960)
    yc = 960 if not C.get("big") else 1180
    cv_out = HK.paste(cv_out, img, 540, yc, 1.0, 1.0)
    cv = Canvas(cv_out)
    if C.get("big"): cv.text(C["big"], 540, 640, 300, YEL, 1.0, maxw=960)
    if C.get("label"): cv.text(C["label"], 540, 1560, 52, "#C9D6EA", 1.0, bold=False, maxw=960)
    cv2.imwrite(out_jpg, cv.out(), [cv2.IMWRITE_JPEG_QUALITY, 90])


def finish(S, final, extra):
    """outputs in the repo: out/<id>.mp4 (already), covers/<id>_cover.jpg, previews/<id>_540.mp4, out/frames/<id>_*.jpg, out/<id>.report.json"""
    sid = S["id"]
    for d_ in ("covers", "previews", "out/frames"): os.makedirs(os.path.join(REPO, d_), exist_ok=True)
    cover(S, final, os.path.join(REPO, "covers", f"{sid}_cover.jpg"))
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", final, "-vf", "scale=540:-2", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", os.path.join(REPO, "previews", f"{sid}_540.mp4")], check=True)
    d = dur_of(final); frames = {}
    for lab, t in (("0", 0.0), ("1", 1.0), ("3", 3.0), ("last", d - 0.05)):
        p = f"out/frames/{sid}_{lab}.jpg"; cv2.imwrite(os.path.join(REPO, p), grab(final, t, 540), [cv2.IMWRITE_JPEG_QUALITY, 85]); frames[lab] = p
    rep = dict(id=sid, account=S.get("account"), template=S.get("template"), hook=S.get("hook"), duration=round(d, 2),
               audio=measure(final), outputs=dict(mp4=f"out/{sid}.mp4", cover=f"covers/{sid}_cover.jpg", preview=f"previews/{sid}_540.mp4"),
               frames=frames, caption_en=S.get("caption_en"), caption_he=S.get("caption_he"), hashtags=S.get("hashtags"),
               sources=S.get("sources"), min_font_px=MINSZ[0] if MINSZ[0] < 999 else None, transcript=None, status="rendered, not posted")
    rep.update(extra)
    json.dump(rep, open(os.path.join(REPO, "out", f"{sid}.report.json"), "w"), indent=1, ensure_ascii=False)
    return rep


def workdir(S):
    w = os.path.join(tempfile.gettempdir(), "tpl_" + S["id"]); os.makedirs(w, exist_ok=True); return w


def snaps(frame, work):
    """local check: TPL_SNAPS=0,3,10 renders those frames to <work>/snaps.png instead of a video"""
    ts = [float(x) for x in os.environ["TPL_SNAPS"].split(",")]
    tiles = [cv2.resize(frame(t), (360, 640), interpolation=cv2.INTER_AREA) for t in ts]
    p = f"{work}/snaps.png"; cv2.imwrite(p, np.hstack(tiles)); print("snaps", p); return {"snaps": p}


def paste_hook(out, hook_img, t, t_end, y=930, scale=1.0):
    """hook card from frame 0 (pops 0.94 -> 1 in 0.2 s), fades out 0.15 to 0.55 s after t_end"""
    a = 1 - seg(t, t_end + 0.15, t_end + 0.55)
    if a <= 0.01: return out
    pop = 0.94 + 0.06 * eout(t / 0.2)
    return HK.paste(out, hook_img, 540, y, scale * pop, a)


def bars_moving_bg(t, speed=18.0):
    """brand gradient with a slowly drifting sky grid (motion from frame 0)"""
    b = BG.copy(); off = int(t * speed) % 120
    g = np.zeros((H, W), np.float32); g[:, (np.arange(W) + off) % 120 == 0] = 1; g[(np.arange(H) + off) % 120 == 0, :] = 1
    return b + g[..., None] * np.array([221, 164, 69], np.float32)[None, None] * 0.07
