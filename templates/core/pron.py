"""templates/core/pron.py: the pronunciation pass. Every TTS script goes through it before Cartesia (Noam, 20 Nov 2026).

English (lexicon templates/lexicon_en.json):
  numbers, years, decimals, percent, $, ordinals, 1990s  -> words ("1948" -> "nineteen forty eight")
  heteronyms by context (live, lives, read, lead, close, tear, tears, wind, record, records, present, minute, content,
  object, desert, wound, bass, row, use)                 -> Cartesia IPA phonemes, e.g. live (dwell) = <<ˈ|l|ɪ|v>>
  names and terms (HAMAS, Hezbollah, Druze, Knesset ...) -> phonemes; "HAMAS's" gets the phonemes + z
  a word already written as <<...>> is kept as is
Hebrew (lexicon templates/lexicon_he.json):
  numbers -> Hebrew words (feminine counting form, masculine before the nouns in lexicon "masc_nouns"; years as words)
  lexicon words -> their fixed vowelised spelling (words the voice got wrong)
  everything else -> automatic nikud from Dicta Nakdan (nakdan API), sentence by sentence
Every original word of the spec ("say") maps to one TTS chunk (one or more words), so the spec text, subtitles and
anchors stay exactly as written; regroup() folds Cartesia's word timestamps back onto the original words.
Spec switches: "pron": {"off": true} (no pass), "pron": {"lex": {"word": "ˈ|w|ɜ|r|d"}} (extra English phonemes),
"pron": {"nikud": false} (Hebrew without nikud); a line's "pron_off": true or "nikud": false applies to that line only.
ElevenLabs is not used for TTS (Cartesia only since Oct 2026); el_alias() gives the ElevenLabs form if it comes back."""
import os, re, json, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__)); TPL = os.path.dirname(HERE)
PUNCT_L = "\"'“‘([«"
PUNCT_R = "\"'”’)]».,!?;:"
HEB = re.compile(r"[֐-׿]")
NIKUD = re.compile(r"[֑-ׇ]")
_LEX = {}


def lex(lang):
    if lang not in _LEX:
        p = os.path.join(TPL, f"lexicon_{lang}.json")
        _LEX[lang] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return _LEX[lang]


def ph(ipa): return f"<<{ipa}>>"


def split_tok(tok):
    """'(live,' -> ('(', 'live', ',')"""
    a = 0
    while a < len(tok) and tok[a] in PUNCT_L: a += 1
    b = len(tok)
    while b > a and tok[b - 1] in PUNCT_R: b -= 1
    return tok[:a], tok[a:b], tok[b:]


def low(w): return re.sub(r"[^\w']", "", w.lower().replace("’", "'"))


# ------------------------------------------------------------------ English numbers
YEAR_PREV = {"in", "since", "by", "of", "from", "until", "till", "to", "before", "after", "around", "circa", "year",
             "the", "and", "between", "early", "late", "mid", "summer", "winter", "spring", "autumn", "fall",
             "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
             "november", "december"}


def _w(s): return re.sub(r"\s+", " ", s.replace("-", " ").replace(",", " ")).strip()


def en_number(core, prev):
    from num2words import num2words as N
    m = re.match(r"^(\$)?(\d[\d,]*)(\.\d+)?(%|k|m|bn|st|nd|rd|th|s|'s)?$", core, re.I)
    if not m: return None
    cur, ip, dec, suf = m.group(1), m.group(2), m.group(3), (m.group(4) or "").lower()
    if "," in ip and not re.match(r"^\d{1,3}(,\d{3})+$", ip): return None
    n = int(ip.replace(",", ""))
    if suf in ("st", "nd", "rd", "th"): return _w(N(n, to="ordinal"))
    if suf in ("s", "'s") and not dec and "," not in ip and 1100 <= n <= 2099 and n % 10 == 0:   # 1990s
        y = _w(N(n, to="year")); return re.sub(r"y$", "ies", y) if y.endswith("y") else y + "s"
    if not dec and not cur and not suf and "," not in ip and 1100 <= n <= 2099 and (prev in YEAR_PREV or prev == ""):
        return _w(N(n, to="year"))
    s = _w(N(float(ip.replace(",", "") + dec) if dec else n))
    if suf == "%": s += " percent"
    elif suf == "k": s += " thousand"
    elif suf == "m": s += " million"
    elif suf == "bn": s += " billion"
    if cur: s += " dollar" if (n == 1 and not dec) else " dollars"
    return s


# ------------------------------------------------------------------ English heteronyms (context rules)
def S(*w): return set(w)


SUBJ = S("i", "we", "you", "they", "people", "who", "families", "jews", "arabs", "citizens", "millions", "thousands",
         "israelis", "palestinians", "druze", "christians", "muslims", "residents", "others", "many", "most", "some",
         "all", "also", "still", "now", "never", "ever", "already", "and", "bedouin", "minorities", "communities", "kids",
         "children", "women", "men", "jewish", "arab", "speakers", "students", "immigrants", "refugees", "that", "where")
MODAL = S("to", "can", "could", "will", "would", "should", "must", "may", "might", "don't", "didn't", "doesn't",
          "do", "did", "let", "let's", "cannot", "can't", "won't", "wouldn't", "please", "we'll", "they'll", "i'll", "you'll")


PAST_CUES = S("yesterday", "ago", "last", "earlier", "already", "once", "had", "was", "were")


def heteronym(w, p1, p2, n1, n2, ctx=frozenset()):
    """w, previous two and next two words (lower case, no punctuation), ctx = the other words of the sentence
    -> IPA key in lexicon 'heteronyms' or None"""
    if w == "live":
        if n1 in S("show", "shows", "music", "stream", "streams", "streaming", "broadcast", "fire", "ammunition", "ammo",
                   "rounds", "coverage", "event", "events", "performance", "audience", "feed", "wire", "concert", "tv",
                   "television", "footage", "updates", "video", "videos", "session", "sessions", "chat", "q", "demo",
                   "translation", "map", "blog", "radio", "action", "bait", "target", "targets") \
                or (n1 == "on" and n2 in S("air", "tv", "television", "camera", "stage", "instagram")) \
                or p1 in S("go", "goes", "went", "going", "gone", "it's", "we're", "they're", "is", "are", "was", "were", "be", "being"):
            return "live_adj"
        return "live_verb"      # default: the verb (lɪv); reels say "live in Israel" far more than "a live show"
    if w == "lives":
        if p1 in S("their", "our", "his", "her", "my", "your", "its", "many", "thousands", "millions", "hundreds", "lost",
                   "save", "saved", "saves", "saving", "innocent", "civilian", "of", "whose", "the", "two", "three",
                   "four", "five", "ten", "nine", "daily", "everyday", "private", "personal", "human", "real", "changed",
                   "change", "risked", "risk", "jewish", "israeli", "palestinian", "arab", "family", "people's"):
            return "lives_noun"
        if p1 in S("he", "she", "it", "who", "that", "family", "which", "still", "now", "where", "also", "everyone", "nobody",
                   "somebody", "someone", "community", "minority", "population") or n1 in S("in", "with", "on", "near", "here", "there", "alongside", "among", "next", "under", "inside", "abroad"):
            return "lives_verb"
        return None
    if w == "read":
        if p1 in S("have", "has", "had", "was", "were", "been", "being", "i've", "you've", "we've", "they've", "already",
                   "is", "are", "be", "well", "widely", "never") or n1 in S("yesterday", "last") \
                or (p1 not in MODAL and ctx & PAST_CUES):
            return "read_past"
        if p1 in MODAL or p1 in S("you", "and", "they", "we", "i") and n1 in S("it", "this", "the", "a", "books", "english", "more", "aloud", "out"):
            return "read_present"
        return None
    if w == "lead":
        if n1 in S("poisoning", "pipes", "pipe", "paint", "bullets", "bullet", "pencil", "pencils") or p1 in S("cast", "made", "heavy"):
            return "lead_metal"
        if p1 in MODAL or p1 in S("they", "we", "you", "i", "who", "to", "and", "will", "the", "a", "take", "took"):
            return "lead_verb"
        return "lead_verb"
    if w == "close":
        if n1 in S("the", "a", "an", "its", "their", "his", "her", "our", "my", "your", "down", "off", "it", "them", "ranks", "deals", "deal", "gaps", "gap", "borders", "schools", "this", "that", "every"):
            return "close_verb"
        if n1 in S("to", "by", "call", "calls", "friend", "friends", "ally", "allies", "relationship", "relationships",
                   "ties", "race", "enough", "second", "together", "range", "contact", "behind", "up") or \
                p1 in S("so", "very", "too", "as", "how", "came", "come", "comes", "coming", "up", "get", "got", "this", "that", "pretty", "really", "stay", "stayed", "remain", "remained"):
            return "close_adj"
        if p1 in MODAL or p1 in S("they", "we", "i", "you"):
            return "close_verb"
        return None
    if w in ("tear", "tears"):
        rip, cry = ("tear_rip", "tear_cry") if w == "tear" else ("tears_rip", "tears_cry")
        if (p1 == "and" and p2 == "wear") or n1 in S("down", "up", "apart", "off", "it", "them", "through", "open", "into"):
            return rip
        if p1 in S("a", "one", "single", "in", "with", "into", "her", "his", "their", "my", "of", "crocodile", "shed", "without", "your", "our", "no", "to") \
                or n1 in S("gas", "drops", "fell", "rolled", "ducts", "streaming", "ran", "running", "of", "in"):
            return cry
        return None
    if w == "wind":
        if n1 in S("down", "up", "back", "the", "it", "its", "their") or p1 in MODAL:
            return "wind_turn"
        return "wind_air"
    if w in ("record", "records"):
        n, v = ("record_noun", "record_verb") if w == "record" else ("records_noun", "records_verb")
        if p1 in S("a", "the", "world", "new", "his", "her", "their", "its", "our", "track", "guinness", "all-time", "criminal",
                   "on", "off", "of", "broke", "broken", "break", "breaks", "breaking", "set", "sets", "historical", "public", "medical", "official", "court", "school", "police", "bank", "unbroken", "this", "that", "these", "those", "your", "my", "best", "voting", "human", "rights") \
                or n1 in S("high", "highs", "low", "lows", "number", "numbers", "year", "years", "levels", "level", "time", "breaking", "holder", "holders", "label", "labels", "show", "shows"):
            return n
        if p1 in MODAL or p1 in S("they", "we", "i", "you", "who", "and", "cameras", "phones") or n1 in S("it", "them", "every", "everything", "this", "your", "yourself", "yourselves", "videos", "video", "voice"):
            return v
        return None
    if w == "present":
        if p1 in S("the", "at", "birthday", "christmas", "past", "until", "up", "a", "hanukkah", "this") or n1 in S("day", "tense", "moment", "situation", "simple", "continuous", "perfect"):
            return "present_noun"
        if p1 in MODAL or p1 in S("we", "they", "i", "and", "you") or n1 in S("the", "it", "them", "their", "his", "her", "its", "evidence", "data", "findings", "yourself", "itself", "themselves", "your", "our", "my", "a", "an"):
            return "present_verb"
        if p1 in S("is", "are", "was", "were", "be", "been", "remain", "remains", "stay") or n1 in S("in", "at", "on"):
            return "present_noun"
        return None
    if w == "minute":
        if n1 in S("details", "detail", "amount", "amounts", "quantity", "quantities", "fraction", "particles", "traces", "differences", "changes"):
            return "minute_tiny"
        return None
    if w == "content":
        if n1 == "with" or p1 in S("be", "am", "is", "are", "was", "were", "feel", "felt", "remain", "stay", "seem", "quite", "perfectly", "not") and n1 in S("to", "with", ""):
            return "content_adj"
        if p1 in S("the", "our", "my", "your", "his", "her", "their", "of", "this", "video", "social", "online", "new", "good", "great", "more", "daily") or n1 in S("creator", "creators", "moderation", "strategy", "warning", "is"):
            return "content_noun"
        return None
    if w == "object":
        if n1 == "to" or p1 in S("they", "we", "i", "you", "will", "would", "didn't", "don't", "may", "might", "can", "could", "to"):
            return "object_verb"
        if p1 in S("an", "the", "this", "that", "any", "every", "flying", "sharp", "no"):
            return "object_noun"
        return None
    if w == "desert":
        if p1 in S("the", "a", "negev", "sinai", "sahara", "judean", "arabian", "open", "empty", "hot", "dry", "syrian") or n1 in S("sand", "heat", "storm", "storms", "island", "climate", "region", "town", "city", "air", "road"):
            return "desert_noun"
        if p1 in MODAL or p1 in S("they", "soldiers", "troops", "will", "would") or n1 in S("their", "his", "her", "the", "us", "them", "you"):
            return "desert_verb"
        return None
    if w == "wound":
        if n1 in S("up", "down", "around", "round", "through", "tight", "tightly"):
            return "wound_turn"
        if p1 in S("a", "the", "gunshot", "stab", "head", "open", "his", "her", "fatal", "old", "deep", "this", "that") or n1 in S("healed", "care", "heals"):
            return "wound_injury"
        return None
    if w == "bass":
        if n1 in S("fishing", "fish") or p1 in S("sea", "striped", "largemouth", "caught", "a", "the") and n1 in S("swam", "fish", "fishing"):
            return "bass_fish"
        return "bass_music"
    if w == "row":
        if p1 == "a" and p2 == "in": return "row_line"
        if p1 in S("diplomatic", "political", "blazing", "furious", "public", "bitter", "huge", "big") or n1 in S("erupted", "broke", "over", "between"):
            return "row_fight"
        if p1 in S("front", "back", "first", "second", "third", "last", "each", "every", "death", "skid", "the") or p1 in MODAL:
            return "row_line"
        return None
    if w == "use":
        if p1 in S("the", "of", "no", "any", "its", "their", "in", "for", "good", "daily", "dual", "military", "civilian",
                   "personal", "this", "that", "what's", "fair", "commercial", "everyday", "widespread", "excessive", "proper") or n1 in S("of", "case", "cases"):
            return "use_noun"
        if p1 in MODAL or p1 in S("we", "they", "i", "you", "and", "people", "who") or n1 in S("the", "a", "an", "it", "them", "this", "his", "her", "their", "our", "your", "my", "these", "those", "words", "english", "ai"):
            return "use_verb"
        return None
    return None


HETERONYMS = ("live", "lives", "read", "lead", "close", "tear", "tears", "wind", "record", "records", "present", "minute",
              "content", "object", "desert", "wound", "bass", "row", "use")


def en_pass(text, extra=None, notes=None):
    """-> [(original word, tts chunk)]"""
    L = lex("en"); H = L.get("heteronyms", {}); NM = {k.lower(): v for k, v in L.get("names", {}).items()}
    CS = L.get("case_sensitive", {}); NM.update({k.lower(): v for k, v in (extra or {}).items()})
    toks = text.split(); lows = [low(split_tok(t)[1]) for t in toks]; out = []
    sent, cur = [], 0                                   # sentence number per word, for context rules
    for t in toks:
        sent.append(cur)
        if re.search(r"[.!?][\"')\]]*$", t): cur += 1
    for i, tok in enumerate(toks):
        a, core, b = split_tok(tok)
        if not core or core.startswith("<<"):
            out.append((tok, tok)); continue
        p1 = lows[i - 1] if i > 0 else ""; p2 = lows[i - 2] if i > 1 else ""
        n1 = lows[i + 1] if i + 1 < len(toks) else ""; n2 = lows[i + 2] if i + 2 < len(toks) else ""
        w = lows[i]; rep = None
        num = en_number(core, p1)
        if num:
            rep = num
        elif core in CS:
            rep = ph(CS[core]["ipa"]) if not CS[core].get("prev") or p1 in CS[core]["prev"] else None
        if rep is None and not num:
            base, poss = w, ""
            if re.search(r"'s$", w) and w[:-2] in NM: base, poss = w[:-2], "|z"
            e = NM.get(base)
            if isinstance(e, dict):
                ok = (not e.get("prev") or p1 in e["prev"]) and (not e.get("next") or n1 in e["next"])
                if ok: rep = ph(e["ipa"] + poss)
            elif isinstance(e, str):
                rep = ph(e + poss)
        if rep is None and w in HETERONYMS:
            ctx = frozenset(lows[j] for j in range(len(toks)) if sent[j] == sent[i] and j != i)
            k = heteronym(w, p1, p2, n1, n2, ctx)
            if k and k in H:
                rep = ph(H[k])
                if notes is not None: notes.append(f"{tok} -> {k}")
        if rep is None:
            out.append((tok, tok)); continue
        if num:
            rep = rep.split(); rep[0] = a + rep[0]; rep[-1] = rep[-1] + b; out.append((tok, " ".join(rep)))
        else:
            out.append((tok, a + rep + b))
    return out


# ------------------------------------------------------------------ Hebrew
def he_number(core, nxt, masc):
    from num2words import num2words as N
    m = re.match(r"^([והבלמשכ]{0,2})[-־]?(\d[\d,]*)(\.\d+)?(%)?$", core)   # ו/ה/ב/ל/מ/ש/כ (+ maqaf) + digits
    if not m: return None
    pre, ip, dec, pc = m.group(1), m.group(2), m.group(3), m.group(4)
    n = int(ip.replace(",", "")); g = "m" if nxt in masc else "f"
    if dec: s = N(float(ip.replace(",", "") + dec), lang="he")
    elif n == 2 and nxt and not pc: s = "שני" if g == "m" else "שתי"      # construct form before a noun
    else:
        try: s = N(n, lang="he", gender=g)
        except TypeError: s = N(n, lang="he")
    if pc: s += " אחוז"
    if pre: s = pre + s
    return re.sub(r"\s+", " ", s.replace(",", " ")).strip()


NAKDAN_URLS = ["https://nakdan-5-3.loadbalancer.dicta.org.il/api", "https://nakdan-2-0.loadbalancer.dicta.org.il/api",
               "https://nakdan-for-morph-analysis.loadbalancer.dicta.org.il/api"]


def _nk_words(js):
    """Dicta answer (several shapes over the years) -> list of vowelised tokens incl. separators"""
    if isinstance(js, dict): js = js.get("data", js.get("result", []))
    out = []
    for it in js:
        if isinstance(it, dict) and "nakdan" in it: it = it["nakdan"]
        if not isinstance(it, dict): continue
        w = it.get("word", it.get("str", ""))
        if it.get("sep") or it.get("separator"): out.append(("sep", w)); continue
        opts = it.get("options") or []
        if opts:
            o = opts[0]
            if isinstance(o, dict): o = o.get("w", o.get("word", w))
            elif isinstance(o, (list, tuple)): o = o[0]
            w = o
        out.append(("w", w))
    return out


def nakdan(text, timeout=40):
    """automatic nikud for one Hebrew sentence or line (Dicta). Returns None on failure."""
    import requests
    body = {"task": "nakdan", "data": text, "genre": "modern", "addmorph": False, "keepqq": False,
            "nodageshdefmem": False, "patachma": False, "keepmetagim": False, "useTokenization": True}
    for u in NAKDAN_URLS:
        try:
            r = requests.post(u, json=body, timeout=timeout)
            if r.status_code != 200: continue
            ws = _nk_words(r.json())
            if not ws: continue
            s = "".join(w for _, w in ws).replace("|", "")
            s = re.sub(r"[֑-ֽ֯׀׃]", "", s)        # cantillation, meteg, paseq, sof pasuq out
            return unicodedata.normalize("NFC", s)
        except Exception:
            continue
    return None


def he_pass(text, use_nikud=True, notes=None):
    """-> [(original word, tts chunk)]. Order: numbers -> lexicon -> nikud for the rest of the line."""
    L = lex("he"); WL = L.get("words", {}); masc = set(L.get("masc_nouns", []))
    toks = text.split(); out = []
    plain = [split_tok(t)[1] for t in toks]
    for i, tok in enumerate(toks):
        a, core, b = split_tok(tok); nxt = NIKUD.sub("", plain[i + 1]) if i + 1 < len(toks) else ""
        num = he_number(core, nxt, masc) if core else None
        key = NIKUD.sub("", core)
        if num: out.append([tok, a + num + b, "num"])
        elif key in WL: out.append([tok, a + WL[key] + b, "lex"])
        elif not HEB.search(core) or NIKUD.search(core): out.append([tok, tok, "keep"])      # English word, digits, or already vowelised
        else: out.append([tok, tok, "auto"])
    if use_nikud and any(k == "auto" for _, _, k in out):
        line = " ".join(c for _, c, _ in out)
        v = nakdan(line)
        vt = v.split() if v else []
        if v and len(vt) == len(line.split()):
            k = 0
            for o in out:
                n = len(o[1].split())
                if o[2] == "auto": o[1] = " ".join(vt[k:k + n])
                k += n
            if notes is not None: notes.append("nikud: dicta")
        elif notes is not None:
            notes.append("nikud: FAILED (" + ("no answer" if not v else f"{len(vt)} vs {len(line.split())} words") + "), plain text sent")
    return [(o[0], o[1]) for o in out]


# ------------------------------------------------------------------ shared
def prepare(text, lang="en", spec_pron=None, line=None, notes=None):
    P = spec_pron or {}; line = line or {}
    if P.get("off") or line.get("pron_off"): return [(t, t) for t in text.split()]
    if lang == "he":
        return he_pass(text, use_nikud=P.get("nikud", True) and line.get("nikud", True), notes=notes)
    return en_pass(text, P.get("lex"), notes)


def tts_text(pairs): return " ".join(c for _, c in pairs)


def regroup(words, pairs):
    """Cartesia words for tts_text(pairs) -> one {word,start,end} per original word. If Cartesia split differently,
    spans are mapped proportionally (logged by the caller)."""
    ns = [len(c.split()) for _, c in pairs]; N = sum(ns); M = len(words); out = []; k = 0
    exact = (N == M)
    for (orig, _), n in zip(pairs, ns):
        if exact: a, b = k, k + n
        else: a, b = int(k * M / N), max(int(k * M / N) + 1, int(round((k + n) * M / N)))
        seg = words[a:min(b, M)] or [words[min(a, M - 1)]]
        out.append({"word": orig, "start": seg[0]["start"], "end": seg[-1]["end"]}); k += n
    return out, exact


def el_alias(pairs):
    """ElevenLabs form (not used now): phoneme chunks -> <phoneme alphabet="ipa" ph="..">word</phoneme> SSML tags"""
    o = []
    for orig, c in pairs:
        m = re.search(r"<<(.+?)>>", c)
        o.append(f'<phoneme alphabet="ipa" ph="{m.group(1).replace("|", "")}">{split_tok(orig)[1]}</phoneme>' if m else c)
    return " ".join(o)


if __name__ == "__main__":       # python3 templates/core/pron.py en "People who live in Israel read it in 1948."
    import sys
    lg, tx = sys.argv[1], " ".join(sys.argv[2:]); nt = []
    for o, c in prepare(tx, lg, notes=nt): print(f"{o:>16}  ->  {c}")
    print("notes:", nt)
