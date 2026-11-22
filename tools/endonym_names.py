"""Pre-render the native country names (shaped with raqm) as yellow PNGs with a soft baked shadow.
Out: templates/data/endonyms/<key>.png (cropped, transparent)."""
import os, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
F = "/home/claude/endo_work/fonts/"
INTER = "/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf"
CJK = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc", 0)   # index 0 = JP
KR = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc", 1)    # 1 = KR
VAR = {"ar": "NotoSansArabic.ttf", "hi": "NotoSansDevanagari.ttf", "ka": "NotoSansGeorgian.ttf", "hy": "NotoSansArmenian.ttf",
       "th": "NotoSansThai.ttf", "he": "NotoSansHebrew.ttf", "bo": "NotoSerifTibetan.ttf"}
NAMES = {  # key: (text, script)
    "deutschland": ("Deutschland", "lat"), "oesterreich": ("Österreich", "lat"), "helvetia": ("Helvetia", "lat"),
    "magyarorszag": ("Magyarország", "lat"), "suomi": ("Suomi", "lat"), "hellas": ("Ελλάς", "lat"),
    "shqiperia": ("Shqipëria", "lat"), "hrvatska": ("Hrvatska", "lat"), "crnagora": ("Црна Гора", "lat"), "eire": ("Éire", "lat"),
    "masr": ("مصر", "ar"), "almaghrib": ("المغرب", "ar"), "aljazair": ("الجزائر", "ar"), "eswatini": ("eSwatini", "lat"),
    "turkiye": ("Türkiye", "lat"), "iran": ("ایران", "ar"), "alurdun": ("الأردن", "ar"), "lubnan": ("لبنان", "ar"),
    "yisrael": ("ישראל", "he"),
    "nihon": ("日本", "cjk"), "zhongguo": ("中国", "cjk"), "hanguk": ("한국", "kr"), "mongoluls": ("Монгол Улс", "lat"),
    "bharat": ("भारत", "hi"), "prathetthai": ("ประเทศไทย", "th"), "sakartvelo": ("საქართველო", "ka"), "hayastan": ("Հայաստան", "hy"), "drukyul": ("འབྲུག་ཡུལ།", "bo"),
}
YEL = (255, 210, 63, 255)


def font(script, sz):
    if script == "lat": return ImageFont.truetype(INTER, sz, layout_engine=ImageFont.Layout.RAQM)
    if script == "cjk": return ImageFont.truetype(CJK[0], sz, index=CJK[1], layout_engine=ImageFont.Layout.RAQM)
    if script == "kr": return ImageFont.truetype(KR[0], sz, index=KR[1], layout_engine=ImageFont.Layout.RAQM)
    f = ImageFont.truetype(F + VAR[script], sz, layout_engine=ImageFont.Layout.RAQM)
    try: f.set_variation_by_axes([800, 100] if len(f.get_variation_axes()) == 2 else [800])
    except Exception as e: print("var", script, e)
    return f


def render(key, txt, script, maxw=1000, sz=230):
    while True:
        f = font(script, sz)
        d = "rtl" if script in ("ar", "he") else None
        bb = f.getbbox(txt, direction=d)
        if bb[2] - bb[0] <= maxw or sz < 80: break
        sz -= 6
    pad = 40; w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0)); ImageDraw.Draw(sh).text((pad - bb[0] + 4, pad - bb[1] + 8), txt, font=f, fill=(0, 0, 0, 150), direction=d)
    sh = sh.filter(ImageFilter.GaussianBlur(9))
    ImageDraw.Draw(sh).text((pad - bb[0], pad - bb[1]), txt, font=f, fill=YEL, direction=d)
    sh.save(os.path.join(OUT, key + ".png"), optimize=True); return sz, w, h


for k, (t, s) in NAMES.items(): print(k, render(k, t, s))
