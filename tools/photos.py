#!/usr/bin/env python3
"""photos.yml worker: photos/list.txt lines "name url" -> photos/<name>.jpg (max 1920 px long side, JPEG q88)
plus photos/<name>.json (source url, and license/author from the Commons API for Wikimedia files).
A name whose .jpg exists and whose .json has the same url is skipped. Failures write {"url","error"} to the json."""
import html, io, json, os, re, sys, time, urllib.parse
import requests
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
UA = {"User-Agent": "reels-drop-photos/1.0 (https://github.com/avinoamroz/reels-drop; noam@expressenglish.co.il)"}
API = "https://commons.wikimedia.org/w/api.php"
MAX = 1920


def strip(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def commons_title(url):
    u = urllib.parse.urlparse(url)
    path = urllib.parse.unquote(u.path)
    if "wikimedia.org" in u.netloc and "/wikipedia/" in path:          # upload.wikimedia.org
        parts = path.split("/")
        name = parts[-2] if "/thumb/" in path else parts[-1]
        return "File:" + name
    if u.netloc.endswith("wikipedia.org") or u.netloc.endswith("wikimedia.org"):
        # .../wiki/File:X.jpg  or  .../wiki/Article#/media/File:X.jpg
        for part in (urllib.parse.unquote(u.fragment), path):
            m = re.search(r"(?:File|Image|Datei|Fichier|קובץ):([^/#?]+)$", part)
            if m:
                return "File:" + m.group(1).replace("_", " ")
    return None


def commons_info(title):
    r = requests.get(API, params={"action": "query", "format": "json", "titles": title, "prop": "imageinfo",
                                  "iiprop": "url|extmetadata|size", "iiurlwidth": MAX}, headers=UA, timeout=60)
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    ii = page["imageinfo"][0]
    md = {k: v.get("value") for k, v in ii.get("extmetadata", {}).items()}
    src = ii["url"]
    odd = src.lower().endswith((".tif", ".tiff", ".svg", ".pdf", ".djvu"))
    if ii.get("thumburl") and (odd or max(ii.get("width", 0), ii.get("height", 0)) > MAX):
        src = ii["thumburl"]  # server side 1920 px rendering: smaller download, and SVG/TIFF become raster
    meta = {"source": "Wikimedia Commons", "title": page.get("title"), "page": ii.get("descriptionurl"),
            "license": strip(md.get("LicenseShortName")), "license_url": md.get("LicenseUrl"),
            "author": strip(md.get("Artist")), "credit": strip(md.get("Credit")),
            "attribution_required": md.get("AttributionRequired"), "usage_terms": strip(md.get("UsageTerms")),
            "date": strip(md.get("DateTimeOriginal") or md.get("DateTime")),
            "description": strip(md.get("ImageDescription"))[:400]}
    meta["credit_line"] = ", ".join(x for x in [meta["author"] or None, meta["license"] or None, "Wikimedia Commons"] if x)
    return src, meta


def get(url):
    for i in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (i + 1)); continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == 3: raise
            time.sleep(5 * (i + 1))
    r.raise_for_status()


def main():
    os.makedirs("photos", exist_ok=True)
    lines = [l.split(None, 1) for l in open("photos/list.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
    for parts in lines:
        if len(parts) != 2:
            continue
        name, url = parts[0], parts[1].strip()
        jp, js = f"photos/{name}.jpg", f"photos/{name}.json"
        if os.path.exists(jp) and os.path.exists(js):
            try:
                if json.load(open(js)).get("url") == url:
                    continue
            except Exception:
                pass
        meta = {"name": name, "url": url}
        try:
            src, title = url, commons_title(url)
            if title:
                src, cm = commons_info(title); meta.update(cm)
            else:
                meta.update({"source": urllib.parse.urlparse(url).netloc, "license": None,
                             "note": "not Commons: check the source page for rights (US federal works are usually public domain)"})
            r = get(src)
            if "image" not in r.headers.get("content-type", "image") and not src.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")):
                raise ValueError(f"not an image: {r.headers.get('content-type')}")
            im = Image.open(io.BytesIO(r.content)); im = ImageOps.exif_transpose(im)
            meta["original_size"] = list(im.size)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA"); bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[-1]); im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")
            im.thumbnail((MAX, MAX), Image.LANCZOS)
            im.save(jp, "JPEG", quality=88, optimize=True, progressive=True)
            meta.update({"size": list(im.size), "downloaded": src, "ok": True})
            print(f"ok {name}: {im.size} {meta.get('license')} | {meta.get('author')}")
        except Exception as e:
            meta.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:500]})
            print(f"FAILED {name}: {meta['error']}")
        json.dump(meta, open(js, "w"), indent=1, ensure_ascii=False)
        time.sleep(1)


if __name__ == "__main__":
    main()
