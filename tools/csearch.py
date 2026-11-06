#!/usr/bin/env python3
"""csearch worker: photos/search_req.txt lines "tag query" -> photos/search/<tag>.json (Commons file search with license, author, size)"""
import html, json, os, re, requests
UA = {"User-Agent": "reels-drop-photos/1.0 (https://github.com/avinoamroz/reels-drop; noam@expressenglish.co.il)"}
API = "https://commons.wikimedia.org/w/api.php"
strip = lambda s: re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()
os.makedirs("photos/search", exist_ok=True)
for l in open("photos/search_req.txt", encoding="utf-8"):
    p = l.strip().split(None, 1)
    if len(p) != 2 or l.startswith("#") or os.path.exists(f"photos/search/{p[0]}.json"): continue
    tag, q = p
    out = []
    try:
        if q.startswith("Category:"):
            r = requests.get(API, params=dict(action="query", format="json", generator="categorymembers", gcmtitle=q, gcmtype="file", gcmlimit=60,
                prop="imageinfo", iiprop="url|extmetadata|size", iiurlwidth=320), headers=UA, timeout=60)
        else:
            r = requests.get(API, params=dict(action="query", format="json", generator="search", gsrsearch=q, gsrnamespace=6, gsrlimit=50,
                prop="imageinfo", iiprop="url|extmetadata|size", iiurlwidth=320), headers=UA, timeout=60)
        for pg in (r.json().get("query", {}).get("pages", {})).values():
            ii = (pg.get("imageinfo") or [{}])[0]; md = {k: v.get("value") for k, v in ii.get("extmetadata", {}).items()}
            out.append(dict(title=pg["title"], w=ii.get("width"), h=ii.get("height"), license=strip(md.get("LicenseShortName")), author=strip(md.get("Artist"))[:80],
                            date=strip(md.get("DateTimeOriginal"))[:30], desc=strip(md.get("ImageDescription"))[:160]))
    except Exception as e:
        out = [dict(error=str(e)[:300])]
    json.dump(out, open(f"photos/search/{tag}.json", "w"), indent=1, ensure_ascii=False)
