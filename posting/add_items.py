#!/usr/bin/env python3
"""Add or move queue items from a small JSON list, then run build_queue.py.
Usage: python3 posting/add_items.py items.json
items.json: [{"id","account":"noam|express","due":"2026-11-12T12:40:00+02:00",
  "video_url","caption","cover_url", optional "collab":[...], "ai":true, "trial":true}]
An existing id is replaced (moved). Payload body is built here, so nothing outside the repo is needed."""
import json, sys, os, subprocess
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queue.json")
q = json.load(open(P)); new = json.load(open(sys.argv[1]))
ids = {n["id"] for n in new}
q["items"] = [i for i in q["items"] if i["id"] not in ids]
for n in new:
    b = {"media_type": "REELS", "video_url": n["video_url"], "caption": n["caption"], "share_to_feed": True,
         "is_ai_generated": n.get("ai", True)}
    if n.get("cover_url"): b["cover_url"] = n["cover_url"]
    if n.get("collab"): b["collaborators"] = n["collab"][:5]
    elif n.get("trial", True): b["trial_params"] = {"graduation_strategy": "SS_PERFORMANCE"}
    q["items"].append({"id": n["id"], "account": n["account"], "due": n["due"],
                       "payload": {"body": json.dumps(b, ensure_ascii=False)}})
json.dump(q, open(P, "w"), ensure_ascii=False, indent=1)
subprocess.run([sys.executable, os.path.join(os.path.dirname(P), "build_queue.py")], check=True)
