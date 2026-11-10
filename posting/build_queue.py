#!/usr/bin/env python3
"""Rebuild the derived fields of posting/queue.json for the Make auto-poster
("IG 5 · Auto-poster from queue", scenario 5035xxx on us1.make.com).

You only ever edit `items` (id, account, due, payload). Then run:
    python3 posting/build_queue.py
and commit + push. This recomputes:
  - due_epoch for each item
  - slots:       {"s<bucket>": {"noam": item, "express": item}}
  - due_buckets: "|<bucket>|<bucket>|..."
where bucket = floor(due_epoch / 1800) (30 minute buckets, UTC epoch).

How Make uses it: every 30 min (06:45 to 01:30 Israel) it fetches this file
(1 operation). It only goes further if the text contains "|B|" where
B = floor((now + 300) / 1800). Make runs at :25 and :55, so each item is
posted at the run nearest its due time (within 15 min); see posts_at. Then it posts slots["s"+B].noam and slots["s"+B].express (at most
one per account), skipping ids already in Make data store "ig_posted".

Rules enforced here:
  - one item per account per 30 min bucket (a clash is an error)
  - account is "noam" or "express"
  - due has a UTC offset, e.g. 2026-11-10T14:00:00+02:00
  - due must be between 06:45 and 01:00 Israel time (the scenario sleeps
    01:30 to 06:45, and an item is picked up in the NEXT bucket)
Removing an item = delete it from items. Items already posted are skipped
by Make anyway (data store), so leaving them is harmless.
"""
import json, sys, os
from datetime import datetime, timezone, timedelta

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queue.json")


def main(path=PATH):
    q = json.load(open(path))
    items = q.get("items", [])
    slots, buckets, errs = {}, [], []
    for it in items:
        for k in ("id", "account", "due", "payload"):
            if k not in it:
                errs.append(f"{it.get('id')}: missing {k}")
        if it.get("account") not in ("noam", "express"):
            errs.append(f"{it.get('id')}: bad account {it.get('account')}")
            continue
        d = datetime.fromisoformat(it["due"])
        if d.tzinfo is None:
            errs.append(f"{it['id']}: due needs a UTC offset")
            continue
        ep = int(d.timestamp())
        it["due_epoch"] = ep
        b = (ep + 300 + 900) // 1800  # nearest Make run (runs at :25 and :55 Israel); Make uses floor((now+300)/1800)
        it["posts_at"] = datetime.fromtimestamp(b * 1800 - 300, timezone(timedelta(hours=2))).strftime("%Y-%m-%d %H:%M")
        key = f"s{b}"
        slot = slots.setdefault(key, {})
        if it["account"] in slot:
            errs.append(f"{it['id']}: same 30 min bucket as {slot[it['account']]['id']} on {it['account']}")
            continue
        p = it["payload"]
        if not isinstance(p.get("body"), str):
            errs.append(f"{it['id']}: payload.body must be the IG1 body JSON string")
        slot[it["account"]] = {"id": it["id"], "due": it["due"], "payload": {"body": p.get("body", "")}}
        buckets.append(b)
    if errs:
        print("ERRORS:\n  " + "\n  ".join(errs))
        sys.exit(1)
    items.sort(key=lambda x: x["due_epoch"])
    q["items"] = items
    q["slots"] = slots
    q["due_buckets"] = "|" + "|".join(str(b) for b in sorted(set(buckets))) + "|" if buckets else "|"
    q["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    json.dump(q, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"ok: {len(items)} items, {len(slots)} buckets")


if __name__ == "__main__":
    main(*sys.argv[1:])
