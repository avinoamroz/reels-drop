# FB page repost: how to post the next N

Target page ONLY: "נועם רז - English Express / תכנית ביצועים באנגלית להייטקיסטים", page id `1154857144379801`.
Never post to any other page (not the English Express page 1792867521003763) or any other account.
Approved by Noam on 12 Nov 2026: repost already-published IG reels here, half Express, half advocacy, about 5 a day (max 10).

Queue: `posting/fb_perf_queue.json` (`queue` is already interleaved Express / advocacy, best first; `posted` and `skipped` are the logs).

## 0. Precondition (blocker as of 12 Nov 2026)
Publishing needs a Facebook connection that can see this page. On 12 Nov the Make connection
4884728 ("Noam's Facebook connection") only listed page 1792867521003763, and the data-store token has no
`pages_manage_posts`. Check first:
- `mcp__Make__rpc_execute` appName `facebook-pages`, appVersion 6, rpcName `Pages`, data `{"__IMTCONN__": 4884728}`.
- If `1154857144379801` is not in the result, STOP and report "Noam must reauthorize Make connection 4884728 and select the page". Do not try other pages or connections.

## 1. Tools (load with ToolSearch)
- `mcp__Make__s5034882_fb_1_facebook_page_video_list_or_publish_page_i` (FB1)
  - list: `page_id=1154857144379801, mode=list, tag=<tag>`
  - publish: `page_id=1154857144379801, mode=publish, file_url=<mp4>, description=<text>, tag=<tag>`
  - Returns nothing inline. Result lands on webhook 2850117:
    `mcp__Make__hook-incomings_list` (hookId 2850117, pg {limit:1, sortBy:created, sortDir:desc}) then `mcp__Make__hook-incomings_get`.
    Publish OK = `{"tag":..,"fb1":"ok","video_id":"<id>"}`; failure = `{"fb1":"error"}`. List = `{"status":"200","response":{"data":[...]}}`.
- `mcp__Make__s5039348_99_ai_graph_get_read_only_returns_response` (read-only Graph GET, body capped at 4000 chars)
  - fresh mp4 + caption: `<ig_id>?fields=media_url,caption,permalink,timestamp`

## 2. Per reel, in queue order (skip ids already in `posted` or `skipped`, and items with a `skip` field)
1. Graph GET `<ig_id>?fields=media_url,caption,permalink,timestamp` right before publishing (CDN links expire).
2. Screen the caption. Move to `skipped` (with reason) if: about Express City; a Hebrew-only promo for an event whose date is past (ExpressX invites etc.); names Adi Barazany, Michal Hanukah or Matan Yehezkel; or a near-duplicate of something already posted.
3. Dedupe: run FB1 list; if any page video has `created_time` within 30 min of the IG `timestamp`, or one of our `posted` fb ids is missing info, treat it as already posted and skip.
4. Description: the IG caption, Hebrew paragraphs first if it has Hebrew; keep at most 3 hashtags; replace every em dash and en dash with a comma or colon. Empty caption = no description.
5. Publish with FB1 (tag `fbperf_<ig_id>`). Read the webhook result.
6. Verify with FB1 list that the new video is there. NEVER retry a publish before a list shows it is not on the page.
7. Append to `posted`: `{ig_id, account, views, fb_video_id, posted_at}`; fill `permalink` and `caption_short` on the queue item.
8. Wait at least 2 minutes before the next publish.

Ordering: keep the interleave (Express, advocacy, Express, ...). Stop at N (default 5, max 10 a day).

## 3. Refill
Queue covers the newest 254 Express and 200 @noamroz__ media. Older media: page with `after` cursors in `coverage`
(`<ig_user_id>/media?fields=media_product_type,insights.metric(views){values}&limit=22&after=<cursor>`; keep limit about 22 so the cursor is not cut off).
IG ids: @noamroz__ 17841400746045382, @expressenglishcom 17841406916610308. Reels only, 300+ views.

## 4. Commit
Commit only `posting/fb_perf_queue.json` and this file. `git -c gc.auto=0 push`. Never prune, gc or reflog expire.
Do not touch `posting/queue.json` or the Launch Map. Never read Make data store 94072, never print tokens.
