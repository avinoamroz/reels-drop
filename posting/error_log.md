# Pipeline error log

One line per issue, written by the silent error watcher: date (UTC), what failed, cause, action.

- 2026-11-14 16:45 UTC: GitHub spec-render runs 43599310825 (homicide_race_vert, life_longest_race_vert) and 43599480940 (pop_race_v6_en_vert), 13:20 to 13:23. Cause: ffprobe returned an empty duration (input media not ready at render time), plus one push ref-lock race. Action: none needed, all three reels re-rendered successfully at 14:51 to 14:54 UTC (superseded), no rerun.
- 2026-11-14 16:45 UTC: Make FB1 (5034882) failed twice (13:41 and 16:41 UTC), InvalidAccessTokenError 401 "user has not authorized application" at UploadVideo. Cause: known external block from fb_perf_next.md, Make connection 4884728 needs Noam to reauthorize Facebook and select page 1154857144379801. Action: logged only, no retry.
