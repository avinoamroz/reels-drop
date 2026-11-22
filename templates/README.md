# Reel templates (spec to reel on GitHub)

One short JSON spec in `specs/<id>.json` becomes one finished reel. Push the spec and the `spec-render` workflow
(`.github/workflows/spec-render.yml`) renders it on GitHub: voice, render, transcript check, then it commits

| file | what |
|---|---|
| `out/<id>.mp4` | the reel, 1080x1920, 30 fps, AAC 48 kHz, mix at -16 LUFS (spec `music.lufs`) |
| `covers/<id>_cover.jpg` | 1080x1920 cover, hook in huge text inside the centre 1080x1440 grid-safe area |
| `previews/<id>_540.mp4` | 540p preview for the review card |
| `out/frames/<id>_{0,1,3,last}.jpg` | frame grabs at 0 s, 1 s, 3 s and the last frame |
| `out/<id>.report.json` | duration, loudness, transcript match score (per line, MISS under 0.8), lint warnings, caption, sources |

Nothing is posted and nothing is approved by this pipeline. Specs named `test_*` are tests and never go to the Launch Map.

## Templates

| template | the format | what it needs |
|---|---|---|
| `map_counter` | motion opener (globe spin, map tear, fold ...) with the hook from frame 0, region zoom, live counter, pins, bar cards, big number, source chips, end card, loop back into the opener | lines + voice, `data.cam0`, counter steps, pins, bars |
| `data_race` | bar race over time from a CSV or inline table, ranks glide, year label, callout card | lines + voice, `data.csv` or `data.table` |
| `scorecard` | live "0 out of 4" scorecard, countries around Israel first, sorted best first, "no data" pills, live score per row | lines + voice, `data.columns` (max 4, each with an icon), `data.rows` |
| `slang_scene` | Express: stick-figure scene first, Hebrew word as a small bottom line, 2 or 3 English options, mini chat, "Save this for your next meeting"; blue = active/speaker, orange = passive, red = new | lines + voice, `data.speaker`, `data.options`, `data.chat` |
| `pun_sequence` | humour: one stick figure ("little Roth") grows up, one pun per stage; each scene slides in, a big label pops on the pun word (changed part in yellow), optional small note; hook over the first scene, loops back | lines + voice, `data.stages` [{scene, at, tag, label, label_at, note}], `subs_replace` for hyphenated puns, `cover.bg: "scene"` |
| `classroom` | story + data: cartoon classroom, every kid a country (flag + name on the desk), hook on the board at frame 0, labelled troublemakers (groups or regimes only) throw paper at the hero who types and blocks, notes to classmates, then the board zooms to full screen: stat card, ranking sorted best first ("no data" pills, hero slot waits), two-number verdict, zoom back for the end card | lines + voice, `data.kids`, `data.bullies`, `data.stat`, `data.ranking`, `data.verdict` (docstring in `classroom.py`) |
| `two_paths` | split screen of two sides with a ground cutaway: soldier, shield, house and shelter on the left; houses, civilians and a tunnel with generic fighters on the right; camera moves split / left / right; sourced quote cards in the top area with focus rings; hook over the split from frame 0, loops back | lines + voice, `data.views`, `dome_at`, `shelter_at`, `tunnel_at`, `fighters_at`, `big`, `quotes`, `focus`, `tags`; `cover.bg: "scene"` |
| `crowd_equals` | data story: "1 Israeli = N neighbours". One sky-blue figure left, a regional map with the beat's country tinted sky (never red), a neutral white crowd of N figures pops in (fractional N: the last figure is cut), the equation counts up; bar cards between beats; emoji/flag icons; loops back to the single figure | lines + voice, `data.beats` [{code, demonym, flag, ratio, txt, tag, lon, lat, at}], `data.bars`, `data.icons` (docstring in `crowd_equals.py`) |
| `versus_card` | data story: one live scorecard per country, "<country> today" vs "at Israel's level" on up to 6 measures; rows fill on their anchors (or fast), a row glows while the voice talks about it, the "Upgrades" score counts up live ("same" rows show =), a yellow sticker (e.g. "16x income") pops; summary bar cards sorted best first; end card with a second balance line; loops back to the hook | lines + voice, `data.measures`, `data.israel`, `data.cards`, `data.bars` (docstring in `versus_card.py`) |
| `story_climb` | first-person story: one minimal female stick figure, scene by scene (film-strip tornado, rock wall on a rope, falls, a closing door, pharmacy and pill boxes, endless peaks, 5% dots, speech bubble, the woman and the little girl), cross dissolves, titles, small notes, credit; loops back | lines + voice, `data.scenes` [{scene, at, tag}], `data.titles`, `data.notes`, `data.credit` (docstring in `story_climb.py`) |
| `stock_chart` | numbers over time as stock charts: map with the hook at frame 0, then one full-screen chart per place (own scale, own year range), the line draws left to right, green rising, red falling, grey flat, ticker header ("IRAQ <150k [down] 90%") counting with the line head, tick sounds; photos between beats, recap grid of sparklines, end card, map + hook again for the loop | lines + voice, `data.map`, `data.charts`, `data.recap`, `data.photos` (docstring in `stock_chart.py`) |
| `zoom_stock` | "zoom out on the chart": dark trading screen, vertical 0 to 100 scale opened zoomed in (gold UTOPIA line at 100, Israel's line + mini map below, hook over the top band), red flaw blocks drop into the gap with band photos, then the scale zooms out to 0 to 100 and the neighbours' lines draw in with right axis price tags, bar cards, end card, zoom back in for the loop | lines + voice, `data.scale`, `data.utopia`, `data.israel`, `data.blocks`, `data.rows`, `data.bars`, `data.photos` (docstring in `zoom_stock.py`) |
| `cine_timeline` | "history in 60 seconds" documentary look: full bleed, no cards. Beats alternate every 2 to 4 s between a dark minimal map (near-black sea, charcoal land, thin borders, small uppercase grey labels; territory fills change in soft blues and muted browns per year; whole land of Israel always drawn, West Bank and Gaza a different shade, Green Line green; slow camera push) and full-bleed REAL archival photos (Ken Burns, muted film grade, grain, tiny credit). Big white karaoke subtitle one word at a time, year label top left (rolls), thin timeline scrubber with a moving dot, small source label under the year, hook from frame 0, `{"follow": true}` line, balance line as plain text, loops to frame 0. Photos are repo paths (e.g. `photos/gl_x.jpg` from photos.yml) and are pulled into the sparse CI checkout automatically | lines + voice, `data.span`, `data.years` [{at, year, label}], `data.beats` [{at, kind: map (cam0, cam1) or photo (file, credit, kb)}], `data.states` [{at, fills}], `data.labels`, `data.marks` (docstring in `cine_timeline.py`; example `specs/golan_100_years_cine.json`) |
| `rehook` | a published reel with a new hook card (max 6 words with a number, frame 0, count-up moving by 1.5 s) instead of its first seconds; body kept; new cover | `data.source` (repo mp4), `data.cut`; no voice |
| `face_hook` | Noam's own face clip opens a published reel: his real voice (lips in sync), clip hflipped, big hook text over his chest, Israel Facts mark, subtitles of exactly `data.said` (timed by faster-whisper); hard cut into the source after its own hook line; rest kept; follow line in the reel clone (ref/follow_line_tts_chain.wav, Noam 22 Nov) replaces the TTS one or is inserted before the final lead-in; face levelled separately, then -16 LUFS; loops into the new frame 0 | `data.clip`, `data.said`, `data.source`, `data.cut` or `data.cut_after`, `data.follow` (docstring in `face_hook.py`) |

Follow line (Noam 22 Nov, replaces the 15 Nov recording): `{"follow": true}` on a line gives say "If you love Israeli facts, follow Noam Roz for more.", screen "Like Israel facts? Follow *@noamroz__*" and wav `ref/follow_line_tts.wav`, a RAW Cartesia take of the reel clone voiced in context (tools/follow_tts.py, workflow follow-tts) with `_words.json`; at render it gets the same house chain, tempo and loudness as the narration, so it blends. Old specs that name `ref/follow_line_noam.wav` are switched to it automatically; `"own_voice": true` keeps Noam's recording.

Recorded lines (15 Nov): a spec line with `"wav": "ref/follow_line_noam.wav"` (and its `say` text for subtitles and the transcript check) is placed from that recording instead of TTS (no house chain, silence trimmed, loudness matched to the voice); use it for Noam's own follow line. map_counter (15 Nov): `clock` (running year clock, `era` for BCE/CE), counter steps with `text`/`src` (ranges, "No reliable count"), highlight `color`/`alpha`/`flash` and `year` timing, `israel_at` (Israel plain until that anchor). Examples: specs/conquest_waves.json, specs/countries_clock.json.

Shared pieces (`core/common.py`): hook card at frame 0 (`opener/hook.py`), burned-in subtitles from the Cartesia word
timings, music bed (`doc_piano`, `journey`, `bounce`, `skip`, `warm` or `file:<repo path>`) at -16 LUFS ducked 9 dB
under speech, synthesised SFX, final two-pass loudnorm, circular ending (the last 0.5 s dissolves into frame 0, or into
the opener's frame 0), end card (balance line + CTA pill), cover generator, 540p preview, frame grabs.
Design: navy gradient, sky #45A4DD, yellow numbers, Inter Display ExtraBold, body text 64 px and up, nothing under 36 px,
max 12 English words per card (lint enforces), whole land of Israel gold with West Bank and Gaza a darker gold and the
Green Line green (`core/base.py` MapR, same as the 9 Nov remakes).

## Spec format

```json
{
 "id": "un_15_of_26",                      // = file name, lower case, digits, _
 "account": "noamroz__",                  // or "expressenglishcom"
 "template": "map_counter",
 "hook": "*15* of *26* UN|condemnations?", // max 6 words, a number, | = line break, *word* = gold
 "opener": {"style": "globe_spin_whip"},  // map_counter only: globe_spin_fast/_reverse/_whip/_night, globe_unpeel, map_tear, map_tear_v2, map_fold, map_puzzle, map_crumple
 "music": {"bed": "doc_piano", "lufs": -16},
 "voice": {"tempo": 1.1},                  // optional; "file": "voice/x.wav" to use a given recording
 "lines": [                                // one per spoken sentence group, in order
  {"say": "Fifteen out of twenty-six U.N. condemnations. One country.", "gap": 0.45},
  {"say": "Every year ... Let's watch the scoreboard.", "screen": "UN General Assembly *2025:* who got condemned?", "at": "L1:watch"}
 ],
 "data": { ... template specific, see the docstring at the top of each template file ... },
 "sources": [{"label": "UN Watch, UNGA resolutions", "year": 2025, "at": "L1", "until": "L7"}],
 "end": {"at": "L10", "balance": "Israel's war in Gaza deserves *scrutiny,* like any war.", "cta": "Comment SAFE for the sources"},
 "caption_en": "...", "caption_he": "...", "hashtags": ["#Israel"],
 "cover": {"t": 14.0, "big": "15 / 26", "text": "UN condemnations:|*one* country?"}
}
```

Time anchors (everywhere a time is asked for): `12.5` seconds, `"L3"` start of line 3 (0 based), `"L3e"` its end,
`"L3:fifteen"` the word, `"L3:one#2"` second "one", `"L3:fifteen+0.2"`, `"END-2"`. The voice's word timestamps
place everything, so a re-voice keeps the sync.

Lint (runs first, fails the job): no em or en dashes anywhere, HAMAS in caps, hook max 6 words, max 12 words per screen
card, file name = id, account valid. Warnings (go into the report): hook without a number, source without a year, no sources.
Your part, not the lint's: per capita by default, real sources and quotes, the CTA keyword is live in ManyChat, recognisable
people have media consent, the cover frame (`cover.t`) has no faces.

## Example spec per template

Full files in `templates/examples/` (the race and scorecard numbers there are placeholders: research and source before use).

**map_counter** (`examples/map_counter.json`, = `specs/test_map_un.json`)
```json
"data": {"cam0": [36.0, 31.5, 24.0],
 "cams": [{"at": "L1-0.2", "view": [18, 30, 205, 610], "dur": 1.4}],
 "highlight": [{"at": "L2", "until": "L7", "code": "RUS"}],
 "pins": [{"at": "L2", "until": "L7", "lon": 37.6, "lat": 55.75, "label": "Russia 6"}],
 "counter": {"label": "Condemnations", "of": 26, "steps": [{"at": "L2", "value": 6}, {"at": "L6:fifteen", "value": 26}]},
 "bars": [{"at": "L7:per", "until": "L9", "rows": [{"label": "Israel", "flag": "IL", "value": 14.9, "hl": true}]}],
 "big": [{"at": "L9", "value": 1100, "from": 1, "suffix": "x", "sub": "the rest of the world"}]}
```

**data_race** (`examples/data_race.json` + `examples/data_race_example.csv`)
```json
"data": {"csv": "templates/examples/data_race_example.csv",
 "meta": {"Israel": {"flag": "IL", "hl": true}, "Egypt": {"flag": "EG"}},
 "at": "L0+1.0", "until": "L5", "holds": {"1967": 1.2}, "top": 5, "dec": 1, "suffix": "M",
 "unit": "population, millions", "callouts": [{"at": "L5", "until": "L6", "text": "*17x* in one lifetime"}]}
```

**scorecard** (`examples/scorecard.json`)
```json
"data": {"columns": [{"label": "Women vote", "icon": "🗳️", "at": "L1:vote"}, {"label": "Criticise", "icon": "📢", "at": "L2:criticise"}],
 "rows": [{"label": "Egypt", "flag": "EG", "cells": ["Y", "N"]}, {"label": "Lebanon", "flag": "LB", "cells": ["Y", null]},
          {"label": "Israel", "flag": "IL", "cells": ["Y", "Y"], "hl": true}],
 "verdict": [{"at": "L5", "until": "L6", "text": "Only *one* gets 4 out of 4"}]}
```

**slang_scene** (`examples/slang_scene.json`, the walla reel)
```json
"data": {"he": "וואלה", "setting": "coffee",
 "speaker": {"name": "Emma", "says": "I'm moving to Tel Aviv!", "at": "L0:i'm"},
 "options": [{"tag": "Casual", "text": "*No way!*", "at": "L1:no"}, {"tag": "At work", "text": "*Wow,* that's big news!", "at": "L3:wow"}],
 "chat": [{"who": "them", "text": "I ran a marathon on Friday.", "at": "L4"}, {"who": "me", "text": "*Really?* How did it go?", "at": "L5"}],
 "save": {"at": "L7"}}
```

**rehook** (`examples/rehook.json`, = `specs/test_rehook_homicide.json`)
```json
"hook": "Who has *4x* Israel's|murder rate?",
"data": {"source": "homicide_race_v4.mp4", "cut": 2.55, "hook_dur": 2.2, "peek": 22.0,
 "count": {"from": 1, "to": 4, "suffix": "x", "label": "Israel's murder rate"}}
```
`cut` = a pause right after the old hook line (look at the source's ASR in `render/done/<id>.asr.txt` or its word timings).

## Running

- GitHub: commit `specs/<id>.json` (no `[skip ci]`) and push. Several specs in one push render in parallel (matrix, up to 20).
  Re-render: Actions, spec-render, Run workflow, ids `a b c`.
- Watch: `gh run list --workflow spec-render.yml`, then the report: `out/<id>.report.json`.
- Local look without a video (cheap, no disk): `TPL_SNAPS=0,3,10 python3 templates/render.py specs/<id>.json`
  (needs the voice locally) writes `/tmp/tpl_<id>/snaps.png`. Lint only: `--lint`.
- Voice: with the repo secret `CARTESIA_API_KEY` the job calls Cartesia directly (sonic-3, voice "Noam Roz - reels"
  `cd06f613-...`, SSE with word timestamps, Cartesia-Version 2025-04-16: the same body as Make TTS1) and commits
  `voice/<id>_vo.wav`, `_words.json` and a script hash, so re-renders of an unchanged script cost nothing. Without the
  secret it uses `voice/<id>_*.wav` + `_words.json` from the repo, or fails with a clear message.
- Other voices (spec `voice.id`): any Cartesia voice id instead of Noam's clone (e.g. a stock female voice for someone else's story);
  add `"chain": "plain"` to skip the Noam pitch/bass chain (atempo only). The voice id is part of the script hash, so a change re-voices.
  Specs without `voice.id` are unchanged. Music bed `soft_piano` (gentle piano + strings, no drums; `music.lift` anchor turns it hopeful).
