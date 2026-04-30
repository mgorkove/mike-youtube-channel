# Required and optional assets for daniel_reed_shorts

## music/  (REQUIRED)

Drop one or more royalty-free background music tracks here (.mp3, .wav,
.m4a, .ogg). One is picked at random per Short and plays underneath the
entire 60-second video, faded in over the first 0.3s and out over the
last second. Volume is attenuated by `satisfying_shorts.music_volume` in
config.yaml (default 0.6).

Tips:
- Calm, ambient, lo-fi, or beats-only — no vocals.
- At least 60 seconds long. Longer files are auto-trimmed.
- Confirm the license permits YouTube monetization if relevant.

The pipeline will refuse to render if this directory is empty.

## intro/  (OPTIONAL)

If non-empty, the pipeline will pick one of these files at random as the
2-second intro instead of fetching from Pexels. Files: any common video
format (.mp4, .mov, .webm, .mkv).

If this directory is empty (or missing), a portrait visually-ASMR clip
is automatically fetched from Pexels using the search terms in
`satisfying_shorts.pexels_intro_queries` (or the built-in defaults:
kinetic sand, slime, paint mixing, ink in water, etc.). This requires
`PEXELS_API_KEY` to be set in the environment.

Pexels stock clips almost never have audio — the music track plays
under the intro just as it does under the photos. If you want actual
ASMR sound under the intro, drop a clip with audio into this directory.
