# MORENA Studio

Say what you want to film, in Kinyarwanda, Swahili, Shona, English, Yoruba, Igbo, Hausa or Pidgin.
Get back a Seedance 2.5 instruction you can post to the API unedited. 503M parameters, on a CPU.

Live: **https://vambo--morena-studio-studio-web.modal.run**

```
Dushaka videwo yerekana uko ibiryo byacu biba: ibiryo biri mu isaro,
hanyuma bikagururwa mu mpande, hanyuma umuntu akarya.          (Kinyarwanda)

  -> Shot 1: A large clay pot simmering with stew in a sunlit Kigali courtyard.
             The camera uses medium shot, overhead, pan.
     Shot 2: A chef's hands plating the steaming stew.
             The camera uses close-up, zoom.
     Shot 3: A person at a wooden table with red laterite walls, eating.
             The camera uses medium shot.
     16:9, 18 seconds, audio on
```

Training pipeline and data: [morena-tools](https://github.com/thisisisheanesu/morena-tools).
The payments demo is a separate product: [morena-pay](https://github.com/thisisisheanesu/morena-pay).

## The camera vocabulary is mined, not invented

`mine_camera_grammar.py` reads 400 screenplays from
[MovieSum](https://huggingface.co/datasets/rohitsaxena/MovieSum), scans **520,378 scene sentences**
and keeps 34 canonical terms with their frequencies and the shot/movement pairs that genuinely
co-occur. Sampling is frequency-weighted, so a wide shot draws a tracking move far more often than
a rack focus, the way real films behave.

The screenplays are not redistributed. Their camera grammar is functional vocabulary and only that
was kept: `camera_grammar.json` holds terms and counts, no script text.

## Measured on 120 held-out briefs

Briefs the model never saw, generated with their own id prefix and split out before the corpus was
built. Scored structurally, never against a reference wording, because a shot has no single right
answer.

| | |
|---|---|
| Chose `seedance_generate` over four other endpoints | **100%** |
| Prompt carries the camera clause Seedance asks for | **100%** |
| Every camera term is one Seedance understands | **100%** |
| Aspect ratio matches what the clip is for | **100%** |
| Duration inside the documented 4 to 30 seconds | **100%** |
| Replied in the language the brief was written in | 99.2% |
| Sequences whose framing actually changes between beats | **100%** |
| **Every check passing on the same brief** | **95.0%** |

The sequence check is not a count of `Shot N` markers. It requires at least two beats and a
different set of camera terms between them, because four identical wide shots labelled Shot 1 to
Shot 4 would pass a marker count while being exactly the thing worth catching.

## The app

One HTML file. The shot renders as a letterboxed frame in the requested aspect ratio, a sequence
renders as one card per beat with its own camera tags and its share of the duration, and there is a
button to copy the JSON. Below it, the checks that decide whether the call would actually run.

## Running it

```bash
modal deploy serve.py
modal volume put morena-pay-models mini-tools-Q4_K_M.gguf
```

Locally:

```bash
llama-server -m mini-tools-Q4_K_M.gguf --path app --no-jinja -c 4096
```

## Licence

Apache 2.0. Independent demo, not affiliated with or endorsed by ByteDance or Seedance.
