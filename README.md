# sanoTTS-es-cl

Chilean Spanish (es_CL) text-to-speech, end to end:

1. **`es_CL-huemul-medium`** — a Piper VITS voice fine-tuned on Google's
   Chilean Spanish corpus (OpenSLR SLR71, male speaker `clm_02121`,
   ~18.8 min), phonemized with `es-419` (Latin American Spanish: seseo +
   yeísmo, which matches the Chilean accent).
2. **`chilean`** — a distilled ~1.5M-parameter sanoTTS browser voice built
   from that teacher with the Root-A recipe (duration student + contextual
   latent student + piperlite decoder student), shipped as a
   `front_f16.bin`/`dec_f16.bin`/`meta.json` bundle for the sanoTTS wasm
   runtime, plus a smaller **~450k-parameter** variant.
3. The browser espeak-ng G2P module (`web/snt_g2p.*`) upgraded to the exact
   espeak-ng version piper 1.8 uses, with the `es-419` voice added and an
   exact-parity gate (`20/20` sentences vs python piper, and `en/de/fr/vi/zh`
   blast-radius checks).

Everything here is scriptable and re-runnable for **any language / speaker**
— see [PIPELINE.md](PIPELINE.md).

## Results

| artifact | size | note |
|---|---|---|
| `es_CL-huemul-medium.onnx` | 63 MB | Piper teacher, UTMOS val_mos 3.46 |
| sanoTTS `chilean` bundle | 3.1 MB | fp16, 1,569,164 params |
| sanoTTS `chilean-small` bundle | 681 KB | fp16, 340,488 params, ~20× realtime |
| piper contribution dir | 61 MB | `es/CL/huemul/medium` for rhasspy/piper-voices |

`samples/` has the same eight sentences rendered three ways each
(`*_teacher.wav` = the Piper teacher, `*_student_full.wav` = the 1.57M
distilled voice, `*_student_small.wav` = the 340k voice). Listen there
before anything else.

Distillation gate (SCOREQ no-reference, 64 held-out rows, synthetic domain):

| lane | SCOREQ | Δ vs teacher |
|---|---|---|
| teacher (full piper ONNX) | 3.97 | — |
| teacher latent → cut piper decoder | 3.97 | **0.00** (exact round-trip) |
| teacher latent → student decoder | 3.37 | −0.60 |
| full student (student duration + acoustic) | 3.15 | −0.83 |

Browser chain: both bundles pass `verify_voice_node.mjs` through the real
wasm runtime (finite audio, no silence; 1.6× realtime at 1.57M, ~20× at
340k). G2P parity gates are 20/20 exact for `es` and `es-419`, plus
en/de/fr/vi/zh blast-radius checks after the espeak-ng upgrade.

Held-out intelligibility (16 reserved Tatoeba sentences, rendered by the
wasm runtime, Whisper small): **CER 0.021, WER 0.082** — the Castilian
`spanish` voice measures 0.147 WER under the same protocol family
(`experiments/evidence/es-cl-tatoeba-20260912.json` in the sanoTTS repo).

## Credits

This recipe exists because the sanoTTS project (**Ampixa/sanoTTS**) is
excellent — a research-grade neural TTS that runs in the browser and on
microcontrollers, with clear docs and an honest evaluation culture. The
distillation pipeline, the wasm runtime and the bundle format are all its
work; this repo only drove it through a language it hadn't met yet, and
reconstructed a handful of uncommitted pipeline tools so the whole thing is
reproducible from a public repo.

Equally: **piper1-gpl** (OHF-Voice) for the teacher training stack, the
espeak-ng project for phonemization, Google's crowdsource team for the
SLR71 dataset, and the Tatoeba community for supplemental text.

## Layout

```
scripts/            every step of the pipeline, in run order
tools/              reimplemented pipeline tools (parity, signatures,
                    feedback loop, summarizer, package + bundle exporters)
PIPELINE.md         the full recipe: other languages, female voice,
                    reduced-size variant, G2P, eval, contribution
ATTRIBUTION.txt     SLR71 / CC BY-SA 4.0 attribution (applies to every model here)
```

## Quickstart (assumes the sanoTTS + piper1-gpl repos are set up)

```bash
# 0. environment (once)
bash scripts/setup_env.sh

# 1. pick a speaker + export the fine-tune dataset
bash scripts/run_speaker_stats.sh        # ranks speakers by duration
bash scripts/run_export_speaker.sh       # writes metadata.csv + eval.txt + wavs

# 2. fine-tune the Piper teacher (~40 min on an RTX 4070)
bash scripts/finetune_huemul.sh
bash scripts/export_huemul.sh            # ONNX + audition samples

# 3. distill into a sanoTTS voice (~2-4 h)
bash scripts/run_phase3.sh

# 4. export the web bundle + the piper contribution package
bash scripts/export_voice_bundle.sh
bash scripts/package_piper_voice.sh
```

Each script has a header comment explaining what it does and what it needs.

## Provenance

- Dataset: [Google Chilean Spanish Low-resource Speech (SLR71)](https://openslr.org/71),
  Copyright 2018, 2019 Google, Inc., CC BY-SA 4.0 — see [ATTRIBUTION.txt](ATTRIBUTION.txt).
- Base checkpoint: `es_ES-davefx-medium` from [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints).
- Teacher training + export: [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) (GPLv3).
- Distillation + runtime: [Ampixa/sanoTTS](https://github.com/Ampixa/sanoTTS) (MIT runtime, GPLv3 training/G2P).

The distilled voice is a derivative of the CC BY-SA 4.0 dataset (via the
teacher) and therefore carries share-alike: the same attribution applies to
every artifact in the release directories.
