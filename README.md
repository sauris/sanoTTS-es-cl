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
| sanoTTS `chilean` bundle | ~3.1 MB | fp16, ~1.5M params |
| sanoTTS `chilean-small` bundle | ~0.9 MB | fp16, ~450k params |
| piper contribution dir | 61 MB | `es/CL/huemul/medium` for rhasspy/piper-voices |

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
