# The es_CL voice pipeline — and how to repeat it for any language

This document is the complete recipe, as executed for `es_CL` (Chilean
Spanish, male speaker `clm_02121`). Every numbered step maps to a script in
[`scripts/`](scripts/), so the whole thing is re-runnable and easy to adapt.

Sections:

1. [Environment](#1-environment)
2. [Data: pick a speaker, export a fine-tune set](#2-data)
3. [Teacher: fine-tune a Piper voice](#3-teacher)
4. [Distillation: teacher → ~1.5M sanoTTS voice](#4-distillation)
5. [Reduced-size variant (~450k params)](#5-reduced-size-variant-300500k)
6. [G2P: make the browser phonemize exactly like the teacher](#6-g2p)
7. [Site wiring](#7-site-wiring)
8. [Evaluation + evidence](#8-evaluation--evidence)
9. [Piper contribution package](#9-piper-contribution-package)
10. [Gotchas learned the hard way](#10-gotchas)

Worked examples at the end: [female es_CL voice](#worked-example-a-female-es_cl-voice),
[a brand-new language](#worked-example-b-a-brand-new-language) and
[three voices in one unattended night](#worked-example-c-three-voices-in-one-unattended-night).

---

## 1. Environment

Hardware used: Windows 11 + RTX 4070 (12 GB), WSL2 Debian 12. Anything with
≥8 GB VRAM works (piper1-gpl reports success on 8 GB).

```bash
# inside WSL (Debian/Ubuntu), as root where needed:
apt install git build-essential cmake ninja-build espeak-ng ffmpeg \
            python3-venv python3-dev
# cmake must be >= 3.26 (Debian 12 ships 3.25) -- the piper venv gets a
# modern one from pip, see below.
```

Two virtualenvs, on purpose:

| venv | purpose | contents |
|---|---|---|
| `~/venvs/piper` | teacher fine-tune + ONNX export | `pip install -e '.[train]'` of a piper1-gpl clone + `torchaudio` (for the val_mos/UTMOS callback) + `cmake>=3.26` + a compiled `espeakbridge` |
| `~/venvs/saanotts` | everything else (packs, students, bundles, eval) | `pip install -e 'SANO_REPO[research]'` + `datasets huggingface_hub` |

`scripts/setup_env.sh` does all of it, including the two non-obvious piper
build steps:

```bash
# piper1-gpl needs its espeak bridge compiled against its own espeak-ng;
# editable installs skip the cmake build, so do it by hand:
cmake -B /tmp/piper-build -G Ninja -DCMAKE_BUILD_TYPE=Release   # from the piper1-gpl clone
cmake --build /tmp/piper-build -j 8
cp /tmp/piper-build/espeakbridge.so src/piper/      # into the clone's src/piper
```

Sanity checks after setup:

```bash
espeak-ng --voices=es          # which Spanish variants exist? (es, es-419; no es-cl)
python -c "import torch; print(torch.cuda.is_available())"
```

**Decision point — espeak voice.** Chilean Spanish has seseo/yeísmo, so the
teacher must be phonemized with `es-419` (espeak's Latin-American voice),
not `es` (Castilian: distinción θ, ʎ). Whatever voice you pick here is baked
into the teacher config and every distillation pack — switching later
invalidates the packs.

## 2. Data

The dataset: [`ylacombe/google-chilean-spanish`](https://huggingface.co/datasets/ylacombe/google-chilean-spanish)
(SLR71 audio + transcripts, 48 kHz, `female`/`male` configs).

1. **Rank speakers by total duration** — `scripts/run_speaker_stats.sh`.
   Gotcha: the numeric `speaker_id` is *not* unique across the female/male
   configs (the gender lives in the filename prefix `clf_`/`clm_`), so key
   stats by `(gender, speaker_id)`.

   | speaker | gender | minutes | utts |
   |---|---|---|---|
   | 2121 | male | 18.8 | 146 | ← picked (≥15 min target)
   | 4310 | female | 16.6 | ~150 | ← top female (see worked example A)

2. **Export** — `scripts/run_export_speaker.sh` (`export_speaker.py`):
   resamples 48 kHz → 22 050 Hz mono 16-bit (`scipy.signal.resample_poly`),
   writes `artifacts/data/<voice>/audio/*.wav` + `metadata.csv`
   (`utt_id|text`, piper1-gpl single-speaker format) + holds out 16
   utterances as `eval.txt` (never trained on, used for auditions) +
   writes `ATTRIBUTION.txt`.

   Cleanup is deliberately minimal: strip whitespace, drop empty/undecodable
   rows and >300-char texts, add missing opening `¿` before question
   segments (espeak uses it for prosody). Unnormalized accents ("no se
   como") stay — they are consistent between teacher and student.

3. **Distillation corpus** — `scripts/run_build_corpus.sh`
   (`build_distill_corpus.py`): 8 000 rows = deduplicated SLR71 transcripts
   (both genders) + Tatoeba `spa` sentences, minus (a) the 16 fine-tune
   holdouts, (b) the sentences used by earlier evidence files, (c) a
   fixed-seed reserve of 16 Tatoeba sentences for the OOD eval
   (`tatoeba_eval16.json`). Deterministic shuffle so the orchestrator's
   front-held-out smoke/eval splits are representative.

## 3. Teacher

`scripts/finetune_huemul.sh` — the whole fine-tune is one piper1-gpl call:

```bash
python -m piper.train fit \
  --data.voice_name huemul \
  --data.csv_path  artifacts/data/es_cl/metadata.csv \
  --data.audio_dir artifacts/data/es_cl/audio \
  --data.espeak_voice es-419 \          # ← the accent decision
  --data.cache_dir artifacts/cache/es_cl \
  --data.config_path artifacts/voices/es_CL/config.json \
  --data.batch_size 8 --data.num_workers 4 \
  --model.sample_rate 22050 \
  --model.warmstart_ckpt $BASE/es_ES/davefx/medium/epoch=5629-step=1605020.ckpt \
  --model.learning_rate 1e-4 --model.learning_rate_d 5e-5 \
  --trainer.max_epochs 1000 --trainer.accelerator gpu --trainer.devices 1 \
  --trainer.default_root_dir artifacts/voices/es_CL/huemul-medium \
  --trainer.check_val_every_n_epoch 5
```

Notes that matter:

- **`--model.warmstart_ckpt`, not `--ckpt_path`.** Warmstart copies every
  matching-shape parameter (including the Spanish phoneme embeddings) into a
  fresh optimizer/epoch counter. `--ckpt_path` is Lightning *resume* — it
  restores optimizer + epoch, and resuming this way after a warmstart
  transiently collapsed val_mos for us. For "train a bit longer", start a
  NEW warmstart from the best checkpoint with a lower LR instead.
- Gender-matched base checkpoint: male → `es_ES-davefx-medium`, female →
  `es_ES-sharvard-medium` (both in rhasspy/piper-checkpoints). Piper
  training resamples audio itself, but exporting the data at 22 050 Hz keeps
  the cache honest.
- 1 000 epochs × 14 steps on a 4070 ≈ 90 min. `val_mos` (UTMOS) is the
  useful metric: ours went 2.30 → 3.46 (epoch 874) while `val_mel`
  plateaued after ~epoch 500. Pick the best `val_mos` checkpoint.
- **Export** — `scripts/export_huemul.sh`:
  `python -m piper.train.export_onnx --checkpoint <best.ckpt> --output-file <name>.onnx`.
  Piper 1.8's exporter needs two patches in our clone (upstream may differ):
  `load_from_checkpoint(..., weights_only=False)` (old checkpoints carry
  `PosixPath`) and `torch.onnx.export(..., dynamo=False)` (the new dynamo
  exporter can't trace VITS). Copy the training config next to the ONNX as
  `<name>.onnx.json`, then hand-fix it:

  ```json
  "language": { "code": "es_CL" },
  "dataset": "google_chilean_spanish_slr71",
  "dataset_attestation": "SLR71, https://openslr.org/71, CC BY-SA 4.0, ..."
  ```

- **Audition gate** — the export script also renders 8 audition wavs
  (held-out sentences + seseo/yeísmo probes + numbers). LISTEN before
  distilling. Ours: "between good and very good" after 1 000 epochs.

## 4. Distillation

`tools/train_voice_from_piper.py` (in the sanoTTS repo) drives every stage:

```bash
bash scripts/run_phase3.sh
# = python tools/train_voice_from_piper.py \
#     --teacher-onnx  artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx \
#     --teacher-config artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx.json \
#     --text artifacts/data/es_cl/distill_corpus.jsonl \
#     --out-dir artifacts/voices/es_CL/run \
#     --package-name huemul --device cuda
```

Stages (all resumable — each writes a done-file and is skipped on re-run):

| stage | what | time (4070) |
|---|---|---|
| s1a–s1d | teacher probe packs (smoke 12 / eval 128 / acoustic 7 860 / decoder 512 rows) | ~25 min (mostly s1c, CPU ONNX) |
| s2 | decoder-cut ONNX (generator input → waveform) | seconds |
| s3 | duration student (hidden 64, depth 3, 4k steps) | ~1 min |
| s4 | acoustic latent student (token_context, hidden 96, 5k steps) | ~3 min |
| s5 | PyTorch↔ONNX decoder parity + `piper-decoder-teacher.pt` | ~12 s |
| s6 | decoder activation-signature pack (512 chunks) | ~2 min |
| s7 | piperlite decoder student (channels 160,80,40,20; teacher-init `importance`; signature losses) | ~40 min |
| s8 | z-mix adaptation (50 % acoustic latents) | ~30 min |
| s9 | joint z finetune (duration+acoustic+decoder) | ~45 min |
| s10 | 5-lane feedback gate (render + SCOREQ) | ~10 min |
| s11 | raw fp16 package export | seconds |

Seven of the tools this orchestrator invokes were **never committed** to the
sanoTTS repo; the working reimplementations live in this repo's
[`tools/`](tools/) and are drop-in compatible (same CLIs/outputs):

- `verify_piper_decoder_torch_parity.py` — rebuilds the Piper generator
  decoder in PyTorch from the cut ONNX initializers (matching shapes +
  graph-read resblock dilations), proves mean-abs ≤ 5e-4 / cosine ≥ 0.9999
  on pack latents, exports `piper-decoder-teacher.pt`.
- `build_piper_vits_decoder_signature_pack.py` — exposes the decoder's
  internal activations as extra ONNX outputs and stores per-latent-frame
  `mean`/`logrms` signatures (`pooled_activation_signature`-exact),
  indexed by `decoder-signature-manifest.jsonl`.
- `run_roota_feedback_loop.py` + `summarize_roota_feedback_loop.py` —
  renders the five gate lanes (teacher / teacher-latent→piper-dec /
  teacher-latent→student-dec / full student / oracle-duration student) via
  the dashboard engine and scores them with SCOREQ.
- `export_roota_self_contained_package.py` — raw fp16 weights + manifest.
- `export_voice_bundle.py` — the browser bundle (see §7).

## 5. Reduced-size variant (~300–500k params)

The shipped `spanish` voice is ~1.56M params (front 832k + dec 731k). A
~450k variant trades some quality for ~3.5× smaller download. Reuse the SAME
packs (s1–s6 are student-size-independent — the signature pack describes the
teacher!) and rerun only the students smaller:

```bash
bash scripts/run_phase3_small.sh
# s3  duration: --hidden 48 --depth 2                      (~47k params)
# s4  acoustic: --hidden 48 --token-depth 2 --depth 2      (~116k params)
# s7  decoder:  --channels 64,32,16,8                      (~290k params)
# s8  z-mix + s9 joint on those checkpoints
```

Parameter arithmetic (piperlite decoder, channels `c0,c1,c2,c3`):

```
pre:  c0*192*7 + c0
ups:  c0*c1*16 + c1 + c1*c2*16 + c2 + c2*c3*8 + c3
bank: per stage 3 branches × (2*ch*ch*k + ch) for k=3,5,7
post: c3*7 + 1
```

`64,32,16,8` lands the decoder at ~290k; `80,40,20,10` (~426k) sounds
noticeably better if you can afford ~650k total. Render the gate lanes and
LISTEN — expect a modest loss of high-frequency detail and occasional
duration wobble vs the 1.5M voice. Then bundle-export with the small
checkpoints (§7) under a distinct key (`huemul-small`).

### int8 storage (optional, half the download again)

The runtime does float math, but the *download* can be int8: the exporter
quantizes each tensor symmetrically (scale = max|w|/127, exported into
`meta.json` as `front_scales`/`dec_scales` in slot order) and the loader
dequantizes to the same float32 blob as always:

```bash
scripts/export_voice_bundle.sh small int8     # 681 KB -> 333 KB
```

Measured on the 340k es_CL voice (same 16 held-out sentences, wasm-rendered,
Whisper small): CER 0.125 int8 vs 0.148 f16 — flat; WER 0.42 vs 0.35 — the
16-sentence WER is noisy (±2-3 word errors), so gate on CER and your ears.
Files: `front_i8.bin`/`dec_i8.bin`, `weights: "int8"` in meta.json. The
browser loader (`web/index.html widenInt8`) and the node gates both handle
f32/f16/int8 transparently. Keep the f16 export alongside for A/B before
committing to int8 — that is exactly what `scripts/int8_quality_gate.sh`
automates (export both, render, RMS-diff, re-score).

## 6. G2P

The browser phonemizer is a wasm espeak-ng (`web/snt_g2p.*`) driven by
`snt_g2p_set_voice(espeak_voice, voice_slot)` — the voice *name* is passed
to `espeak_ng_SetVoiceByName` at runtime, and `voice_slot` picks the Piper
phoneme-id table (the id map is universal across piper voices, so slot 11 =
`es` works for `es-419` too).

The stock bundle only shipped `lang/roa/es`, and its vendored espeak-ng
sources were older than what piper 1.8 builds against. Getting **exact**
parity (the browser must produce byte-identical ids to the packs the teacher
was distilled from) took three steps:

1. **Vendor espeak-ng @ `724808c`** (the commit piper1-gpl's CMake pins,
   2026-04-06) into `mcu/ports/wasm/espeak/`:
   `scripts/vendor_espeak_724808c.sh` clones it, checks out the commit +
   submodules, and copies the same pruned 27 libespeak-ng `.c` + 6
   ucd-tools `.c` files (+ headers) the wasm build always used.
2. **Sync the data** to piper 1.8's `espeak-ng-data` — same 18-language
   prune plus `lang/roa/es-419` (a ~170-byte voice file; the seseo lives in
   the compiled `es-la` phoneme table that is already inside `phondata`).
   `scripts/rebuild_g2p_v3.sh`.
3. **Patch the shim** (`mcu/ports/wasm/snt_g2p_wasm.c`): use
   `espeak_TextToPhonemesWithTerminator` + `espeakCHARS_AUTO` and map the
   returned `CLAUSE_*` code to punctuation exactly like piper's
   `espeakbridge` does. The old text-scanning heuristic dropped commas after
   the espeak-ng upgrade.

Build (emsdk ≥ 6; `scripts/install_emsdk.sh`):

```bash
source ~/work/emsdk/emsdk_env.sh
mcu/ports/wasm/build_g2p.sh     # → web/snt_g2p.{js,wasm,data}
```

Gates (must be 100 % exact — these are the training-time contract):

```bash
node mcu/ports/wasm/verify_g2p_es419_node.mjs    # es + es-419, 20/20 sentences
node mcu/ports/wasm/verify_g2p_blast_node.mjs    # en/de/fr/vi/zh spot check
node mcu/ports/wasm/verify_g2p_node.mjs          # the original per-language gate
```

For a NEW language: add the voice file (and dictionary, if separate) to
`espeak-data-multi`, pick the right id-table slot (`cp_id_tables_multi.h`),
extend the gate, rebuild. If a sentence mismatches, diff phoneme-by-phoneme
with `scripts/debug_g2p_mismatch.py` — every mismatch we saw was either
sentence-split framing (test harness) or version skew, never an id-map gap.

## 7. Site wiring

`tools/export_voice_bundle.py` (this repo) writes the wasm blob pair + meta:

```bash
python tools/export_voice_bundle.py \
  --key huemul --espeak-voice es-419 --g2p-voice-slot 11 \
  --duration-checkpoint  run/duration/duration-student.pt \
  --acoustic-checkpoint  run/joint/latent-student.pt \
  --decoder-checkpoint   run/joint/decoder-student.pt \
  --out-dir web/voices/huemul
```

The blob format (documented in the tool's docstring, mirrored from
`mcu/src/snt_front_f32.c` + `snt_piperlite.c`): `meta.bin` header (dims +
n_tensors at a fixed offset + per-slot offset table) followed by fp16
weights in the C runtime's slot order; `meta.json` carries the dims,
`front_widened_sha256`/`dec_widened_sha256` (sha256 of the fp32-widened
blob — the loader verifies it) and the `weights: "f16"` opt-in.

Then in `web/index.html`:

- add a `VOICES` entry `{key:"huemul", label:"Chilean", flag:"🇨🇱", ...}`
- add `DEFAULT_TEXT.huemul` (a sentence that shows off the accent)
- add the mascot/picker entry

Serve locally (`?voices=local` pins the local host) and click-test every
sentence. Test `?voices=hf` too after publishing — HF serves `no-store`,
which is why the page keeps its own Cache Storage copy.

## 8. Evaluation -- evidence

The site's language table reads `experiments/evidence/ood-*.json`. The
protocol (see `experiments/evidence/ood-tatoeba-20260908.json`): 16 Tatoeba
sentences rendered by the numpy runtime, transcribed by Whisper small, CER
against the source. The reserved `tatoeba_eval16.json` from §2 is the es_CL
input; `tools/eval_scorecard.py` runs the harness. Ship the JSON with the
same fields (or show `—` — several voices ship unmeasured).

## 9. Piper contribution package

`scripts/package_piper_voice.sh` assembles
`artifacts/release/piper-voices/es/CL/huemul/medium/`:

```
es_CL-huemul-medium.onnx        # the exported teacher
es_CL-huemul-medium.onnx.json   # training config + hand-fixed language/dataset/attribution
MODEL_CARD                      # language, speakers, quality, dataset, license, lineage
samples/01_eval.wav ...         # audition wavs for reviewers
```

Contribution path: PR to `rhasspy/piper-voices` (add the dir under
`es/CL/huemul/medium/`) or upload to the `rhasspy/piper-voices` HF repo —
plus one line in `docs/VOICES.md` ("Español, Chile (Spanish, es_CL)").

## 10. Gotchas

- **A custom player must call `set_voice` before the first phonemize call.**
  The g2p wasm keeps whatever espeak voice it was last given — on a fresh
  page that is the `en-us` default baked into `snt_g2p_init`. The demo page
  shipped without the switch, so Spanish text was phonemized as *English*
  and came out as garble while the main site (which sets the voice inside
  `synthVoice`) was always clear — bit-identical output between the two only
  after `snt_g2p_set_voice(meta.espeak_voice, meta.g2p_voice_slot)` runs
  before the first `snt_g2p_text_to_ids`, failing hard if the switch is
  rejected. Debugged with `demo/cdp_speak.cjs` (drives the real Speak path
  headlessly and saves the WAV; the md5 matching the main site's output was
  the proof) and `demo/cdp_selftest.cjs` (dumps ids/hashes from inside the
  page). If a custom player sounds "like the wrong language", this is the
  first thing to check.
- **Split text into per-sentence chunks before synthesizing.** The students
  are trained on single utterances; feeding a whole paragraph as one id
  sequence degrades to near-garbage. The main site's `splitChunks()` (ported
  verbatim into the demo) is the reference implementation.

- **PowerShell → wsl.exe eats `$` and quotes.** Never inline complex bash in
  `wsl -d Debian -- bash -lc "..."` from PowerShell; write a script file and
  run `wsl -d Debian -- bash /mnt/c/.../script.sh`. This repo's `scripts/`
  exist partly because of that.
- **drvfs (/mnt/c) is flaky under emcc** — `file_packager` intermittently
  fails rewriting `snt_g2p.data` in place. `build_g2p.sh` now builds into a
  `mktemp -d` on ext4 and copies over.
- **Editable installs of scikit-build projects don't compile C extensions**
  — hence the manual cmake + `cp espeakbridge.so` dance in §1.
- **Lightning resume after warmstart** transiently tanked val_mos (1.75 vs
  3.46). If you extend training, warmstart from the best checkpoint at a
  lower LR instead of resuming.
- **Old checkpoints + torch ≥ 2.6**: `weights_only=False` on load.
- **torch 2.14 ONNX export defaults to dynamo** — VITS needs
  `dynamo=False` (the legacy TorchScript exporter).
- **`speaker_id` is not gender-unique** in SLR71 (§2).
- **Tatoeba exports moved** to `downloads.tatoeba.org/exports/per_language/<lang>/`.

---

## Worked example A: a female es_CL voice

Only §2 and §3 change:

1. `export_speaker.py`: `SPEAKER_ID = 4310`, load the `"female"` config.
   (4 310 is the top female speaker at 16.6 min / ~150 utts; runners-up:
   7508 at 16.5 min, 7049 at 16.3 min — all ≥ the 15 min target.)
2. Base checkpoint: `es/es_ES/sharvard/medium/epoch=4899-step=215600.ckpt`
   (the only female es_ES medium checkpoint in rhasspy/piper-checkpoints).
3. Everything downstream is byte-identical in shape: distillation, g2p
   (`es-419` again), bundle export with `--key huemul-f` (or a name like
   `tenca`), site entry with a different mascot, evidence run.
4. Budget ~2.5 h wall-clock for the teacher + ~2 h for the 1.5M distill
   (or ~40 min for the small variant).

## Worked example B: a brand-new language

Using Quechua (quy) as a sketch, since es_CL covered all the es-specific
parts:

1. **Data**: any mono-speaker corpus with transcripts (Common Voice per-speaker
   split, a fine-tunable HF dataset, or OpenSLR). Target ≥ 15 min after
   cleanup; the export script only assumes `audio` + `text` columns (or swap
   in a loader for your source).
2. **espeak support check**: `espeak-ng --voices=quy`. If the language (or
   the accent variant you need) is missing, espeak-ng must gain it first —
   that is an upstream contribution, not a training problem.
3. **Base checkpoint**: closest high-quality piper voice in rhasspy-checkpoints
   (language match > gender match > quality match). For a language with NO
   piper voice, warmstart from `en_US-lessac-medium` and double the epochs.
4. **espeak voice string**: exactly what `--voices` printed (e.g. `quy`,
   `es-419`, `pt-br` — case and separator matter).
5. **Corpus language**: the Tatoeba filter is a 3-letter code
   (`scripts/build_distill_corpus.py`); if Tatoeba coverage is thin, top up
   from FLORES/opus (mind the license) or just use the transcripts you have —
   the acoustic student generalizes from ~2k rows upward (id/vi shipped on
   1 908).
6. **G2P slot**: `cp_id_tables_multi.h` maps slots → id tables; the tables
   are the universal piper map, so any voice with the standard 256-symbol
   map can reuse an existing slot. A genuinely new *table* only appears if
   you trained with a non-standard map (piper hasn't, historically).
7. Everything else — orchestrator, gates, bundle, evidence, packaging — is
   language-agnostic.

## Worked example C: three voices in one unattended night

2026-09-13, executed end-to-end by the parametrized scripts in
[`scripts/`](scripts/) (the huemul-specific ones above stay as the reference
walkthrough). Three voices, one GPU, one detached master loop:

| voice | lang | speaker | base ckpt | best val_mos | OOD CER / WER |
|---|---|---|---|---|---|
| copihue | es_CL | SLR71 female clf_04310 (~16.6 min) | es_ES-sharvard | 3.54 @ ep 869 | 0.039 / 0.196 |
| vueltiao | es_CO | SLR72 male com_06136 (~18.1 min) | es_ES-davefx | 3.25 @ ep 894 | 0.053 / 0.193 |
| chande | es_CO | SLR72 female cof_02484 (~19.8 min) | es_ES-sharvard | 3.72 @ ep 814 | 0.022 / 0.127 |

(huemul for comparison: val_mos 3.46, CER 0.021 / WER 0.082 — the
16-sentence WER is noisy, gate on CER + ears. Names follow the endemic-thing
convention; the two Colombian voices are named from the Sincelejo savannas:
the chandé dance and the sombrero vueltiao woven in San Andrés de Sotavento.)

Per voice (~1.9 h wall on the 4070): finetune 1000 epochs ≈ 105 min →
ONNX export + audition render ≈ 15 s → distill s1–s11 ≈ 45 min → bundle
export + wasm-render evidence ≈ 30 s. All stages resumable via
`artifacts/es_cl/pipeline/<voice>/*.done`; the master is
`scripts/run_voice_pipeline.sh`, run detached:

```bash
wsl -d Debian -- bash -lc "setsid nohup bash /tmp/run_voice_pipeline.sh \
  > artifacts/es_cl/logs/pipeline_master.log 2>&1 < /dev/null & disown; sleep 8"
```

New scripts and what they generalize:

- `export_slr71_speaker.py --voice V --config female --speaker 4310` — any
  SLR71 speaker (copihue = top female at 16.6 min, from the §2 stats table).
- `stats_es_co.py` / `export_es_co_speaker.py` — SLR72 has no HF dataset;
  rank speakers by wav-header durations straight inside the openslr zips,
  then export from the zip (same cleanup rules; TSV transcripts). Same
  gender-id-collision gotcha as SLR71 (com_04310 exists alongside
  clf_04310!).
- `build_distill_corpus2.py --voice V` — SLR71 + SLR72 + Tatoeba top-up to
  8 000 rows; excludes every voice's eval holdouts, the es_ES evidence
  sentences and the shared `tatoeba_eval16.json` reserve (loaded, never
  regenerated, so all voices eval on the same 16 sentences).
- `finetune_voice.sh V LANG GENDER` — picks the gender-matched base ckpt,
  per-voice cache dir, per-language config copy (es_CO reuses the es_CL
  config verbatim: espeak es-419 + standard 256 map).
- `export_teacher.sh V LANG` — picks the best val_mos checkpoint by filename,
  exports ONNX, patches language/dataset/attestation into the config copy,
  renders 8 auditions, then trims checkpoints (disk guard, see below).
- `run_phase3_voice.sh` / `export_bundle_voice.sh` / `run_evidence_voice.py`
  — the §4/§7/§8 flow with teacher/corpus/key as arguments.
- `package_piper_voice.sh` per voice (or the ps1 equivalent) →
  `artifacts/release/piper-voices/es/{CL/copihue,CO/vueltiao,CO/chande}/medium/`.

Site wiring: one `VOICES` entry per voice (copihue joins the `chilean` tile;
vueltiao + chande form a new `colombian` tile 🇨🇴), `DEFAULT_TEXT` entries,
and a Spanish–Colombia row in the language table fed by the three new
`experiments/evidence/*-tatoeba-20260913.json` files.

Gotchas added to the wall of pain:

- **Lightning val checkpoints are ~1 GB each** and one lands every val epoch;
  three finetunes filled the boot drive to 0 bytes (and a 0-byte
  `<text>.audio.pt` cache entry then poisons every later run of that voice —
  `PytorchStreamReader failed reading zip archive` in `prepare_data`; delete
  the voice's cache dir to rebuild). `export_teacher.sh` now trims all but
  the best val_mos ckpt + last.ckpt, and a detached `disk_guard.sh` prunes
  val_mel ckpts + all-but-top-3 val_mos during training.
- **A checkpoint-pruning guard must exempt files younger than ~10 min** —
  `fsspec`'s atomic save (move → copystat) races any deleter, and the run
  dies with `FileNotFoundError` in `copystat`.
- **wsl.exe-hosted background processes die with their console** — long jobs
  must be launched with `setsid nohup ... & disown` *inside* WSL (and the
  launcher must `sleep` a few seconds so the new session exists before
  wsl.exe exits). If the WSL VM itself wedges (`HCS_E_CONNECTION_TIMEOUT`),
  only an elevated `Stop-Process -Name vmmemWSL` / service restart or a
  reboot clears it; everything on /mnt/c survives and the master resumes
  from done-files.
- **`pkill -f` matches its own launcher's command line** — use
  `ps | grep -F | awk | xargs kill` instead.
