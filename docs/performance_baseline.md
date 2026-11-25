# Performance baseline

Reference timings for judging speed changes (chip size, batching, beam count, model loading).
Re-run the same measurement after each change and add a column or row.

## Setup

- **Machine:** Windows 11 on ARM (Snapdragon X), CPU only, `torch 2.14.1+cpu` under x86 emulation.
- **Model:** Florence-2-base, `<OPEN_VOCABULARY_DETECTION>`, `num_beams=3`.
- **Targets:** `disaster_response` profile, one target: `building`.
- **Chips:** 512 x 512 px, stride 384.
- **Image:** `data/raw/synthetic_vantor_targets.tif`, 4000 x 4000 px synthetic noise with 7 injected objects. It can show speed only: Florence-2 draws one box around each whole chip, so detection quality is not measured here.

## 2026-11-25 baseline (before item 2/3 changes)

`python -m src.run_pipeline --input synthetic_vantor_targets.tif --ground-truth synthetic_ground_truth.json`

| Measure | Value |
|---|---|
| Chips created | 121 |
| Chips kept by filter | 121 (0 rejected) |
| Chipping | 3.6 s |
| Filtering | 2.7 s |
| Model load | 11.1 s (6–8 s when the weights are already in the OS file cache) |
| Detection | 955 s (15 min 55 s) |
| **Seconds per chip per target** | **7.9** |
| Total | 976 s (16 min 16 s) |
| Detections | 121, all whole-chip boxes |
| Confidence (min / median / max) | 0.907 / 0.994 / 0.996 |
| Exported to GeoPackage | 100 (duplicate removal across overlapping chips) |

**Where the time goes:** detection is 98 % of the run. Model loading is about 1 %, so preloading weights is low priority; fewer chips, fewer beams and batching are where the gains are.

## 2026-11-25 after skipping unusable chips (items 2 and 3)

Test image with NoData corners (same scene, 68 % of pixels inside a diamond footprint):

| Measure | Value |
|---|---|
| Chips created | 100 (21 fully empty windows skipped by the chipper) |
| Chips rejected by filter before the model | 16 (`low_valid_pixels`) |
| Chips sent to the model | 84 |
| Detection | 12 min 58 s total for `run_detection.py`, 8.3 s per chip |

Before these changes, `run_detection.py` without `--manifest` ran on every `.tif` in `data/interim`: these 100 chips plus any left over from other images (121 here), about 221 chips or roughly 29 minutes.

The re-run of 10 chips on the baseline image gave identical boxes and confidences to the baseline (7.4 s per chip).

## 2026-11-25 speed experiments (items 4–7)

**Run-to-run noise:** the same 10 chips with the same settings took 6.0–9.2 s per chip across runs on this laptop. Differences under about 15 % are not meaningful here; compare runs made back to back.

Same 10 chips (512 px) unless noted:

| Setting | s per chip | Results vs baseline | Verdict |
|---|---|---|---|
| Reference: 3 beams, batch 1 | 8.1 | identical | — |
| `--num-beams 1` | 8.0 | same boxes; confidence differs in the 5th decimal | No gain on CPU: image encoding dominates and outputs are short (one box). May help on real imagery with many objects per chip, where outputs are longer. |
| `--batch-size 2` | 7.9 | identical | No gain on CPU |
| `--batch-size 5` | 7.5 | identical | ~8 %, within noise on CPU. Expected to help on a GPU; worth trying 4–8 there. |
| `--tile-size 768` (stride 576), 10 chips back to back with 512 | 6.7 vs 6.0 | different chips | Same cost per chip, because Florence-2 resizes every chip to 768 x 768 either way |
| `--tile-size 768`, whole image | 49 chips, 202 s detection (3 min 38 s total) | different chips | **2.5x fewer chips** than 512 px (121), so roughly 2.5x faster per image. Each object is seen at 1:1 instead of enlarged 1.5x, which may cost recall on very small objects: check on real imagery before making it the default. |

**Model load (item 7):** 6–12 s across eight runs, under 1 % of a whole-image run. Preloading weights or loading once for several images is not worth building yet.

**Defaults after these experiments:** 3 beams and batch 1 unchanged; chip size changed to 768 px (next section). All three are now options on `run_pipeline` (`--tile-size`, `--stride`, `--num-beams`, `--batch-size`) and settings in `config.py` (`DEFAULT_TILE_SIZE`, `FLORENCE2_NUM_BEAMS`, `FLORENCE2_BATCH_SIZE`).

## 2026-11-25 new defaults (768 px chips, image read once per chip, post-processing)

**Reading each chip's image once for all targets** (3 chips, 3 catalogue targets, same session): 47.0 s per chip before, 30.2 s after, with identical outputs. The saving grows with each extra target; it is smaller than the 83 % encoder share suggests because the longer catalogue prompts take more time to write out.

**Whole image, new defaults** (`python -m src.run_pipeline --input synthetic_vantor_targets.tif --ground-truth synthetic_ground_truth.json`, one target `building`):

| Measure | 512 px baseline | 768 px, new defaults |
|---|---|---|
| Chips | 121 | 49 |
| Detection time | 955 s | 164 s |
| Total time | 16 min 16 s | 3 min 13 s |
| Boxes from the model | 121 (all whole-chip) | 49 (all whole-chip) |
| Removed by post-processing | — | 49 |
| Exported false positives | 100 | 0 |
| Objects found (of 7) | 0 | 0 |

Post-processing now removes the whole-chip outlines Florence-2 draws on this noise image. Finding none of the 7 injected objects is expected on random noise and says nothing about real imagery; recall must be measured on real scenes with ground truth.
