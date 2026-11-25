# Note for Andrew — 25 November 2026

From Charles (written with Claude Code). Summarises today's changes to `main`, the decisions behind them, and the items that need your call.

## Where the project stands

- **`python -m src.run_pipeline --input <raster>` runs end to end:** chip → filter → Florence-2 → post-process → remove duplicates → GeoPackage. With `--ground-truth`, it also scores the result.
- **All testing so far is on the synthetic noise image** (`synthetic_vantor_targets.tif`). That shows the pipeline runs and how fast, but not how accurate it is. On noise, Florence-2 only ever outlines whole chips. Post-processing now removes those, so the output is empty: no false positives, but none of the 7 injected objects found either.
- **The next real milestone is a real scene with known objects.** It's needed to measure recall, to confirm the 768 px chip size, and to tune any thresholds.
- **Still written but not connected:**
  - Steps 1–2: archive indexing, incident buffers, SAR, resampling.
  - The ensemble, calibration and verification parts of steps 5–6.
  - The spatial overlay.
- **Grounding DINO is still blocked** by the `transformers==4.38` pin.
- **Flowchart:** `docs/object_detection_workflow.png` shows the full design, with each module marked as running, not wired in, or blocked.

## When you pull

1. **Chips are now 768 px (stride 576) by default**, not 512. Use `--tile-size 512` to go back.
2. **`run_detection.py` with no arguments uses the chipper's latest manifest**, not every `.tif` in `data/interim`.
   - Use `--all-files` for the old behaviour.
   - Use `--no-filter` to skip the tile filter.
3. **Florence-2 result JSONs have real fields again.** The fields you commented out in `florence2_adapter.py` are restored.
   - `confidence` is no longer a fixed 0.85. It is now Florence-2's own per-box score: the average certainty of the box's 4 edge positions.
   - That measures how precisely a box is placed, not whether the object is there.
   - If you were changing the record format on purpose, tell us and we'll build on that.
4. **If nothing survives post-processing, no GeoPackage is written**, and any earlier one for that raster is deleted.
5. **If `git pull` complains about `detection_run.log`**, run `git restore detection_run.log` first. The log is no longer tracked.

## Decisions made today

| Area | Decision | Why |
|---|---|---|
| Florence-2 output | Restored box fields; confidence = per-box score (`avg_loc_scores`) | Empty records broke every run; a fixed score made thresholds, duplicate removal and ranking meaningless |
| Empty chips | Filter runs inside `BatchInferenceRunner` before any model call; chipper skips windows < 5 % valid | Every entry point now skips empty, cloudy or featureless chips |
| Brightness | Contrast stretch computed once per scene; real pixels written 1–255, NoData always 0 | Same ground looks identical in neighbouring chips; dark real pixels are no longer counted as NoData |
| Chip size | **768 px default** (Florence-2's input size) | 2.5× fewer chips, same time per chip. Whole test image: 16 min → 3 min 13 s on the laptop |
| Multiple targets | Each chip's image is read once and reused for every target | Reading the image is ~83 % of a model call. 3 targets: 47 s → 30 s per chip, identical output |
| Batching | Added (`FLORENCE2_BATCH_SIZE`, `--batch-size`), default 1 | Identical results; no gain on the laptop CPU; expected to help on a GPU |
| Beam count | Kept at 3 (`FLORENCE2_NUM_BEAMS`, `--num-beams`) | 1 beam was no faster on CPU, with the same boxes |
| Model loading | Not optimised | 6–23 s per run, about 1 % of a full run |
| Post-processing | `VLMPostProcessor` runs on each chip's results in `run_pipeline` | Applies catalogue score thresholds; drops boxes covering > 90 % of the chip (`MAX_BOX_CHIP_FRACTION`), < 10 px² or > 10:1 |
| Duplicates | `VectorWriter` removes duplicates on boxes *before* converting tents, vehicles and aircraft to points; also catches boxes cut off at a chip edge | Previously, duplicate points never overlapped, so they were all kept; a cut-off box and the whole object could both survive |
| Evaluation | Scores the exported objects, per catalogue class plus a location-only row | Previously scored pre-deduplication boxes with class ignored |
| Scrapped | Florence-2-large; native ARM64 Python on the laptop; thread tuning; deleting the duplicate weights file | Not worth the effort now, or too uncertain |

Numbers and test details: `docs/performance_baseline.md`. Full change log: section 13 of `docs/CODEBASE_OVERVIEW.md`.

## Open items for your decision

1. **Target profile names (A3).**
   - `TARGET_PROMPTS["disaster_response"]` is now just `["building"]`. That isn't one of the catalogue's 8 classes, so Florence-2 gets the raw word, with no synonyms and the default 0.25 threshold.
   - For evaluation, `building` is matched to `damaged_buildings`.
   - Using catalogue keys (`damaged_buildings`, `tents`, …) gets the tuned prompts and thresholds.
   - Was `building` deliberate (e.g. for testing), or should the profiles use catalogue keys again?
2. **A check that the object really exists (A7, potential).**
   - Today's confidence only measures how precisely a box is placed.
   - The real existence check could be either:
     - the `ZeroShotVerifier`, which needs example image crops per class in `models/prototypes/<class>/`; or
     - a Florence-2 + Grounding DINO ensemble, which needs the `transformers` upgrade below.
   - Which direction, if either?
3. **Grounding DINO / `transformers` upgrade** (parked earlier until you weigh in). DINO needs `transformers >= 4.40`, but the environment pins 4.38.
   - Was 4.38 pinned for a reason?
   - If not, we'd test the upgrade in a copy of the environment first, then update the offline wheels.
4. **GPU runs (E3).** Could you run the pipeline on your GPU machine with `--batch-size 4` to `8` and compare the time per chip? Batching should pay off there. It needs a GPU build of PyTorch; this laptop is CPU only.
5. **Chip only around incidents (E4).**
   - Connecting steps 1–2 (incident point → 1 km buffer → pixel window) would mean chipping only the area of interest instead of whole scenes. For large scenes that's roughly 10–100× less work.
   - It's a medium-to-large piece of work, and it depends on how you want incidents and the archive to feed in.
   - Should it be next?
6. **768 px chips and small objects.** At 768 px, objects are no longer enlarged 1.5× before the model sees them, so very small objects (single tents, vehicles) may be missed more often.
   - Can you help get a real scene with known objects, so we can compare 512 and 768 px?
   - `--tile-size 512` remains available either way.
7. **Smaller things to be aware of:**
   - **mAP in `eval_metrics.py`** is a simplified estimate (precision × recall), not true average precision.
   - **`nms_handler.py`'s edge rule** assumes non-overlapping chips. The working version of that idea now lives in `vector_writer.py`.
   - **Duplicate modules:** `chipper.py` and `parallel_chipper.py` write different manifest formats, and there are two `CoordinateTransformer` classes. Which should stay?

## Suggested next steps

1. Get a real scene with known objects into `data/raw/`, then run `run_pipeline --ground-truth` at 768 and 512 px.
2. Decide items 1–3 above, then choose between the verifier and the ensemble.
3. Test on the GPU (item 4), then decide whether to connect steps 1–2 (item 5).
