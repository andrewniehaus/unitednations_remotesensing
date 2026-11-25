"""
Generates docs/object_detection_workflow.png's HTML source. Edit STEPS below to update the chart.

Render (Windows, from the repository root):
    python docs/gen_workflow_chart.py workflow.html
    msedge --headless=new --hide-scrollbars --force-device-scale-factor=2.37 --window-size=1520,2700 --screenshot=chart.png workflow.html
Then crop the blank space below the notes.
"""
import html
import sys
from pathlib import Path

RUN, NOT_WIRED, BLOCKED, HUMAN = "run", "notwired", "blocked", "human"
BADGES = {
    RUN: ("RUNS IN PIPELINE", "b-run"),
    NOT_WIRED: ("WRITTEN, NOT WIRED IN", "b-nw"),
    BLOCKED: ("BLOCKED", "b-blk"),
    HUMAN: ("HUMAN IN THE LOOP", "b-hum"),
}

INPUTS = [
    ("data/raw/", ["Very-high-resolution GeoTIFFs: Maxar/Vantor, Airbus PlÃ©iades/SPOT (optical), Capella (SAR radar)"]),
    ("Incident reports", ["Points (longitude/latitude + date), e.g. from Liveuamap or manual reports"]),
    ("models/", ["Local weights: Florence-2 (in use), Grounding DINO, ResNet-50 + prototype images per class"]),
    ("src/config.py", ["Paths, CPU/GPU, 512 px chips every 384 px, 3 beams, batch size 1, target profiles"]),
]

# (number, colour, title, question, explanation, modules, passes_on)
STEPS = [
    (1, "blue", "Find the right imagery", "Which images show this place at this time?",
     "Builds a catalogue of every scene, then picks the scenes and pixel areas that cover each incident.",
     [("ingestion/local_archive_indexer.py", "LocalArchiveIndexer", NOT_WIRED,
       ["Reads each scene's footprint (WGS84), size, resolution, CRS",
        "Date from sidecar â†’ TIFF tag â†’ filename; vendor â†’ optical / SAR",
        "Catalogue â†’ GeoJSON / SQLite; <i>query_buffer()</i> by area, time window, sensor type"]),
      ("ingestion/spatial_buffer.py", "SpatialBufferEngine", NOT_WIRED,
       ["Each incident point â†’ square buffer (default 1 km, in local UTM metres)",
        "Intersects buffers with scene footprints",
        "Outputs the exact pixel window to cut from each matching scene"])],
     "Selected GeoTIFFs + pixel windows (today: one raster passed with <code>--input</code>)"),
    (2, "blue", "Standardise the imagery", "Make every image look alike to the model.",
     "Different satellites differ in resolution, bands and brightness. Evening these out means a tent looks the same size and colour to the model, whichever satellite took the picture.",
     [("ingestion/sar_processor.py", "SARProcessor (radar only)", NOT_WIRED,
       ["Raw values â†’ backscatter in dB (âˆ’30 â€¦ +5)", "Speckle removed (median / Gaussian filter)",
        "Dual-pol ratio band: tents vs buildings"]),
      ("tiling/gsd_resampler.py", "GSDResampler", NOT_WIRED,
       ["Resamples to one ground resolution: 0.5 m per pixel", "Skips scenes already within 5 %"]),
      ("tiling/band_selector.py Â· normalizer.py", "BandSelector Â· ArrayNormalizer", RUN,
       ["4-band, 8-band or SAR â†’ 3-channel 8-bit RGB (what the models expect)",
        "2â€“98 % contrast stretch per chip, <b>NoData left out of the stretch</b>",
        "Normalizer runs inside the chipper; BandSelector not wired"])],
     "Uniform 8-bit RGB rasters"),
    (3, "blue", "Cut into chips & drop empty ones", "The models see small squares, not whole scenes.",
     "Scenes are far too large for the model, so they are cut into overlapping chips. Each chip keeps an \"address\" so its results can be put back on the map. Blank, cloudy or featureless chips are skipped before the model runs.",
     [("tiling/chipper.py", "RasterChipper", RUN,
       ["512 Ã— 512 px windows every 384 px (128 px overlap); last row/column flush to the edge",
        "Windows under 5 % valid pixels are not written",
        "Writes <b>tile_manifest.json</b>: each chip's pixel window and geotransform",
        "Re-chipping clears that raster's old chips and results"]),
      ("tiling/adaptive_overlap.py", "AdaptiveOverlapCalculator", NOT_WIRED,
       ["Overlap set by target shape: trenches 50 %, tent camps 35 %, vehicles 20 %",
        "So objects aren't cut in half at chip seams"]),
      ("tiling/spatial_filter.py", "SpatialTileFilter", RUN,
       ["Drops chips with &lt; 50 % valid pixels, &gt; 60 % cloud, or entropy &lt; 3.5 (empty desert, water)",
        "Runs before every model call (run_pipeline, run_detection)",
        "Reasons logged to tile_rejection_manifest.json"])],
     "Usable 512 px RGB chips + chip manifest (the map \"address book\")"),
    (4, "purple", "Detect & label objects", "Where are the tents, vehicles, trenches â€¦?",
     "A vision-language model is asked in plain English to find each target type. It returns a box, a label and a confidence score for everything it finds, with no task-specific training needed.",
     [("detection/prompt_manager.py", "PromptManager Â· 8 target classes", RUN,
       ["tents Â· trenchlines Â· damaged_buildings Â· aircraft Â· military_compounds Â· guard_towers Â· military_vehicles Â· emergency_assets",
        "Each: prompt + synonyms + own score threshold (0.20â€“0.35)",
        "Profile names not in the catalogue are used as raw prompts"]),
      ("detection/florence2_adapter.py", "Florence2Adapter", RUN,
       ["Task &lt;OPEN_VOCABULARY_DETECTION&gt;, one query per target",
        "<b>Confidence</b> = how sure the model is of the box's 4 edge positions (not whether the object exists)",
        "Chips with the same target batched per call (FLORENCE2_BATCH_SIZE)"]),
      ("detection/grounding_dino_adapter.py", "GroundingDINOAdapter", BLOCKED,
       ["Prompt \"tent . refugee tent . â€¦\"", "Returns boxes with real confidence scores, filtered per class",
        "âš  Needs transformers â‰¥ 4.40; environment pins 4.38. Deferred"]),
      ("detection/batch_runner.py", "BatchInferenceRunner", RUN,
       ["Filters chips first, then runs the model on the rest",
        "One JSON per chip â†’ data/interim/vlm_results/",
        "Resumable: skips chips that already have results"])],
     "Per chip: [ {bbox, label, confidence} ] in chip pixel coordinates"),
    (5, "orange", "Clean up & combine", "Turn raw guesses from two models into one list.",
     "Labels are standardised, implausible boxes are removed, and overlapping answers from both models are merged. Agreement between the models raises the score. Duplicates from overlapping chips are removed.",
     [("detection/post_processor.py", "VLMPostProcessor", NOT_WIRED,
       ["Free-text labels â†’ standard class names", "Per-class score cut-off (default 0.30)",
        "Removes boxes outside 10â€“250 000 pxÂ² or aspect &gt; 10:1 (would remove whole-chip boxes)"]),
      ("detection/ensemble_detector.py", "EnsembleDetector Â· Weighted Box Fusion", NOT_WIRED,
       ["Merges boxes that overlap â‰¥ 50 % (IoU)", "Model weights: Grounding DINO 1.2, Florence-2 1.0",
        "+0.15 score when both models agree"]),
      ("tiling/nms_handler.py", "TiledNMSHandler", NOT_WIRED,
       ["Removes duplicates where chips overlapped (IoU 0.45, optional Soft-NMS)",
        "Boxes touching a chip edge scored Ã— 0.70, so whole objects win"])],
     "One de-duplicated detection table: box, class, fused score, agreeing models"),
    (6, "orange", "Verify the classification", "Is it really what the model says it is?",
     "Scores are turned into real probabilities. Each object is checked against example images of its class and against common-sense geographic rules. Uncertain cases go to a person.",
     [("detection/confidence_calibrator.py", "ConfidenceCalibrator", NOT_WIRED,
       ["Platt scaling: score â†’ probability (fitted on validation data)",
        "Up to 30 % discount in suspiciously dense clusters (100 m)"]),
      ("detection/zero_shot_verifier.py", "ZeroShotVerifier", NOT_WIRED,
       ["Crops each box â†’ ResNet-50 embedding", "Compared with example images in models/prototypes/&lt;class&gt;/",
        "Final = 0.5 Ã— model score + 0.5 Ã— similarity"]),
      ("detection/spatial_heuristics.py", "SpatialHeuristicsEngine", NOT_WIRED,
       ["No land objects on water", "Related targets near an anchor (e.g. vehicles near compounds)",
        "Trench segments must connect; realistic size in mÂ²"]),
      ("detection/hard_negative_miner.py", "HardNegativeMiner Â· human in the loop", HUMAN,
       ["Uncertain (0.25â€“0.40) and isolated (outlier) detections",
        "Crops + HTML gallery for an analyst to confirm or reject",
        "â†º Feeds better prototypes, thresholds and future training data back into steps 4â€“6"])],
     "Verified objects: class + calibrated probability (still in pixel coordinates)"),
    (7, "green", "Place on the map & export", "Where on Earth is each object?",
     "Each chip's address in the manifest turns pixel boxes into real-world coordinates. Countable objects become points; infrastructure stays as outlines.",
     [("georeferencing/vector_writer.py", "VectorWriter", RUN,
       ["Chip geotransform (from manifest) Ã— pixel box â†’ map polygon",
        "Tents, aircraft, vehicles â†’ points; trenches, compounds â†’ polygons",
        "Removes duplicates across overlapping chips (IoU 0.4), keeping the higher score",
        "Adds prompt used, acquisition date and source scene"]),
      ("georeferencing/crs_manager.py Â· coord_transformer.py", "CRSManager Â· CoordinateTransformer", NOT_WIRED,
       ["Pixel â†’ map coordinates via affine transform",
        "Reprojects everything to one consistent CRS (EPSG:4326 fallback)"]),
      ("tiling/rebuilder.py", "RasterRebuilder (optional)", NOT_WIRED,
       ["Stitches chip results back into one GeoTIFF or GeoPackage for whole-scene viewing"])],
     "data/processed/&lt;raster&gt;_detections.gpkg"),
    (8, "green", "Add context & measure accuracy", "Who is affected, and how good are the results?",
     "Detections are linked to buildings and population to show human impact, then scored against known correct answers.",
     [("ingestion/spatial_overlay.py", "SpatialOverlayEngine", NOT_WIRED,
       ["Optionally keeps only detections touching a building footprint (removes non-structural false positives)",
        "Sums WorldPop population inside each object â†’ people exposed"]),
      ("Testing/eval_metrics.py", "SpatialEvalMetrics", RUN,
       ["One-to-one matching with ground truth (Hungarian algorithm)",
        "Precision, recall, F1, mAP@50 and mAP@50:95",
        "Run with <code>--ground-truth</code>; currently location only, class ignored"])],
     None),
]


def module_card(path, cls, status, bullets, colour):
    badge_text, badge_cls = BADGES[status]
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return (f'<div class="card {status}"><div class="path {colour}">{html.escape(path)}</div>'
            f'<div class="cls">{html.escape(cls)}</div><span class="badge {badge_cls}">{badge_text}</span>'
            f'<ul>{items}</ul></div>')


def build() -> str:
    rows = []
    inputs = "".join(
        f'<div class="card input"><div class="mono">{html.escape(t)}</div><ul>{"".join(f"<li>{b}</li>" for b in bs)}</ul></div>'
        for t, bs in INPUTS)
    rows.append(f'<div class="row"><div class="concept start"><h2>START Â· Inputs</h2>'
                f'<p>Everything is held locally; nothing is downloaded while running.</p></div>'
                f'<div class="mods" style="--n:4">{inputs}</div></div>')
    rows.append('<div class="row pass"><div class="arrow"></div><div><span class="pill"><b>PASSES ON</b> Raw scenes + incident points</span></div></div>')

    for num, colour, title, question, text, modules, passes in STEPS:
        cards = "".join(module_card(p, c, s, b, colour) for p, c, s, b in modules)
        rows.append(f'<div class="row"><div class="concept {colour}"><div class="head"><span class="num {colour}">{num}</span>'
                    f'<h2>{title}</h2></div><p class="q">"{question}"</p><p>{text}</p></div>'
                    f'<div class="mods" style="--n:{len(modules)}">{cards}</div></div>')
        if passes:
            rows.append(f'<div class="row pass"><div class="arrow"></div><div><span class="pill"><b>PASSES ON</b> {passes}</span></div></div>')
    rows.append('<div class="row pass"><div class="arrow"></div><div></div></div>')

    rows.append('<div class="row"><div class="concept end"><h2>END Â· Classified object map</h2>'
                '<p>Ready for analysts in QGIS / ArcGIS</p></div>'
                '<div class="card endcard"><div class="mono green">Each object in the GeoPackage carries</div><ul>'
                '<li><b>What:</b> class (e.g. tent, military vehicle, damaged building) + confidence</li>'
                '<li><b>Where:</b> point or polygon in real-world coordinates</li>'
                '<li><b>When &amp; from what:</b> acquisition date, source scene, prompt used</li>'
                '<li><b>Impact:</b> estimated people exposed Â· <b>Quality:</b> precision / recall (planned)</li>'
                '</ul></div></div>')

    return TEMPLATE.replace("{{ROWS}}", "\n".join(rows))


TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Object Detection Workflow</title>
<style>
:root { --ink:#1f2937; --muted:#5b6472; --line:#c9cfd8; --bg:#f6f8fb;
  --blue:#2f6db3; --blue-bg:#e8f0fa; --purple:#7a3fb5; --purple-bg:#f1eafa;
  --orange:#c0661c; --orange-bg:#fcf0e3; --green:#2e8b57; --green-bg:#e6f3ea; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:"Segoe UI", system-ui, sans-serif; font-size:13px; line-height:1.45; }
.page { width:1520px; padding:36px 46px 30px; }
h1 { font-size:31px; margin:0 0 6px; letter-spacing:-0.3px; }
.sub { color:var(--muted); font-size:14.5px; margin:0 0 10px; }
.run-strip { background:#fff; border:1px solid var(--line); border-radius:8px; padding:9px 14px; margin:0 0 12px; font-size:13px; }
.run-strip code { font-family:Consolas, monospace; background:#eef1f5; padding:1px 5px; border-radius:4px; }
.legend { display:flex; flex-wrap:wrap; gap:18px; color:var(--muted); font-size:12.5px; margin-bottom:22px; align-items:center; }
.legend i { display:inline-block; width:13px; height:13px; border-radius:3px; margin-right:6px; vertical-align:-2px; }
.cols { display:grid; grid-template-columns:362px 1fr; gap:22px; font-size:11.5px; letter-spacing:1px; color:var(--muted); font-weight:600; margin-bottom:8px; }
.row { display:grid; grid-template-columns:362px 1fr; gap:22px; }
.concept { border-radius:9px; padding:16px 18px; }
.concept h2 { margin:0; font-size:19px; }
.concept p { margin:6px 0 0; }
.concept .q { font-style:italic; color:var(--muted); }
.head { display:flex; align-items:center; gap:12px; }
.num { width:32px; height:32px; border-radius:50%; color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:15px; flex:none; }
.concept.blue { background:var(--blue-bg); } .num.blue { background:var(--blue); }
.concept.purple { background:var(--purple-bg); } .num.purple { background:var(--purple); }
.concept.orange { background:var(--orange-bg); } .num.orange { background:var(--orange); }
.concept.green { background:var(--green-bg); } .num.green { background:var(--green); }
.concept.start { background:#243041; color:#fff; } .concept.end { background:var(--green); color:#fff; }
.mods { display:grid; grid-template-columns:repeat(var(--n), 1fr); gap:12px; }
.card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:11px 13px; }
.card.notwired { background:#fbfbfc; border-style:dashed; }
.card.notwired ul { color:#6b7280; }
.card.blocked { border-color:#d14343; background:#fff7f7; }
.card.human { border-style:dashed; border-color:#d9b98f; background:#fffaf2; }
.path { font-family:Consolas, monospace; font-weight:700; font-size:12.5px; }
.path.blue { color:var(--blue); } .path.purple { color:var(--purple); } .path.orange { color:var(--orange); } .path.green { color:var(--green); }
.mono { font-family:Consolas, monospace; font-weight:700; font-size:12.5px; }
.mono.green { color:var(--green); }
.cls { color:var(--muted); font-size:12px; }
.badge { display:inline-block; font-size:9.5px; font-weight:700; letter-spacing:0.6px; padding:1px 7px; border-radius:10px; margin:5px 0 1px; }
.b-run { background:#dcf1e3; color:#1e6b40; }
.b-nw { background:#eceef2; color:#5b6472; }
.b-blk { background:#fbdada; color:#a12828; }
.b-hum { background:#f8e8d2; color:#8a5a1c; }
ul { margin:6px 0 0; padding-left:16px; }
li { margin:2px 0; }
li::marker { color:#9aa3af; }
.pass { min-height:40px; align-items:center; }
.arrow { height:40px; margin-left:34px; border-left:2px solid #7d8794; position:relative; }
.arrow::after { content:""; position:absolute; left:-7px; bottom:-2px; border:6px solid transparent; border-top:8px solid #7d8794; }
.pill { display:inline-block; border:1px solid #9aa3af; border-radius:14px; padding:3px 12px; font-size:12px; background:#fff; }
.pill b { font-size:10px; letter-spacing:1px; color:var(--muted); margin-right:5px; }
.pill code { font-family:Consolas, monospace; }
.endcard { border:2px solid var(--green); }
.notes { border-top:1px solid var(--line); margin-top:26px; padding-top:12px; color:var(--muted); font-size:12px; }
.notes b { color:var(--ink); }
</style></head>
<body><div class="page">
<h1>Satellite Object Detection &amp; Classification Workflow</h1>
<p class="sub">unitednations_remotesensing Â· offline ("air-gapped") pipeline: from raw satellite scenes to a classified, map-ready layer of objects. Shows the full design in <code>src/</code> and which parts run today. Updated 25 Nov 2026.</p>
<div class="run-strip"><b>Run it today:</b> <code>python -m src.run_pipeline --input &lt;raster&gt; [--profile disaster_response] [--ground-truth gt.json] [--tile-size 768] [--batch-size N]</code> runs steps 3 â†’ 4 â†’ 7 (â†’ 8 when ground truth is given). Steps 1â€“2, 5â€“6 and the overlay in step 8 are written but not yet connected.</div>
<div class="legend">
<span><i style="background:var(--blue)"></i>Prepare the imagery</span>
<span><i style="background:var(--purple)"></i>Detect objects with AI models</span>
<span><i style="background:var(--orange)"></i>Refine &amp; verify</span>
<span><i style="background:var(--green)"></i>Deliver results</span>
<span><span class="badge b-run">RUNS IN PIPELINE</span></span>
<span><span class="badge b-nw">WRITTEN, NOT WIRED IN</span></span>
<span><span class="badge b-blk">BLOCKED</span></span>
<span><span class="badge b-hum">HUMAN IN THE LOOP</span></span>
</div>
<div class="cols"><div>CONCEPTUALLY â€” WHAT &amp; WHY</div><div>TECHNICALLY â€” MODULE Â· METHOD Â· KEY SETTINGS</div></div>
{{ROWS}}
<div class="notes"><b>Notes.</b> Tested end to end on synthetic imagery only; real imagery is the next step. On the laptop CPU, Florence-2 takes about 6â€“9 s per chip per target, 83 % of it reading the image; a 4000 Ã— 4000 px scene is 121 chips at 512 px or 49 at 768 px (<code>docs/performance_baseline.md</code>). Because steps 5â€“6 are not wired in, boxes covering a whole chip currently reach the output. Confidence measures how precisely a box is placed, not whether the object exists. <b>Known gaps:</b> Grounding DINO needs a newer <i>transformers</i>; <i>chipper.py</i> and <i>parallel_chipper.py</i> write different manifest formats; there are two <i>CoordinateTransformer</i> classes. <b>Not shown:</b> a parallel change-detection track (<i>src/change_detection/</i>) that compares before and after images using Clay foundation-model embeddings.</div>
</div></body></html>
"""

if __name__ == "__main__":
    Path(sys.argv[1]).write_text(build(), encoding="utf-8")
    print(f"wrote {sys.argv[1]}")

