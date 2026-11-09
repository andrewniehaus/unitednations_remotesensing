"""
End-to-end offline object detection run for a single raster.

Tiles the raster, drops unusable tiles, runs the selected VLM, and exports a georeferenced
GeoPackage to config.PROCESSED_DIR, optionally scoring it against ground truth.
The detection refinement modules (post-processing, ensembling, verification) are not run yet.

Run from the repository root:
    python -m src.run_pipeline --input synthetic_vantor_targets.tif --profile military_infrastructure
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src import config
from src.detection.batch_runner import BatchInferenceRunner
from src.detection.run_detection import load_adapter
from src.georeferencing.vector_writer import VectorWriter
from src.Testing.eval_metrics import SpatialEvalMetrics
from src.tiling.chipper import RasterChipper
from src.tiling.spatial_filter import SpatialTileFilter

logger = logging.getLogger(__name__)

FILTERED_MANIFEST_FILENAME = "tile_manifest_filtered.json"


def resolve_input(input_arg: str) -> Path:
    """Finds the input raster as given, or by name inside config.RAW_DIR."""
    input_path = Path(input_arg)
    if not input_path.exists():
        input_path = config.RAW_DIR / input_arg
    if not input_path.exists():
        logger.critical(f"Input raster not found: {input_arg} (also looked in {config.RAW_DIR})")
        sys.exit(1)
    return input_path.resolve()


def filter_tiles(manifest: Dict[str, Any], max_tiles: int = 0) -> Path:
    """Evaluates every manifest tile and writes a manifest holding only the usable ones."""
    tiles = sorted(manifest["tiles"], key=lambda t: (t["row_idx"], t["col_idx"]))
    tile_filter = SpatialTileFilter()

    with ThreadPoolExecutor() as executor:
        outcomes = list(executor.map(
            lambda tile: tile_filter.evaluate_tile(config.INTERIM_DIR / tile["filename"]), tiles
        ))

    kept = [tile for tile, (is_valid, _) in zip(tiles, outcomes) if is_valid]
    rejected = {tile["filename"]: reason for tile, (is_valid, reason) in zip(tiles, outcomes) if not is_valid}

    with open(config.PROCESSED_DIR / "tile_rejection_manifest.json", "w", encoding="utf-8") as f:
        json.dump(rejected, f, indent=4)
    logger.info(f"Tile filter: {len(kept)} kept, {len(rejected)} rejected.")

    if not kept:
        logger.critical("No tiles passed the filter. See tile_rejection_manifest.json.")
        sys.exit(1)

    if max_tiles:
        kept = kept[:max_tiles]
        logger.warning(f"--max-tiles set: detecting on the first {len(kept)} tiles only.")

    filtered_path = config.INTERIM_DIR / FILTERED_MANIFEST_FILENAME
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(dict(manifest, tiles=kept, total_tiles=len(kept)), f, indent=4)
    return filtered_path


def collect_detections(
    writer: VectorWriter, tiles: List[Dict[str, Any]], results_dir: Path
) -> Tuple[List[Dict[str, Any]], int]:
    """Loads per-tile VLM results; returns georeferencing-ready detections and the missing tile count."""
    detections: List[Dict[str, Any]] = []
    missing = 0
    for tile in tiles:
        result_path = results_dir / Path(tile["filename"]).with_suffix(".json")
        if not result_path.exists():
            missing += 1
            continue
        with open(result_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        detections += writer.parse_inference_outputs(
            tile["filename"],
            [r["bbox"] for r in results],
            [r["label"] for r in results],
            [r["confidence"] for r in results],
        )
    return detections, missing


def evaluate(
    writer: VectorWriter, detections: List[Dict[str, Any]], ground_truth_path: Path
) -> pd.DataFrame:
    """Scores detections against ground truth using each object's location only, ignoring class."""
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    columns = ["label", "xmin", "ymin", "xmax", "ymax"]
    gt_df = pd.DataFrame(
        [("object", g["geo_bbox_utm"][0], g["geo_bbox_utm"][3], g["geo_bbox_utm"][2], g["geo_bbox_utm"][1])
         for g in ground_truth],
        columns=columns,
    )

    rows = []
    for det in detections:
        xmin, ymin, xmax, ymax = writer._pixel_bbox_to_geo_polygon(det["local_bbox"], det["transform"]).bounds
        rows.append(("object", xmin, ymin, xmax, ymax, det["confidence"]))
    pred_df = pd.DataFrame(rows, columns=columns + ["confidence"])

    return SpatialEvalMetrics().evaluate_pipeline_output(gt_df, pred_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline object detection on one raster, end to end.")
    parser.add_argument("--input", required=True, help="GeoTIFF path, or a file name inside data/raw.")
    parser.add_argument("--model", choices=["florence2", "groundingdino"], default="florence2")
    parser.add_argument("--profile", choices=sorted(config.TARGET_PROMPTS), default="disaster_response",
                        help="Target prompt profile defined in src/config.py.")
    parser.add_argument("--acquisition-date", default="Unknown", help="Stored on every exported detection.")
    parser.add_argument("--ground-truth", default=None,
                        help="Optional ground truth JSON (as written by create_synthetic_tiff) to score against.")
    parser.add_argument("--max-tiles", type=int, default=0, help="Detect on only the first N usable tiles.")
    parser.add_argument("--skip-chip", action="store_true",
                        help="Reuse existing tiles and resume detection. Without this, re-chipping clears earlier results.")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    targets = config.TARGET_PROMPTS[args.profile]
    manifest_path = config.INTERIM_DIR / config.MANIFEST_FILENAME

    if args.skip_chip:
        if not manifest_path.exists():
            logger.critical(f"--skip-chip needs an existing manifest at {manifest_path}.")
            sys.exit(1)
    else:
        logger.info(f"Chipping {input_path.name}...")
        RasterChipper().chip_raster(input_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if Path(manifest["source_raster"]) != input_path:
        logger.critical(f"Manifest belongs to {manifest['source_raster']}, not {input_path}. Re-run without --skip-chip.")
        sys.exit(1)
    logger.info(f"Manifest lists {len(manifest['tiles'])} tiles.")

    filtered_manifest_path = filter_tiles(manifest, args.max_tiles)
    with open(filtered_manifest_path, "r", encoding="utf-8") as f:
        tiles = json.load(f)["tiles"]

    logger.info(f"Running {args.model} with profile '{args.profile}': {targets}")
    results_dir = config.INTERIM_DIR / "vlm_results"
    BatchInferenceRunner(model_adapter=load_adapter(args.model)).run_on_manifest(
        manifest_path=filtered_manifest_path, targets=targets, output_dir=results_dir
    )

    writer = VectorWriter()
    detections, missing = collect_detections(writer, tiles, results_dir)
    if missing == len(tiles):
        logger.critical("Detection produced no results for any tile. See detection_run.log.")
        sys.exit(1)
    if missing:
        logger.warning(f"{missing} of {len(tiles)} tiles have no detection results and are excluded.")
    logger.info(f"Collected {len(detections)} detections.")

    output_path = writer.export_to_vector(
        detections,
        prompt_used=", ".join(targets),
        acquisition_date=args.acquisition_date,
        output_filename=f"{input_path.stem}_detections.gpkg",
    )
    if output_path:
        logger.info(f"Exported detections to {output_path}")

    if args.ground_truth:
        ground_truth_path = Path(args.ground_truth)
        if not ground_truth_path.exists():
            ground_truth_path = config.RAW_DIR / args.ground_truth
        logger.info("Location-only evaluation against ground truth:\n" + evaluate(writer, detections, ground_truth_path).to_string(index=False))

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
