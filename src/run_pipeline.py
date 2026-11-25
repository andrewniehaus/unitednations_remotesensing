"""
End-to-end offline object detection run for a single raster.

Tiles the raster, drops unusable tiles, runs the selected VLM, post-processes each chip's
results (class score thresholds, implausible box sizes), removes duplicates from overlapping
chips, and exports a georeferenced GeoPackage to config.PROCESSED_DIR, optionally scoring it
against ground truth. Ensembling, calibration and verification are not run yet.

Run from the repository root:
    python -m src.run_pipeline --input synthetic_vantor_targets.tif --profile disaster_response
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src import config
from src.detection.batch_runner import BatchInferenceRunner
from src.detection.post_processor import VLMPostProcessor
from src.detection.prompt_manager import PromptManager
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
    kept_paths, rejected = SpatialTileFilter().filter_paths(
        [config.INTERIM_DIR / tile["filename"] for tile in tiles]
    )
    kept_names = {path.name for path in kept_paths}
    kept = [tile for tile in tiles if tile["filename"] in kept_names]
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


def build_post_processor(targets: List[str], prompt_manager: PromptManager) -> VLMPostProcessor:
    """Post-processor using each target's catalogue score threshold (0.25 for targets not in the catalogue)."""
    return VLMPostProcessor(
        class_score_thresh={
            target.lower(): prompt_manager.get_target(target)["box_threshold"] for target in targets
        },
    )


def refine_tile_results(
    results: List[Dict[str, Any]], post_processor: VLMPostProcessor, tile_size: int
) -> List[Dict[str, Any]]:
    """
    Applies the per-class score threshold and drops implausible boxes from one chip's results:
    under 10 px², aspect ratio over 10:1, or covering more than config.MAX_BOX_CHIP_FRACTION
    of the chip (Florence-2 sometimes outlines the whole chip).
    """
    if not results:
        return []
    df = pd.DataFrame(
        [(*r["bbox"], r["label"], r["confidence"]) for r in results],
        columns=["xmin", "ymin", "xmax", "ymax", "label", "confidence"],
    )
    df = post_processor.apply_semantic_mapping(df)
    df = post_processor.filter_by_confidence(df)
    df = post_processor.morphological_filter(df, max_area=config.MAX_BOX_CHIP_FRACTION * tile_size ** 2)
    return [
        {"bbox": [row.xmin, row.ymin, row.xmax, row.ymax], "label": row.label, "confidence": row.confidence}
        for row in df.itertuples(index=False)
    ]


def collect_detections(
    writer: VectorWriter,
    tiles: List[Dict[str, Any]],
    results_dir: Path,
    post_processor: VLMPostProcessor,
    tile_size: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Loads and post-processes per-tile VLM results.

    Returns:
        Georeferencing-ready detections, the count of tiles without results, and the
        count of raw detections removed by post-processing.
    """
    detections: List[Dict[str, Any]] = []
    missing = removed = 0
    for tile in tiles:
        result_path = results_dir / Path(tile["filename"]).with_suffix(".json")
        if not result_path.exists():
            missing += 1
            continue
        with open(result_path, "r", encoding="utf-8") as f:
            raw_results = json.load(f)
        results = refine_tile_results(raw_results, post_processor, tile_size)
        removed += len(raw_results) - len(results)
        detections += writer.parse_inference_outputs(
            tile["filename"],
            [r["bbox"] for r in results],
            [r["label"] for r in results],
            [r["confidence"] for r in results],
        )
    return detections, missing, removed


def evaluate(
    exported: List[Dict[str, Any]], ground_truth_path: Path, prompt_manager: PromptManager
) -> pd.DataFrame:
    """
    Scores the exported detections against ground truth, per class and overall.

    Labels on both sides are mapped to catalogue classes (e.g. 'military tent' -> 'tents');
    labels with no catalogue match are compared as written. The 'ALL (location only)' row
    ignores class.
    """
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    def to_class(label: str) -> str:
        return prompt_manager.match_target(label) or label.lower().strip()

    columns = ["label", "xmin", "ymin", "xmax", "ymax"]
    gt_df = pd.DataFrame(
        [(to_class(g["label"]), g["geo_bbox_utm"][0], g["geo_bbox_utm"][3], g["geo_bbox_utm"][2], g["geo_bbox_utm"][1])
         for g in ground_truth],
        columns=columns,
    )
    pred_df = pd.DataFrame(
        [(to_class(det["label"]), *det["box_geometry"].bounds, det["confidence"]) for det in exported],
        columns=columns + ["confidence"],
    )

    metrics = SpatialEvalMetrics()
    per_class = metrics.evaluate_pipeline_output(gt_df, pred_df)
    overall = metrics.evaluate_pipeline_output(gt_df.assign(label="ALL (location only)"),
                                               pred_df.assign(label="ALL (location only)"))
    return pd.concat([per_class, overall], ignore_index=True)


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
    parser.add_argument("--tile-size", type=int, default=config.DEFAULT_TILE_SIZE,
                        help="Chip width/height in pixels. Florence-2 resizes every chip to 768 x 768.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Pixels between chip starts. Defaults to config.DEFAULT_STRIDE, scaled with --tile-size.")
    parser.add_argument("--num-beams", type=int, default=None,
                        help="Florence-2 beam search width (config.FLORENCE2_NUM_BEAMS). 1 is faster.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Chips per Florence-2 model call (config.FLORENCE2_BATCH_SIZE).")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    targets = config.TARGET_PROMPTS[args.profile]
    manifest_path = config.INTERIM_DIR / config.MANIFEST_FILENAME

    if args.skip_chip:
        if not manifest_path.exists():
            logger.critical(f"--skip-chip needs an existing manifest at {manifest_path}.")
            sys.exit(1)
    else:
        # Keep the configured stride-to-size ratio (75 %, i.e. 25 % overlap) when only the tile size changes
        stride = args.stride or round(args.tile_size * config.DEFAULT_STRIDE / config.DEFAULT_TILE_SIZE)
        logger.info(f"Chipping {input_path.name} into {args.tile_size} px chips, stride {stride}...")
        RasterChipper(tile_size=args.tile_size, stride=stride).chip_raster(input_path)

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
    adapter = load_adapter(args.model)
    for setting in ("num_beams", "batch_size"):
        value = getattr(args, setting)
        if value is not None:
            if hasattr(adapter, setting):
                setattr(adapter, setting, value)
            else:
                logger.warning(f"--{setting.replace('_', '-')} is not supported by {args.model}; ignored.")
    BatchInferenceRunner(model_adapter=adapter).run_on_manifest(
        manifest_path=filtered_manifest_path, targets=targets, output_dir=results_dir,
        filter_tiles=False,  # already filtered above
    )

    writer = VectorWriter()
    prompt_manager = PromptManager()
    detections, missing, removed = collect_detections(
        writer, tiles, results_dir, build_post_processor(targets, prompt_manager), manifest["tile_size"]
    )
    if missing == len(tiles):
        logger.critical("Detection produced no results for any tile. See detection_run.log.")
        sys.exit(1)
    if missing:
        logger.warning(f"{missing} of {len(tiles)} tiles have no detection results and are excluded.")
    exported = writer.deduplicate(detections)
    logger.info(
        f"Detections: {len(detections) + removed} from the model, {removed} removed by post-processing, "
        f"{len(detections) - len(exported)} duplicates from overlapping chips removed, {len(exported)} kept."
    )

    output_path = config.PROCESSED_DIR / f"{input_path.stem}_detections.gpkg"
    if exported:
        writer.export_to_vector(
            detections,
            prompt_used=", ".join(targets),
            acquisition_date=args.acquisition_date,
            output_filename=output_path.name,
        )
        logger.info(f"Exported detections to {output_path}")
    else:
        output_path.unlink(missing_ok=True)  # don't leave an earlier run's output looking current
        logger.warning("No detections left to export; no GeoPackage written.")

    if args.ground_truth:
        ground_truth_path = Path(args.ground_truth)
        if not ground_truth_path.exists():
            ground_truth_path = config.RAW_DIR / args.ground_truth
        logger.info("Evaluation of exported detections against ground truth:\n"
                    + evaluate(exported, ground_truth_path, prompt_manager).to_string(index=False))

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
