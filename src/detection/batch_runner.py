"""
Batch Inference Runner for Air-Gapped Vision-Language Models.
Orchestrates tile-by-tile detection execution across intermediate raster tiles,
manages GPU VRAM garbage collection, and exports tile-level prediction JSONs
formatted for RasterRebuilder.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm import tqdm

from src import config

logger = logging.getLogger(__name__)


class BatchInferenceRunner:
    """
    Orchestrates batch inference over tiled imagery directories or chipping manifests.
    Executes VLM prediction, manages GPU memory cleanup, and persists individual
    detection outputs to disk.
    """

    def __init__(self, model_adapter: Any):
        self.adapter = model_adapter

    def _clear_vram(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_on_manifest(
        self,
        manifest_path: Path,
        targets: Union[str, List[str]],
        output_dir: Optional[Path] = None,
        clear_vram_frequency: int = getattr(config, 'DEFAULT_CLEAR_VRAM_FREQ', 10),
        **kwargs: Any,
    ) -> Path:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        tiles_info: List[Dict[str, Any]] = manifest.get("tiles", [])
        out_dir = output_dir or (config.INTERIM_DIR / "vlm_results")
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting batch inference over {len(tiles_info)} tiles.")

        overwrite_allowed = getattr(config, 'DEFAULT_OVERWRITE', False)

        for idx, tile_record in enumerate(tqdm(tiles_info, desc="VLM Tile Inference")):
            raw_path = Path(tile_record["filename"])
            
            # Safely resolve tile path (prioritize absolute, then manifest relative, then interim relative)
            if raw_path.is_absolute():
                tile_path = raw_path
            elif (manifest_path.parent / raw_path).exists():
                tile_path = manifest_path.parent / raw_path
            else:
                tile_path = config.INTERIM_DIR / raw_path

            # Preserve directory structure in output
            json_filename = raw_path.with_suffix(".json")
            json_output_path = out_dir / json_filename

            if json_output_path.exists() and not overwrite_allowed:
                continue

            if not tile_path.exists():
                logger.warning(f"Tile image missing: {tile_path}. Skipping.")
                continue

            # Ensure subdirectories exist before writing
            json_output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                results = self.adapter.predict(
                    image_data=tile_path,
                    targets=targets,
                    **kwargs,
                )

                with open(json_output_path, "w", encoding="utf-8") as jf:
                    json.dump(results, jf, indent=2)

            except Exception as e:
                logger.error(f"Error processing tile {tile_path.name}: {e}")
                continue

            if (idx + 1) % clear_vram_frequency == 0:
                self._clear_vram()

        self._clear_vram()
        logger.info(f"Batch inference complete. Results stored in {out_dir}")
        return out_dir

    def run_on_directory(
        self,
        input_dir: Path,
        targets: Union[str, List[str]],
        output_dir: Optional[Path] = None,
        file_extension: str = ".tif",
        clear_vram_frequency: int = getattr(config, 'DEFAULT_CLEAR_VRAM_FREQ', 10),
        **kwargs: Any,
    ) -> Path:
        # Use rglob to capture potential subdirectories and ensure relative structures are maintained
        tile_paths = list(input_dir.rglob(f"*{file_extension}"))
        out_dir = output_dir or (config.INTERIM_DIR / "vlm_results")
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Found {len(tile_paths)} tiles matching '{file_extension}' in {input_dir}")

        overwrite_allowed = getattr(config, 'DEFAULT_OVERWRITE', False)

        for idx, tile_path in enumerate(tqdm(tile_paths, desc="Directory Inference")):
            # Extract relative path to replicate structure in the output directory
            relative_path = tile_path.relative_to(input_dir)
            json_output_path = out_dir / relative_path.with_suffix(".json")

            if json_output_path.exists() and not overwrite_allowed:
                continue
                
            # Ensure subdirectories exist before writing
            json_output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                results = self.adapter.predict(
                    image_data=tile_path,
                    targets=targets,
                    **kwargs,
                )

                with open(json_output_path, "w", encoding="utf-8") as jf:
                    json.dump(results, jf, indent=2)

            except Exception as e:
                logger.error(f"Failed processing tile {tile_path.name}: {e}")
                continue

            if (idx + 1) % clear_vram_frequency == 0:
                self._clear_vram()

        self._clear_vram()
        logger.info(f"Directory processing finished. Output directory: {out_dir}")
        return out_dir