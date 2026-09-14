import json
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import box

from src import config


class RasterRebuilder:
    """Reconstructs full-extent georeferenced rasters and vector datasets from processed tiles."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initializes the rebuilder with output location."""
        self.output_dir = Path(output_dir) if output_dir else config.PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prep_tile_array(self, tile_file: Path, padding_dict: Dict[str, int]) -> np.ndarray:
        """Worker function to read and strip padding from a tile array concurrently."""
        with rasterio.open(tile_file) as t_src:
            tile_data = t_src.read()

        pad_bottom = padding_dict["bottom"]
        pad_right = padding_dict["right"]

        _, h, w = tile_data.shape
        valid_h = h - pad_bottom
        valid_w = w - pad_right

        return tile_data[:, :valid_h, :valid_w]

    def rebuild_raster(
        self,
        manifest_path: Path,
        processed_tiles_dir: Path,
        output_filename: str,
        dtype: str = "uint8",
        num_bands: int = 3,
        nodata_value: float = 0.0,
        build_overviews: bool = True,
        metadata_tags: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Stitches processed raster tiles (heatmaps/masks) into a single georeferenced GeoTIFF."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        source_raster_path = Path(manifest["source_raster"])

        with rasterio.open(source_raster_path) as src:
            base_profile = src.profile.copy()

        base_profile.update(
            {
                "driver": "GTiff",
                "dtype": dtype,
                "count": num_bands,
                "nodata": nodata_value,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": manifest.get("tile_size", config.DEFAULT_TILE_SIZE),
                "blockysize": manifest.get("tile_size", config.DEFAULT_TILE_SIZE),
            }
        )

        output_path = self.output_dir / output_filename

        # Sequential write with concurrent array preparation
        with rasterio.open(output_path, "w", **base_profile) as dst:
            if metadata_tags:
                dst.update_tags(**metadata_tags)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_window = {}
                
                for tile_record in manifest["tiles"]:
                    tile_file = processed_tiles_dir / tile_record["filename"]
                    if not tile_file.exists():
                        continue
                    
                    window_dict = tile_record["window"]
                    window = Window(
                        col_off=window_dict["col_off"],
                        row_off=window_dict["row_off"],
                        width=window_dict["width"],
                        height=window_dict["height"],
                    )
                    
                    future = executor.submit(self._prep_tile_array, tile_file, tile_record["padding"])
                    future_to_window[future] = window

                for future in concurrent.futures.as_completed(future_to_window):
                    window = future_to_window[future]
                    unpadded_data = future.result()
                    dst.write(unpadded_data, window=window)

        # Build overviews (pyramids) for rapid desktop GIS rendering
        if build_overviews:
            with rasterio.open(output_path, "r+") as dst:
                factors = [2, 4, 8, 16]
                dst.build_overviews(factors, Resampling.nearest)
                dst.update_tags(ns="rio_overview", resampling="nearest")

        return output_path

    def rebuild_vectors(
        self, 
        manifest_path: Path, 
        vlm_results_dir: Path, 
        output_filename: str
    ) -> Path:
        """
        Translates local tile-level bounding boxes (from VLM inference) into a global GeoPackage.
        Assumes VLM outputs are saved as JSON files mirroring the tile filenames (e.g., tile_0_0.json).
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        crs = manifest["crs"]
        global_geometries = []
        labels = []
        confidences = []

        for tile_record in manifest["tiles"]:
            # Match the tile filename to its corresponding VLM JSON output
            json_filename = Path(tile_record["filename"]).with_suffix(".json")
            json_path = vlm_results_dir / json_filename

            if not json_path.exists():
                continue

            with open(json_path, "r", encoding="utf-8") as jf:
                detections = json.load(jf)

            # Reconstruct the affine transform for this specific tile
            t = tile_record["transform"]
            tile_transform = rasterio.Affine(t[0], t[1], t[2], t[3], t[4], t[5])

            for det in detections:
                # Assuming VLM output format: [xmin, ymin, xmax, ymax] in local pixel coordinates
                px_min, py_min, px_max, py_max = det["bbox"]
                
                # Transform pixel coordinates to global spatial coordinates
                global_xmin, global_ymax = tile_transform * (px_min, py_min)
                global_xmax, global_ymin = tile_transform * (px_max, py_max)
                
                geom = box(global_xmin, global_ymin, global_xmax, global_ymax)
                
                global_geometries.append(geom)
                labels.append(det.get("label", "unknown"))
                confidences.append(det.get("confidence", 0.0))

        if not global_geometries:
            raise ValueError("No valid detections found to rebuild.")

        gdf = gpd.GeoDataFrame(
            {"label": labels, "confidence": confidences}, 
            geometry=global_geometries, 
            crs=crs
        )
        
        output_path = self.output_dir / output_filename
        gdf.to_file(output_path, driver="GPKG")
        
        return output_path