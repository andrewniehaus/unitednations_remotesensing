"""
src/change_detection/change_pipeline.py

Orchestrator for End-to-End Latent Temporal Change Detection.

Coordinates batch loading of pre-event (T1) and post-event (T2) tile pairs from 
config.INTERIM_DIR, extracts Clay embeddings, executes similarity evaluation, 
generates change polygons, and logs execution metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import geopandas as gpd
import torch

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config
from src.change_detection.clay_encoder import ClayEncoder
from src.change_detection.similarity_engine import LatentSimilarityEngine
from src.change_detection.mask_generator import ChangeMaskGenerator
from src.georeferencing.coord_transformer import CoordinateTransformer


class ChangeDetectionPipeline:
    """
    Main orchestrator class for air-gapped temporal change detection.
    """

    def __init__(self, use_otsu_thresh: bool = True):
        """
        Initialize module dependencies and models.
        """
        self.encoder = ClayEncoder()
        self.similarity_engine = LatentSimilarityEngine(metric="cosine")
        self.mask_generator = ChangeMaskGenerator()
        self.transformer = CoordinateTransformer()
        self.use_otsu_thresh = use_otsu_thresh

    def verify_tile_pair_alignment(
        self, t1_path: Path, t2_path: Path
    ) -> bool:
        """
        Feature 1: T1/T2 Tile Pair Alignment Verification.
        Ensures temporal image tile pairs match in spatial aspect ratio and dimensions.
        """
        from PIL import Image

        if not (t1_path.exists() and t2_path.exists()):
            return False

        with Image.open(t1_path) as img1, Image.open(t2_path) as img2:
            return img1.size == img2.size

    def process_tile_pair(
        self,
        t1_path: Path,
        t2_path: Path,
        x_offset: int = 0,
        y_offset: int = 0
    ) -> pd.DataFrame:
        """
        Executes change detection pipeline across a single pair of T1/T2 tiles.
        """
        if not self.verify_tile_pair_alignment(t1_path, t2_path):
            return pd.DataFrame()

        # Extract spatial feature maps
        feat_t1 = self.encoder.extract_spatial_embedding_map(t1_path)
        feat_t2 = self.encoder.extract_spatial_embedding_map(t2_path)

        # Calculate spatial change map
        change_grid = self.similarity_engine.compute_spatial_change_map(feat_t1, feat_t2)
        clean_grid = self.similarity_engine.apply_local_variance_normalization(change_grid)

        # Generate change polygons
        df_polygons = self.mask_generator.extract_change_polygons(
            clean_grid,
            use_otsu=self.use_otsu_thresh,
            x_offset=x_offset,
            y_offset=y_offset
        )

        return df_polygons

    def run_pipeline(
        self,
        tile_manifest: List[Dict[str, Any]],
        source_raster_path: Optional[Path] = None
    ) -> gpd.GeoDataFrame:
        """
        Feature 2 & 3: Batch Execution, Memory Cleanup, and GeoPandas Export.

        Args:
            tile_manifest (List[Dict[str, Any]]): List of dicts containing 't1_path', 't2_path', 'x_offset', 'y_offset'.
            source_raster_path (Optional[Path]): Path to raw source GeoTIFF for georeferencing.

        Returns:
            gpd.GeoDataFrame: Georeferenced temporal change detection vector polygons.
        """
        all_results = []
        total_pairs = len(tile_manifest)

        for idx, pair in enumerate(tile_manifest):
            df_changes = self.process_tile_pair(
                t1_path=Path(pair["t1_path"]),
                t2_path=Path(pair["t2_path"]),
                x_offset=pair.get("x_offset", 0),
                y_offset=pair.get("y_offset", 0)
            )

            if not df_changes.empty:
                all_results.append(df_changes)

            # Feature 2: Batch Memory Cleanup
            if idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all_results:
            empty_gdf = gpd.GeoDataFrame(columns=["xmin", "ymin", "xmax", "ymax", "label", "confidence", "geometry"])
            return empty_gdf

        consolidated_df = pd.concat(all_results, ignore_index=True)

        # Feature 3: Vector Georeferencing
        if source_raster_path and source_raster_path.exists():
            gdf = self.transformer.transform_detections(
                df=consolidated_df,
                raster_path=source_raster_path,
                target_crs="EPSG:4326"
            )
        else:
            gdf = gpd.GeoDataFrame(consolidated_df, geometry="geometry", crs="EPSG:4326")

        # Feature 4: Audit Manifest Generation
        self._write_summary_manifest(gdf, total_pairs)

        return gdf

    def _write_summary_manifest(self, gdf: gpd.GeoDataFrame, total_pairs_processed: int) -> None:
        """
        Feature 4: Logs summary report to config.PROCESSED_DIR / 'change_summary_manifest.json'.
        """
        manifest_path = config.PROCESSED_DIR / "change_summary_manifest.json"
        summary = {
            "total_tile_pairs_processed": total_pairs_processed,
            "total_change_polygons_detected": len(gdf),
            "severity_counts": gdf["label"].value_counts().to_dict() if not gdf.empty else {},
        }

        with open(manifest_path, "w") as f:
            json.dump(summary, f, indent=4)