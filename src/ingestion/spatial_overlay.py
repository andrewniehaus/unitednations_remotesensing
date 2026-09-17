"""
src/georeferencing/spatial_overlay.py

Contextual Vector Overlay and Population Exposure Enrichment Module.

Intersects model prediction vectors (e.g., detected change or object polygons) 
with offline contextual datasets (e.g., Microsoft Building Footprints, WorldPop rasters) 
in config.RAW_DIR to calculate population exposure and filter non-structural false positives.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Polygon

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class SpatialOverlayEngine:
    """
    Enriches geospatial vector predictions with contextual demographic 
    and building footprint vector datasets.
    """

    def __init__(
        self,
        building_footprints_path: Optional[Path] = None,
        population_raster_path: Optional[Path] = None
    ):
        """
        Initialize contextual overlay file paths from local configuration directories.

        Args:
            building_footprints_path (Optional[Path]): Path to local building shapefile/GeoPackage.
            population_raster_path (Optional[Path]): Path to local WorldPop GeoTIFF raster.
        """
        self.building_path = building_footprints_path or (
            config.RAW_DIR / "context" / "building_footprints.gpkg"
        )
        self.population_path = population_raster_path or (
            config.RAW_DIR / "context" / "worldpop_population.tif"
        )

    def filter_by_building_footprints(
        self,
        predictions_gdf: gpd.GeoDataFrame,
        min_overlap_ratio: float = 0.20
    ) -> gpd.GeoDataFrame:
        """
        Features 2 & 3: Building Footprint Intersection & CRS Auto-Harmonization.
        Intersects detected polygons with local vector building footprints to eliminate 
        non-structural false positives (e.g., soil shifts or shadows).

        Args:
            predictions_gdf (gpd.GeoDataFrame): Model prediction vectors.
            min_overlap_ratio (float): Minimum area overlap ratio with a building footprint.

        Returns:
            gpd.GeoDataFrame: Cleaned GeoDataFrame containing confirmed structural detections.
        """
        if predictions_gdf.empty or not self.building_path.exists():
            return predictions_gdf

        # Load offline vector footprints
        buildings_gdf = gpd.read_file(self.building_path)

        # Feature 3: Dynamic CRS Harmonization
        if buildings_gdf.crs != predictions_gdf.crs:
            buildings_gdf = buildings_gdf.to_crs(predictions_gdf.crs)

        # Perform spatial intersection overlay
        spatial_join = gpd.sjoin(
            predictions_gdf,
            buildings_gdf,
            how="inner",
            predicate="intersects"
        )

        if spatial_join.empty:
            return gpd.GeoDataFrame(columns=predictions_gdf.columns, crs=predictions_gdf.crs)

        # Remove spatial join duplicate indices
        confirmed_ids = spatial_join.index.unique()
        filtered_gdf = predictions_gdf.loc[confirmed_ids].reset_index(drop=True)

        return filtered_gdf

    def estimate_population_exposure(
        self,
        predictions_gdf: gpd.GeoDataFrame,
        stats_type: str = "sum"
    ) -> gpd.GeoDataFrame:
        """
        Feature 1: Zonal Population Exposure Estimation.
        Extracts zonal statistics from local WorldPop raster grids to estimate total 
        impacted human population within target detection/change polygons.

        Args:
            predictions_gdf (gpd.GeoDataFrame): Target detection vector layer.
            stats_type (str): Zonal stat calculation ('sum', 'mean', 'max').

        Returns:
            gpd.GeoDataFrame: Enriched GeoDataFrame with 'population_count' column.
        """
        if predictions_gdf.empty or not self.population_path.exists():
            predictions_gdf["population_count"] = 0.0
            return predictions_gdf

        gdf_copy = predictions_gdf.copy()

        # Ensure prediction vectors match population raster CRS prior to zonal extraction
        with rasterio.open(self.population_path) as src:
            pop_crs = src.crs

        if gdf_copy.crs != pop_crs:
            gdf_copy = gdf_copy.to_crs(pop_crs)

        # Execute offline zonal statistics using rasterstats
        stats = zonal_stats(
            gdf_copy,
            str(self.population_path),
            stats=stats_type,
            nodata=-9999
        )

        pop_counts = [round(item.get(stats_type) or 0.0, 2) for item in stats]
        
        # Attach population estimates to original prediction DataFrame
        predictions_gdf["population_count"] = pop_counts
        return predictions_gdf

    def export_contextual_layer(
        self,
        enriched_gdf: gpd.GeoDataFrame,
        output_filename: str = "enriched_impact_analysis.gpkg"
    ) -> Path:
        """
        Features 4 & 5: Spatial Conflict Resolution & Offline Output Persistence.
        Saves contextualized detection vectors directly to config.PROCESSED_DIR.

        Args:
            enriched_gdf (gpd.GeoDataFrame): Contextualized GeoDataFrame.
            output_filename (str): Output vector filename (.gpkg or .geojson).

        Returns:
            Path: Path to saved output layer in config.PROCESSED_DIR.
        """
        output_path = config.PROCESSED_DIR / output_filename
        
        if enriched_gdf.empty:
            return output_path

        # Feature 5: Save directly to processed directory using GeoPackage format
        enriched_gdf.to_file(output_path, driver="GPKG")
        return output_path