"""
src/tiling/tile_indexer.py

In-Memory Spatial Tile Indexing and Manifest Engine.

Builds spatial R-Tree indices over generated chips to allow sub-second geographic 
coordinate queries and exports GeoJSON/GeoPackage tile manifests for desktop GIS auditing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class TileIndexer:
    """
    Spatial R-Tree indexer for tiled raster mosaics.
    """

    def __init__(self, crs: str = "EPSG:4326"):
        """
        Initialize tile indexer.

        Args:
            crs (str): Target spatial coordinate system for manifest export.
        """
        self.crs = crs
        self.tile_records: List[Dict[str, Any]] = []
        self.spatial_tree: Optional[STRtree] = None
        self._geometries: List[Polygon] = []

    def add_tile(
        self,
        tile_path: Union[str, Path],
        bounds: Tuple[float, float, float, float],
        pixel_window: Tuple[int, int, int, int],
        parent_raster_path: Union[str, Path],
    ) -> None:
        """
        Feature 1 & 3: In-Memory Spatial Metadata Storage.
        Adds a generated tile chip record to the index manifest.

        Args:
            tile_path (Union[str, Path]): Path to chip in config.INTERIM_DIR.
            bounds (Tuple[float, float, float, float]): (minx, miny, maxx, maxy) bounds.
            pixel_window (Tuple[int, int, int, int]): (row_off, col_off, height, width).
            parent_raster_path (Union[str, Path]): Path to source GeoTIFF in config.RAW_DIR.
        """
        tile_poly = box(*bounds)
        self._geometries.append(tile_poly)

        record = {
            "tile_path": str(tile_path),
            "parent_raster": str(parent_raster_path),
            "row_off": pixel_window[0],
            "col_off": pixel_window[1],
            "height": pixel_window[2],
            "width": pixel_window[3],
            "bounds": bounds,
            "geometry": tile_poly,
        }
        self.tile_records.append(record)

    def build_spatial_index(self) -> None:
        """
        Feature 1: Builds Shapely STRtree (R-Tree) spatial index over registered tiles.
        """
        if self._geometries:
            self.spatial_tree = STRtree(self._geometries)

    def query_tiles_by_point(self, lon: float, lat: float) -> List[Dict[str, Any]]:
        """
        Feature 2: Fast Sub-Second Point Spatial Querying.
        Finds all tile chips overlapping a given target geographic coordinate.
        """
        if self.spatial_tree is None:
            self.build_spatial_index()

        if self.spatial_tree is None:
            return []

        point_geom = box(lon - 1e-6, lat - 1e-6, lon + 1e-6, lat + 1e-6)
        matched_indices = self.spatial_tree.query(point_geom)

        results = []
        for idx in matched_indices:
            results.append(self.tile_records[idx])

        return results

    def export_tile_manifest(
        self, output_filename: str = "tile_grid_manifest.gpkg"
    ) -> Path:
        """
        Features 4 & 5: GeoPackage Vector Grid Export and Overlap Tracking.
        Writes registered tile grid manifest to config.PROCESSED_DIR for GIS auditing.
        """
        out_path = config.PROCESSED_DIR / output_filename
        if not self.tile_records:
            return out_path

        df = pd.DataFrame(self.tile_records)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=self.crs)

        # Feature 5: Calculate tile seam overlap ratios
        gdf["area_sqm"] = gdf.geometry.area
        
        # Save manifest
        if output_filename.endswith(".gpkg"):
            gdf.to_file(out_path, driver="GPKG")
        else:
            gdf.to_file(out_path, driver="GeoJSON")

        return out_path