"""
src/georeferencing/coord_transformer.py

Pixel to Geospatial Coordinate Transformer Module.

Transforms pixel-space bounding box detection coordinates (xmin, ymin, xmax, ymax) 
back into spatial map projections (e.g., EPSG:4326 or native UTM CRS) using affine 
geotransforms from raw source GeoTIFF rasters in config.RAW_DIR.
"""

from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
from pyproj import Transformer

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class CoordinateTransformer:
    """
    Transforms pixel bounding box coordinates to spatial vector geometries.
    """

    def __init__(self):
        """
        Initialize the coordinate transformer with an empty metadata cache.
        """
        # Feature 1: Batch GeoTIFF Metadata Caching
        self._meta_cache: Dict[str, Dict[str, Any]] = {}

    def get_raster_metadata(self, raster_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Retrieves and caches affine transform, CRS, and bounds for a GeoTIFF.

        Args:
            raster_path (Union[str, Path]): Path to raw source GeoTIFF.

        Returns:
            Dict[str, Any]: Metadata containing transform matrix and CRS.
        """
        path_str = str(raster_path)
        if path_str in self._meta_cache:
            return self._meta_cache[path_str]

        with rasterio.open(raster_path) as src:
            meta = {
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "bounds": src.bounds,
            }
            self._meta_cache[path_str] = meta
            return meta

    def pixel_to_spatial_polygon(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        transform: rasterio.Affine
    ) -> Polygon:
        """
        Feature 4: Converts pixel bounding box corners into a 4-vertex spatial polygon 
        using the affine transform matrix ($x_{geo} = c + a \\cdot x_{pixel} + b \\cdot y_{pixel}$).

        Args:
            xmin (float): Minimum pixel X.
            ymin (float): Minimum pixel Y.
            xmax (float): Maximum pixel X.
            ymax (float): Maximum pixel Y.
            transform (rasterio.Affine): GeoTIFF affine transform matrix.

        Returns:
            Polygon: Shapely polygon in native map coordinates.
        """
        # Top-left, top-right, bottom-right, bottom-left corners in pixel space
        px_coords = [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
        ]

        # Apply Affine Transformation matrix
        map_coords = [transform * (px, py) for px, py in px_coords]
        return Polygon(map_coords)

    def transform_detections(
        self,
        df: pd.DataFrame,
        raster_path: Union[str, Path],
        target_crs: Optional[str] = "EPSG:4326",
        bbox_cols: Tuple[str, str, str, str] = ("xmin", "ymin", "xmax", "ymax"),
    ) -> gpd.GeoDataFrame:
        """
        Features 2, 3 & 5: Converts a detection DataFrame to a GeoPandas GeoDataFrame 
        and reprojects coordinates to the specified target CRS.

        Args:
            df (pd.DataFrame): Pixel detection records.
            raster_path (Union[str, Path]): Path to source GeoTIFF in config.RAW_DIR.
            target_crs (Optional[str]): Target CRS string (e.g., 'EPSG:4326').
            bbox_cols (Tuple[str, str, str, str]): Column names for pixel bounding boxes.

        Returns:
            gpd.GeoDataFrame: Vector detections with spatial polygon geometries and CRS.
        """
        if df.empty:
            return gpd.GeoDataFrame(df, geometry=[], crs=target_crs or "EPSG:4326")

        meta = self.get_raster_metadata(raster_path)
        transform = meta["transform"]
        native_crs = meta["crs"]

        # Feature 4: Generate spatial polygons for each pixel bounding box
        geometries = []
        for _, row in df.iterrows():
            poly = self.pixel_to_spatial_polygon(
                xmin=row[bbox_cols[0]],
                ymin=row[bbox_cols[1]],
                xmax=row[bbox_cols[2]],
                ymax=row[bbox_cols[3]],
                transform=transform,
            )
            geometries.append(poly)

        # Build GeoDataFrame in native raster CRS
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geometries, crs=native_crs)

        # Feature 2: On-the-Fly Reprojection to target CRS
        if target_crs and native_crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        return gdf