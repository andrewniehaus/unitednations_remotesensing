"""
src/ingestion/spatial_buffer.py

Spatial-Temporal Incident Buffering and Imagery Queuing Engine.

Converts geolocated incident point coordinates (e.g., from Liveuamap or manual reports) 
into planar metric spatial bounding boxes. Intersects these buffers with GeoTIFF raster 
extents in config.RAW_DIR to automatically queue target raster windows for downstream chipping.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point, Polygon, box
from pyproj import CRS, Transformer

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class SpatialBufferEngine:
    """
    Transforms point-based event reports into spatial metric bounding boxes 
    and identifies matching imagery extents in config.RAW_DIR.
    """

    def __init__(
        self,
        default_buffer_meters: float = 1000.0,
        default_crs: str = "EPSG:4326"
    ):
        """
        Initialize the spatial buffering engine.

        Args:
            default_buffer_meters (float): Radial buffer distance in meters around event points.
            default_crs (str): Native Coordinate Reference System of input coordinates.
        """
        self.default_buffer_meters = default_buffer_meters
        self.default_crs = default_crs

    def _estimate_utm_crs(self, lon: float, lat: float) -> str:
        """
        Feature 1: Auto-UTM Projection Selection.
        Calculates the appropriate local UTM zone EPSG code for accurate metric buffering.
        """
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = "north" if lat >= 0 else "south"
        epsg_code = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        return f"EPSG:{epsg_code}"

    def create_metric_buffer(
        self,
        lon: float,
        lat: float,
        buffer_meters: Optional[float] = None
    ) -> Polygon:
        """
        Feature 1: Planar Metric Buffering.
        Reprojects EPSG:4326 lat/lon to local projected UTM, buffers in meters, 
        and transforms the bounding box back to target geographic space.

        Args:
            lon (float): Longitude coordinate.
            lat (float): Latitude coordinate.
            buffer_meters (Optional[float]): Custom buffer distance in meters.

        Returns:
            Polygon: Bounding box polygon in EPSG:4326 geographic coordinates.
        """
        dist = buffer_meters if buffer_meters is not None else self.default_buffer_meters
        utm_crs = self._estimate_utm_crs(lon, lat)

        # Build point geometry and reproject to metric UTM
        gdf_point = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)],
            crs=self.default_crs
        ).to_crs(utm_crs)

        # Apply metric buffer and get bounding box extent
        buffered_geom = gdf_point.buffer(dist).iloc[0]
        minx, miny, maxx, maxy = buffered_geom.bounds
        bbox_utm = box(minx, miny, maxx, maxy)

        # Reproject bounding box back to EPSG:4326
        gdf_bbox = gpd.GeoDataFrame(geometry=[bbox_utm], crs=utm_crs).to_crs(self.default_crs)
        return gdf_bbox.geometry.iloc[0]

    def scan_raw_imagery_extents(self) -> List[Dict[str, Union[str, Path, Polygon]]]:
        """
        Feature 2: Lightweight Metadata Indexing.
        Scans config.RAW_DIR for GeoTIFF files and extracts spatial bounding boxes 
        without loading heavy pixel arrays into RAM.

        Returns:
            List[Dict]: List of metadata dicts containing 'path', 'crs', and 'geometry' (Polygon).
        """
        raw_dir = Path(config.RAW_DIR)
        raster_files = list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff"))
        index = []

        for rpath in raster_files:
            try:
                with rasterio.open(rpath) as src:
                    bounds = src.bounds
                    poly_native = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                    
                    # Convert bounds to EPSG:4326 for uniform spatial index matching
                    gdf_poly = gpd.GeoDataFrame(geometry=[poly_native], crs=src.crs)
                    gdf_wgs84 = gdf_poly.to_crs(self.default_crs)

                    index.append({
                        "path": rpath,
                        "crs": str(src.crs),
                        "geometry": gdf_wgs84.geometry.iloc[0],
                        "width": src.width,
                        "height": src.height,
                    })
            except Exception:
                continue

        return index

    def match_incidents_to_imagery(
        self,
        incidents_df: pd.DataFrame,
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        buffer_meters: Optional[float] = None
    ) -> List[Dict[str, Union[str, Path, Window, int]]]:
        """
        Features 3 & 5: Batch Incident Queueing and Window Calculation.
        Intersects buffered incident locations against raw imagery extents and computes 
        exact pixel window slices for downstream chipping in src/tiling/chipper.py.

        Args:
            incidents_df (pd.DataFrame): DataFrame containing incident point events.
            lon_col (str): Longitude column name.
            lat_col (str): Latitude column name.
            buffer_meters (Optional[float]): Metric buffer distance.

        Returns:
            List[Dict]: Manifest list of target pixel windows and raster paths ready for extraction.
        """
        if incidents_df.empty:
            return []

        raster_index = self.scan_raw_imagery_extents()
        if not raster_index:
            return []

        extraction_manifest = []

        for idx, row in incidents_df.iterrows():
            lon, lat = row[lon_col], row[lat_col]
            buffer_poly = self.create_metric_buffer(lon, lat, buffer_meters)

            # Spatial intersection check against indexed raw imagery
            for r_info in raster_index:
                r_poly = r_info["geometry"]
                if buffer_poly.intersects(r_poly):
                    r_path = Path(r_info["path"])

                    # Compute pixel window offsets using rasterio
                    with rasterio.open(r_path) as src:
                        # Reproject buffer polygon into native raster CRS
                        gdf_buf = gpd.GeoDataFrame(geometry=[buffer_poly], crs=self.default_crs)
                        gdf_native = gdf_buf.to_crs(src.crs)
                        poly_native = gdf_native.geometry.iloc[0]

                        # Calculate intersecting pixel window
                        win = rasterio.features.geometry_window(src, [poly_native])

                        extraction_manifest.append({
                            "incident_id": row.get("incident_id", idx),
                            "raster_path": r_path,
                            "window_row_off": win.row_off,
                            "window_col_off": win.col_off,
                            "window_height": win.height,
                            "window_width": win.width,
                            "target_crs": str(src.crs),
                        })

        return extraction_manifest