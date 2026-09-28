"""
CRS Manager Module
==================
Standardizes spatial projections across disparate satellite sources (Vantor, Airbus, Capella)
and vector datasets (building footprints, OpenStreetMap extracts) to ensure exact spatial 
alignment during offline change detection and mask generation.
"""

import logging
from typing import Union
import rasterio
import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import base
from shapely.ops import transform

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

class CRSManager:
    def __init__(self, target_epsg: int = 4326):
        """
        Initializes the CRS Manager. Defaults to WGS84 (EPSG:4326) for global compatibility,
        but can be instantiated with local projected UTM zones for accurate area/distance calculations.
        """
        self.target_crs = CRS.from_epsg(target_epsg)
        self.target_epsg_str = f"EPSG:{target_epsg}"

    def standardize_vector_crs(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Checks and reprojects a GeoDataFrame to the standard target CRS.
        Crucial for aligning external building footprint datasets with detection outputs.
        """
        if gdf.crs is None:
            logger.warning("GeoDataFrame lacks CRS. Assuming EPSG:4326.")
            gdf.set_crs(epsg=4326, inplace=True)
            
        if gdf.crs != self.target_crs:
            return gdf.to_crs(self.target_crs)
        return gdf

    def get_raster_crs(self, raster_path: str) -> CRS:
        """
        Extracts the native CRS from a source TIFF.
        """
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise ValueError(f"Raster at {raster_path} has no defined CRS.")
            return src.crs

    def transform_geometry(self, geom: base.BaseGeometry, src_crs: Union[str, CRS]) -> base.BaseGeometry:
        """
        Transforms a standalone Shapely geometry (e.g., an incident buffer) from its native CRS
        to the standardized target CRS.
        """
        source_crs = CRS.from_user_input(src_crs)
        if source_crs == self.target_crs:
            return geom
        
        project = Transformer.from_crs(source_crs, self.target_crs, always_xy=True).transform
        return transform(project, geom)