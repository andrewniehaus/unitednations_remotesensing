"""
Coordinate Transformer Module
=============================
Translates local tile pixel coordinates (bounding boxes output by Florence-2 or Grounding DINO) 
back into global geographic or projected spatial coordinates (EPSG standards) using the source 
raster's affine geotransform. 
"""

import logging
from typing import List, Tuple, Dict, Any
import rasterio
from rasterio.transform import xy

from src.config import DEFAULT_TILE_SIZE

logger = logging.getLogger(__name__)

class CoordinateTransformer:
    def __init__(self, source_raster_path: str):
        """
        Initializes the transformer by extracting the spatial metadata and affine 
        transform directly from the unchipped, original satellite imagery.
        """
        self.source_path = source_raster_path
        with rasterio.open(self.source_path) as src:
            self.transform = src.transform
            self.crs = src.crs

    def pixel_bbox_to_spatial(self, xmin: int, ymin: int, xmax: int, ymax: int) -> Tuple[float, float, float, float]:
        """
        Transforms a pixel bounding box into spatial coordinates.
        Calculates the top-left and bottom-right spatial coordinates accounting for pixel resolution and origin.
        
        Returns: (min_lon, min_lat, max_lon, max_lat) or equivalent projected coords.
        """
        # Calculate spatial coordinates for the Top-Left and Bottom-Right corners of the bounding box
        tl_x, tl_y = xy(self.transform, ymin, xmin, offset='ul')
        # [EDIT 2026-09-28 | Claude Code for charliefp03-dg] offset='dr' is not a valid rasterio option;
        # changed to 'lr' (lower-right corner of the pixel).
        br_x, br_y = xy(self.transform, ymax, xmax, offset='lr')
        
        return (min(tl_x, br_x), min(tl_y, br_y), max(tl_x, br_x), max(tl_y, br_y))

    def batch_transform_detections(self, detections: List[Dict[str, Any]], tile_offset_x: int = 0, tile_offset_y: int = 0) -> List[Dict[str, Any]]:
        """
        Transforms a batch of VLM inferences from local chip/tile space into absolute global spatial coordinates.
        
        :param detections: List of dictionaries containing 'xmin', 'ymin', 'xmax', 'ymax'.
        :param tile_offset_x: Pixel X-offset of the tile relative to the original full-size raster.
        :param tile_offset_y: Pixel Y-offset of the tile relative to the original full-size raster.
        """
        spatial_detections = []
        for det in detections:
            # Map local tile bounding boxes to the global pixel space of the original massive TIFF
            global_xmin = det['xmin'] + tile_offset_x
            global_ymin = det['ymin'] + tile_offset_y
            global_xmax = det['xmax'] + tile_offset_x
            global_ymax = det['ymax'] + tile_offset_y
            
            spatial_bbox = self.pixel_bbox_to_spatial(global_xmin, global_ymin, global_xmax, global_ymax)
            
            # Create immutable copy with spatial metadata
            updated_det = det.copy()
            updated_det['spatial_bbox'] = spatial_bbox
            updated_det['crs'] = self.crs.to_string() if self.crs else "UNKNOWN"
            spatial_detections.append(updated_det)
            
        return spatial_detections