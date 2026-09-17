"""
src/detection/spatial_heuristics.py

Spatial Logic and Heuristics Engine.

Applies geographic rules, topological relationships, and physical scale constraints 
to filter out false positives from Vision-Language Model predictions (e.g., dropping 
tents placed in water, or isolated trench fragments).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class SpatialHeuristicsEngine:
    """
    Validates model detections against physical and relational geospatial logic.
    """

    def __init__(self, context_dir: Optional[Path] = None):
        self.context_dir = context_dir or (config.RAW_DIR / "context")
        self.water_mask_path = self.context_dir / "water_bodies.gpkg"

    def filter_by_land_water_mask(
        self, gdf: gpd.GeoDataFrame, land_only_classes: List[str]
    ) -> gpd.GeoDataFrame:
        """
        Feature 1: Context Masking.
        Removes land-only objects (e.g., 'tent', 'trench') that intersect water bodies.
        """
        if gdf.empty or not self.water_mask_path.exists():
            return gdf

        water_gdf = gpd.read_file(self.water_mask_path).to_crs(gdf.crs)
        
        # Separate classes that need masking from those that don't (e.g., 'ship')
        mask_target_gdf = gdf[gdf["label"].isin(land_only_classes)]
        safe_gdf = gdf[~gdf["label"].isin(land_only_classes)]

        if mask_target_gdf.empty:
            return gdf

        # Spatial difference: keep only land targets that DO NOT intersect water
        valid_land_gdf = gpd.sjoin(mask_target_gdf, water_gdf, how="left", predicate="intersects")
        valid_land_gdf = valid_land_gdf[valid_land_gdf["index_right"].isna()].drop(columns=["index_right"])

        return pd.concat([safe_gdf, valid_land_gdf], ignore_index=True)

    def enforce_relational_distance(
        self, gdf: gpd.GeoDataFrame, target_class: str, anchor_class: str, max_distance_meters: float
    ) -> gpd.GeoDataFrame:
        """
        Feature 2: Inter-Class Distance Constraints.
        Ensures a 'target' (e.g., vehicle) is within X meters of an 'anchor' (e.g., compound).
        Requires a metric projected CRS (UTM).
        """
        if gdf.empty:
            return gdf

        targets = gdf[gdf["label"] == target_class]
        anchors = gdf[gdf["label"] == anchor_class]

        if anchors.empty:
            # If no anchors exist, all targets are invalid
            return gdf[gdf["label"] != target_class]

        valid_targets = []
        for idx, target_row in targets.iterrows():
            # Check distance to nearest anchor
            min_dist = anchors.geometry.distance(target_row.geometry).min()
            if min_dist <= max_distance_meters:
                valid_targets.append(target_row)

        valid_targets_gdf = gpd.GeoDataFrame(valid_targets, crs=gdf.crs) if valid_targets else gpd.GeoDataFrame(columns=gdf.columns)
        other_classes = gdf[gdf["label"] != target_class]

        return pd.concat([other_classes, valid_targets_gdf], ignore_index=True)

    def validate_trench_continuity(self, gdf: gpd.GeoDataFrame, max_gap_meters: float = 15.0) -> gpd.GeoDataFrame:
        """
        Feature 3: Topological Graph Building.
        Builds a network graph of trench segment detections to drop isolated, spurious fragments.
        """
        trench_gdf = gdf[gdf["label"].str.contains("trench")].copy()
        if trench_gdf.empty:
            return gdf

        # Build adjacency graph
        G = nx.Graph()
        for i, geom1 in trench_gdf.geometry.items():
            G.add_node(i)
            for j, geom2 in trench_gdf.geometry.items():
                if i < j and geom1.distance(geom2) <= max_gap_meters:
                    G.add_edge(i, j)

        # Find connected components; isolated segments (length 1) are dropped
        valid_indices = []
        for component in nx.connected_components(G):
            if len(component) > 1:  # Must be connected to at least one other segment
                valid_indices.extend(list(component))

        valid_trenches = trench_gdf.loc[valid_indices]
        other_gdf = gdf[~gdf["label"].str.contains("trench")]
        
        return pd.concat([other_gdf, valid_trenches], ignore_index=True)

    def enforce_physical_scale(self, df: pd.DataFrame, gsd_meters: float, class_scales: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """
        Feature 4: Physical Scale Constraints.
        Drops detections whose pixel dimensions translate to impossible physical sizes.
        class_scales format: {"tent": (min_sq_meters, max_sq_meters)}
        """
        if df.empty:
            return df

        df = df.copy()
        valid_mask = np.ones(len(df), dtype=bool)

        # Area in pixels * (meters/pixel)^2 = Area in square meters
        pixel_areas = (df["xmax"] - df["xmin"]) * (df["ymax"] - df["ymin"])
        physical_areas = pixel_areas * (gsd_meters ** 2)

        for cls, (min_area, max_area) in class_scales.items():
            cls_mask = df["label"] == cls
            size_invalid = cls_mask & ((physical_areas < min_area) | (physical_areas > max_area))
            valid_mask[size_invalid] = False

        return df[valid_mask].reset_index(drop=True)