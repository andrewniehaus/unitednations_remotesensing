"""
src/detection/post_processor.py

Post-Processor for Vision-Language Model Detections.

Normalizes raw open-vocabulary outputs from models like Florence-2 or Grounding DINO.
Applies semantic prompt mapping, geometric filtering (area and aspect ratio), and 
coordinate denormalization based on local configuration parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config

class VLMPostProcessor:
    """
    Cleans and standardizes raw bounding box outputs from Vision-Language Models.
    """

    def __init__(
        self,
        semantic_map: Optional[Dict[str, str]] = None,
        default_score_thresh: float = 0.30,
        class_score_thresh: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the post-processor with mapping and threshold rules.

        Args:
            semantic_map (Dict[str, str]): Maps raw VLM text to standard class names.
            default_score_thresh (float): Baseline confidence threshold.
            class_score_thresh (Dict[str, float]): Overriding thresholds for specific classes.
        """
        self.semantic_map = semantic_map or {}
        self.default_score_thresh = default_score_thresh
        self.class_score_thresh = class_score_thresh or {}

    def denormalize_coordinates(
        self, df: pd.DataFrame, tile_size: int = config.DEFAULT_TILE_SIZE
    ) -> pd.DataFrame:
        """
        Feature: Coordinate Denormalization.
        Scales [0, 1] relative coordinates to absolute pixel coordinates.
        """
        if df.empty or "xmin" not in df.columns:
            return df
            
        # Check if coordinates are normalized (max value <= 1.0)
        if df[["xmin", "ymin", "xmax", "ymax"]].max().max() <= 1.0:
            df[["xmin", "xmax"]] = df[["xmin", "xmax"]] * tile_size
            df[["ymin", "ymax"]] = df[["ymin", "ymax"]] * tile_size
            
        return df

    def apply_semantic_mapping(self, df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        """
        Feature: Semantic Prompt Mapping.
        Consolidates varying VLM text outputs into standardized class categories.
        """
        if df.empty or label_col not in df.columns:
            return df
            
        # Lowercase for uniform matching
        df[label_col] = df[label_col].str.lower().str.strip()
        
        # Apply mapping if the raw string matches a key
        df[label_col] = df[label_col].apply(
            lambda x: self.semantic_map.get(x, x)
        )
        return df

    def filter_by_confidence(
        self, df: pd.DataFrame, label_col: str = "label", score_col: str = "confidence"
    ) -> pd.DataFrame:
        """
        Feature: Dynamic Class Thresholding.
        Applies different confidence thresholds based on the target class.
        """
        if df.empty or score_col not in df.columns:
            return df

        def check_threshold(row):
            cls = row[label_col]
            score = row[score_col]
            thresh = self.class_score_thresh.get(cls, self.default_score_thresh)
            return score >= thresh

        mask = df.apply(check_threshold, axis=1)
        return df[mask].reset_index(drop=True)

    def morphological_filter(
        self, 
        df: pd.DataFrame, 
        min_area: float = 10.0, 
        max_area: float = 250000.0,
        max_aspect_ratio: float = 10.0
    ) -> pd.DataFrame:
        """
        Feature: Morphological Filtering.
        Removes detections based on physical geometry (Area and Aspect Ratio).
        Area = width * height
        Aspect Ratio = max(width, height) / min(width, height)
        """
        if df.empty:
            return df

        width = df["xmax"] - df["xmin"]
        height = df["ymax"] - df["ymin"]
        
        # Calculate Area ($Area = width \times height$)
        area = width * height
        
        # Calculate Aspect Ratio
        max_dim = np.maximum(width, height)
        min_dim = np.minimum(width, height)
        # Avoid division by zero
        min_dim = np.where(min_dim == 0, 1e-6, min_dim)
        aspect_ratio = max_dim / min_dim

        # Create boolean mask for valid geometry
        valid_area = (area >= min_area) & (area <= max_area)
        valid_ar = aspect_ratio <= max_aspect_ratio
        
        mask = valid_area & valid_ar
        return df[mask].reset_index(drop=True)

    def process(self, raw_detections: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline to execute all post-processing steps.
        """
        df = raw_detections.copy()
        df = self.apply_semantic_mapping(df)
        df = self.filter_by_confidence(df)
        df = self.denormalize_coordinates(df)
        df = self.morphological_filter(df)
        
        return df