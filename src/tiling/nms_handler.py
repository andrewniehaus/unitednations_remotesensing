"""
src/tiling/nms_handler.py

Spatial Non-Maximum Suppression (NMS) Handler for Tiled Inference Mosaics.

Resolves duplicate and overlapping bounding box predictions generated across tile 
seams when processing large GeoTIFF rasters. Fully integrated with PyTorch, 
TorchVision CUDA kernels, and GeoPandas data structures.
"""

from typing import Union, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torchvision.ops as ops
from shapely.geometry import box

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class TiledNMSHandler:
    """
    Handles Non-Maximum Suppression across tiled raster inferences with support 
    for Soft-NMS, edge-fragment penalization, and spatial GeoDataFrames.
    """

    def __init__(
        self,
        iou_threshold: float = 0.45,
        score_threshold: float = 0.25,
        class_agnostic: bool = False,
        device: torch.device = config.DEVICE,
    ):
        """
        Initialize NMS parameters.

        Args:
            iou_threshold (float): Intersection over Union (IoU) overlap threshold.
            score_threshold (float): Minimum confidence score filter.
            class_agnostic (bool): If True, suppresses overlaps regardless of class label.
            device (torch.device): PyTorch device setting from config.
        """
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.class_agnostic = class_agnostic
        self.device = device

    def penalize_tile_edge_boxes(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        tile_size: int = config.DEFAULT_TILE_SIZE,
        margin: int = 8,
        penalty_factor: float = 0.70,
    ) -> torch.Tensor:
        """
        Feature 3: Identifies bounding boxes touching tile seams in relative tile space 
        and penalizes confidence scores to prevent partial edge crops from outranking 
        complete objects in adjacent overlapping tiles.

        Args:
            boxes (torch.Tensor): Tensor of shape (N, 4) in [xmin, ymin, xmax, ymax].
            scores (torch.Tensor): Tensor of shape (N,) with confidence scores.
            tile_size (int): Tile pixel dimension from config.
            margin (int): Pixel margin width defining the tile border zone.
            penalty_factor (float): Multiplier applied to score for edge-touching boxes.

        Returns:
            torch.Tensor: Penalized confidence scores.
        """
        if boxes.numel() == 0:
            return scores

        # Local tile pixel coordinates relative to individual tile grids
        rem_xmin = torch.remainder(boxes[:, 0], tile_size)
        rem_ymin = torch.remainder(boxes[:, 1], tile_size)
        rem_xmax = torch.remainder(boxes[:, 2], tile_size)
        rem_ymax = torch.remainder(boxes[:, 3], tile_size)

        # Detect boxes extending into boundary margins
        touches_left = rem_xmin <= margin
        touches_top = rem_ymin <= margin
        touches_right = rem_xmax >= (tile_size - margin)
        touches_bottom = rem_ymax >= (tile_size - margin)

        is_edge_box = touches_left | touches_top | touches_right | touches_bottom

        # Apply score penalty
        adjusted_scores = scores.clone()
        adjusted_scores[is_edge_box] *= penalty_factor

        return adjusted_scores

    def soft_nms_pytorch(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        sigma: float = 0.5,
        method: str = "gaussian",
    ) -> torch.Tensor:
        """
        Feature 2: Soft-NMS implementation for high-density object clusters.
        Decays detection scores based on overlap rather than hard deletion.
        Decay function: $S_i = S_i \exp(-\frac{\text{IoU}^2}{\sigma})$

        Args:
            boxes (torch.Tensor): Tensor of shape (N, 4) [xmin, ymin, xmax, ymax].
            scores (torch.Tensor): Tensor of shape (N,) confidence scores.
            sigma (float): Variance parameter for Gaussian decay.
            method (str): 'gaussian' or 'linear'.

        Returns:
            torch.Tensor: Kept indices after score decay filtering.
        """
        N = boxes.shape[0]
        if N == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        updated_scores = scores.clone()
        updated_boxes = boxes.clone()
        indices = torch.arange(N, device=self.device)

        for i in range(N):
            max_idx = torch.argmax(updated_scores[i:]) + i
            # Swap current element with max element
            updated_boxes[[i, max_idx]] = updated_boxes[[max_idx, i]]
            updated_scores[[i, max_idx]] = updated_scores[[max_idx, i]]
            indices[[i, max_idx]] = indices[[max_idx, i]]

            pos_boxes = updated_boxes[i + 1 :]
            if pos_boxes.shape[0] == 0:
                break

            # Compute IoU between current box and remaining boxes
            max_box = updated_boxes[i : i + 1]
            iou = ops.box_iou(max_box, pos_boxes).squeeze(0)

            # Apply decay weight
            if method == "gaussian":
                weight = torch.exp(-(iou ** 2) / sigma)
            elif method == "linear":
                weight = torch.where(iou > self.iou_threshold, 1.0 - iou, torch.ones_like(iou))
            else:
                weight = torch.ones_like(iou)

            updated_scores[i + 1 :] *= weight

        # Filter out boxes below confidence threshold
        keep = indices[updated_scores >= self.score_threshold]
        return keep

    def apply_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_soft_nms: bool = False,
        penalize_edges: bool = True,
    ) -> torch.Tensor:
        """
        Feature 1 & 4: Executes Batched GPU NMS or Soft-NMS.

        Args:
            boxes (torch.Tensor): Tensor (N, 4) in pixel coordinates.
            scores (torch.Tensor): Tensor (N,) confidence scores.
            labels (Optional[torch.Tensor]): Tensor (N,) class IDs.
            use_soft_nms (bool): Enables Soft-NMS instead of standard hard suppression.
            penalize_edges (bool): Applies edge boundary penalties prior to NMS.

        Returns:
            torch.Tensor: Indices of boxes retained after suppression.
        """
        if boxes.numel() == 0:
            return torch.tensor([], dtype=torch.int64, device=self.device)

        # Move tensors to hardware specified in config
        boxes = boxes.to(device=self.device, dtype=config.TORCH_DTYPE)
        scores = scores.to(device=self.device, dtype=config.TORCH_DTYPE)

        # Edge boundary fragment penalization
        if penalize_edges:
            scores = self.penalize_tile_edge_boxes(boxes, scores)

        # Initial confidence filtering
        valid_mask = scores >= self.score_threshold
        valid_indices = torch.nonzero(valid_mask).squeeze(1)

        if valid_indices.numel() == 0:
            return torch.tensor([], dtype=torch.int64, device=self.device)

        boxes = boxes[valid_indices]
        scores = scores[valid_indices]

        if labels is not None:
            labels = labels[valid_indices].to(device=self.device)
        else:
            labels = torch.zeros(boxes.shape[0], dtype=torch.int64, device=self.device)

        # Execute Soft-NMS or Standard Batched NMS
        if use_soft_nms:
            kept_relative_indices = self.soft_nms_pytorch(boxes, scores)
        else:
            if self.class_agnostic:
                # Class-agnostic: pass uniform class IDs
                dummy_labels = torch.zeros_like(labels)
                kept_relative_indices = ops.batched_nms(
                    boxes, scores, dummy_labels, self.iou_threshold
                )
            else:
                # Class-aware suppression
                kept_relative_indices = ops.batched_nms(
                    boxes, scores, labels, self.iou_threshold
                )

        final_indices = valid_indices[kept_relative_indices]
        return final_indices

    def process_dataframe(
        self,
        df: Union[pd.DataFrame, gpd.GeoDataFrame],
        bbox_cols: List[str] = ["xmin", "ymin", "xmax", "ymax"],
        score_col: str = "confidence",
        label_col: Optional[str] = "label",
        use_soft_nms: bool = False,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Feature 5: High-level entry point for processing Pandas/GeoPandas 
        DataFrames generated downstream from tile inference engines.

        Args:
            df (Union[pd.DataFrame, gpd.GeoDataFrame]): Input detection records.
            bbox_cols (List[str]): Columns storing pixel or bounding box coordinates.
            score_col (str): Column storing confidence scores.
            label_col (Optional[str]): Column storing class string names or IDs.
            use_soft_nms (bool): Whether to use Soft-NMS.

        Returns:
            Union[pd.DataFrame, gpd.GeoDataFrame]: Cleaned DataFrame with duplicate 
            tile-seam detections removed.
        """
        if df.empty:
            return df

        boxes_tensor = torch.tensor(df[bbox_cols].values, dtype=torch.float32)
        scores_tensor = torch.tensor(df[score_col].values, dtype=torch.float32)

        labels_tensor = None
        if label_col and label_col in df.columns:
            # Map categorical text labels to integer IDs for batched_nms kernel
            unique_labels = {lbl: idx for idx, lbl in enumerate(df[label_col].unique())}
            labels_tensor = torch.tensor(
                df[label_col].map(unique_labels).values, dtype=torch.int64
            )

        kept_indices = self.apply_nms(
            boxes=boxes_tensor,
            scores=scores_tensor,
            labels=labels_tensor,
            use_soft_nms=use_soft_nms,
        )

        cpu_indices = kept_indices.cpu().numpy()
        return df.iloc[cpu_indices].reset_index(drop=True)