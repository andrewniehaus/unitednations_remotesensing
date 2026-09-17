"""
src/detection/ensemble_detector.py

Multi-Model Ensemble and Prediction Reconciler.

Combines, deduplicates, and fuses overlapping bounding box predictions from multiple 
independent Vision-Language Model adapters (e.g., Florence-2, Grounding DINO). 
Employs Weighted Box Fusion (WBF), consensus scoring, and cross-model label alignment.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torchvision.ops as ops

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class EnsembleDetector:
    """
    Reconciles and fuses multi-model bounding box predictions across VLM backbones.
    """

    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        iou_threshold: float = 0.50,
        consensus_boost: float = 0.15,
    ):
        """
        Initialize ensemble fusion parameters.

        Args:
            model_weights (Optional[Dict[str, float]]): Relative reliability weights per model backbone.
            iou_threshold (float): IoU threshold for matching candidate boxes across models.
            consensus_boost (float): Multiplicative confidence bonus applied when multiple models agree.
        """
        self.model_weights = model_weights or {"florence2": 1.0, "grounding_dino": 1.2}
        self.iou_threshold = iou_threshold
        self.consensus_boost = consensus_boost
        self.device = config.DEVICE

    def weighted_box_fusion(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        model_ids: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Feature 1: Weighted Box Fusion (WBF).
        Blends coordinates of overlapping bounding boxes weighted by confidence scores 
        and model backbone reliability rather than suppressing lower-scoring detections.

        Formula: $X_{\text{fused}} = \frac{\sum w_i \cdot S_i \cdot X_i}{\sum w_i \cdot S_i}$

        Args:
            boxes (torch.Tensor): Box tensor of shape (N, 4) in [xmin, ymin, xmax, ymax].
            scores (torch.Tensor): Confidence score tensor of shape (N,).
            model_ids (List[str]): List of source model names corresponding to each box.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[int]]: (Fused Boxes, Fused Scores, Agreement Counts)
        """
        if boxes.numel() == 0:
            return boxes, scores, []

        weights = torch.tensor(
            [self.model_weights.get(m, 1.0) for m in model_ids],
            dtype=config.TORCH_DTYPE,
            device=self.device,
        )

        # Compute pairwise IoU matrix
        iou_matrix = ops.box_iou(boxes, boxes)
        visited = torch.zeros(len(boxes), dtype=torch.bool, device=self.device)

        fused_boxes = []
        fused_scores = []
        agreement_counts = []

        for i in range(len(boxes)):
            if visited[i]:
                continue

            # Identify matching overlapping boxes from cluster
            matches = torch.nonzero(iou_matrix[i] >= self.iou_threshold).squeeze(1)
            visited[matches] = True

            cluster_boxes = boxes[matches]
            cluster_scores = scores[matches]
            cluster_weights = weights[matches]

            # Weighted combination factor: $W_i = w_i \cdot S_i$
            combined_weights = (cluster_scores * cluster_weights).unsqueeze(1)
            weight_sum = combined_weights.sum(dim=0)

            # Feature 1: Coordinate Fusion
            weighted_box = (cluster_boxes * combined_weights).sum(dim=0) / torch.clamp(weight_sum, min=1e-6)
            
            # Feature 2: Consensus Scoring
            unique_models = len(set([model_ids[m.item()] for m in matches]))
            base_score = torch.max(cluster_scores)
            
            if unique_models > 1:
                boosted_score = torch.clamp(base_score * (1.0 + self.consensus_boost * (unique_models - 1)), max=1.0)
            else:
                boosted_score = base_score

            fused_boxes.append(weighted_box)
            fused_scores.append(boosted_score)
            agreement_counts.append(unique_models)

        return torch.stack(fused_boxes), torch.stack(fused_scores), agreement_counts

    def combine_predictions(
        self,
        predictions_df: pd.DataFrame,
        bbox_cols: Tuple[str, str, str, str] = ("xmin", "ymin", "xmax", "ymax"),
        score_col: str = "confidence",
        model_col: str = "model_source",
        label_col: str = "label",
    ) -> pd.DataFrame:
        """
        Feature 3, 4 & 5: Executes cross-model label consolidation, spatial ensemble 
        fusion, and source metadata provenance tracking over prediction DataFrames.

        Args:
            predictions_df (pd.DataFrame): Combined raw prediction outputs from all models.
            bbox_cols (Tuple[str, str, str, str]): Bounding box coordinate column names.
            score_col (str): Confidence score column name.
            model_col (str): Model source provenance column name.
            label_col (str): Target class string column name.

        Returns:
            pd.DataFrame: Deduplicated, fused prediction DataFrame with consensus scores.
        """
        if predictions_df.empty:
            return predictions_df

        fused_records = []

        # Process each class label group independently
        for class_label, group in predictions_df.groupby(label_col):
            boxes_tensor = torch.tensor(
                group[list(bbox_cols)].values, dtype=config.TORCH_DTYPE, device=self.device
            )
            scores_tensor = torch.tensor(
                group[score_col].values, dtype=config.TORCH_DTYPE, device=self.device
            )
            model_sources = group[model_col].tolist()

            fused_b, fused_s, counts = self.weighted_box_fusion(
                boxes=boxes_tensor,
                scores=scores_tensor,
                model_ids=model_sources,
            )

            cpu_boxes = fused_b.cpu().numpy()
            cpu_scores = fused_s.cpu().numpy()

            for idx in range(len(cpu_boxes)):
                fused_records.append({
                    bbox_cols[0]: float(cpu_boxes[idx][0]),
                    bbox_cols[1]: float(cpu_boxes[idx][1]),
                    bbox_cols[2]: float(cpu_boxes[idx][2]),
                    bbox_cols[3]: float(cpu_boxes[idx][3]),
                    label_col: class_label,
                    score_col: float(cpu_scores[idx]),
                    "consensus_models": counts[idx],
                    "ensemble_method": "weighted_box_fusion",
                })

        return pd.DataFrame(fused_records)