"""
src/Testing/eval_metrics.py

Quantitative Evaluation and Validation Metrics for Geospatial AI Workflows.

Computes IoU, Precision, Recall, F1-Score, mAP@50, mAP@50:95, and area error metrics 
for object detection and spatial change validation. Integrates Hungarian bipartite 
matching to eliminate double-counting of detections.
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
from scipy.optimize import linear_sum_assignment

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config


class SpatialEvalMetrics:
    """
    Evaluates geospatial AI detection performance against ground truth datasets.
    """

    def __init__(self, iou_threshold: float = 0.50):
        """
        Initialize evaluation metrics with default IoU threshold.

        Args:
            iou_threshold (float): IoU threshold for counting a detection as True Positive.
        """
        self.iou_threshold = iou_threshold

    @staticmethod
    def calculate_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculates Intersection over Union (IoU) between two boxes [xmin, ymin, xmax, ymax].

        Args:
            box1 (np.ndarray): Bounding box 1.
            box2 (np.ndarray): Bounding box 2.

        Returns:
            float: IoU value in [0, 1].
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def match_predictions_hungarian(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        iou_thresh: float = 0.50
    ) -> Tuple[int, int, int]:
        """
        Feature 1: Optimal Bipartite Matching using the Hungarian Algorithm.
        
        Args:
            gt_boxes (np.ndarray): Array of shape (M, 4) ground truth boxes.
            pred_boxes (np.ndarray): Array of shape (N, 4) predicted boxes.
            iou_thresh (float): Minimum IoU threshold to consider a match valid.

        Returns:
            Tuple[int, int, int]: (True Positives, False Positives, False Negatives)
        """
        num_gt = len(gt_boxes)
        num_pred = len(pred_boxes)

        if num_gt == 0 and num_pred == 0:
            return 0, 0, 0
        if num_gt == 0:
            return 0, num_pred, 0
        if num_pred == 0:
            return 0, 0, num_gt

        # Build Cost Matrix (1 - IoU)
        iou_matrix = np.zeros((num_gt, num_pred))
        for i in range(num_gt):
            for j in range(num_pred):
                iou_matrix[i, j] = self.calculate_box_iou(gt_boxes[i], pred_boxes[j])

        cost_matrix = 1.0 - iou_matrix
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

        tp = 0
        for g, p in zip(gt_indices, pred_indices):
            if iou_matrix[g, p] >= iou_thresh:
                tp += 1

        fp = num_pred - tp
        fn = num_gt - tp
        return tp, fp, fn

    def compute_precision_recall_f1(
        self, tp: int, fp: int, fn: int
    ) -> Dict[str, float]:
        """
        Computes Precision, Recall, and F1-Score from counts.

        Returns:
            Dict[str, float]: Metrics dictionary containing precision, recall, and f1.
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    def compute_map_at_thresholds(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        iou_thresholds: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Feature 2: Calculates Average Precision (mAP@50 and mAP@50:95).

        Args:
            gt_boxes (np.ndarray): Ground truth boxes (M, 4).
            pred_boxes (np.ndarray): Predicted boxes (N, 4).
            pred_scores (np.ndarray): Prediction confidence scores (N,).
            iou_thresholds (Optional[List[float]]): List of IoU thresholds.

        Returns:
            Dict[str, float]: Calculated mAP metrics.
        """
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.50, 1.00, 0.05).tolist()

        # Sort predictions by confidence score descending
        sort_idx = np.argsort(-pred_scores)
        sorted_pred_boxes = pred_boxes[sort_idx] if len(pred_boxes) > 0 else pred_boxes

        ap_scores = []
        for thresh in iou_thresholds:
            tp, fp, fn = self.match_predictions_hungarian(gt_boxes, sorted_pred_boxes, iou_thresh=thresh)
            metrics = self.compute_precision_recall_f1(tp, fp, fn)
            ap_scores.append(metrics["precision"] * metrics["recall"])  # Simplified AP estimate

        map_50 = ap_scores[0] if len(ap_scores) > 0 else 0.0
        map_50_95 = float(np.mean(ap_scores)) if len(ap_scores) > 0 else 0.0

        return {
            "mAP_50": round(map_50, 4),
            "mAP_50_95": round(map_50_95, 4),
        }

    def compute_area_bias_metrics(
        self,
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        bbox_cols: List[str] = ["xmin", "ymin", "xmax", "ymax"]
    ) -> Dict[str, float]:
        """
        Feature 4: Measures target surface area estimation error.

        Args:
            gt_df (pd.DataFrame): Ground truth detections.
            pred_df (pd.DataFrame): Predicted detections.
            bbox_cols (List[str]): Bounding box coordinate columns.

        Returns:
            Dict[str, float]: Total area metrics and area ratio bias.
        """
        def calc_total_area(df):
            if df.empty:
                return 0.0
            widths = df[bbox_cols[2]] - df[bbox_cols[0]]
            heights = df[bbox_cols[3]] - df[bbox_cols[1]]
            return float((widths * heights).sum())

        gt_area = calc_total_area(gt_df)
        pred_area = calc_total_area(pred_df)
        area_diff = pred_area - gt_area
        area_ratio = pred_area / gt_area if gt_area > 0 else 0.0

        return {
            "gt_total_area": round(gt_area, 2),
            "pred_total_area": round(pred_area, 2),
            "area_difference": round(area_diff, 2),
            "area_ratio": round(area_ratio, 4),
        }

    def evaluate_pipeline_output(
        self,
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        label_col: str = "label",
        bbox_cols: List[str] = ["xmin", "ymin", "xmax", "ymax"],
        score_col: str = "confidence"
    ) -> pd.DataFrame:
        """
        Feature 3: Class-Wise Evaluation Summary Report.

        Args:
            gt_df (pd.DataFrame): Ground truth records.
            pred_df (pd.DataFrame): Model detection records.
            label_col (str): Target class name column.
            bbox_cols (List[str]): Bounding box column names.
            score_col (str): Confidence score column name.

        Returns:
            pd.DataFrame: Evaluation report aggregated by class category.
        """
        classes = sorted(list(set(gt_df[label_col].unique()).union(set(pred_df[label_col].unique()))))
        summary_rows = []

        for cls in classes:
            sub_gt = gt_df[gt_df[label_col] == cls] if not gt_df.empty else pd.DataFrame()
            sub_pred = pred_df[pred_df[label_col] == cls] if not pred_df.empty else pd.DataFrame()

            gt_b = sub_gt[bbox_cols].values if not sub_gt.empty else np.empty((0, 4))
            pred_b = sub_pred[bbox_cols].values if not sub_pred.empty else np.empty((0, 4))
            pred_s = sub_pred[score_col].values if (not sub_pred.empty and score_col in sub_pred.columns) else np.ones(len(pred_b))

            tp, fp, fn = self.match_predictions_hungarian(gt_b, pred_b, self.iou_threshold)
            prf = self.compute_precision_recall_f1(tp, fp, fn)
            maps = self.compute_map_at_thresholds(gt_b, pred_b, pred_s)
            area_m = self.compute_area_bias_metrics(sub_gt, sub_pred, bbox_cols)

            row = {"class": cls, **prf, **maps, **area_m}
            summary_rows.append(row)

        return pd.DataFrame(summary_rows)