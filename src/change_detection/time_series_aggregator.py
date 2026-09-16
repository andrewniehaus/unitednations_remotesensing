"""
src/change_detection/time_series_aggregator.py

Multi-Temporal Sequence Analysis Module.

Tracks latent similarity decay across a chronological stack of imagery to pinpoint 
the exact time window an asset was altered, damaged, or destroyed, addressing 
long-term structural monitoring requirements.
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config
from src.change_detection.clay_encoder import ClayEncoder
from src.change_detection.similarity_engine import LatentSimilarityEngine


class TimeSeriesAggregator:
    """
    Parses chronological image stacks to pinpoint temporal events.
    """

    def __init__(self, baseline_threshold: float = 0.85):
        """
        Initialize dependencies.
        
        Args:
            baseline_threshold (float): Similarity threshold below which an object 
                                        is considered destroyed or heavily altered.
        """
        self.encoder = ClayEncoder()
        self.similarity_engine = LatentSimilarityEngine(metric="cosine")
        self.baseline_threshold = baseline_threshold

    def analyze_sequence(
        self, chronological_tiles: List[Dict[str, str]]
    ) -> pd.DataFrame:
        """
        Extracts embeddings for a chronological sequence and tracks similarity against T1.

        Args:
            chronological_tiles (List[Dict[str, str]]): List of dicts with 'date' and 'path'.
                Example: [{'date': '2023-01-01', 'path': '/path/to/t1.tif'}, ...]

        Returns:
            pd.DataFrame: Sequence analysis tracking similarity scores and event flagging.
        """
        if len(chronological_tiles) < 2:
            return pd.DataFrame()

        results = []
        baseline_embedding = None

        for idx, item in enumerate(chronological_tiles):
            date_str = item["date"]
            tile_path = Path(item["path"])
            
            # Extract standard feature embedding (1, D)
            current_embedding = self.encoder.extract_features(tile_path)

            if idx == 0:
                baseline_embedding = current_embedding
                results.append({
                    "date": date_str,
                    "similarity_to_baseline": 1.0,
                    "status": "baseline"
                })
                continue

            # Compute similarity against the established baseline ($T_n$ vs $T_1$)
            sim_score = self.similarity_engine.compute_similarity(
                baseline_embedding, current_embedding
            ).item()

            status = "intact" if sim_score >= self.baseline_threshold else "altered/destroyed"

            results.append({
                "date": date_str,
                "similarity_to_baseline": round(sim_score, 4),
                "status": status
            })

            # Clean up local memory loop
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        df_results = pd.DataFrame(results)
        return df_results

    def identify_event_window(self, df_sequence: pd.DataFrame) -> Optional[Dict[str, str]]:
        """
        Identifies the exact date window where the structural collapse/change occurred.
        """
        if df_sequence.empty:
            return None

        # Find the first date where status shifted from intact to altered
        altered_rows = df_sequence[df_sequence["status"] == "altered/destroyed"]
        
        if altered_rows.empty:
            return {"event_window": "No significant change detected in sequence."}

        # The event occurred between the date preceding the drop and the date of the drop
        drop_idx = altered_rows.index[0]
        if drop_idx > 0:
            pre_event_date = df_sequence.iloc[drop_idx - 1]["date"]
            post_event_date = df_sequence.iloc[drop_idx]["date"]
            
            return {
                "pre_event": pre_event_date,
                "post_event": post_event_date,
                "event_window": f"{pre_event_date} to {post_event_date}"
            }
        
        return {"event_window": "Change detected immediately after baseline."}