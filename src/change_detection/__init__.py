"""
src/change_detection/__init__.py

Temporal Change Detection Submodule.
Exposes foundational encoding, latent similarity calculation, change mask generation,
and full-pipeline orchestration for offline change detection.
"""

from .clay_encoder import ClayEncoder
from .similarity_engine import LatentSimilarityEngine
from .mask_generator import ChangeMaskGenerator
from .change_pipeline import ChangeDetectionPipeline

__all__ = [
    "ClayEncoder",
    "LatentSimilarityEngine",
    "ChangeMaskGenerator",
    "ChangeDetectionPipeline",
]