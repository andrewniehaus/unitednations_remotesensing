"""
Prompt Manager for Open-Vocabulary Vision-Language Models (VLMs).
Provides structured target catalogs, prompt formatting, synonym expansion,
and class-specific detection thresholds for air-gapped VLM workflows.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src import config

logger = logging.getLogger(__name__)


# Default domain-specific target catalog optimized for Remote Sensing, ISR, and HADR
DEFAULT_TARGET_CATALOG: Dict[str, Dict[str, Any]] = {
    "tents": {
        "primary_prompt": "refugee tent, humanitarian tent, military encampment tent",
        "synonyms": [
            "canvas shelter",
            "temporary housing tent",
            "encampment structure",
            "displaced person shelter",
        ],
        "box_threshold": 0.25,
        "text_threshold": 0.25,
    },
    "trenchlines": {
        "primary_prompt": "military trench line, defensive earthworks, zigzag trench, field fortification",
        "synonyms": [
            "earthen trench",
            "defensive ditch",
            "fortified trench",
            "excavated trench",
        ],
        "box_threshold": 0.20,
        "text_threshold": 0.20,
    },
    "damaged_buildings": {
        "primary_prompt": "damaged building, destroyed structure, collapsed roof, rubble, debris",
        "synonyms": [
            "bomb damaged structure",
            "earthquake damaged building",
            "partially collapsed building",
            "ruined building",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
    },
    "aircraft": {
        "primary_prompt": "airplane, fighter jet, cargo aircraft, helicopter, military plane",
        "synonyms": [
            "fixed wing aircraft",
            "rotary wing aircraft",
            "tarmac airplane",
            "runway plane",
        ],
        "box_threshold": 0.35,
        "text_threshold": 0.30,
    },
    "military_compounds": {
        "primary_prompt": "military base compound, perimeter wall, guard tower, fortified outpost",
        "synonyms": [
            "secure compound",
            "military facility",
            "walled compound",
            "command post",
        ],
        "box_threshold": 0.25,
        "text_threshold": 0.25,
    },
    "guard_towers": {
        "primary_prompt": "guard tower, observation tower, watchtower, security tower",
        "synonyms": [
            "elevated watchtower",
            "perimeter tower",
            "surveillance tower",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
    },
    "military_vehicles": {
        "primary_prompt": "armored vehicle, military truck, tank, missile launcher, convoy vehicle",
        "synonyms": [
            "armored personnel carrier",
            "tracked vehicle",
            "camouflage vehicle",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
    },
    "emergency_assets": {
        "primary_prompt": "emergency response vehicle, ambulance, fire truck, aid supply stack",
        "synonyms": [
            "humanitarian aid crate",
            "disaster response vehicle",
            "emergency generator",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
    },
}


class PromptManager:
    """
    Manages structured prompt templates, synonym expansion, and target-specific thresholds
    for open-vocabulary detection models (e.g., Florence-2, Grounding DINO).
    """

    def __init__(self, catalog_path: Optional[Path] = None):
        """
        Initializes the PromptManager with a default or custom target catalog.

        Args:
            catalog_path: Optional Path to a custom JSON target catalog file.
        """
        self.catalog_path = catalog_path or (config.PROCESSED_DIR / "target_catalog.json")
        self.catalog: Dict[str, Dict[str, Any]] = {}

        if self.catalog_path.exists():
            self.load_catalog(self.catalog_path)
        else:
            self.catalog = DEFAULT_TARGET_CATALOG.copy()

    def get_target(self, target_key: str) -> Dict[str, Any]:
        """
        Retrieves full configuration metadata for a specific target key.

        Args:
            target_key: Key corresponding to the target category (e.g., 'tents').

        Returns:
            Dictionary containing primary prompt, synonyms, and default thresholds.
        """
        key = target_key.lower().strip()
        if key in self.catalog:
            return self.catalog[key]

        logger.warning(f"Target key '{target_key}' not found in catalog. Using as raw prompt string.")
        return {
            "primary_prompt": target_key,
            "synonyms": [],
            "box_threshold": 0.25,
            "text_threshold": 0.25,
        }

    def format_for_florence2(
        self,
        target_query: str,
        task_token: str = "<CAPTION_TO_PHRASE>",
        include_synonyms: bool = False,
    ) -> str:
        """
        Formats a query or target key specifically for Microsoft Florence-2 syntax.

        Args:
            target_query: Catalog key (e.g., 'tents') or raw ad-hoc text string.
            task_token: Task prefix token (e.g., '<CAPTION_TO_PHRASE>' or '<OD>').
            include_synonyms: Whether to append catalog synonyms to the prompt.

        Returns:
            Formatted prompt string ready for the Florence-2 processor.
        """
        target_info = self.get_target(target_query)
        prompt_text = target_info["primary_prompt"]

        if include_synonyms and target_info.get("synonyms"):
            synonyms_str = ", ".join(target_info["synonyms"])
            prompt_text = f"{prompt_text}, {synonyms_str}"

        return f"{task_token} {prompt_text}".strip()

    def format_for_grounding_dino(
        self,
        targets: Union[str, List[str]],
        include_synonyms: bool = False,
    ) -> str:
        """
        Formats queries for Grounding DINO syntax (dot-separated phrases).

        Args:
            targets: Single target key/string or list of target keys/strings.
            include_synonyms: Whether to include expanded synonyms in the query.

        Returns:
            Dot-separated string formatted for Grounding DINO (e.g., "tent . building .").
        """
        if isinstance(targets, str):
            targets = [targets]

        phrases: List[str] = []
        for t in targets:
            info = self.get_target(t)
            phrases.append(info["primary_prompt"])
            if include_synonyms and info.get("synonyms"):
                phrases.extend(info["synonyms"])

        # Grounding DINO requires period separators between distinct class queries
        formatted_prompt = " . ".join(phrases)
        if not formatted_prompt.endswith("."):
            formatted_prompt += " ."

        return formatted_prompt

    def get_thresholds(
        self,
        target_key: str,
        override_box: Optional[float] = None,
        override_text: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Retrieves recommended box and text confidence thresholds for a target.

        Args:
            target_key: Key corresponding to the target category.
            override_box: Optional custom box threshold.
            override_text: Optional custom text threshold.

        Returns:
            Dictionary with 'box_threshold' and 'text_threshold'.
        """
        info = self.get_target(target_key)
        return {
            "box_threshold": override_box if override_box is not None else info.get("box_threshold", 0.25),
            "text_threshold": override_text if override_text is not None else info.get("text_threshold", 0.25),
        }

    def add_target(
        self,
        key: str,
        primary_prompt: str,
        synonyms: Optional[List[str]] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> None:
        """
        Adds or updates a target category in the catalog during runtime.
        """
        self.catalog[key.lower().strip()] = {
            "primary_prompt": primary_prompt,
            "synonyms": synonyms or [],
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        }

    def save_catalog(self, output_path: Optional[Path] = None) -> Path:
        """
        Saves the current catalog to a local JSON file in data/processed/.
        """
        path = output_path or self.catalog_path
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.catalog, f, indent=4)

        logger.info(f"Target catalog saved to {path}")
        return path

    def load_catalog(self, input_path: Path) -> None:
        """
        Loads a target catalog from a local JSON file.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)
        logger.info(f"Target catalog loaded from {input_path}")