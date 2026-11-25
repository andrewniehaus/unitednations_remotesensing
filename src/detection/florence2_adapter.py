"""
src/detection/florence2_adapter.py

Florence-2 Vision-Language Model Inference Adapter.

Provides offline, air-gapped integration for Microsoft Florence-2 models. Loads local 
weights from config.MODELS_DIR / "florence2" and executes multi-task prompt queries 
(Open-Vocabulary Detection, Phrase Grounding) over tiled image inputs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from unittest.mock import patch
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports

# Dynamic Configuration Import (Air-Gapped Workstation Standard)
from src import config

logger = logging.getLogger(__name__)

# Used only when per-box scores can't be read from the model output (a warning is logged)
FALLBACK_CONFIDENCE = 0.85


def _get_imports_without_flash_attn(filename: Union[str, Path]) -> List[str]:
    """
    Florence-2's bundled modeling_florence2.py imports flash_attn only when it is
    installed, but transformers' pre-load check treats it as a hard requirement.
    flash_attn needs an NVIDIA GPU and is hard to build on Windows, so drop it from
    the check for that file; the model then uses standard attention.
    """
    imports = get_imports(filename)
    if str(filename).endswith("modeling_florence2.py") and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


class Florence2Adapter:
    """
    Adapter for running local, air-gapped Florence-2 Vision-Language Model tasks.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize and load Florence-2 model and processor from local configuration paths.

        Args:
            model_dir (Optional[Path]): Directory containing local Florence-2 model weights.
        """
        self.model_dir = model_dir or (config.MODELS_DIR / "florence2")
        self.device = config.DEVICE
        self.torch_dtype = config.TORCH_DTYPE
        self.num_beams = 3

        # Load offline model weights and processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, 
            local_files_only=True,
            trust_remote_code=True
        )
        # Load without requiring flash_attn; "eager" attention runs on any CPU/GPU
        with patch("transformers.dynamic_module_utils.get_imports", _get_imports_without_flash_attn):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=self.torch_dtype,
                local_files_only=True,
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(self.device)

        self.model.eval()

    def predict_tile(
        self,
        image_input: Union[str, Path, Image.Image],
        task_prompt: str = "<OPEN_VOCABULARY_DETECTION>",
        text_input: str = "military compound . guard tower . trench line",
        x_offset: int = 0,
        y_offset: int = 0,
    ) -> pd.DataFrame:
        """
        Runs vision-language inference on a single tile or image path.

        Args:
            image_input (Union[str, Path, Image.Image]): PIL Image or image file path.
            task_prompt (str): Task identifier prompt (e.g., '<OPEN_VOCABULARY_DETECTION>').
            text_input (str): Target object queries separated by periods or commas.
            x_offset (int): X-coordinate offset for tile mosaic reconstruction.
            y_offset (int): Y-coordinate offset for tile mosaic reconstruction.

        Returns:
            pd.DataFrame: Detection records with pixel coordinates and labels.
        """
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        prompt = task_prompt + text_input if text_input else task_prompt

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            device=self.device, dtype=self.torch_dtype
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=self.num_beams,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_text = self.processor.batch_decode(
            generated_ids.sequences, skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        box_scores = self._box_scores(generated_ids, (image.width, image.height))

        return self._format_parsed_output(parsed_answer, task_prompt, x_offset, y_offset, box_scores)

    def _box_scores(self, generated: Any, image_size: Tuple[int, int]) -> Optional[List[float]]:
        """
        Per-box confidence from the probabilities of the generated tokens, in output order.

        Florence-2 writes each box as four <loc_N> tokens. A box's confidence is the geometric
        mean of those four tokens' probabilities, which is Florence-2's own 'avg_loc_scores'
        measure. Its <OD> parser applies this, but <OPEN_VOCABULARY_DETECTION> doesn't, so it
        is computed here. Returns None if the scores can't be lined up with the boxes.
        """
        try:
            transition_scores = self.model.compute_transition_scores(
                sequences=generated.sequences,
                scores=generated.scores,
                beam_indices=getattr(generated, "beam_indices", None),
                normalize_logits=self.num_beams == 1,  # beam search scores are already log-probabilities
            )[0].tolist()
            token_ids = generated.sequences[0].tolist()[1:]  # drop the decoder start token
            if len(token_ids) != len(transition_scores):
                logger.warning(f"{len(token_ids)} tokens but {len(transition_scores)} token scores.")
                return None

            post_processor = self.processor.post_processor
            text, spans = post_processor.decode_with_spans(self.processor.tokenizer, token_ids)
            if "<poly>" in text:
                return None  # polygon output; no boxes to score
            instances = post_processor.parse_description_with_bboxes_from_text_and_spans(
                text, spans=spans, scores=transition_scores, score_mode="avg_loc_scores", image_size=image_size
            )
            return [instance["score"] for instance in instances]
        except Exception as e:
            logger.warning(f"Could not compute Florence-2 box scores: {e}")
            return None

    def _format_parsed_output(
        self,
        parsed_data: Dict[str, Any],
        task_prompt: str,
        x_offset: int,
        y_offset: int,
        box_scores: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Formats raw model output dictionaries into standard Pandas DataFrames.
        """
        records = []

        if task_prompt in parsed_data:
            data = parsed_data[task_prompt]
            bboxes = data.get("bboxes", [])
            # <OD> returns "labels"; <OPEN_VOCABULARY_DETECTION> returns "bboxes_labels"
            labels = data.get("labels", data.get("bboxes_labels", []))

            if bboxes and (box_scores is None or len(box_scores) != len(bboxes)):
                logger.warning(
                    f"No per-box scores for {len(bboxes)} boxes; using fallback confidence {FALLBACK_CONFIDENCE}."
                )
                box_scores = None

            for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                # Florence-2 box format: [xmin, ymin, xmax, ymax]
                xmin = float(bbox[0]) + x_offset
                ymin = float(bbox[1]) + y_offset
                xmax = float(bbox[2]) + x_offset
                ymax = float(bbox[3]) + y_offset

                records.append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "label": str(label).strip(),
                    "confidence": box_scores[i] if box_scores is not None else FALLBACK_CONFIDENCE,
                })

        # Feature 4: Clear VRAM cache after execution
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pd.DataFrame(records)

    def predict_batch(
        self,
        tile_manifest: List[Dict[str, Any]],
        task_prompt: str = "<OPEN_VOCABULARY_DETECTION>",
        text_input: str = "military compound . guard tower"
    ) -> pd.DataFrame:
        """
        Feature 2: Processes a batch list of tile paths and offsets.

        Args:
            tile_manifest (List[Dict[str, Any]]): List of dicts containing 'path', 'x_offset', 'y_offset'.
            task_prompt (str): Task identifier prompt.
            text_input (str): Target detection query string.

        Returns:
            pd.DataFrame: Consolidated batch detections.
        """
        batch_results = []
        for item in tile_manifest:
            df = self.predict_tile(
                image_input=item["path"],
                task_prompt=task_prompt,
                text_input=text_input,
                x_offset=item.get("x_offset", 0),
                y_offset=item.get("y_offset", 0)
            )
            if not df.empty:
                batch_results.append(df)

        if not batch_results:
            return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "label", "confidence"])

        return pd.concat(batch_results, ignore_index=True)

    # [EDIT 2026-09-28 | Claude Code for charliefp03-dg] Added predict() so this adapter satisfies the
    # interface BatchInferenceRunner calls: predict(image_data=..., targets=..., **kwargs) returning
    # [{"bbox": [xmin, ymin, xmax, ymax], "label": str, "confidence": float}]. It wraps predict_tile().
    def predict(
        self,
        image_data: Union[str, Path, Image.Image],
        targets: Union[str, List[str]],
        task_prompt: str = "<OPEN_VOCABULARY_DETECTION>",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Standard detection interface used by BatchInferenceRunner.

        Args:
            image_data: Tile path or PIL Image.
            targets: Catalog key(s) from PromptManager (e.g. 'tents') or raw prompt text.
            task_prompt: Florence-2 task token.
            **kwargs: Ignored; accepted for interface compatibility (e.g. box_threshold).

        Returns:
            List of detection dicts in local tile pixel coordinates.
        """
        from src.detection.prompt_manager import PromptManager

        target_list = [targets] if isinstance(targets, str) else list(targets)
        prompt_manager = PromptManager()

        # Florence-2 treats a joined prompt as one phrase, so query each target separately
        # and label detections with the target that produced them.
        detections = []
        for target in target_list:
            df = self.predict_tile(
                image_input=image_data,
                task_prompt=task_prompt,
                text_input=prompt_manager.get_target(target)["primary_prompt"],
            )
            detections.extend(
                {
                    "bbox": [row.xmin, row.ymin, row.xmax, row.ymax],
                    "label": target,
                    "confidence": row.confidence,
                }
                for row in df.itertuples(index=False)
            )

        return detections