"""
Main execution entry point for offline VLM object detection.
Instantiates the selected model adapter with local weights and orchestrates 
batch inference over chipped intermediate imagery.
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from src import config
from src.detection.batch_runner import BatchInferenceRunner


# Configure logging for robust debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.PROJECT_ROOT / "detection_run.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


def load_adapter(model_choice: str):
    """
    Safely loads the requested model adapter and verifies local offline weights exist.
    """
    if model_choice.lower() == "florence2":
        if not config.FLORENCE2_DIR.exists() or not any(config.FLORENCE2_DIR.iterdir()):
            logger.critical(f"Florence-2 model weights missing in {config.FLORENCE2_DIR}. Ensure they are downloaded for air-gapped use.")
            sys.exit(1)
        
        try:
            from src.detection.florence2_adapter import Florence2Adapter
            logger.info(f"Initializing Florence-2 Adapter on {config.DEVICE}...")
            return Florence2Adapter(model_dir=config.FLORENCE2_DIR)
        except ImportError as e:
            logger.critical(f"Failed to import Florence2Adapter. Ensure src/detection/florence2_adapter.py exists. Error: {e}")
            sys.exit(1)

    elif model_choice.lower() == "groundingdino":
        if not config.GROUNDING_DINO_DIR.exists() or not any(config.GROUNDING_DINO_DIR.iterdir()):
            logger.critical(f"Grounding DINO model weights missing in {config.GROUNDING_DINO_DIR}.")
            sys.exit(1)

        try:
            from src.detection.grounding_dino_adapter import GroundingDINOAdapter
            logger.info(f"Initializing Grounding DINO Adapter on {config.DEVICE}...")
            # Adapter reads device/dtype from config and resolves the folder inside models/
            return GroundingDINOAdapter(model_directory_name=config.GROUNDING_DINO_DIR.name)
        except ImportError as e:
            logger.critical(f"Failed to import GroundingDINOAdapter. Error: {e}")
            sys.exit(1)
    else:
        logger.error(f"Unsupported model choice: {model_choice}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Offline VLM Batch Inference Execution")
    parser.add_argument("--model", type=str, choices=["florence2", "groundingdino"], default="florence2",
                        help="Select the VLM to use for detection.")
    parser.add_argument("--profile", type=str, default="disaster_response",
                        help="Target prompt profile defined in src/config.py (e.g., disaster_response, military_infrastructure).")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Optional: Path to a specific chipping manifest JSON. If not provided, runs on all files in the interim directory.")
    parser.add_argument("--ext", type=str, default=".tif",
                        help="File extension to look for if running on a directory instead of a manifest.")
    
    args = parser.parse_args()

    # Resolve target prompts
    if args.profile in config.TARGET_PROMPTS:
        targets = config.TARGET_PROMPTS[args.profile]
        logger.info(f"Loaded target profile '{args.profile}': {targets}")
    else:
        logger.warning(f"Profile '{args.profile}' not found in config.TARGET_PROMPTS. Defaulting to DEFAULT_TARGETS.")
        targets = getattr(config, 'DEFAULT_TARGETS', [])
        if not targets:
            logger.critical("No target prompts resolved. Exiting.")
            sys.exit(1)

    # Initialize Adapter and Runner
    try:
        adapter = load_adapter(args.model)
        runner = BatchInferenceRunner(model_adapter=adapter)
    except Exception as e:
        logger.critical(f"Failed to initialize model architecture: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

    # Execute inference pipeline
    try:
        if args.manifest:
            manifest_path = Path(args.manifest)
            if not manifest_path.is_absolute():
                manifest_path = config.INTERIM_DIR / manifest_path
                
            logger.info(f"Starting execution based on manifest: {manifest_path}")
            runner.run_on_manifest(
                manifest_path=manifest_path,
                targets=targets,
                output_dir=config.INTERIM_DIR / "vlm_results"
            )
        else:
            logger.info(f"Starting execution on directory: {config.INTERIM_DIR}")
            runner.run_on_directory(
                input_dir=config.INTERIM_DIR,
                targets=targets,
                file_extension=args.ext,
                output_dir=config.INTERIM_DIR / "vlm_results"
            )
    except Exception as e:
        logger.error(f"Execution failed during batch processing: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("Pipeline execution finished. Checking for unreleased VRAM...")
        if hasattr(runner, '_clear_vram'):
            runner._clear_vram()

if __name__ == "__main__":
    main()