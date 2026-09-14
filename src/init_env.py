from pathlib import Path
import config  # Assuming this runs from src/ or your module paths are set up

def create_project_directories():
    """Checks for and creates required data and model directories."""
    
    # Referencing the exact variables established in your config.py
    required_dirs = [
        config.MODELS_DIR,
        config.RAW_DIR,
        config.INTERIM_DIR,
        config.PROCESSED_DIR
    ]
    
    for dir_path in required_dirs:
        path_obj = Path(dir_path)
        if not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)
            print(f"Created missing directory: {path_obj}")
        else:
            print(f"Directory verified: {path_obj}")

if __name__ == "__main__":
    create_project_directories()