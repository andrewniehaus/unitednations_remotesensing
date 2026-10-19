import sys
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Ensure project root is in python path to resolve src/config.py
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import src.config as config


def inject_synthetic_objects(data: np.ndarray, transform):
    """
    Injects synthetic target features into a 3-band array (C, H, W) based on 0.5m GSD:
    - Tent cluster (bright 10x10 px squares)
    - Trench line (dark 3px-wide jagged vector line)
    - Military compound (100x100 px outline with interior structures)
    - Vehicles/Aircraft (small high-contrast 6x12 px rectangles)
    
    Returns array and a list of ground truth bounding boxes.
    """
    ground_truth = []

    # 1. Tent Cluster (3 tents, 10x10 px each at 0.5m GSD = 5m x 5m)
    tent_coords = [(800, 800), (800, 825), (825, 800)]
    for idx, (r, c) in enumerate(tent_coords):
        # Bright canvas with dark shadow edge
        data[:, r:r+10, c:c+10] = np.array([210, 205, 190], dtype=np.uint8)[:, None, None]
        data[:, r+9:r+10, c:c+10] = 30  # Shadow line
        
        # Ground truth tracking [ymin, xmin, ymax, xmax] in pixel space
        ground_truth.append({
            "id": f"tent_{idx+1}",
            "label": "military tent",
            "pixel_bbox": [r, c, r+10, c+10]
        })

    # 2. Trench Line (Zig-zagging dark line, 3px wide)
    trench_points = [(1500, 1200), (1550, 1280), (1520, 1350), (1600, 1450)]
    for i in range(len(trench_points) - 1):
        r0, c0 = trench_points[i]
        r1, c1 = trench_points[i+1]
        num_steps = max(abs(r1 - r0), abs(c1 - c0))
        r_steps = np.linspace(r0, r1, num_steps, dtype=int)
        c_steps = np.linspace(c0, c1, num_steps, dtype=int)
        for r, c in zip(r_steps, c_steps):
            data[:, max(0, r-1):r+2, max(0, c-1):c+2] = 25  # Dark excavated line
            
    ground_truth.append({
        "id": "trench_1",
        "label": "trench line",
        "pixel_bbox": [1500, 1200, 1600, 1450]
    })

    # 3. Military Compound (100x100 px = 50m x 50m compound wall + internal buildings)
    cr, cc = 2500, 2500
    # Perimeter Wall (3px thick, bright concrete)
    data[:, cr:cr+100, cc:cc+3] = 190
    data[:, cr:cr+100, cc+97:cc+100] = 190
    data[:, cr:cr+3, cc:cc+100] = 190
    data[:, cr+97:cr+100, cc:cc+100] = 190
    
    # Internal structures (Two 20x20 px buildings inside compound)
    data[:, cr+15:cr+35, cc+15:cc+35] = np.array([160, 150, 140], dtype=np.uint8)[:, None, None]
    data[:, cr+50:cr+70, cc+50:cc+70] = np.array([160, 150, 140], dtype=np.uint8)[:, None, None]
    
    ground_truth.append({
        "id": "compound_1",
        "label": "military compound",
        "pixel_bbox": [cr, cc, cr+100, cc+100]
    })

    # 4. Vehicles (Two 6x12 px high-contrast rectangles near compound)
    vehicle_coords = [(2620, 2510), (2620, 2530)]
    for idx, (r, c) in enumerate(vehicle_coords):
        data[:, r:r+6, c:c+12] = np.array([230, 230, 240], dtype=np.uint8)[:, None, None]
        ground_truth.append({
            "id": f"vehicle_{idx+1}",
            "label": "vehicle",
            "pixel_bbox": [r, c, r+6, c+12]
        })

    # Convert pixel bboxes to UTM geospatial bounding coordinates using Affine transform
    for gt in ground_truth:
        ymin, xmin, ymax, xmax = gt["pixel_bbox"]
        ul_x, ul_y = transform * (xmin, ymin)
        lr_x, lr_y = transform * (xmax, ymax)
        gt["geo_bbox_utm"] = [ul_x, ul_y, lr_x, lr_y]

    return data, ground_truth


def create_synthetic_tiff(
    output_filename: str = "synthetic_vantor_targets.tif",
    gt_filename: str = "synthetic_ground_truth.json",
    width: int = 4000,
    height: int = 4000,
):
    """Generates synthetic GeoTIFF with embedded targets and exports ground truth metadata."""
    output_path = config.RAW_DIR / output_filename
    gt_path = config.RAW_DIR / gt_filename

    # GeoTransform: 0.5m GSD in UTM Zone 36N (EPSG:32636)
    transform = from_origin(500000.0, 5300000.0, 0.5, 0.5)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:32636",
        "transform": transform,
        "nodata": 0,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }

    print(f"Generating base synthetic array ({width}x{height})...")
    # Base terrain background noise
    data = np.random.randint(60, 140, size=(3, height, width), dtype=np.uint8)

    print("Injecting fake objects (tents, trench lines, compounds, vehicles)...")
    data, ground_truth = inject_synthetic_objects(data, transform)

    print(f"Writing synthetic GeoTIFF to {output_path}...")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")

    print(f"Saving ground truth bounding boxes to {gt_path}...")
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=4)

    print("Synthetic imagery and ground truth generation complete.")


if __name__ == "__main__":
    create_synthetic_tiff()