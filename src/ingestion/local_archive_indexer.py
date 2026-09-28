"""
Local Archive Indexer Module
============================
Parses, indexes, and queries un-indexed local satellite imagery archives 
(Vantor/Maxar GEGD Pro, Airbus, Capella SAR, etc.) stored within the air-gapped 
data directories.

Features:
- Multi-threaded raster scanning using Rasterio/GDAL.
- Provider auto-detection (Vantor, Airbus, Capella SAR) via TIFF tags and sidecar metadata (.xml, .json).
- Automatic Coordinate Reference System (CRS) reprojection to WGS84 (EPSG:4326) bounding boxes.
- Spatial-temporal indexing with GeoJSON and SQLite output options.
- Buffer-matching utility to query local images against Liveuamap or incident coordinate buffers.
"""

import os
import json
import re
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box, shape, Polygon

# Import system configuration paths and hardware constants
from src.config import RAW_DIR, PROCESSED_DIR, INTERIM_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LocalArchiveIndexer:
    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        output_dir: Path = PROCESSED_DIR,
        max_workers: int = 8,
    ):
        """
        Initializes the archive indexer.

        :param raw_dir: Path to directory containing raw imagery archives.
        :param output_dir: Path to export index outputs (GeoJSON/SQLite).
        :param max_workers: Number of concurrent threads for scanning files.
        """
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.catalog: List[Dict[str, Any]] = []

    def _extract_timestamp(self, filepath: Path, tags: Dict[str, Any]) -> str:
        """Extract acquisition timestamp from sidecar metadata, TIFF tags, or filename heuristics."""
        # 1. Check sidecar XML/JSON metadata
        sidecar_xml = filepath.with_suffix(".xml")
        sidecar_json = filepath.with_suffix(".json")

        if sidecar_json.exists():
            try:
                with open(sidecar_json, "r") as f:
                    data = json.load(f)
                    # Capella SAR format
                    if "collect" in data and "start_time" in data["collect"]:
                        return data["collect"]["start_time"]
                    # Generic JSON
                    for key in ["acquisition_time", "datetime", "startTime"]:
                        if key in data:
                            return str(data[key])
            except Exception:
                pass

        if sidecar_xml.exists():
            try:
                content = sidecar_xml.read_text()
                # Airbus / Maxar XML tags
                match = re.search(r"<(?:FIRST_LINE_TIME|IMAGERY_DATE|ACQUISITION_DATE|firstLineTime)>(.*?)</", content)
                if match:
                    return match.group(1).strip()
            except Exception:
                pass

        # 2. Check TIFF Metadata Tags
        for tag_key in ["TIFFTAG_DATETIME", "AREA_OR_POINT", "acquisition_date"]:
            if tag_key in tags:
                return str(tags[tag_key])

        # 3. Fallback: Regex parsing from standard vendor filename conventions
        filename = filepath.name
        # Match YYYY-MM-DD or YYYYMMDD_HHMMSS
        date_match = re.search(r"(\d{4}[-_]?\d{2}[-_]?\d{2}(?:T\d{2}[-_]?\d{2}[-_]?\d{2})?)", filename)
        if date_match:
            raw_str = date_match.group(1).replace("-", "").replace("_", "").replace("T", "")
            try:
                if len(raw_str) >= 14:
                    dt = datetime.strptime(raw_str[:14], "%Y%m%d%H%M%S")
                else:
                    dt = datetime.strptime(raw_str[:8], "%Y%m%d")
                return dt.replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass

        # 4. Ultimate fallback: File creation time
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    def _detect_provider_and_modality(self, filepath: Path, tags: Dict[str, Any], count: int) -> Tuple[str, str]:
        """Identifies vendor source (Vantor, Airbus, Capella) and sensor modality (Optical vs SAR)."""
        filename = filepath.name.lower()
        path_str = str(filepath).lower()

        # Vendor Detection
        if "capella" in filename or "capella" in path_str:
            provider = "Capella Space"
            modality = "SAR"
        elif "airbus" in filename or "pleiades" in filename or "spot" in filename or "phr" in filename:
            provider = "Airbus"
            modality = "Optical"
        elif "maxar" in filename or "wv" in filename or "gegd" in path_str or "vantor" in path_str:
            provider = "Vantor / Maxar"
            modality = "Optical"
        else:
            provider = "Unknown Vendor"
            modality = "SAR" if count == 1 and ("sar" in filename or "hh" in filename or "vv" in filename) else "Optical"

        return provider, modality

    def parse_raster_metadata(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Parses spatial boundaries, CRS, and vendor attributes from a single raster file."""
        if not filepath.suffix.lower() in [".tif", ".tiff"]:
            return None

        try:
            with rasterio.open(filepath) as src:
                bounds = src.bounds
                crs = src.crs
                
                if crs is None:
                    logger.warning(f"Skipping {filepath.name}: Missing CRS projection.")
                    return None

                # Transform bounds to EPSG:4326 (WGS84) for standardized indexing
                wgs84_bounds = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
                
                # Calculate ground sample distance (GSD) / resolution in meters
                res_x, res_y = abs(src.transform.a), abs(src.transform.e)
                gsd_meters = round((res_x + res_y) / 2.0, 3)

                tags = src.tags()
                timestamp = self._extract_timestamp(filepath, tags)
                provider, modality = self._detect_provider_and_modality(filepath, tags, src.count)

                geom = box(*wgs84_bounds)

                return {
                    "filepath": str(filepath.resolve()),
                    "filename": filepath.name,
                    "provider": provider,
                    "modality": modality,
                    "bands": src.count,
                    "height": src.height,
                    "width": src.width,
                    "gsd_meters": gsd_meters,
                    "crs": crs.to_string(),
                    "timestamp": timestamp,
                    "bbox_wgs84": list(wgs84_bounds),
                    "geometry": geom.__geo_interface__,
                }

        except Exception as e:
            logger.error(f"Error parsing {filepath.name}: {str(e)}")
            return None

    def build_index(self, scan_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Recursively scans the directory and builds the catalog in memory using multi-threading."""
        target_dir = Path(scan_dir) if scan_dir else self.raw_dir
        logger.info(f"Scanning archive directory: {target_dir}")

        image_paths = [p for p in target_dir.rglob("*") if p.suffix.lower() in [".tif", ".tiff"]]
        logger.info(f"Found {len(image_paths)} raster candidates. Parsing metadata...")

        self.catalog = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.parse_raster_metadata, path): path for path in image_paths}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.catalog.append(result)

        logger.info(f"Successfully indexed {len(self.catalog)} imagery archives.")
        return self.catalog

    def export_geojson(self, output_filename: str = "archive_catalog.geojson") -> Path:
        """Exports catalog as a standardized GeoJSON FeatureCollection."""
        features = []
        for item in self.catalog:
            feature = {
                "type": "Feature",
                "geometry": item["geometry"],
                "properties": {k: v for k, v in item.items() if k != "geometry"},
            }
            features.append(feature)

        geojson_data = {"type": "FeatureCollection", "features": features}
        out_path = self.output_dir / output_filename
        with open(out_path, "w") as f:
            json.dump(geojson_data, f, indent=2)

        logger.info(f"Exported catalog to GeoJSON: {out_path}")
        return out_path

    def export_sqlite(self, db_filename: str = "archive_index.sqlite") -> Path:
        """Exports catalog to a local SQLite database for fast air-gapped SQL queries."""
        db_path = self.output_dir / db_filename
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS imagery_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE,
                filename TEXT,
                provider TEXT,
                modality TEXT,
                bands INTEGER,
                height INTEGER,
                width INTEGER,
                gsd_meters REAL,
                crs TEXT,
                timestamp TEXT,
                min_lon REAL,
                min_lat REAL,
                max_lon REAL,
                max_lat REAL
            )
        """)

        for item in self.catalog:
            b = item["bbox_wgs84"]
            cursor.execute("""
                INSERT OR REPLACE INTO imagery_index (
                    filepath, filename, provider, modality, bands, height, width,
                    gsd_meters, crs, timestamp, min_lon, min_lat, max_lon, max_lat
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item["filepath"], item["filename"], item["provider"], item["modality"],
                item["bands"], item["height"], item["width"], item["gsd_meters"],
                item["crs"], item["timestamp"], b[0], b[1], b[2], b[3]
            ))

        conn.commit()
        conn.close()
        logger.info(f"Exported catalog to SQLite database: {db_path}")
        return db_path

    def query_buffer(
        self,
        target_bbox: Tuple[float, float, float, float],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        required_modality: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Matches cataloged imagery against an incident spatial buffer and optional temporal window.

        :param target_bbox: Tuple of (min_lon, min_lat, max_lon, max_lat).
        :param start_time: ISO-formatted start datetime string.
        :param end_time: ISO-formatted end datetime string.
        :param required_modality: Filter for 'Optical' or 'SAR'.
        :return: List of catalog items intersecting the spatial-temporal criteria.
        """
        target_poly = box(*target_bbox)
        results = []

        # [EDIT 2026-09-28 | Claude Code for charliefp03-dg] Normalize all datetimes to timezone-aware UTC.
        # Comparing naive and aware datetimes raised an uncaught TypeError. Naive inputs are assumed UTC.
        def _as_utc(dt: datetime) -> datetime:
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

        t_start = _as_utc(datetime.fromisoformat(start_time)) if start_time else None
        t_end = _as_utc(datetime.fromisoformat(end_time)) if end_time else None

        for item in self.catalog:
            if required_modality and item["modality"].lower() != required_modality.lower():
                continue

            # Temporal filtering
            if t_start or t_end:
                try:
                    item_dt = _as_utc(datetime.fromisoformat(item["timestamp"]))  # [EDIT 2026-09-28 | Claude Code for charliefp03-dg]
                    if t_start and item_dt < t_start:
                        continue
                    if t_end and item_dt > t_end:
                        continue
                except ValueError:
                    pass

            # Spatial intersection filtering
            item_poly = shape(item["geometry"])
            if target_poly.intersects(item_poly):
                intersection_area = target_poly.intersection(item_poly).area
                coverage_pct = round((intersection_area / target_poly.area) * 100, 2)
                
                matched_item = item.copy()
                matched_item["buffer_coverage_pct"] = coverage_pct
                results.append(matched_item)

        # Sort results by coverage percentage descending
        results.sort(key=lambda x: x.get("buffer_coverage_pct", 0), reverse=True)
        return results


if __name__ == "__main__":
    # Test script execution using paths from src.config
    indexer = LocalArchiveIndexer(raw_dir=RAW_DIR, output_dir=PROCESSED_DIR)
    catalog = indexer.build_index()
    
    if catalog:
        indexer.export_geojson()
        indexer.export_sqlite()
        
        # Example buffer match query (e.g., sample spatial window)
        sample_query_bbox = (-66.8, 10.4, -66.7, 10.5)  # Venezuela area example
        matches = indexer.query_buffer(target_bbox=sample_query_bbox)
        logger.info(f"Found {len(matches)} local rasters covering the target query buffer.")