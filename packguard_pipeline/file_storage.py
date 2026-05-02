"""
Per-lot file storage. Saves uploaded bytes to `data/uploads/<lot_id>/<filename>`
and returns the absolute paths.

Day 1 only stored filenames. Day 6 task per the brief — actually persist bytes
so checkpoints can re-read images and CSVs from disk.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import IO, Optional

UPLOAD_ROOT = Path(__file__).resolve().parents[1] / "data" / "uploads"


def lot_upload_dir(lot_id: str) -> Path:
    d = UPLOAD_ROOT / lot_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_upload(lot_id: str, filename: str, source: IO[bytes]) -> Path:
    """Save an uploaded file's bytes to disk under the lot's upload directory."""
    safe_name = Path(filename).name  # strip any path components from filename
    dest = lot_upload_dir(lot_id) / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(source, f)
    return dest


def categorize_uploads(saved_paths: list[Path]) -> dict[str, list[str]]:
    """
    Bucket saved paths into the categories LotState.input_files cares about.
    Heuristic by filename — fine for hackathon, replace with explicit form fields later.
    """
    cats: dict[str, list[str]] = {
        "xray_images": [],
        "aoi_images": [],
        "reflow_csv": [],
        "bond_force_log": [],
        "test_data_csv": [],
        "material_spec_json": [],
    }
    for p in saved_paths:
        n = p.name.lower()
        s = str(p)
        if "xray" in n or "solder" in n or "void" in n:
            cats["xray_images"].append(s)
        elif "aoi" in n or "die" in n:
            cats["aoi_images"].append(s)
        elif "reflow" in n and n.endswith(".csv"):
            cats["reflow_csv"].append(s)
        elif "bond" in n and n.endswith(".csv"):
            cats["bond_force_log"].append(s)
        elif "burn" in n or "test" in n:
            cats["test_data_csv"].append(s)
        elif n.endswith(".json"):
            cats["material_spec_json"].append(s)
    return cats


def reduce_to_input_files(saved_paths: list[Path]) -> dict:
    """Convert categorized lists to the shape `InputFiles` expects."""
    cats = categorize_uploads(saved_paths)
    return {
        "xray_images": cats["xray_images"],
        "aoi_images": cats["aoi_images"],
        "reflow_csv": cats["reflow_csv"][0] if cats["reflow_csv"] else None,
        "bond_force_log": cats["bond_force_log"][0] if cats["bond_force_log"] else None,
        "test_data_csv": cats["test_data_csv"][0] if cats["test_data_csv"] else None,
        "material_spec_json": cats["material_spec_json"][0] if cats["material_spec_json"] else None,
    }


__all__ = [
    "save_upload",
    "lot_upload_dir",
    "categorize_uploads",
    "reduce_to_input_files",
    "UPLOAD_ROOT",
]
