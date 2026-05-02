"""Tests for file upload storage and categorization."""

import io
import shutil
from pathlib import Path

import pytest

from packguard_pipeline.file_storage import (
    UPLOAD_ROOT,
    categorize_uploads,
    lot_upload_dir,
    reduce_to_input_files,
    save_upload,
)


@pytest.fixture
def tmp_lot(monkeypatch, tmp_path):
    """Redirect UPLOAD_ROOT to a per-test tmp dir so we don't pollute repo state."""
    monkeypatch.setattr("packguard_pipeline.file_storage.UPLOAD_ROOT", tmp_path)
    return tmp_path


def test_save_upload_writes_bytes(tmp_lot):
    body = b"PNG_BYTES_PRETEND"
    p = save_upload("LOT-2026-001", "die_000.png", io.BytesIO(body))
    assert p.exists()
    assert p.read_bytes() == body


def test_save_upload_strips_path_components(tmp_lot):
    """Filename shouldn't allow path traversal."""
    p = save_upload("LOT-2026-001", "../../etc/passwd.png", io.BytesIO(b"x"))
    # Result lives strictly inside the lot dir.
    assert p.parent == tmp_lot / "LOT-2026-001"
    assert p.name == "passwd.png"


def test_categorize_uploads_routes_by_filename(tmp_lot):
    paths = [
        Path("data/uploads/L/xray_die0.png"),
        Path("data/uploads/L/aoi_die1.png"),
        Path("data/uploads/L/reflow.csv"),
        Path("data/uploads/L/bond_force.csv"),
        Path("data/uploads/L/burn_in.csv"),
        Path("data/uploads/L/material.json"),
    ]
    cats = categorize_uploads(paths)
    assert any("xray" in p for p in cats["xray_images"])
    assert any("aoi" in p for p in cats["aoi_images"])
    assert any("reflow" in p for p in cats["reflow_csv"])
    assert any("bond" in p for p in cats["bond_force_log"])
    assert any("burn" in p for p in cats["test_data_csv"])
    assert any("material" in p for p in cats["material_spec_json"])


def test_reduce_to_input_files_shape(tmp_lot):
    paths = [
        Path("/tmp/xray_die.png"),
        Path("/tmp/reflow.csv"),
    ]
    out = reduce_to_input_files(paths)
    assert "xray_images" in out
    assert isinstance(out["xray_images"], list)
    assert out["reflow_csv"] is not None
    assert out["bond_force_log"] is None  # not provided
