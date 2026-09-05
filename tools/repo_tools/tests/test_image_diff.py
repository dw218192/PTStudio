"""Exercise existing-image FLIP comparisons without an editor or legacy shim."""

from pathlib import Path

from PIL import Image
from repo_tools.image_diff.diff import _run_case
from repo_tools.image_diff.common import BakeConfig, Case, ImageDiffConfig


def test_existing_capture_is_preserved_and_compared(tmp_path: Path):
    reference = tmp_path / "reference.png"
    capture = tmp_path / "capture.png"
    Image.new("RGB", (96, 64), (80, 100, 120)).save(reference)
    Image.new("RGB", (96, 64), (80, 100, 120)).save(capture)
    before = capture.read_bytes()
    case = Case("compare", tmp_path / "absent.usda", "/Camera", "Forward", reference, 0.3)
    cfg = ImageDiffConfig(32, 0.3, tmp_path, tmp_path / "out", BakeConfig("Path Trace", 1), [case])
    # There is no scene or task dispatcher here: this must only compare images.
    result = _run_case(case, cfg, tmp_path, "Debug", tmp_path / "logs", False, capture)
    assert result.passed
    assert result.score == 0.0
    assert result.heatmap.exists()
    assert capture.read_bytes() == before
    Image.new("RGB", (96, 64), (255, 255, 255)).save(capture)
    result = _run_case(case, cfg, tmp_path, "Debug", tmp_path / "logs", False, capture)
    assert not result.passed
    assert result.score > 0.3
