from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from tools.dataset_labeler.core import sha256_file
from tools.dataset_labeler.deblur_dataset import (
    BLUR_TYPES,
    DeblurGenerationConfig,
    _blur_parameters,
    apply_motion_blur,
    generate_deblur_dataset,
    motion_kernel,
    validate_deblur_dataset,
)


def _sources(root: Path, count: int = 5) -> Path:
    root.mkdir()
    for index in range(count):
        image = Image.new("RGB", (90 + index, 70 + index), (20 + index * 15, 80, 130))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8 + index, 12, 45, 48 + index), fill=(220, 30 + index * 10, 40))
        draw.line((0, index * 3 + 1, image.width - 1, image.height - 5), fill=(30, 240, 100), width=2)
        image.save(root / f"capture-{index}.png")
    return root


def _config(seed: int = 42, variants: int = 2) -> DeblurGenerationConfig:
    return DeblurGenerationConfig.from_mapping({
        "size": 64, "variants": variants, "seed": seed,
        "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        "moderate": {"path_length": [5, 9]}, "severe": {"path_length": [11, 17]},
    })


def test_fixed_seed_determinism_and_different_seed_variation(tmp_path: Path) -> None:
    source = _sources(tmp_path / "source")
    first = tmp_path / "first"; second = tmp_path / "second"; third = tmp_path / "third"
    generate_deblur_dataset(source, first, _config(42))
    generate_deblur_dataset(source, second, _config(42))
    generate_deblur_dataset(source, third, _config(43))
    first_rows = json.loads((first / "manifests/pairs.json").read_text())["pairs"]
    second_rows = json.loads((second / "manifests/pairs.json").read_text())["pairs"]
    third_rows = json.loads((third / "manifests/pairs.json").read_text())["pairs"]
    assert [row["blurred_output_hash"] for row in first_rows] == [row["blurred_output_hash"] for row in second_rows]
    assert [row["blurred_output_hash"] for row in first_rows] != [row["blurred_output_hash"] for row in third_rows]


def test_pair_alignment_dimensions_and_complete_manifest(tmp_path: Path) -> None:
    source = _sources(tmp_path / "source", 3)
    output = tmp_path / "dataset"
    result = generate_deblur_dataset(source, output, _config(7, 1))
    report = validate_deblur_dataset(output)
    assert result["generated_pairs"] == 3
    assert report["valid"], report["errors"]
    rows = json.loads((output / "manifests/pairs.json").read_text())["pairs"]
    required = {"pair_id", "trajectory", "spatial_variation", "exposure_settings", "noise_settings", "sharp_output_hash", "blurred_output_hash"}
    for row in rows:
        assert required <= set(row)
        sharp = Image.open(output / row["split"] / "sharp" / row["file_name"])
        blurred = Image.open(output / row["split"] / "blur" / row["file_name"])
        assert sharp.size == blurred.size == (64, 64)
        expected = ImageOps.fit(Image.open(source / row["source_path"]).convert("RGB"), (64, 64), method=Image.Resampling.LANCZOS)
        assert np.array_equal(np.asarray(sharp), np.asarray(expected))


def test_motion_psfs_cover_required_types() -> None:
    sharp = np.zeros((65, 65, 3), dtype=np.float32); sharp[32, 32] = 1.0
    cases = {
        "horizontal": {"angle_degrees": 0.0, "curvature": 0.0},
        "diagonal": {"angle_degrees": 15.0, "curvature": 0.0},
        "tangential": {"angle_degrees": 0.0, "curvature": 0.35},
        "spatially_varying": {"angle_degrees": 6.0, "curvature": 0.2},
    }
    outputs = {}
    for name, values in cases.items():
        params = {"blur_type": name, "severity": "moderate", "path_length": 15, "exposure_ev": 0.0, "brightness": 0.0, "contrast": 1.0, "noise_sigma": 0.0, **values}
        outputs[name], detail = apply_motion_blur(sharp, params, np.random.default_rng(1))
        assert np.isfinite(outputs[name]).all() and not np.array_equal(outputs[name], sharp)
        assert len(detail["trajectory"]) > 2
    horizontal = outputs["horizontal"][..., 0]
    assert np.count_nonzero(horizontal[32]) > np.count_nonzero(horizontal[:, 32])
    assert not np.allclose(outputs["horizontal"], outputs["diagonal"])
    assert not np.allclose(outputs["tangential"], outputs["diagonal"])
    assert not np.allclose(outputs["spatially_varying"], outputs["tangential"])


def test_moderate_and_severe_ranges_and_kernels() -> None:
    config = _config()
    observed = {"moderate": [], "severe": []}
    rng = np.random.default_rng(17)
    for _ in range(300):
        parameters = _blur_parameters(rng, config)
        observed[parameters["severity"]].append(parameters["path_length"])
        low, high = getattr(config, parameters["severity"]).path_length
        assert low <= parameters["path_length"] <= high
        assert parameters["blur_type"] in BLUR_TYPES
    assert observed["moderate"] and observed["severe"]
    kernel, trajectory = motion_kernel(13, 12, 0.3)
    assert abs(float(kernel.sum()) - 1.0) < 1e-6 and len(trajectory) > 2


def test_split_leakage_prevention_for_duplicates_and_capture_groups(tmp_path: Path) -> None:
    source = _sources(tmp_path / "source", 6)
    shutil.copy2(source / "capture-0.png", source / "capture-0-copy.png")
    (source / "source_groups.json").write_text(json.dumps([
        {"relative_path": "capture-1.png", "capture_group": "session-a"},
        {"relative_path": "capture-2.png", "capture_group": "session-a"},
    ]), encoding="utf-8")
    output = tmp_path / "dataset"
    generate_deblur_dataset(source, output, _config(42, 1))
    rows = json.loads((output / "manifests/pairs.json").read_text())["pairs"]
    by_source = {row["source_path"]: row["split"] for row in rows}
    assert by_source["capture-0.png"] == by_source["capture-0-copy.png"]
    assert by_source["capture-1.png"] == by_source["capture-2.png"]
    assert validate_deblur_dataset(output)["valid"]


def test_atomic_resume_cancellation_and_overwrite_protection(tmp_path: Path) -> None:
    source = _sources(tmp_path / "source", 4)
    output = tmp_path / "dataset"
    calls = 0

    def cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    cancelled = generate_deblur_dataset(source, output, _config(5, 2), cancel_check=cancel)
    assert cancelled["state"] == "cancelled" and cancelled["generated_pairs"] == 1
    assert not list(output.rglob("*.tmp"))
    resumed = generate_deblur_dataset(source, output, _config(5, 2), resume=True)
    assert resumed["state"] == "complete" and resumed["generated_pairs"] == resumed["planned_pairs"]
    before = sha256_file(output / "manifests/pairs.json")
    try:
        generate_deblur_dataset(source, output, _config(5, 2), resume=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing output should not be overwritten silently.")
    assert sha256_file(output / "manifests/pairs.json") == before
