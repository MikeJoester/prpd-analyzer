from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Diffusion.prpd_diffusion.evaluation.metrics import (
    distribution_shift,
    feature_matrix,
    mean_absolute_error,
    mmd_rbf,
    paired_metrics,
    psnr,
    standardize,
)


def _matrices(count: int, high: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, high, size=(count, 128, 600), dtype=np.uint8)


def test_feature_matrix_shape_and_definition():
    matrices = _matrices(5, 200, 0)
    features = feature_matrix(matrices)
    assert features.shape == (5, 256)
    assert np.allclose(features[0, :128], matrices[0].mean(axis=1), atol=1e-4)
    assert np.allclose(features[0, 128:], matrices[0].max(axis=1))


def test_feature_matrix_accepts_single_matrix():
    assert feature_matrix(_matrices(1, 200, 1)[0]).shape == (1, 256)


def test_identical_inputs_have_zero_error():
    matrix = _matrices(1, 200, 2)[0]
    assert mean_absolute_error(matrix, matrix) == 0.0
    assert psnr(matrix, matrix) == float("inf")


def test_paired_metrics_reports_improvement_over_baseline():
    clean = _matrices(1, 120, 3)[0]
    noise = _matrices(1, 255, 4)[0]
    noisy = np.maximum(clean, noise)
    restored = np.maximum(clean, (noise * 0.2).astype(np.uint8))
    metrics = paired_metrics(restored, clean, noisy)
    assert metrics["mae"] < metrics["baseline_mae"]
    assert metrics["mae_improvement"] > 0
    assert metrics["mean_profile_mae"] >= 0


def test_standardize_uses_reference_statistics():
    reference = np.array([[0.0, 10.0], [2.0, 10.0]])
    other = np.array([[1.0, 10.0]])
    scaled_reference, scaled_other = standardize(reference, other)
    assert np.allclose(scaled_reference.mean(axis=0), 0.0)
    assert np.allclose(scaled_other, [[0.0, 0.0]])   # 상수 feature는 0으로 유지


def test_mmd_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(5)
    left = rng.normal(size=(60, 16))
    right = rng.normal(size=(60, 16))
    assert abs(mmd_rbf(left, right)) < 0.05


def test_distribution_shift_prefers_closer_restoration():
    clean = _matrices(8, 120, 6)
    noise = _matrices(8, 255, 7)
    noisy = np.maximum(clean, noise)
    restored = np.maximum(clean, (noise * 0.2).astype(np.uint8))
    shift = distribution_shift(feature_matrix(clean), feature_matrix(noisy), feature_matrix(restored))
    assert shift["frechet_after"] < shift["frechet_before"]
    assert shift["mmd_after"] <= shift["mmd_before"]
