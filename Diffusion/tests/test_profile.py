"""ARDD 1D 트랙의 profile 표현 테스트.

가장 중요한 단언: `to_feature_vector`가 Analyzer의 `make_features`와 **정확히 같은 벡터**를
낸다는 것. 이 정의가 어긋나면 unpaired 분포 지표(Fréchet/MMD)가 무의미해진다.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PRPD_Analyzer.prpd_analyzer.features import make_features
from Diffusion.prpd_diffusion.data.profile import (
    PHASE_BINS,
    feature_matrix,
    from_model_output,
    profile_from_matrix,
    profiles_from_matrices,
    resolve_kind,
    to_feature_vector,
    to_model_input,
    to_profile,
)


def _matrix(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(128, 3600), dtype=np.uint8)


def test_profile_shape_and_definition():
    matrix = _matrix()
    profile = profile_from_matrix(matrix)
    assert profile.shape == (2, PHASE_BINS)
    assert profile.dtype == np.float32
    assert np.allclose(profile[0], matrix.mean(axis=1), atol=1e-4)
    assert np.array_equal(profile[1], matrix.max(axis=1).astype(np.float32))


def test_value_round_trip_is_lossless():
    profile = profile_from_matrix(_matrix(1))
    restored = from_model_output(to_model_input(profile))
    assert np.allclose(profile, restored, atol=1e-3)


def test_model_input_range():
    values = to_model_input(profile_from_matrix(_matrix(2)))
    assert values.min() >= -1.0
    assert values.max() <= 1.0
    assert values.dtype == np.float32


def test_model_output_is_clipped():
    out_of_range = np.array([[-5.0] * PHASE_BINS, [5.0] * PHASE_BINS], dtype=np.float32)
    restored = from_model_output(out_of_range)
    assert restored.min() == 0.0
    assert restored.max() == 255.0


def test_to_profile_combines_extraction_and_scaling():
    matrix = _matrix(3)
    assert np.allclose(to_profile(matrix), to_model_input(profile_from_matrix(matrix)))


def test_feature_vector_matches_analyzer_definition():
    matrix = _matrix(4)
    profile = profile_from_matrix(matrix)
    ours = to_feature_vector(profile)
    theirs = make_features(
        matrix.mean(axis=1)[None, :].astype(np.float32),
        matrix.max(axis=1)[None, :].astype(np.float32),
    )[0]
    assert ours.shape == (256,)
    assert np.array_equal(ours, theirs)


def test_feature_matrix_batches():
    matrices = np.stack([_matrix(5), _matrix(6)])
    profiles = profiles_from_matrices(matrices)
    assert profiles.shape == (2, 2, PHASE_BINS)
    features = feature_matrix(profiles)
    assert features.shape == (2, 256)
    assert np.array_equal(features[1], to_feature_vector(profiles[1]))


def test_single_matrix_is_promoted_to_batch():
    assert profiles_from_matrices(_matrix(7)).shape == (1, 2, PHASE_BINS)


def test_bad_shape_is_rejected():
    try:
        profile_from_matrix(np.zeros((64, 3600), dtype=np.uint8))
    except ValueError:
        return
    raise AssertionError("위상 bin 수가 다른 입력을 거부해야 합니다.")


def test_phase_row_kind_is_not_implemented_yet():
    assert resolve_kind("profile_v1") == "profile_v1"
    try:
        resolve_kind("phase_row_v1")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("phase_row_v1은 아직 미구현이어야 합니다.")
    try:
        resolve_kind("nope")
    except ValueError:
        return
    raise AssertionError("알 수 없는 kind를 거부해야 합니다.")
