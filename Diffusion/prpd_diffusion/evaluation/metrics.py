"""노이즈 제거 성능 지표.

두 축으로 평가한다.

1. **paired (합성 test 쌍)** — 정답 clean이 있으므로 픽셀·profile 오차를 직접 계산한다.
   MAE / PSNR / 위상 profile 오차.
2. **unpaired (실제 Field PD)** — 정답이 없으므로 분포로 평가한다. Analyzer의 256차원
   mean/max feature 공간에서 복원 결과가 Lab PD 분포에 얼마나 가까워졌는지 본다
   (Fréchet distance, MMD). "복원 전 → 복원 후" 거리가 줄어야 의미가 있다.

feature 정의는 Analyzer와 동일해야 비교가 성립하므로 `prpd_analyzer.features`를 그대로 쓴다.
(256 feature는 평가·통계용이며 딥러닝 입력으로는 사용하지 않는다 — CLAUDE.md 규칙)
"""

from __future__ import annotations

import numpy as np

from PRPD_Analyzer.prpd_analyzer.features import make_features

from ..data.representation import phase_profiles


# ------------------------------------------------------------- paired 지표


def mean_absolute_error(restored: np.ndarray, clean: np.ndarray) -> float:
    return float(np.mean(np.abs(restored.astype(np.float64) - clean.astype(np.float64))))


def psnr(restored: np.ndarray, clean: np.ndarray, data_range: float = 255.0) -> float:
    mse = float(np.mean((restored.astype(np.float64) - clean.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def profile_errors(restored: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    """위상별 평균/최대 profile 오차. PRPD 해석에서 실제로 쓰이는 축의 오차."""
    restored_mean, restored_max = phase_profiles(restored)
    clean_mean, clean_max = phase_profiles(clean)
    return {
        "mean_profile_mae": float(np.mean(np.abs(restored_mean - clean_mean))),
        "max_profile_mae": float(np.mean(np.abs(restored_max - clean_max))),
    }


def paired_metrics(restored: np.ndarray, clean: np.ndarray, noisy: np.ndarray | None = None) -> dict:
    """복원 결과 하나에 대한 paired 지표 묶음.

    `noisy`를 주면 "아무것도 하지 않았을 때"(=입력 그대로)의 오차를 함께 계산한다.
    복원 MAE가 이 baseline보다 낮지 않으면 모델이 아무 도움이 안 된 것이다.
    """
    metrics = {"mae": mean_absolute_error(restored, clean), "psnr": psnr(restored, clean)}
    metrics.update(profile_errors(restored, clean))
    if noisy is not None:
        metrics["baseline_mae"] = mean_absolute_error(noisy, clean)
        metrics["mae_improvement"] = metrics["baseline_mae"] - metrics["mae"]
    return metrics


# ----------------------------------------------------------- 분포(unpaired)


def feature_matrix(matrices: np.ndarray) -> np.ndarray:
    """`(N, 128, 3600)` → Analyzer와 동일한 `(N, 256)` mean/max feature."""
    matrices = np.asarray(matrices)
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
    means = np.stack([matrix.mean(axis=1) for matrix in matrices])
    maxima = np.stack([matrix.max(axis=1) for matrix in matrices])
    return make_features(means, maxima)


def frechet_distance(left: np.ndarray, right: np.ndarray, eps: float = 1e-6) -> float:
    """두 feature 집합 사이의 Fréchet distance (FID와 같은 정의, feature만 다름).

    표본 수가 feature 차원(256)보다 작으면 공분산이 특이해져 값이 불안정하다.
    그런 경우에는 `mmd_rbf`를 함께 보고 판단한다.
    """
    from scipy import linalg  # scipy는 scikit-learn 의존성으로 이미 설치되어 있다.

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mu_left, mu_right = left.mean(axis=0), right.mean(axis=0)
    offset = np.eye(left.shape[1]) * eps  # 특이 공분산에서 sqrtm이 발산하지 않도록
    cov_left = np.cov(left, rowvar=False) + offset
    cov_right = np.cov(right, rowvar=False) + offset

    result = linalg.sqrtm(cov_left.dot(cov_right))  # scipy 버전별로 튜플/배열
    covmean = result[0] if isinstance(result, tuple) else result
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    difference = mu_left - mu_right
    return float(difference.dot(difference) + np.trace(cov_left + cov_right - 2.0 * covmean))


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(left**2, axis=1)[:, None] + np.sum(right**2, axis=1)[None, :] - 2.0 * left.dot(right.T)
    )
    return np.maximum(squared, 0.0)


def mmd_rbf(left: np.ndarray, right: np.ndarray, gamma: float | None = None) -> float:
    """RBF kernel MMD^2. 표본 수가 적을 때 Fréchet distance보다 안정적이다.

    `gamma`를 주지 않으면 두 집합을 합친 거리의 중앙값으로 결정한다(median heuristic).
    feature 스케일이 0~255이므로 고정 gamma를 쓰면 커널이 포화해 값이 무의미해진다.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if gamma is None:
        pooled = np.concatenate([left, right], axis=0)
        median = float(np.median(_squared_distances(pooled, pooled)))
        gamma = 1.0 / median if median > 0 else 1.0

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-gamma * _squared_distances(a, b))

    def off_diagonal_mean(matrix: np.ndarray) -> float:
        size = matrix.shape[0]
        if size < 2:
            return 0.0
        return float((matrix.sum() - np.trace(matrix)) / (size * (size - 1)))

    # 불편(unbiased) 추정량 — 표본이 적을 때 대각항(=1)이 값을 지배하지 않도록 제외한다.
    return float(
        off_diagonal_mean(kernel(left, left))
        + off_diagonal_mean(kernel(right, right))
        - 2.0 * kernel(left, right).mean()
    )


def standardize(reference: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """reference의 평균/표준편차로 모든 집합을 표준화한다.

    Analyzer의 t-SNE 경로와 동일하게 StandardScaler 기준을 맞추되, 통계는 항상
    기준 집합(보통 train Lab PD)에서만 계산한다.
    """
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    std[std == 0] = 1.0
    return [(array - mean) / std for array in (reference, *others)]


def distribution_shift(
    reference: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> dict:
    """복원 전/후가 기준 분포(reference, 보통 Lab PD)에 얼마나 가까운지 비교한다.

    `*_after`가 `*_before`보다 작아야 노이즈 제거가 분포 관점에서도 효과가 있다는 뜻이다.
    """
    scaled_reference, scaled_before, scaled_after = standardize(reference, before, after)
    return {
        "frechet_before": frechet_distance(scaled_reference, scaled_before),
        "frechet_after": frechet_distance(scaled_reference, scaled_after),
        "mmd_before": mmd_rbf(scaled_reference, scaled_before),
        "mmd_after": mmd_rbf(scaled_reference, scaled_after),
    }
