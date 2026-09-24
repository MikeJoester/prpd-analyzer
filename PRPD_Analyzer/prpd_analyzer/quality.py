"""Category 품질 점검 — pooled shrinkage Mahalanobis 기반 검토 우선순위 산출.

`claude.md` 6절. **category를 자동으로 바꾸지 않는다.** 점수는 "사람이 원본 PRPD를 다시
볼 순서"일 뿐이며, 최종 판단은 별도 승인 절차로 한다.

왜 Mahalanobis인가 — 256차원 위상 feature는 인접 bin이 강하게 상관되어 있어 등방성
유클리드 거리가 성립하지 않는다. `ai_data_20260825_101830`(3,172개, 날짜단위 GroupKFold)
에서 Lab PD label 4종 분류 macroF1:

    feature set     euclid-centroid   maha-pooled   maha-perclass
    mean(128)            0.476           0.552          0.379
    max(128)             0.522           0.540          0.280
    mean+max(256)        0.545           0.644          0.349

두 가지가 여기서 결정된다.

* **공분산은 pooled로 추정한다.** 셀별 추정은 표본이 256차원을 감당하지 못해(셀당
  286~562개) 유클리드보다도 나쁘다.
* **mean+max > max > mean.** 256차원 결합이 실제로 기여한다.

shrinkage는 LedoitWolf가 아니라 고정값을 쓴다. LW는 이 데이터에서 under-shrink 해
pooled에서 0.013을 골랐고 macroF1 0.644에 그쳤다. log1p 변환 후 고정 α 스윕:

    alpha      0.2     0.4     0.5     0.6     0.7     0.9
    Lab PD   0.717   0.727   0.728   0.725   0.720   0.694

α≈0.5가 넓은 평탄역의 최적점이라 기본값으로 둔다.

임계값에 이론 chi-square를 쓰지 않는다. 강한 shrinkage가 거리를 압축해 실측 d² 분포가
chi2(256)보다 훨씬 조밀하기 때문이다(실측 median 70.9 / p99 314.2 vs chi2 median≈255 /
p99 311.6). **경험적 percentile을 쓴다.**
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import ShrunkCovariance
from sklearn.metrics import f1_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

MIN_CELL_COUNT = 10
DEFAULT_SHRINKAGE = 0.5
DEFAULT_TRANSFORM = "log1p"
TRANSFORMS = ("log1p", "none")
DEFAULT_PERCENTILES = (0.99, 0.95)

Cell = tuple[str, str]  # (group, label)


def apply_transform(features: np.ndarray, transform: str) -> np.ndarray:
    """점수 계산 전용 변환. 저장되는 256차원 feature 자체는 바꾸지 않는다."""
    if transform == "log1p":
        return np.log1p(np.asarray(features, dtype=np.float64))
    if transform == "none":
        return np.asarray(features, dtype=np.float64)
    raise ValueError(f"알 수 없는 transform: {transform} (가능: {TRANSFORMS})")


# ------------------------------------------------------------------ 기준 모델


@dataclass(frozen=True)
class CellModel:
    """(group, label) 셀별 평균 + 전 셀 공통 pooled precision.

    `precision`이 셀마다 다르지 않은 것은 의도다. 모듈 docstring의 실측 표 참조.
    """

    means: dict[Cell, np.ndarray]
    counts: dict[Cell, int]
    precision: np.ndarray
    shrinkage: float
    transform: str
    excluded_cells: tuple[Cell, ...] = ()
    warnings: tuple[str, ...] = field(default=())

    @property
    def dimension(self) -> int:
        return int(self.precision.shape[0])

    def cells_of_group(self, group: str) -> list[Cell]:
        return [cell for cell in self.means if cell[0] == group]

    def to_dict(self) -> dict:
        return {
            "shrinkage": self.shrinkage,
            "transform": self.transform,
            "dimension": self.dimension,
            "reference_cells": {f"{g} | {l}": self.counts[(g, l)] for g, l in sorted(self.means)},
            "excluded_cells": [f"{g} | {l}" for g, l in self.excluded_cells],
            "warnings": list(self.warnings),
        }


def fit_cell_model(
    features: np.ndarray,
    table: pd.DataFrame,
    shrinkage: float = DEFAULT_SHRINKAGE,
    transform: str = DEFAULT_TRANSFORM,
    min_count: int = MIN_CELL_COUNT,
) -> CellModel:
    """(group, label) 셀 평균과 pooled shrinkage precision을 추정한다.

    `min_count` 미만인 셀은 기준에서 제외한다. 평균이 사실상 자기 자신이 되어 d²가 0에
    가까워지고, 그 결과 **문제 있는 파일을 "정상"으로 잘못 안심시키기** 때문이다.
    제외된 셀의 파일도 리포트에는 남지만 `reference_status="insufficient"`로 표시된다.
    """
    if shrinkage < 0.0 or shrinkage > 1.0:
        raise ValueError(f"shrinkage는 0~1이어야 합니다: {shrinkage}")
    values = apply_transform(features, transform)
    if len(values) != len(table):
        raise ValueError(f"feature {len(values)}행 vs table {len(table)}행")

    keys = list(zip(table["group"].astype(str), table["label"].astype(str)))
    sizes: dict[Cell, int] = {}
    for cell in keys:
        sizes[cell] = sizes.get(cell, 0) + 1

    means: dict[Cell, np.ndarray] = {}
    counts: dict[Cell, int] = {}
    excluded: list[Cell] = []
    residuals: list[np.ndarray] = []
    for cell, size in sorted(sizes.items()):
        rows = np.fromiter((k == cell for k in keys), dtype=bool, count=len(keys))
        if size < min_count:
            excluded.append(cell)
            continue
        block = values[rows]
        means[cell] = block.mean(axis=0)
        counts[cell] = size
        residuals.append(block - means[cell])

    if not means:
        raise ValueError(
            f"기준으로 쓸 수 있는 (group, label) 셀이 없습니다. 최소 {min_count}개가 필요합니다."
        )

    pooled = np.vstack(residuals)
    precision = ShrunkCovariance(shrinkage=shrinkage, assume_centered=True).fit(pooled).precision_

    messages: list[str] = []
    dimension = values.shape[1]
    if len(pooled) < 2 * dimension:
        messages.append(
            f"공분산 추정 표본이 {len(pooled)}개로 차원({dimension})의 2배 미만입니다. "
            "기준 분포가 불안정할 수 있으니 필터를 넓혀 다시 실행하세요."
        )
    for cell in excluded:
        messages.append(f"셀 '{cell[0]} | {cell[1]}'은 표본 {sizes[cell]}개로 기준에서 제외했습니다.")

    return CellModel(
        means=means,
        counts=counts,
        precision=precision,
        shrinkage=float(shrinkage),
        transform=transform,
        excluded_cells=tuple(excluded),
        warnings=tuple(messages),
    )


# ------------------------------------------------------------------ 거리 계산


def _mahalanobis(values: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = values - mean
    return np.einsum("ij,jk,ik->i", delta, precision, delta)


def cell_distances(model: CellModel, features: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    """자기 셀 d²(leave-one-out 보정)와 같은 group 내 최근접 다른 label까지의 d².

    자기 셀 평균에는 그 샘플이 포함되어 있으므로 `μ_loo = (n·μ − x) / (n − 1)`로 빼고
    계산한다. 보정하지 않으면 작은 셀(`Field PD / Corona` n=33)의 d²가 체계적으로 낮게
    나와, 정작 표본이 부족한 셀이 가장 안전해 보이는 역설이 생긴다.

    경쟁 후보는 **같은 group 안의 다른 label**로 제한한다. group을 섞으면 Lab/Field
    도메인 격차가 오라벨 신호로 둔갑한다.
    """
    values = apply_transform(features, model.transform)
    groups = table["group"].astype(str).to_numpy()
    labels = table["label"].astype(str).to_numpy()
    total = len(values)

    own = np.full(total, np.nan)
    nearest_distance = np.full(total, np.nan)
    nearest_label = np.array([""] * total, dtype=object)
    status = np.array(["ok"] * total, dtype=object)

    for index in range(total):
        cell: Cell = (groups[index], labels[index])
        sample = values[index]
        if cell in model.means:
            count = model.counts[cell]
            if count > 1:
                loo_mean = (count * model.means[cell] - sample) / (count - 1)
            else:  # min_count가 1로 낮춰진 경우에만 도달한다
                loo_mean = model.means[cell]
            own[index] = _mahalanobis(sample[None, :], loo_mean, model.precision)[0]
        else:
            status[index] = "insufficient"

        rivals = [c for c in model.cells_of_group(groups[index]) if c != cell]
        if rivals:
            distances = [
                (_mahalanobis(sample[None, :], model.means[c], model.precision)[0], c[1])
                for c in rivals
            ]
            best_distance, best_label = min(distances, key=lambda item: item[0])
            nearest_distance[index] = best_distance
            nearest_label[index] = best_label

    return pd.DataFrame(
        {
            "suspect_score": own,
            "d2_nearest": nearest_distance,
            "nearest_category": nearest_label,
            "margin": own - nearest_distance,
            "reference_status": status,
        }
    )


# ------------------------------------------------------------------ 임계값


def add_suspect_flags(
    report: pd.DataFrame,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
) -> tuple[pd.DataFrame, dict]:
    """경험적 percentile로 상위 구간 flag를 붙인다. 이론 chi-square는 쓰지 않는다.

    전역 percentile을 주 기준으로 삼아 검토 인력이 실제로 문제가 많은 셀에 집중되게 하고,
    `cell_percentile`로 자기 셀 안에서의 상대 위치를 함께 제공한다.

    Returns:
        `(flag가 추가된 report, 적용된 컷값 dict)` — 컷값은 재현을 위해 기록한다.
    """
    result = report.copy()
    cuts: dict[str, float] = {}
    for column in ("suspect_score", "margin"):
        values = result[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        for percentile in percentiles:
            name = f"{'suspect' if column == 'suspect_score' else 'margin'}_flag_p{int(percentile * 100)}"
            if finite.size == 0:
                result[name] = False
                continue
            cut = float(np.quantile(finite, percentile))
            cuts[name] = cut
            result[name] = np.isfinite(values) & (values >= cut)

    if "group" in result.columns and "current_category" in result.columns:
        scores = result["suspect_score"]
        result["cell_percentile"] = (
            scores.groupby([result["group"], result["current_category"]]).rank(pct=True)
        )
    else:
        result["cell_percentile"] = result["suspect_score"].rank(pct=True)
    return result, cuts


# ------------------------------------------------------------------ 리포트


def build_category_report(
    features: np.ndarray,
    table: pd.DataFrame,
    model: CellModel | None = None,
    shrinkage: float = DEFAULT_SHRINKAGE,
    transform: str = DEFAULT_TRANSFORM,
    min_count: int = MIN_CELL_COUNT,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
) -> tuple[pd.DataFrame, CellModel, dict]:
    """검토 우선순위 리포트를 만든다.

    Returns:
        `(report, model, cuts)`. `report`는 `suspect_score` 내림차순이며,
        점수를 낼 수 없는 파일(`reference_status="insufficient"`)이 뒤로 간다.
    """
    if model is None:
        model = fit_cell_model(features, table, shrinkage, transform, min_count)

    base = table.reset_index(drop=True)
    distances = cell_distances(model, features, base)
    report = pd.DataFrame(
        {
            "file": base["file"],
            "path": base["path"],
            "group": base["group"],
            "current_category": base["label"].astype(str),
            "date": base["date"],
        }
    ).join(distances)

    # 단일 label group(Lab/Field Noise, Synthetic)은 경쟁 후보가 없다. 이때 margin은 NaN이며
    # "자기 category가 가장 가깝다"고 쓰면 비교를 한 것처럼 읽혀 오해를 부른다.
    no_rival = report["nearest_category"].astype(str).eq("")
    report["suspect_reason"] = np.select(
        [
            report["reference_status"].eq("insufficient"),
            no_rival,
            report["margin"] > 0,
        ],
        [
            "기준 셀 표본 부족 — 점수 계산 불가",
            "group 내 비교 가능한 다른 category 없음 — 자기 분포 이탈도만 반영",
            "자기 category보다 " + report["nearest_category"].astype(str) + " 중심이 더 가까움",
        ],
        default="자기 category 중심이 가장 가까움",
    )
    report["review_status"] = "Unreviewed"
    report, cuts = add_suspect_flags(report, percentiles)
    report = report.sort_values("suspect_score", ascending=False, na_position="last").reset_index(drop=True)
    return report, model, cuts


# ------------------------------------------------------------------ feature 품질 진단

FEATURE_SETS = {
    "mean(128)": slice(0, 128),
    "max(128)": slice(128, 256),
    "mean+max(256)": slice(0, 256),
}
DIAGNOSTIC_SCOPES = ("ALL", "Lab PD", "Field PD", "PD only")


def _scope_mask(table: pd.DataFrame, scope: str) -> np.ndarray:
    groups = table["group"].astype(str)
    if scope == "ALL":
        return np.ones(len(table), dtype=bool)
    if scope == "PD only":
        return groups.isin(("Lab PD", "Field PD")).to_numpy()
    return groups.eq(scope).to_numpy()


def _cv_macro_f1(values: np.ndarray, labels: np.ndarray, dates: np.ndarray, shrinkage: float) -> float:
    """날짜단위 GroupKFold + pooled shrinkage Mahalanobis 최근접 셀 분류의 macro F1.

    파일 단위 무작위 분할을 쓰지 않는 이유는 `claude.md` Phase 7과 같다 — 같은 날짜의
    데이터가 train/test에 섞이면 성능이 과대평가된다.
    """
    classes = sorted(set(labels))
    if len(classes) < 2:
        return float("nan")
    splits = min(5, len(set(dates)), min(int((labels == c).sum()) for c in classes))
    if splits < 2:
        return float("nan")

    truth: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    for train, test in GroupKFold(n_splits=splits).split(values, labels, groups=dates):
        present = sorted(set(labels[train]))
        means = {c: values[train][labels[train] == c].mean(axis=0) for c in present}
        residuals = np.vstack([values[train][labels[train] == c] - means[c] for c in present])
        precision = ShrunkCovariance(shrinkage=shrinkage, assume_centered=True).fit(residuals).precision_
        distances = np.stack([_mahalanobis(values[test], means[c], precision) for c in present], axis=1)
        truth.append(labels[test])
        predicted.append(np.array(present)[distances.argmin(axis=1)])
    return float(f1_score(np.concatenate(truth), np.concatenate(predicted), average="macro"))


def feature_set_diagnostics(
    features: np.ndarray,
    table: pd.DataFrame,
    shrinkage: float = DEFAULT_SHRINKAGE,
    transforms: tuple[str, ...] = TRANSFORMS,
    scopes: tuple[str, ...] = DIAGNOSTIC_SCOPES,
) -> pd.DataFrame:
    """Mean / Max / Mean+Max가 PD class를 얼마나 분리하는지 비교한다.

    **이 표는 feature 품질 진단 전용이다.** 분류기로 배포하지 않고 `suspect_score`에도
    쓰지 않는다.

    `silhouette`과 `cv_macro_f1`을 함께 싣는 이유 — silhouette은 이 데이터에서 어떤
    전처리로도 0.06을 넘지 못하고, 특히 `max`(0.008)와 `mean+max`(0.012)를 구분하지
    못한다. 클래스가 구형으로 뭉쳐 있지 않기 때문이며, 바로 그래서 Mahalanobis가 필요하다.
    feature set의 우열은 `cv_macro_f1`(0.540 vs 0.644)이 가른다.
    """
    base = table.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for scope in scopes:
        mask = _scope_mask(base, scope)
        if mask.sum() < 2 * len(FEATURE_SETS):
            continue
        labels = base.loc[mask, "label"].astype(str).to_numpy()
        groups = base.loc[mask, "group"].astype(str).to_numpy()
        dates = base.loc[mask, "date"].astype(str).to_numpy()
        for transform in transforms:
            values_all = apply_transform(features[mask], transform)
            for name, columns in FEATURE_SETS.items():
                values = values_all[:, columns]
                scaled = StandardScaler().fit_transform(values)
                rows.append(
                    {
                        "scope": scope,
                        "transform": transform,
                        "feature_set": name,
                        "n": int(mask.sum()),
                        "silhouette_label": (
                            float(silhouette_score(scaled, labels)) if len(set(labels)) > 1 else np.nan
                        ),
                        "silhouette_group": (
                            float(silhouette_score(scaled, groups)) if len(set(groups)) > 1 else np.nan
                        ),
                        "cv_macro_f1": _cv_macro_f1(values, labels, dates, shrinkage),
                    }
                )
    return pd.DataFrame(rows)
