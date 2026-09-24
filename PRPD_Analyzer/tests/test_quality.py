"""Category 품질 점검(`prpd_analyzer.quality`) 검증.

점수가 "검토 우선순위"로 쓸 만한지를 본다 — 오라벨을 잡는가, 표본이 부족한 셀을 조용히
정상으로 넘기지 않는가, 같은 입력에서 같은 값이 나오는가.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _skip import Skip
from PRPD_Analyzer.prpd_analyzer.quality import (
    DEFAULT_SHRINKAGE,
    FEATURE_SETS,
    MIN_CELL_COUNT,
    TRANSFORMS,
    add_suspect_flags,
    apply_transform,
    build_category_report,
    cell_distances,
    feature_set_diagnostics,
    fit_cell_model,
)

ROOT = Path(__file__).resolve().parent.parent
DIMENSION = 256


# ------------------------------------------------------------------ 합성 데이터


def _synthetic(seed: int = 0, per_cell: int = 60, dimension: int = DIMENSION):
    """(group, label) 셀마다 중심이 다른 정규 표본. 값은 0 이상(log1p 적용 가능)."""
    rng = np.random.default_rng(seed)
    cells = [("Lab PD", "Corona"), ("Lab PD", "Void"), ("Lab PD", "Floating")]
    blocks, rows = [], []
    for offset, (group, label) in enumerate(cells):
        centre = np.zeros(dimension)
        centre[offset * 20 : offset * 20 + 20] = 40.0
        block = np.clip(centre + rng.normal(0.0, 3.0, size=(per_cell, dimension)), 0.0, None)
        blocks.append(block)
        for index in range(per_cell):
            rows.append(
                {
                    "file": f"{label}_{index}.dat",
                    "path": f"/data/{label}_{index}.dat",
                    "group": group,
                    "label": label,
                    "date": f"2024010{index % 5 + 1}",
                }
            )
    return np.vstack(blocks).astype(np.float32), pd.DataFrame(rows)


# ------------------------------------------------------------------ 오라벨 탐지


def test_mislabelled_sample_gets_positive_margin():
    """label을 바꾼 샘플은 margin이 양수가 되고 원래 category를 가리켜야 한다."""
    features, table = _synthetic()
    victim = 5  # Corona 블록의 한 샘플
    assert table.loc[victim, "label"] == "Corona"
    table.loc[victim, "label"] = "Void"

    report, _, _ = build_category_report(features, table)
    row = report[report["file"] == "Corona_5.dat"].iloc[0]
    assert row["margin"] > 0, f"margin={row['margin']}"
    assert row["nearest_category"] == "Corona"
    # 정상 샘플 대부분은 자기 중심이 가장 가깝다.
    healthy = report[report["file"] != "Corona_5.dat"]
    assert (healthy["margin"] <= 0).mean() > 0.9


def test_clean_data_flags_almost_nothing_by_margin():
    features, table = _synthetic(seed=3)
    report, _, _ = build_category_report(features, table)
    assert (report["margin"] > 0).sum() == 0


# ------------------------------------------------------------------ 공분산 추정


def test_pooled_precision_is_symmetric_and_positive_definite():
    features, table = _synthetic()
    model = fit_cell_model(features, table)
    precision = model.precision
    assert precision.shape == (DIMENSION, DIMENSION)
    assert np.allclose(precision, precision.T, atol=1e-8)
    assert float(np.linalg.eigvalsh(precision).min()) > 0.0


def test_zero_shrinkage_matches_empirical_covariance():
    features, table = _synthetic(per_cell=400, dimension=8)
    model = fit_cell_model(features, table, shrinkage=0.0)
    values = apply_transform(features, model.transform)
    residuals = np.vstack(
        [
            values[(table["group"] == g) & (table["label"] == l)]
            - values[(table["group"] == g) & (table["label"] == l)].mean(axis=0)
            for g, l in model.means
        ]
    )
    empirical = residuals.T.dot(residuals) / len(residuals)
    assert np.allclose(np.linalg.inv(empirical), model.precision, rtol=1e-5, atol=1e-6)


def test_shrinkage_outside_unit_interval_raises():
    features, table = _synthetic(per_cell=20, dimension=8)
    for bad in (-0.1, 1.5):
        try:
            fit_cell_model(features, table, shrinkage=bad)
        except ValueError:
            continue
        raise AssertionError(f"shrinkage={bad}가 통과했습니다.")


# ------------------------------------------------------------------ 표본 부족 셀


def test_small_cell_is_excluded_and_marked():
    features, table = _synthetic()
    tiny_features = np.clip(np.full((3, DIMENSION), 5.0), 0.0, None).astype(np.float32)
    tiny = pd.DataFrame(
        [
            {
                "file": f"tiny_{i}.dat",
                "path": f"/data/tiny_{i}.dat",
                "group": "Lab PD",
                "label": "Particle",
                "date": "20240110",
            }
            for i in range(3)
        ]
    )
    features = np.vstack([features, tiny_features])
    table = pd.concat([table, tiny], ignore_index=True)

    report, model, _ = build_category_report(features, table)
    assert ("Lab PD", "Particle") in model.excluded_cells
    assert any("Particle" in message for message in model.warnings)

    rows = report[report["current_category"] == "Particle"]
    assert len(rows) == 3
    assert rows["reference_status"].eq("insufficient").all()
    assert rows["suspect_score"].isna().all()
    # 자기 점수는 없어도 어느 category에 가까운지는 남긴다.
    assert rows["nearest_category"].ne("").all()


def test_min_cell_count_default_is_documented_value():
    assert MIN_CELL_COUNT == 10


# ------------------------------------------------------------------ LOO 보정


def test_leave_one_out_increases_own_distance():
    """자기 샘플이 자기 평균에 포함되면 거리가 낮게 나온다. 보정본이 항상 더 커야 한다."""
    features, table = _synthetic(per_cell=15)
    model = fit_cell_model(features, table)
    corrected = cell_distances(model, features, table)["suspect_score"].to_numpy()

    values = apply_transform(features, model.transform)
    naive = np.empty(len(values))
    for index, (group, label) in enumerate(zip(table["group"], table["label"])):
        delta = values[index] - model.means[(group, label)]
        naive[index] = float(delta.dot(model.precision).dot(delta))
    assert np.all(corrected >= naive - 1e-9)
    assert corrected.mean() > naive.mean()


# ------------------------------------------------------------------ 임계값


def test_percentile_flags_select_expected_fraction():
    features, table = _synthetic(per_cell=200)
    report, _, cuts = build_category_report(features, table)
    scorable = report[report["reference_status"].eq("ok")]
    assert abs(scorable["suspect_flag_p99"].mean() - 0.01) < 0.005
    assert abs(scorable["suspect_flag_p95"].mean() - 0.05) < 0.01
    assert cuts["suspect_flag_p99"] > cuts["suspect_flag_p95"]


def test_flags_are_empty_when_no_finite_scores():
    report = pd.DataFrame({"suspect_score": [np.nan, np.nan], "margin": [np.nan, np.nan]})
    flagged, cuts = add_suspect_flags(report)
    assert not flagged["suspect_flag_p99"].any()
    assert cuts == {}


def test_cell_percentile_is_within_unit_interval():
    features, table = _synthetic()
    report, _, _ = build_category_report(features, table)
    values = report["cell_percentile"].dropna()
    assert float(values.min()) > 0.0 and float(values.max()) <= 1.0


# ------------------------------------------------------------------ 재현성


def test_report_is_reproducible():
    features, table = _synthetic()
    first, _, cuts_first = build_category_report(features, table)
    second, _, cuts_second = build_category_report(features, table)
    assert np.allclose(first["suspect_score"], second["suspect_score"], equal_nan=True)
    assert cuts_first == cuts_second


def test_transform_does_not_mutate_input():
    features, _ = _synthetic(per_cell=10, dimension=8)
    original = features.copy()
    apply_transform(features, "log1p")
    assert np.array_equal(features, original)


def test_unknown_transform_raises():
    try:
        apply_transform(np.zeros((2, 4)), "sqrt")
    except ValueError:
        return
    raise AssertionError("알 수 없는 transform이 통과했습니다.")


# ------------------------------------------------------------------ feature 진단


def test_diagnostics_cover_every_feature_set_and_transform():
    features, table = _synthetic(per_cell=40)
    diagnostics = feature_set_diagnostics(features, table, scopes=("Lab PD",))
    assert set(diagnostics["feature_set"]) == set(FEATURE_SETS)
    assert set(diagnostics["transform"]) == set(TRANSFORMS)
    values = diagnostics["cv_macro_f1"].dropna()
    assert float(values.min()) >= 0.0 and float(values.max()) <= 1.0
    assert diagnostics["silhouette_label"].between(-1.0, 1.0).all()


def test_diagnostics_separate_well_on_separable_synthetic_data():
    """셀 중심이 다른 차원에 놓인 합성 데이터에서는 macro F1이 높아야 한다."""
    features, table = _synthetic(per_cell=40)
    diagnostics = feature_set_diagnostics(features, table, scopes=("Lab PD",))
    combined = diagnostics[diagnostics["feature_set"] == "mean+max(256)"]
    assert float(combined["cv_macro_f1"].max()) > 0.9


# ------------------------------------------------------------------ 실데이터 회귀


def _real_dataset():
    root = ROOT.parent / "artifacts"
    folders = sorted(path for path in root.glob("ai_data_*") if path.is_dir())
    folders = [path for path in folders if not (path / "STALE_AXIS_BUG.md").is_file()]
    if not folders:
        raise Skip("유효한 artifacts/ai_data_* 폴더가 없습니다.")
    sys.path.insert(0, str(ROOT.parent))
    from Diffusion.prpd_diffusion.contract import AiDataset

    dataset = AiDataset.load(folders[-1])
    metadata = dataset.metadata.copy()
    features = np.zeros((len(metadata), DIMENSION), dtype=np.float32)
    for group in dataset.groups:
        array = dataset.group_array(group)
        index = metadata.index[metadata["group"] == group].to_numpy()
        block = array[metadata.loc[index, "tensor_index"].to_numpy()]
        features[index, :128] = block.mean(axis=2)
        features[index, 128:] = block.max(axis=2)
    return features, metadata


def test_real_data_combined_feature_beats_mean_alone():
    """mean+max가 mean 단독보다 잘 분리한다 — 256차원 결합의 근거."""
    features, metadata = _real_dataset()
    diagnostics = feature_set_diagnostics(
        features, metadata, transforms=("log1p",), scopes=("Lab PD",)
    )
    scores = diagnostics.set_index("feature_set")["cv_macro_f1"]
    assert scores["mean+max(256)"] > scores["mean(128)"], scores.to_dict()
    assert scores["mean+max(256)"] > scores["max(128)"], scores.to_dict()


def test_real_data_score_is_not_a_proxy_for_cell_size():
    """작은 셀이라는 이유만으로 점수가 높아지면 리포트가 무의미해진다."""
    features, metadata = _real_dataset()
    report, _, _ = build_category_report(features, metadata)
    scorable = report[report["reference_status"].eq("ok")]
    summary = scorable.groupby(["group", "current_category"]).agg(
        size=("suspect_score", "size"), median=("suspect_score", "median")
    )
    correlation = float(np.corrcoef(summary["size"], summary["median"])[0, 1])
    assert abs(correlation) < 0.5, f"셀 크기와 중앙 d²의 상관 {correlation:.3f}"
