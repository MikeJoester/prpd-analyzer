"""ARDD 1D 트랙의 리포트 그림. plotly만 사용(Analyzer와 동일 의존성).

HTML 작성 자체는 2D 트랙의 `evaluation.report.write_html_report`를 그대로 쓴다.
그림만 profile(2, 128) 표현에 맞게 새로 만든다.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..data.profile import CHANNELS


def profile_pair_figure(
    noisy: np.ndarray,
    restored: np.ndarray,
    clean: np.ndarray | None = None,
    title: str = "위상 profile 복원 비교",
) -> go.Figure:
    """`(2, 128)` profile 하나의 복원 전/후 비교. 복원이 clean 쪽으로 갔는지 본다."""
    figure = go.Figure()
    series = [("noisy", noisy), ("restored", restored)]
    if clean is not None:
        series.append(("clean", clean))
    dashes = {"mean": "solid", "max": "dot"}
    for name, profile in series:
        array = np.asarray(profile)
        for index, channel in enumerate(CHANNELS):
            figure.add_trace(
                go.Scatter(
                    y=array[index],
                    name=f"{name} {channel}",
                    mode="lines",
                    line=dict(dash=dashes[channel]),
                )
            )
    figure.update_layout(
        title=title,
        xaxis_title="phase bin (0-127)",
        yaxis_title="amplitude (0-255)",
        height=430,
    )
    return figure


def channel_improvement_figure(metrics: dict, title: str = "채널별 MAE 개선폭") -> go.Figure:
    """mean/max 채널의 baseline 대비 개선폭.

    **채널을 합치지 않는 이유**를 그림으로 보여 준다. mean 채널은 시간 평균이라 노이즈에
    둔감하므로 개선폭이 0에 가까울 수 있고, 그것이 곧 실패는 아니다.
    """
    names = list(CHANNELS)
    baseline = [metrics.get(f"baseline_{name}_profile_mae", 0.0) for name in names]
    restored = [metrics.get(f"{name}_profile_mae", 0.0) for name in names]
    figure = go.Figure()
    figure.add_trace(go.Bar(x=names, y=baseline, name="baseline (무처리)"))
    figure.add_trace(go.Bar(x=names, y=restored, name="복원 후"))
    figure.update_layout(
        title=title, barmode="group", yaxis_title="MAE (0-255 스케일)", height=380
    )
    return figure


def snr_sweep_figure(rows: list[dict], title: str = "SNR별 성능 (ARDD-2025 프로토콜)") -> go.Figure:
    """SNR 스윕 결과 곡선. `identity` baseline과 함께 읽어 무너지는 지점을 본다."""
    figure = go.Figure()
    snr_values = [row["snr_db"] for row in rows]
    for channel in CHANNELS:
        figure.add_trace(
            go.Scatter(
                x=snr_values,
                y=[row.get(f"baseline_{channel}_profile_mae") for row in rows],
                name=f"{channel} baseline",
                mode="lines+markers",
                line=dict(dash="dot"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=snr_values,
                y=[row.get(f"{channel}_profile_mae") for row in rows],
                name=f"{channel} 복원",
                mode="lines+markers",
            )
        )
    figure.update_layout(
        title=title, xaxis_title="SNR (dB)", yaxis_title="MAE (0-255 스케일)", height=430
    )
    figure.update_xaxes(autorange="reversed")  # +3 → -12 방향으로 읽는다
    return figure


def missing_sweep_figure(rows: list[dict], title: str = "결측 비율별 성능") -> go.Figure:
    figure = go.Figure()
    fractions = [row["fraction"] for row in rows]
    for channel in CHANNELS:
        figure.add_trace(
            go.Scatter(
                x=fractions,
                y=[row.get(f"baseline_{channel}_profile_mae") for row in rows],
                name=f"{channel} baseline",
                mode="lines+markers",
                line=dict(dash="dot"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=fractions,
                y=[row.get(f"{channel}_profile_mae") for row in rows],
                name=f"{channel} 복원",
                mode="lines+markers",
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="연속 결측 비율 (시간축)",
        yaxis_title="MAE (0-255 스케일)",
        height=430,
    )
    return figure
