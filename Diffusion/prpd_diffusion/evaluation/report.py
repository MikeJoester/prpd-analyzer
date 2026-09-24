"""복원 결과 HTML 리포트. plotly만 사용한다(Analyzer와 동일 의존성)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from ..data.representation import phase_profiles, time_pool


def profile_comparison_figure(
    noisy: np.ndarray, restored: np.ndarray, clean: np.ndarray | None = None
) -> go.Figure:
    """위상별 평균/최대 profile 비교. 복원이 clean 쪽으로 이동했는지 본다."""
    figure = go.Figure()
    series = [("noisy", noisy), ("restored", restored)]
    if clean is not None:
        series.append(("clean", clean))
    for name, matrix in series:
        mean_profile, max_profile = phase_profiles(matrix)
        figure.add_trace(go.Scatter(y=mean_profile, name=f"{name} mean", mode="lines"))
        figure.add_trace(
            go.Scatter(y=max_profile, name=f"{name} max", mode="lines", line=dict(dash="dot"))
        )
    figure.update_layout(
        title="위상별 평균/최대 profile",
        xaxis_title="phase bin (0-127)",
        yaxis_title="amplitude",
        height=420,
    )
    return figure


def prpd_heatmap_figure(matrix: np.ndarray, title: str, time_factor: int = 30) -> go.Figure:
    """PRPS 히트맵. 화면 표시용으로 시간축을 축소한다(분석 값은 변경하지 않음)."""
    display = time_pool(np.asarray(matrix), time_factor, mode="max")
    figure = go.Figure(data=go.Heatmap(z=display, colorscale="Turbo", zmin=0, zmax=255))
    figure.update_layout(
        title=title, xaxis_title=f"time (÷{time_factor})", yaxis_title="phase bin", height=380
    )
    return figure


def write_html_report(
    path: Path,
    title: str,
    metrics: dict,
    figures: list[go.Figure],
    notes: list[str] | None = None,
) -> Path:
    """자체 포함 HTML(plotly.js 인라인)로 저장한다. 사내망에서도 열린다."""
    blocks = [f"<h1>{title}</h1>"]
    if notes:
        blocks.append("<ul>" + "".join(f"<li>{line}</li>" for line in notes) + "</ul>")
    blocks.append("<h2>지표</h2><pre>" + json.dumps(metrics, ensure_ascii=False, indent=2) + "</pre>")
    for index, figure in enumerate(figures):
        blocks.append(
            figure.to_html(full_html=False, include_plotlyjs=True if index == 0 else False)
        )
    document = (
        "<html><head><meta charset='utf-8'><title>"
        + title
        + "</title><style>body{font-family:sans-serif;margin:2rem;max-width:1100px}"
        "pre{background:#f5f5f5;padding:1rem;overflow-x:auto}</style></head><body>"
        + "".join(blocks)
        + "</body></html>"
    )
    Path(path).write_text(document, encoding="utf-8")
    return Path(path)
