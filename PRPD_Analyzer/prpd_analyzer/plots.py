from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .io import PHASE_BINS


def profile_figure(table: pd.DataFrame, profile_column: str, title: str) -> go.Figure:
    figure = go.Figure()
    for group, subset in table.groupby("group"):
        profiles = np.stack(subset[profile_column].to_numpy())
        figure.add_trace(
            go.Scatter(
                x=np.arange(1, PHASE_BINS + 1),
                y=profiles.mean(axis=0),
                mode="lines",
                name=group,
            )
        )
    figure.update_layout(title=title, xaxis_title="Phase bin", yaxis_title="Amplitude")
    return figure
