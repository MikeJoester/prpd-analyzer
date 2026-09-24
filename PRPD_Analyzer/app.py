from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PRPD_Analyzer.prpd_analyzer.embedding import run_tsne
from PRPD_Analyzer.prpd_analyzer.dataset import generate_ai_dataset
from PRPD_Analyzer.prpd_analyzer.features import make_features
from PRPD_Analyzer.prpd_analyzer.io import PHASE_BINS, discover_files, load_samples as load_sample_records
from PRPD_Analyzer.prpd_analyzer.metadata import GROUPS, LABELS, infer_metadata
from PRPD_Analyzer.prpd_analyzer.plots import profile_figure
from PRPD_Analyzer.prpd_analyzer.quality import (
    DEFAULT_SHRINKAGE,
    build_category_report,
    feature_set_diagnostics,
)
from PRPD_Analyzer.prpd_analyzer.statistics import build_count_table, build_date_table


@st.cache_data(show_spinner=False)
def load_samples(root_text: str, selected_files: tuple[str, ...]) -> pd.DataFrame:
    return load_sample_records(root_text, selected_files)


st.set_page_config(page_title="PRPD Analyzer", page_icon="◌", layout="wide")
st.title("PRPD Data Analyzer")
st.caption("128 phase bins × 3600 time samples | mean/max 256-feature t-SNE")

with st.sidebar:
    root_text = st.text_input("Data Folder", value=str(Path(__file__).parent / "Data" / "by_type"))
    root = Path(root_text)
    files = discover_files(root) if root.is_dir() else []
    st.write(f"Discovered files: {len(files):,}")
    analyze_all = st.checkbox("Analyze all data (ignore filters)", value=True)
    selected_labels = st.multiselect("Fault Types", LABELS, default=list(LABELS))
    selected_groups = st.multiselect("Data Groups", GROUPS[:-1], default=list(GROUPS[:-1]))
    available_dates = sorted({infer_metadata(path, root)[2] for path in files})
    selected_dates = st.multiselect("Measurement Date", available_dates, default=available_dates)
    max_samples = st.number_input(
        "Max Samples",
        min_value=1,
        max_value=max(1, len(files)),
        value=max(1, len(files)),
        step=50,
    )
    perplexity = st.slider("t-SNE perplexity", 5.0, 50.0, 30.0, step=1.0)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    ai_groups = st.multiselect("Include Groups for AI Dataset", GROUPS[:-1], default=list(GROUPS[:-1]))
    ai_output_root_text = st.text_input(
        "AI Dataset Output Folder",
        value=str(REPO_ROOT / "artifacts"),
    )

if not files:
    st.error("Data folder not found. Please select Data/by_type or Data/by_date folder.")
    st.stop()

if analyze_all:
    filtered_files = files
else:
    filtered_files = [
        path
        for path in files
        if infer_metadata(path, root)[0] in selected_labels
        and infer_metadata(path, root)[1] in selected_groups
        and infer_metadata(path, root)[2] in selected_dates
    ]
file_pool = filtered_files
if not analyze_all and len(file_pool) > max_samples:
    rng = np.random.default_rng(int(seed))
    file_pool = sorted(rng.choice(file_pool, size=int(max_samples), replace=False).tolist())

st.info(
    f"Total search: {len(files):,} | Analysis target: {len(file_pool):,} "
    f"| Mode: {'All data' if analyze_all else 'Filter applied'} "
    f"| Selected groups: {', '.join(selected_groups) or 'None'} "
    "| Each file is converted into 256-dimensional (mean 128 + max 128) Features."
)

if st.button("Run Data Analysis and t-SNE", type="primary"):
    if len(file_pool) < 3:
        st.error("At least 3 valid files are required to run t-SNE.")
        st.stop()
    with st.spinner(f"Reading {len(file_pool):,} files and calculating Features and t-SNE..."):
        table = load_samples(str(root), tuple(str(path) for path in file_pool))
        valid = table[table["valid"]].copy()
        if len(valid) < 3:
            st.error("Less than 3 readable .dat files found. Please check the error list.")
            st.stop()
        mean_profiles = np.stack(valid["mean_profile"].to_numpy())
        max_profiles = np.stack(valid["max_profile"].to_numpy())
        features = make_features(mean_profiles, max_profiles)
        embedding = run_tsne(features, perplexity, int(seed))
        valid["t-SNE 1"] = embedding[:, 0]
        valid["t-SNE 2"] = embedding[:, 1]
        valid["feature_dim"] = features.shape[1]
        st.session_state["result"] = valid

result = st.session_state.get("result")
if result is None:
    st.info("Set conditions on the left and click `Run Data Analysis and t-SNE`.")
else:
    analysis_table = table if "table" in locals() else result
    legacy_noise = result["group"].isin(("Lab Noise", "Field Noise", "Synthetic")) & result["label"].eq("Unknown")
    if legacy_noise.any():
        result = result.copy()
        result.loc[legacy_noise, "label"] = "Noise"
        st.session_state["result"] = result
    summary_tab, pattern_tab, tsne_tab, data_tab, report_tab, ai_tab = st.tabs(
        ["Summary", "PRPD / Feature", "t-SNE", "Data", "Category Report", "AI Dataset"]
    )

    with summary_tab:
        metric_columns = st.columns(5)
        metric_columns[0].metric("Valid Analyzed Files", f"{len(result):,}")
        error_count = len(table) - len(result) if "table" in locals() else 0
        metric_columns[1].metric("Error Files", f"{error_count:,}")
        metric_columns[2].metric("Fault Types", f"{result['label'].nunique():,}")
        metric_columns[3].metric("Data Groups", f"{result['group'].nunique():,}")
        metric_columns[4].metric("Feature Dimension", "256")
        count_table = build_count_table(result)
        date_table = build_date_table(result)
        st.plotly_chart(
            px.bar(
                count_table,
                x="group",
                y="count",
                color="label",
                barmode="group",
                category_orders={"group": list(GROUPS[:-1]), "label": list(LABELS)},
                title="Fault Types Count by Data Group",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(date_table, x="date", y="count", color="group", markers=True, title="Data Count by Date"),
            use_container_width=True,
        )
        st.dataframe(
            date_table.pivot(index="date", columns="group", values="count").fillna(0).astype(int),
            use_container_width=True,
        )
        noise_table = result[result["group"].isin(("Lab Noise", "Field Noise"))]
        if not noise_table.empty:
            st.subheader("Noise Group Comparison")
            noise_counts = noise_table.groupby(["date", "group"], as_index=False).size().rename(columns={"size": "count"})
            noise_columns = st.columns(2)
            noise_columns[0].metric("Lab Noise", f"{(noise_table['group'] == 'Lab Noise').sum():,}")
            noise_columns[1].metric("Field Noise", f"{(noise_table['group'] == 'Field Noise').sum():,}")
            st.plotly_chart(
                px.bar(noise_counts, x="date", y="count", color="group", barmode="group", title="Lab Noise / Field Noise Comparison by Date"),
                use_container_width=True,
            )

    with pattern_tab:
        detail_groups = [group for group in GROUPS[:-1] if group in result["group"].unique()]
        detail_group = st.selectbox("Detailed Data Group", detail_groups)
        group_table = result[result["group"] == detail_group].reset_index(drop=True)
        detail_labels = [label for label in LABELS if label in group_table["label"].unique()]
        detail_label = st.selectbox("Fault Types", detail_labels)
        detail_table = group_table[group_table["label"] == detail_label].reset_index(drop=True)
        detail_options = detail_table.apply(
            lambda row: f"{row['file']} | {row['date']} | {row['label']}", axis=1
        ).tolist()
        selected_option = st.selectbox("Detailed File", detail_options)
        selected_row = detail_table.iloc[detail_options.index(selected_option)]
        st.subheader(f"{detail_group} / {detail_label}")
        st.write(f"File: {selected_row['file']} | Date: {selected_row['date']}")
        st.plotly_chart(
            profile_figure(detail_table, "mean_profile", f"{detail_group} / {detail_label} Phase Mean Value"),
            use_container_width=True,
        )
        st.plotly_chart(
            profile_figure(detail_table, "max_profile", f"{detail_group} / {detail_label} Phase Max Value"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.imshow(selected_row["matrix"], aspect="auto", color_continuous_scale="Viridis", labels={"x": "Time", "y": "Phase", "color": "Amplitude"}),
            use_container_width=True,
        )
        feature_table = pd.DataFrame(
            {
                "phase": np.arange(1, PHASE_BINS + 1),
                "mean": selected_row["mean_profile"],
                "max": selected_row["max_profile"],
            }
        )
        st.dataframe(feature_table, use_container_width=True, hide_index=True)

    with tsne_tab:
        figure = px.scatter(
            result,
            x="t-SNE 1",
            y="t-SNE 2",
            color="group",
            symbol="label",
            hover_data=["group", "label", "file", "date"],
            category_orders={"group": list(GROUPS[:-1]), "label": list(LABELS)},
            height=680,
        )
        figure.update_traces(marker={"size": 9, "opacity": 0.8})
        st.plotly_chart(figure, use_container_width=True)
        tsne_export = result[["file", "path", "label", "group", "date", "feature_dim", "t-SNE 1", "t-SNE 2"]]
        st.download_button("t-SNE Result CSV", tsne_export.to_csv(index=False).encode("utf-8-sig"), "prpd_tsne.csv", "text/csv")

    with data_tab:
        display_columns = ["file", "label", "group", "date", "path", "t-SNE 1", "t-SNE 2"]
        st.dataframe(result[display_columns], use_container_width=True, hide_index=True)
        errors = table[~table["valid"]] if "table" in locals() else pd.DataFrame()
        if not errors.empty:
            st.warning(f"Read Error Files: {len(errors):,}")
            st.dataframe(errors[["path", "error"]], use_container_width=True, hide_index=True)

    with report_tab:
        report_features = make_features(
            np.stack(result["mean_profile"].to_numpy()),
            np.stack(result["max_profile"].to_numpy()),
        )
        st.subheader("Category Quality Review Target")
        st.caption(
            "Scores are for manual review prioritization, not automatic reclassification. "
            "Based on pooled shrinkage Mahalanobis distance per (group, label) cell, "
            "threshold is empirical percentile."
        )
        try:
            report, cell_model, cuts = build_category_report(report_features, result)
        except ValueError as error:
            st.error(f"Cannot build reference distribution: {error}")
            st.stop()

        for message in cell_model.warnings:
            st.warning(message)

        scorable = report[report["reference_status"].eq("ok")]
        model_columns = st.columns(4)
        model_columns[0].metric("Scorable", f"{len(scorable):,}")
        model_columns[1].metric("Reference Cell", f"{len(cell_model.means):,}")
        model_columns[2].metric("shrinkage", f"{cell_model.shrinkage:g}")
        model_columns[3].metric("Transform", cell_model.transform)

        view_choice = st.radio(
            "Review Scope",
            ("Top 1% (d²)", "Top 5% (d²)", "Closer to another category (margin > 0)", "All"),
            horizontal=True,
        )
        if view_choice.startswith("Top 1%"):
            suspects = scorable[scorable["suspect_flag_p99"]]
        elif view_choice.startswith("Top 5%"):
            suspects = scorable[scorable["suspect_flag_p95"]]
        elif view_choice.startswith("Closer"):
            suspects = scorable[scorable["margin"] > 0]
        else:
            suspects = scorable

        cut_text = " | ".join(f"{name} ≥ {value:.1f}" for name, value in cuts.items())
        st.write(f"Review Target: {len(suspects):,} / Scorable: {len(scorable):,}")
        st.caption(f"Applied Cuts — {cut_text}")
        review_columns = [
            "file", "group", "current_category", "date", "suspect_score",
            "nearest_category", "d2_nearest", "margin", "cell_percentile",
            "suspect_reason", "review_status", "path",
        ]
        st.dataframe(
            suspects[review_columns].style.format(
                {
                    "suspect_score": "{:.1f}",
                    "d2_nearest": "{:.1f}",
                    "margin": "{:+.1f}",
                    "cell_percentile": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        unscorable = report[~report["reference_status"].eq("ok")]
        if not unscorable.empty:
            with st.expander(f"Insufficient Reference Sample — Cannot calculate score ({len(unscorable):,})"):
                st.caption(
                    "Cells with insufficient samples have their mean close to themselves, so the distance "
                    "becomes near 0. They are separated to prevent being mistaken as 'normal'."
                )
                st.dataframe(
                    unscorable[["file", "group", "current_category", "date", "nearest_category", "path"]],
                    use_container_width=True,
                    hide_index=True,
                )

        st.download_button(
            "Category Report CSV",
            report.to_csv(index=False).encode("utf-8-sig"),
            "prpd_category_review_report.csv",
            "text/csv",
        )

        st.divider()
        st.subheader("Feature Quality Diagnostics")
        st.caption(
            "This is a **diagnostic metric** comparing how well Mean / Max / Mean+Max separate PD classes. "
            "It is not a classifier and is not used for the suspect scores above. "
            "Since silhouette cannot distinguish max and mean+max in this data, "
            "CV macro F1 by date is also considered."
        )
        if st.button("Run Feature Quality Diagnostics"):
            with st.spinner("Cross-validating feature set × transform × scope combination..."):
                st.session_state["diagnostics"] = feature_set_diagnostics(
                    report_features, result, shrinkage=DEFAULT_SHRINKAGE
                )
        diagnostics = st.session_state.get("diagnostics")
        if diagnostics is not None and not diagnostics.empty:
            st.dataframe(
                diagnostics.style.format(
                    {"silhouette_label": "{:+.3f}", "silhouette_group": "{:+.3f}", "cv_macro_f1": "{:.3f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Feature Diagnostics CSV",
                diagnostics.to_csv(index=False).encode("utf-8-sig"),
                "prpd_feature_diagnostics.csv",
                "text/csv",
            )

    with ai_tab:
        st.subheader("Generate AI Training Dataset based on raw data")
        st.caption("Saves the original 128×3600 raw tensor instead of 256-dimensional features.")
        st.write(f"Included Groups: {', '.join(ai_groups) if ai_groups else 'None'}")
        st.write(f"Output Location: {Path(ai_output_root_text).resolve()}")
        if st.button("Generate AI Dataset for Selected Groups", type="primary"):
            if not ai_groups:
                st.error("Please select at least one group.")
            else:
                try:
                    output_folder = generate_ai_dataset(
                        analysis_table,
                        root,
                        ai_groups,
                        Path(ai_output_root_text),
                        int(seed),
                    )
                    st.success(f"AI Dataset Generation Complete: {output_folder}")
                    st.session_state["ai_output_folder"] = str(output_folder)
                except (OSError, ValueError) as error:
                    st.error(f"AI Dataset Generation Failed: {error}")
        if st.session_state.get("ai_output_folder"):
            st.code(st.session_state["ai_output_folder"])