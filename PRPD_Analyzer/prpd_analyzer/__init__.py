"""Reusable analysis modules for the PRPD web analyzer."""

from .embedding import run_tsne
from .dataset import generate_ai_dataset
from .features import make_features
from .io import discover_files, load_samples, read_dat
from .metadata import GROUPS, LABELS, infer_metadata
from .quality import (
    CellModel,
    build_category_report,
    cell_distances,
    feature_set_diagnostics,
    fit_cell_model,
)

__all__ = [
    "GROUPS",
    "LABELS",
    "CellModel",
    "build_category_report",
    "cell_distances",
    "feature_set_diagnostics",
    "fit_cell_model",
    "generate_ai_dataset",
    "discover_files",
    "infer_metadata",
    "load_samples",
    "make_features",
    "read_dat",
    "run_tsne",
]
