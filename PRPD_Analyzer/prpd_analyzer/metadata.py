from __future__ import annotations

from pathlib import Path
import re

LABELS = ("Corona", "Floating", "Noise", "Particle", "Void")
GROUPS = ("Lab PD", "Field PD", "Lab Noise", "Field Noise", "Synthetic", "Unknown")

_GROUP_BY_FOLDER = {
    "Lab": "Lab PD",
    "Lab_PD": "Lab PD",
    "Lab_Noise": "Lab Noise",
    "Field": "Field PD",
    "Field_PD": "Field PD",
    "Field_Noise": "Field Noise",
    "Artificially_Generated": "Synthetic",
    "Randomly_Generated": "Synthetic",
}


def infer_metadata(path: Path, root: Path) -> tuple[str, str, str]:
    relative_parts = path.relative_to(root).parts
    group = next(
        (_GROUP_BY_FOLDER[part] for part in relative_parts if part in _GROUP_BY_FOLDER),
        "Unknown",
    )
    label = next((part for part in relative_parts if part in LABELS), "Unknown")
    if label == "Unknown" and group in ("Lab Noise", "Field Noise", "Synthetic"):
        label = "Noise"
    date_match = re.search(r"(?<!\d)(20\d{6})(?!\d)", path.name)
    date = date_match.group(1) if date_match else next(
        (part for part in relative_parts if re.fullmatch(r"20\d{6}", part)),
        "unknown_date",
    )
    return label, group, date
