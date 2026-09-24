from __future__ import annotations

import pandas as pd


def build_count_table(table: pd.DataFrame) -> pd.DataFrame:
    return table.groupby(["group", "label"], as_index=False).size().rename(columns={"size": "count"})


def build_date_table(table: pd.DataFrame) -> pd.DataFrame:
    return table.groupby(["date", "group"], as_index=False).size().rename(columns={"size": "count"})
