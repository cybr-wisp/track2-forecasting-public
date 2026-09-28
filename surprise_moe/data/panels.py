"""Point-in-time safe panel loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ASSET_COLUMNS = ("asset", "asset_id")


def read_panels(panels_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all parquet panels available to the forecast unit."""
    found = sorted(panels_dir.glob("*.parquet"))

    if not found and panels_dir.parent.is_dir():
        found = sorted(panels_dir.parent.glob("*.parquet"))

    if not found:
        raise ValueError(f"No parquet panels found under {panels_dir}")

    return {path.stem: pd.read_parquet(path) for path in found}


def asset_column(df: pd.DataFrame) -> str | None:
    """Return the asset identifier column used by a panel."""
    return next((column for column in ASSET_COLUMNS if column in df.columns), None)


def get_asset_history(
    panels: dict[str, pd.DataFrame],
    asset: str,
    asof: str,
) -> pd.Series:
    """Return one asset's history using observations at or before ``asof`` only."""
    cutoff = pd.Timestamp(asof)

    for frame in panels.values():
        column = asset_column(frame)

        if column is None:
            continue

        subset = frame.loc[frame[column].astype(str) == str(asset)].copy()

        if subset.empty:
            continue

        subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
        subset = subset.loc[
            subset["date"].notna() & (subset["date"] <= cutoff)
        ].sort_values("date")

        if subset.empty:
            continue

        values = pd.to_numeric(subset["value"], errors="coerce")
        valid = values.notna()

        return pd.Series(
            values.loc[valid].to_numpy(dtype=float),
            index=subset.loc[valid, "date"],
            name=str(asset),
        )

    raise ValueError(f"Asset {asset!r} unavailable at or before {asof}")
