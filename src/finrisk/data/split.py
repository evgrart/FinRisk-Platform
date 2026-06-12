"""Leakage-safe temporal splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Three chronological partitions and their boundary timestamps."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def temporal_split(
    frame: pd.DataFrame,
    *,
    time_column: str,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> TemporalSplit:
    """Split rows chronologically using quantile-derived time boundaries."""

    if time_column not in frame.columns:
        raise KeyError(f"missing time column: {time_column}")
    if len(frame) < 3:
        raise ValueError("at least three rows are required for a temporal split")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    timestamps = pd.to_datetime(frame[time_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{time_column} contains invalid or missing timestamps")

    ordered = frame.assign(_temporal_split_time=timestamps).sort_values(
        "_temporal_split_time", kind="stable"
    )
    train_end = timestamps.quantile(train_ratio)
    validation_end = timestamps.quantile(train_ratio + validation_ratio)
    if train_end >= validation_end:
        raise ValueError("time boundaries are not strictly increasing")

    train = ordered[ordered["_temporal_split_time"] < train_end]
    validation = ordered[
        (ordered["_temporal_split_time"] >= train_end)
        & (ordered["_temporal_split_time"] < validation_end)
    ]
    test = ordered[ordered["_temporal_split_time"] >= validation_end]
    partitions = [train, validation, test]
    if any(partition.empty for partition in partitions):
        raise ValueError("temporal boundaries produced an empty partition")

    return TemporalSplit(
        train=train.drop(columns="_temporal_split_time"),
        validation=validation.drop(columns="_temporal_split_time"),
        test=test.drop(columns="_temporal_split_time"),
        train_end=pd.Timestamp(train_end),
        validation_end=pd.Timestamp(validation_end),
    )


def split_summary(
    split: TemporalSplit, *, time_column: str, target_column: str | None = None
) -> pd.DataFrame:
    """Return a compact summary suitable for experiment logs."""

    records = []
    for name, partition in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        times = pd.to_datetime(partition[time_column])
        record = {"split": name, "rows": len(partition), "time_min": times.min(), "time_max": times.max()}
        if target_column is not None:
            record["target_rate"] = float(partition[target_column].mean())
        records.append(record)
    return pd.DataFrame(records)
