"""
Objective function and cross-validation splits.

Dieng et al. minimise RMSE between simulated and observed malaria incidence
aggregated over the whole district, with cross-validation to prevent
overfitting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rmse(sim: pd.Series, obs: pd.Series) -> float:
    """RMSE over the overlapping months only. Returns inf if no overlap."""
    joined = pd.concat([sim.rename("sim"), obs.rename("obs")], axis=1, join="inner").dropna()
    if joined.empty:
        return float("inf")
    diff = joined["sim"].to_numpy(float) - joined["obs"].to_numpy(float)
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(sim: pd.Series, obs: pd.Series) -> float:
    joined = pd.concat([sim.rename("sim"), obs.rename("obs")], axis=1, join="inner").dropna()
    if joined.empty:
        return float("inf")
    return float(np.mean(np.abs(joined["sim"] - joined["obs"])))


def bias(sim: pd.Series, obs: pd.Series) -> float:
    joined = pd.concat([sim.rename("sim"), obs.rename("obs")], axis=1, join="inner").dropna()
    if joined.empty:
        return float("nan")
    return float(np.mean(joined["sim"] - joined["obs"]))


def pearson_r(sim: pd.Series, obs: pd.Series) -> float:
    joined = pd.concat([sim.rename("sim"), obs.rename("obs")], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return float("nan")
    a = joined["sim"].to_numpy(float)
    b = joined["obs"].to_numpy(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def skill_table(sim: pd.Series, obs: pd.Series) -> dict[str, float]:
    """Full validation metric set, matching what the paper reports."""
    return {
        "rmse": rmse(sim, obs),
        "mae": mae(sim, obs),
        "bias": bias(sim, obs),
        "pearson_r": pearson_r(sim, obs),
        "n_months": int(
            len(pd.concat([sim, obs], axis=1, join="inner").dropna())
        ),
    }


@dataclass(frozen=True)
class Fold:
    """One cross-validation fold: which years train, which years validate."""

    name: str
    train_years: tuple[int, ...]
    test_years: tuple[int, ...]


def blocked_year_folds(years: list[int], k: int = 4) -> list[Fold]:
    """
    Blocked k-fold over whole YEARS.

    Malaria time series are strongly autocorrelated within a year (seasonal
    cycle + immunity carry-over). Splitting individual months at random leaks
    information between train and test and flatters the model. Always split on
    contiguous year blocks.
    """
    years = sorted(set(int(y) for y in years))
    if k < 2 or k > len(years):
        raise ValueError(f"k must be in [2, {len(years)}], got {k}")
    blocks = np.array_split(np.array(years), k)
    folds: list[Fold] = []
    for i, block in enumerate(blocks):
        test = tuple(int(y) for y in block)
        train = tuple(y for y in years if y not in test)
        folds.append(Fold(name=f"fold{i + 1}", train_years=train, test_years=test))
    return folds


def subset_years(s: pd.Series, years: tuple[int, ...]) -> pd.Series:
    """Select the months belonging to the given years from a PeriodIndex series."""
    mask = np.isin(s.index.year, np.array(years))
    return s[mask]
