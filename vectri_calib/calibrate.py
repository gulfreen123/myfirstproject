"""
Calibration driver: wires model + objective + GA together.

This is the module that actually produces the "Best fit" column of Table 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .ga import GAConfig, GAResult, run_ga
from .model import MalariaModel
from .objective import Fold, blocked_year_folds, rmse, skill_table, subset_years
from .parameters import DEFAULTS, LOWER, PARAMETERS, SYMBOLS, UPPER


def genome_to_params(genome: Sequence[float]) -> dict[str, float]:
    """Map a GA genome (ordered array) to a named VECTRI parameter dict."""
    if len(genome) != len(SYMBOLS):
        raise ValueError(f"genome length {len(genome)} != {len(SYMBOLS)}")
    out: dict[str, float] = {}
    for value, spec in zip(genome, PARAMETERS):
        out[spec.symbol] = float(round(value)) if spec.integer else float(value)
    return out


def params_to_genome(params: dict[str, float]) -> np.ndarray:
    return np.array([params[s] for s in SYMBOLS], dtype=float)


@dataclass
class CalibrationResult:
    ga: GAResult
    best_params: dict[str, float]
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    fold_metrics: list[dict[str, object]]
    folds: list[Fold]


def make_fitness(
    model: MalariaModel,
    forcing: pd.DataFrame,
    obs: pd.Series,
    train_years: tuple[int, ...],
    penalty: float = 1e9,
) -> callable:
    """
    Build the objective the GA minimises: RMSE on the TRAINING months only.

    A crashing or non-finite model run scores `penalty` rather than raising,
    so one bad parameter vector cannot kill a 3200-run calibration.
    """
    obs_train = subset_years(obs, train_years)

    def fitness(genome: np.ndarray) -> float:
        try:
            params = genome_to_params(genome)
            sim = model.run(params, forcing)
            if sim.isna().all() or not np.isfinite(sim.to_numpy(float)).any():
                return penalty
            score = rmse(sim, obs_train)
            return score if np.isfinite(score) else penalty
        except Exception:
            return penalty

    return fitness


def calibrate(
    model: MalariaModel,
    forcing: pd.DataFrame,
    obs: pd.Series,
    config: GAConfig | None = None,
    holdout_years: tuple[int, ...] | None = None,
    n_folds: int = 4,
    seed_with_defaults: bool = True,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Full calibration, matching the paper's protocol.

    holdout_years : years withheld from the GA entirely and used only for the
                    final independent evaluation. Choose these BEFORE running.
    n_folds       : blocked-year CV folds evaluated with the final best genome
                    to expose overfitting.
    """
    all_years = sorted(set(int(y) for y in obs.index.year))
    holdout = tuple(holdout_years or ())
    for y in holdout:
        if y not in all_years:
            raise ValueError(f"holdout year {y} not present in observations")
    train_years = tuple(y for y in all_years if y not in holdout)
    if not train_years:
        raise ValueError("no training years left after removing the holdout")

    if verbose:
        print(f"Training years : {train_years[0]}-{train_years[-1]} ({len(train_years)} yr)")
        print(f"Holdout years  : {holdout if holdout else 'none'}")

    fitness_fn = make_fitness(model, forcing, obs, train_years)
    seeds = [list(DEFAULTS)] if seed_with_defaults else None

    if verbose:
        cfg = config or GAConfig()
        print(f"GA: {cfg.n_ens} members x {cfg.n_gen} generations "
              f"= {cfg.evaluations()} model evaluations")

    ga_result = run_ga(
        fitness_fn, LOWER, UPPER, config=config,
        seed_genomes=seeds, verbose=verbose,
    )

    best_params = genome_to_params(ga_result.best_genome)
    sim = model.run(best_params, forcing)

    train_metrics = skill_table(sim, subset_years(obs, train_years))
    test_metrics = (skill_table(sim, subset_years(obs, holdout))
                    if holdout else {})

    # Blocked-year CV diagnostics using the final genome. A large gap between
    # in-fold and out-of-fold RMSE is the overfitting signal.
    folds = blocked_year_folds(list(train_years), k=min(n_folds, len(train_years)))
    fold_metrics: list[dict[str, object]] = []
    for fold in folds:
        fold_metrics.append({
            "fold": fold.name,
            "test_years": fold.test_years,
            "train_rmse": rmse(sim, subset_years(obs, fold.train_years)),
            "test_rmse": rmse(sim, subset_years(obs, fold.test_years)),
        })

    return CalibrationResult(
        ga=ga_result,
        best_params=best_params,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        fold_metrics=fold_metrics,
        folds=folds,
    )


def itn_experiment(
    model: MalariaModel,
    forcing: pd.DataFrame,
    params: dict[str, float],
    obs: pd.Series | None = None,
) -> dict[str, object]:
    """
    Reproduce the paper's headline with/without-ITN contrast.

    Targets from Dieng et al.: ~58% EIR reduction, ~41% incidence error
    reduction, +/-100-150 cases per month mean deviation.
    """
    forcing_off = forcing.copy()
    forcing_off["itn_cover"] = 0.0

    sim_on = model.run(params, forcing)
    sim_off = model.run(params, forcing_off)

    total_on = float(sim_on.sum())
    total_off = float(sim_off.sum())
    incidence_reduction = (
        100.0 * (total_off - total_on) / total_off if total_off > 0 else float("nan")
    )

    result: dict[str, object] = {
        "total_cases_with_itn": total_on,
        "total_cases_without_itn": total_off,
        "incidence_reduction_pct": incidence_reduction,
        "mean_monthly_cases_avoided": float((sim_off - sim_on).mean()),
    }

    if obs is not None:
        m_on = skill_table(sim_on, obs)
        m_off = skill_table(sim_off, obs)
        result["rmse_with_itn"] = m_on["rmse"]
        result["rmse_without_itn"] = m_off["rmse"]
        result["error_reduction_pct"] = (
            100.0 * (m_off["rmse"] - m_on["rmse"]) / m_off["rmse"]
            if m_off["rmse"] > 0 else float("nan")
        )
        result["mean_abs_monthly_deviation"] = m_on["mae"]
    return result
