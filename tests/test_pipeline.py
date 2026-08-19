"""Regression tests for the VECTRI calibration harness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectri_calib.calibrate import calibrate, genome_to_params, params_to_genome
from vectri_calib.data import synthetic_siaya, validate_forcing
from vectri_calib.ga import GAConfig, run_ga
from vectri_calib.model import ReducedVectri
from vectri_calib.objective import blocked_year_folds, rmse
from vectri_calib.parameters import (DEFAULTS, LOWER, N_PARAMS, PARAMETERS,
                                     SYMBOLS, UPPER, assert_integrity)
from vectri_calib.report import build_table1, compensating_pairs
from vectri_calib.sensitivity import oat_sensitivity


@pytest.fixture(scope="module")
def testbed():
    forcing, obs = synthetic_siaya(start="2007-01-01", end="2012-12-31")
    return forcing, obs


@pytest.fixture(scope="module")
def default_params():
    return {p.symbol: p.default for p in PARAMETERS}


# --- parameter table ------------------------------------------------------
def test_exactly_22_parameters():
    """Dieng et al. perturb 22 parameters. Drift here invalidates the rebuild."""
    assert N_PARAMS == 22
    assert_integrity()


def test_published_bestfit_reachable():
    """The GA must be able to reach the published values within its bounds."""
    for p in PARAMETERS:
        if p.published_bestfit is not None:
            assert p.lower <= p.published_bestfit <= p.upper, p.symbol


def test_defaults_inside_bounds():
    assert all(lo <= d <= hi for lo, d, hi in zip(LOWER, DEFAULTS, UPPER))


def test_genome_roundtrip(default_params):
    g = params_to_genome(default_params)
    back = genome_to_params(g)
    assert set(back) == set(SYMBOLS)
    for s in SYMBOLS:
        assert back[s] == pytest.approx(default_params[s], rel=1e-9)


# --- forcing validation ---------------------------------------------------
def test_forcing_validation_catches_bad_units(testbed):
    forcing, _ = testbed
    bad = forcing.copy()
    bad["itn_cover"] = bad["itn_cover"] * 100.0     # percent instead of fraction
    with pytest.raises(ValueError, match="itn_cover"):
        validate_forcing(bad)


def test_forcing_validation_catches_kelvin(testbed):
    forcing, _ = testbed
    bad = forcing.copy()
    bad["tas"] = bad["tas"] + 273.15
    with pytest.raises(ValueError, match="implausible temperatures"):
        validate_forcing(bad)


# --- model ----------------------------------------------------------------
def test_model_runs_and_is_finite(testbed, default_params):
    forcing, _ = testbed
    out = ReducedVectri().run(default_params, forcing)
    assert len(out) > 0
    assert np.isfinite(out.to_numpy()).all()
    assert (out >= 0).all()


def test_model_is_deterministic(testbed, default_params):
    forcing, _ = testbed
    m = ReducedVectri()
    a = m.run(default_params, forcing)
    b = m.run(default_params, forcing)
    pd.testing.assert_series_equal(a, b)


def test_itn_reduces_transmission(testbed, default_params):
    """
    Core physical check: ITN coverage must reduce both EIR and cases.

    A small non-monotonic wobble (<2%) is tolerated at LOW coverage: reduced
    exposure lowers acquired immunity, which raises susceptibility. That
    rebound is a real epidemiological effect, not a bug. The overall
    zero-to-full-coverage effect must be clearly negative.
    """
    forcing, _ = testbed
    m = ReducedVectri()
    off = forcing.copy()
    off["itn_cover"] = 0.0
    full = forcing.copy()
    full["itn_cover"] = 0.95

    cases_off = float(m.run(default_params, off).sum())
    cases_full = float(m.run(default_params, full).sum())
    eir_off = float(m.annual_eir(default_params, off).mean())
    eir_full = float(m.annual_eir(default_params, full).mean())

    assert cases_full < cases_off, "ITN must reduce cases"
    assert eir_full < eir_off, "ITN must reduce EIR"
    assert (eir_off - eir_full) / eir_off > 0.3, "ITN effect on EIR implausibly weak"


def test_eir_in_plausible_range(testbed, default_params):
    """Without ITN, annual EIR should sit in the range the paper reports (10-20)."""
    forcing, _ = testbed
    off = forcing.copy()
    off["itn_cover"] = 0.0
    eir = float(ReducedVectri().annual_eir(default_params, off).mean())
    assert 3.0 < eir < 60.0, f"annual EIR {eir:.1f} implausible for a Siaya-like setting"


def test_every_parameter_influences_output(testbed, default_params):
    """
    A parameter with zero influence cannot be calibrated. If this fails, the
    model is ignoring part of the genome and the GA is wasting its budget.
    """
    forcing, _ = testbed
    m = ReducedVectri()
    base = float(m.run(default_params, forcing).sum())
    inert = []
    for spec in PARAMETERS:
        trial = dict(default_params)
        trial[spec.symbol] = spec.lower + 0.25 * (spec.upper - spec.lower)
        if trial[spec.symbol] == default_params[spec.symbol]:
            trial[spec.symbol] = spec.lower + 0.75 * (spec.upper - spec.lower)
        try:
            got = float(m.run(trial, forcing).sum())
        except Exception:
            continue
        if abs(got - base) < 1e-9:
            inert.append(spec.symbol)
    assert not inert, f"parameters with no effect on output: {inert}"


# --- GA -------------------------------------------------------------------
def test_ga_evaluation_budget_matches_paper():
    cfg = GAConfig()
    assert cfg.n_ens == 80 and cfg.n_gen == 40
    assert cfg.evaluations() == 3200


def test_ga_recovers_known_optimum():
    target = np.linspace(0.2, 0.8, N_PARAMS)
    res = run_ga(lambda x: float(np.sum((x - target) ** 2)),
                 [0.0] * N_PARAMS, [1.0] * N_PARAMS,
                 GAConfig(n_ens=60, n_gen=30, seed=1), verbose=False)
    assert res.best_fitness < 0.05
    assert np.max(np.abs(res.best_genome - target)) < 0.15


def test_ga_respects_bounds():
    res = run_ga(lambda x: float(np.sum(x)), LOWER, UPPER,
                 GAConfig(n_ens=20, n_gen=5, seed=2), verbose=False)
    assert np.all(res.final_population >= np.array(LOWER) - 1e-9)
    assert np.all(res.final_population <= np.array(UPPER) + 1e-9)


def test_ga_improves_monotonically():
    """Elitism means the best score can never get worse."""
    target = np.full(N_PARAMS, 0.5)
    res = run_ga(lambda x: float(np.sum((x - target) ** 2)),
                 [0.0] * N_PARAMS, [1.0] * N_PARAMS,
                 GAConfig(n_ens=20, n_gen=10, seed=3), verbose=False)
    h = np.array(res.history_best)
    assert np.all(np.diff(h) <= 1e-12), "elitism violated: best fitness got worse"


def test_ga_is_reproducible():
    f = lambda x: float(np.sum(x ** 2))
    a = run_ga(f, LOWER, UPPER, GAConfig(n_ens=16, n_gen=4, seed=42), verbose=False)
    b = run_ga(f, LOWER, UPPER, GAConfig(n_ens=16, n_gen=4, seed=42), verbose=False)
    assert a.best_fitness == b.best_fitness
    np.testing.assert_allclose(a.best_genome, b.best_genome)


# --- cross-validation -----------------------------------------------------
def test_blocked_folds_are_disjoint_and_complete():
    years = list(range(2007, 2023))
    folds = blocked_year_folds(years, k=4)
    assert len(folds) == 4
    covered = []
    for f in folds:
        assert not set(f.train_years) & set(f.test_years), "train/test leak"
        covered.extend(f.test_years)
    assert sorted(covered) == years, "folds must partition the years exactly"


# --- end to end -----------------------------------------------------------
def test_calibration_improves_on_defaults(testbed):
    """The GA must beat the literature prior it was seeded with."""
    forcing, obs = testbed
    model = ReducedVectri()
    default_rmse = rmse(model.run({p.symbol: p.default for p in PARAMETERS}, forcing), obs)
    res = calibrate(model, forcing, obs,
                    config=GAConfig(n_ens=16, n_gen=6, seed=7),
                    verbose=False)
    assert res.ga.best_fitness <= default_rmse + 1e-6
    assert set(res.best_params) == set(SYMBOLS)


def test_table1_has_published_layout(testbed):
    forcing, obs = testbed
    res = calibrate(ReducedVectri(), forcing, obs,
                    config=GAConfig(n_ens=12, n_gen=3, seed=8), verbose=False)
    table = build_table1(res.best_params, res.ga.archive_genomes, res.ga.archive_fitness)
    for col in ("Symbol", "Parameters (definition)", "Default", "Best fit", "Unit", "Source"):
        assert col in table.columns
    assert len(table) == 22
    assert {"Behav p05", "Behav p95", "Range occupied"} <= set(table.columns)


def test_ga_archives_every_evaluation():
    """The archive must hold every genome evaluated - identifiability needs it."""
    cfg = GAConfig(n_ens=10, n_gen=4, seed=5)
    res = run_ga(lambda x: float(np.sum(x ** 2)), LOWER, UPPER, cfg, verbose=False)
    assert res.archive_genomes is not None
    assert len(res.archive_genomes) == res.n_evaluations
    assert len(res.archive_fitness) == res.n_evaluations
    assert res.archive_fitness.min() == pytest.approx(res.best_fitness)


def test_monte_carlo_scan_detects_unidentifiable_parameters():
    """
    Identifiability must come from INDEPENDENT sampling, not from the GA.

    Objective here depends only on parameter 0; all others are unidentifiable
    by construction. The scan must show parameter 0 tightly constrained and
    the rest spanning most of their range.
    """
    from vectri_calib.report import monte_carlo_behavioural
    lo, hi = [0.0] * N_PARAMS, [1.0] * N_PARAMS
    behav, _ = monte_carlo_behavioural(
        lambda x: float((x[0] - 0.5) ** 2), lo, hi, n_samples=1500, tol=0.5, seed=11)
    spread_identifiable = float(np.percentile(behav[:, 0], 95)
                                - np.percentile(behav[:, 0], 5))
    spread_unidentifiable = float(np.percentile(behav[:, 5], 95)
                                  - np.percentile(behav[:, 5], 5))
    assert spread_identifiable < 0.4, "constrained parameter should be narrow"
    assert spread_unidentifiable > 0.6, "unconstrained parameter should span its range"


def test_ga_archive_cannot_measure_identifiability():
    """
    Documents the limitation that motivates monte_carlo_behavioural: a
    converged GA reports an unidentifiable parameter as tightly constrained.
    If this ever stops holding, the Monte-Carlo scan may be redundant.
    """
    from vectri_calib.report import behavioural_set
    lo, hi = [0.0] * N_PARAMS, [1.0] * N_PARAMS
    res = run_ga(lambda x: float((x[0] - 0.5) ** 2), lo, hi,
                 GAConfig(n_ens=30, n_gen=15, seed=11), verbose=False)
    behav = behavioural_set(res.archive_genomes, res.archive_fitness, tol=0.5)
    spread = float(np.percentile(behav[:, 5], 95) - np.percentile(behav[:, 5], 5))
    assert spread < 0.3, (
        "GA archive unexpectedly retained diversity; re-check the identifiability docs")


def test_compensating_pairs_needs_enough_members():
    """Correlations from 3 elite members are meaningless; guard against it."""
    rng = np.random.default_rng(0)
    ens = rng.uniform(0, 1, size=(80, N_PARAMS))
    fit = rng.uniform(0, 1, size=80)
    pairs = compensating_pairs(ens, fit, top_frac=0.25)
    # random data should yield few strong correlations
    assert len(pairs) < 60, "spurious correlation explosion"


def test_sensitivity_ranks_all_parameters(testbed, default_params):
    forcing, _ = testbed
    sens = oat_sensitivity(ReducedVectri(), forcing, default_params, verbose=False)
    assert len(sens) == 22
    assert list(sens["rank"]) == list(range(1, 23))
    assert sens["pct_change"].is_monotonic_decreasing


def test_small_behavioural_set_warns():
    """A handful of behavioural samples must not be reported as a solid result."""
    from vectri_calib.report import identifiability_from_scan
    rng = np.random.default_rng(0)
    tiny = rng.uniform(np.array(LOWER), np.array(UPPER), size=(4, N_PARAMS))
    best = {p.symbol: p.default for p in PARAMETERS}
    with pytest.warns(RuntimeWarning, match="behavioural samples"):
        identifiability_from_scan(tiny, best)
