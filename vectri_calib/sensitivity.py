"""
Sensitivity analysis - the separate step Dieng et al. run alongside calibration.

Their finding to reproduce: TEMPERATURE-dependent parameters dominate, i.e.
groups G (gonotrophic), H (sporogonic) and the thermal members of E
(rlarv_tmax / rlarv_tmin). If your ranking does not put those on top,
something is wrong with your setup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibrate import genome_to_params
from .model import MalariaModel
from .parameters import BY_SYMBOL, LOWER, PARAMETERS, SYMBOLS, UPPER


def oat_sensitivity(
    model: MalariaModel,
    forcing: pd.DataFrame,
    base_params: dict[str, float],
    perturbation: float = 0.10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    One-at-a-time (OAT) sensitivity around the calibrated optimum.

    Each parameter is nudged +/- `perturbation` of its BOUND RANGE (not of its
    own value - that would make small-valued parameters look artificially
    insensitive). Returns a table ranked by elasticity.
    """
    base_sim = model.run(base_params, forcing)
    base_total = float(base_sim.sum())
    if base_total <= 0:
        raise RuntimeError("Baseline run produced zero cases; cannot compute sensitivity")

    rows = []
    for spec in PARAMETERS:
        span = spec.upper - spec.lower
        step = perturbation * span
        results = {}
        for direction, sign in (("up", 1.0), ("down", -1.0)):
            trial = dict(base_params)
            value = np.clip(base_params[spec.symbol] + sign * step, spec.lower, spec.upper)
            trial[spec.symbol] = float(round(value)) if spec.integer else float(value)
            try:
                total = float(model.run(trial, forcing).sum())
            except Exception:
                total = float("nan")
            results[direction] = total

        up, down = results["up"], results["down"]
        delta = abs(up - down)
        rows.append({
            "symbol": spec.symbol,
            "group": spec.group,
            "definition": spec.definition,
            "base_value": base_params[spec.symbol],
            "cases_up": up,
            "cases_down": down,
            "abs_change": delta,
            "pct_change": 100.0 * delta / base_total,
            "elasticity": (100.0 * delta / base_total) / (2.0 * perturbation * 100.0),
        })
        if verbose:
            print(f"  {spec.symbol:22s} {100.0 * delta / base_total:8.2f}% change")

    df = pd.DataFrame(rows).sort_values("pct_change", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)


def morris_screening(
    model: MalariaModel,
    forcing: pd.DataFrame,
    n_trajectories: int = 10,
    n_levels: int = 4,
    seed: int = 20260119,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Morris elementary-effects screening - a global alternative to OAT.

    OAT only probes the neighbourhood of one point. Morris samples across the
    whole bounded space, so it catches parameters whose influence depends on
    where you are. mu_star ranks importance; sigma flags interaction /
    nonlinearity.

    Cost: n_trajectories x (n_params + 1) model runs.
    """
    rng = np.random.default_rng(seed)
    lower = np.asarray(LOWER, dtype=float)
    upper = np.asarray(UPPER, dtype=float)
    k = len(SYMBOLS)
    delta = n_levels / (2.0 * (n_levels - 1))

    effects: list[np.ndarray] = []
    total_runs = n_trajectories * (k + 1)
    run = 0

    for _ in range(n_trajectories):
        base = rng.integers(0, n_levels - 1, size=k) / (n_levels - 1.0)
        order = rng.permutation(k)
        current = base.copy()

        def evaluate(unit_vec: np.ndarray) -> float:
            genome = lower + unit_vec * (upper - lower)
            try:
                return float(model.run(genome_to_params(genome), forcing).sum())
            except Exception:
                return float("nan")

        y_prev = evaluate(current)
        run += 1
        traj = np.full(k, np.nan)
        for idx in order:
            nxt = current.copy()
            nxt[idx] = min(nxt[idx] + delta, 1.0) if nxt[idx] + delta <= 1.0 \
                else max(nxt[idx] - delta, 0.0)
            y_next = evaluate(nxt)
            run += 1
            step = nxt[idx] - current[idx]
            traj[idx] = (y_next - y_prev) / step if step != 0 else np.nan
            current, y_prev = nxt, y_next
        effects.append(traj)
        if verbose:
            print(f"  Morris trajectory complete ({run}/{total_runs} runs)")

    arr = np.array(effects, dtype=float)
    df = pd.DataFrame({
        "symbol": SYMBOLS,
        "group": [BY_SYMBOL[s].group for s in SYMBOLS],
        "mu": np.nanmean(arr, axis=0),
        "mu_star": np.nanmean(np.abs(arr), axis=0),
        "sigma": np.nanstd(arr, axis=0),
    }).sort_values("mu_star", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)


def temperature_dominance_check(sens: pd.DataFrame, top_n: int = 6) -> dict[str, object]:
    """
    Test the paper's central sensitivity claim: temperature parameters dominate.
    """
    thermal = {"G_gono", "H_sporo"}
    thermal_symbols = {"rlarv_tmax", "rlarv_tmin"}
    top = sens.head(top_n)
    is_thermal = top.apply(
        lambda r: r["group"] in thermal or r["symbol"] in thermal_symbols, axis=1
    )
    n_thermal = int(is_thermal.sum())
    return {
        "top_n": top_n,
        "n_temperature_related_in_top": n_thermal,
        "fraction": n_thermal / top_n,
        "top_symbols": top["symbol"].tolist(),
        "matches_paper_claim": n_thermal >= top_n / 2,
    }
