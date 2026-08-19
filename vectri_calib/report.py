"""
Build the Table 1 deliverable, plus the diagnostics that should accompany it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .parameters import GROUPS, PARAMETERS, UNIT_NOTES


def behavioural_set(
    genomes: np.ndarray, fitness: np.ndarray, tol: float = 0.10
) -> np.ndarray:
    """
    GLUE-style behavioural set: every genome scoring within `tol` (fractional)
    of the best RMSE ever found.

    Using the GA's FINAL POPULATION for this is wrong: elitism and selection
    collapse it onto a single point, so the spread measures how converged the
    GA is, not how well the data constrains the parameter. The behavioural set
    is drawn from every genome ever evaluated, so a parameter that the data
    genuinely does not pin down shows a wide spread.
    """
    finite = fitness[np.isfinite(fitness)]
    if finite.size == 0:
        raise ValueError("no finite fitness values in archive")
    best = float(np.nanmin(finite))
    # Scale-free threshold. A purely multiplicative band (best * (1 + tol))
    # collapses to a single point whenever the best score approaches zero, and
    # would then report every parameter as perfectly constrained. Anchoring the
    # band to the spread between best and median keeps it meaningful for any
    # objective scale.
    median = float(np.nanpercentile(finite, 50))
    threshold = max(best * (1.0 + tol), best + tol * (median - best))
    keep = np.isfinite(fitness) & (fitness <= threshold)
    if keep.sum() < 5:
        keep = np.isfinite(fitness) & (fitness <= np.nanpercentile(finite, 5))
    return genomes[keep]


def build_table1(
    best_params: dict[str, float],
    ensemble: np.ndarray | None = None,
    ensemble_fitness: np.ndarray | None = None,
    top_frac: float = 0.25,
) -> pd.DataFrame:
    """
    Assemble Table 1 in the published layout, plus columns the original lacks.

    ensemble / ensemble_fitness: the GA's final population. Supplying them adds
    a credible interval derived from the best-performing `top_frac` of members.
    The published table reports point estimates only; adding spread is the
    single biggest improvement you can make over the original.
    """
    rows = []
    behav = None
    if ensemble is not None and ensemble_fitness is not None:
        behav = behavioural_set(ensemble, ensemble_fitness)

    for j, spec in enumerate(PARAMETERS):
        fitted = best_params[spec.symbol]
        row = {
            "Symbol": spec.symbol,
            "Parameters (definition)": spec.definition,
            "Default": spec.default,
            "Best fit": round(fitted, 4),
            "Unit": spec.unit,
            "Source": ",".join(str(s) for s in spec.sources),
            "Group": GROUPS[spec.group],
            "Search lower": spec.lower,
            "Search upper": spec.upper,
            "Ratio (fit/default)": round(fitted / spec.default, 3) if spec.default else np.nan,
            "Published best fit": spec.published_bestfit,
        }
        if behav is not None:
            vals = behav[:, j]
            row["Behav p05"] = round(float(np.percentile(vals, 5)), 4)
            row["Behav p95"] = round(float(np.percentile(vals, 95)), 4)
            span = spec.upper - spec.lower
            # Fraction of the search range that well-fitting solutions still
            # span. Near 1.0 = the data does NOT constrain this parameter and
            # its "best fit" value is close to arbitrary.
            row["Range occupied"] = round(
                float(np.percentile(vals, 95) - np.percentile(vals, 5)) / span, 3
            )
            row["N behavioural"] = int(len(behav))
        rows.append(row)

    return pd.DataFrame(rows)


def identifiability_report(table: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Flag parameters the observations do not actually constrain.

    A HIGH 'Range occupied' means well-fitting solutions span most of the
    allowed range for that parameter: the calibration is not really estimating
    it, and its "best fit" value is close to arbitrary. Do not quote such a
    value as an estimate of a biological quantity.
    """
    if "Range occupied" not in table.columns:
        raise ValueError("Run build_table1 with the GA archive to get identifiability")
    out = table[["Symbol", "Group", "Best fit",
                 "Behav p05", "Behav p95", "Range occupied"]].copy()
    out["Verdict"] = np.where(
        out["Range occupied"] <= threshold, "constrained", "POORLY CONSTRAINED"
    )
    return out.sort_values("Range occupied", ascending=False)


def compensating_pairs(ensemble: np.ndarray, fitness: np.ndarray,
                       top_frac: float = 0.25, min_abs_r: float = 0.5) -> pd.DataFrame:
    """
    Detect equifinality: parameter pairs that trade off against each other
    among well-fitting solutions.

    This is the diagnostic that exposes the compensating structure visible in
    the published table (e.g. rbiteratio down but rpthost2vect_I up). Strongly
    correlated pairs mean the two values are only meaningful TOGETHER.
    """
    n_elite = max(3, int(len(fitness) * top_frac))
    elite = ensemble[np.argsort(fitness)[:n_elite]]
    symbols = [p.symbol for p in PARAMETERS]

    rows = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            a, b = elite[:, i], elite[:, j]
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if abs(r) >= min_abs_r:
                rows.append({
                    "param_a": symbols[i],
                    "param_b": symbols[j],
                    "correlation": round(r, 3),
                    "relationship": "compensating (trade-off)" if r < 0 else "reinforcing",
                })
    df = pd.DataFrame(rows)
    return df.reindex(df["correlation"].abs().sort_values(ascending=False).index) \
        if not df.empty else df


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no tabulate dep)."""
    cols = [str(c) for c in df.columns]

    def fmt(v: object) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def to_markdown(table: pd.DataFrame, path: str | Path, title: str = "Table 1") -> None:
    cols = ["Symbol", "Parameters (definition)", "Default", "Best fit", "Unit", "Source"]
    extra = [c for c in ("Behav p05", "Behav p95", "Range occupied") if c in table.columns]
    lines = [
        f"# {title}",
        "",
        "Rebuilt VECTRI parameter set. `Default` = literature prior (input); "
        "`Best fit` = genetic-algorithm result (output).",
        "",
        _md_table(table[cols + extra]),
        "",
        "## Unit corrections applied",
        "",
        "The published Table 1 contains unit errors. Corrected here:",
        "",
    ]
    for sym, note in UNIT_NOTES.items():
        lines.append(f"- **{sym}** - {note}")
    lines += [
        "",
        "## How to read this table",
        "",
        "- Best-fit values are only valid **as a set**. Do not transplant single "
        "rows into another model.",
        "- Check the identifiability report before quoting any individual value.",
        "- Parameter pairs flagged as compensating are not independently "
        "estimated; they trade off against each other.",
    ]
    Path(path).write_text("\n".join(lines))


def convergence_frame(history_best: list[float], history_mean: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "generation": range(len(history_best)),
        "best_rmse": history_best,
        "mean_rmse": history_mean,
    })


def monte_carlo_behavioural(
    fitness_fn,
    lower,
    upper,
    n_samples: int = 2000,
    tol: float = 0.10,
    seed: int = 20260119,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GLUE-style identifiability scan by INDEPENDENT random sampling.

    Why this exists
    ---------------
    A genetic algorithm is a point estimator. Selection and crossover collapse
    the population onto the incumbent best - including along dimensions the
    data does not constrain at all, where the collapse is pure genetic drift.
    So neither the GA's final population NOR its full evaluation archive can
    tell you how well a parameter is identified: both will report a
    near-arbitrary parameter as tightly constrained.

    Measuring identifiability requires samples drawn independently of the
    optimisation. This function draws `n_samples` vectors uniformly within the
    bounds, scores them, and returns the behavioural subset (within `tol` of
    the best score found in the scan). A parameter that the observations do
    not constrain will span most of its range in that subset.

    Cost: `n_samples` extra model evaluations. Budget for it separately from
    the 3200-evaluation calibration.

    Returns
    -------
    (behavioural_genomes, all_fitness)
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.uniform(lower, upper, size=(n_samples, len(lower)))
    scores = np.array([fitness_fn(s) for s in samples], dtype=float)
    return behavioural_set(samples, scores, tol=tol), scores


def identifiability_from_scan(
    behavioural: np.ndarray, best_params: dict[str, float]
) -> pd.DataFrame:
    """Build the identifiability table from a Monte-Carlo behavioural set."""
    rows = []
    for j, spec in enumerate(PARAMETERS):
        vals = behavioural[:, j]
        p05 = float(np.percentile(vals, 5))
        p95 = float(np.percentile(vals, 95))
        occupied = (p95 - p05) / (spec.upper - spec.lower)
        rows.append({
            "Symbol": spec.symbol,
            "Group": GROUPS[spec.group],
            "Best fit": round(best_params[spec.symbol], 4),
            "Behav p05": round(p05, 4),
            "Behav p95": round(p95, 4),
            "Range occupied": round(occupied, 3),
            "N behavioural": int(len(behavioural)),
            "Verdict": "constrained" if occupied <= 0.5 else "POORLY CONSTRAINED",
        })
    return pd.DataFrame(rows).sort_values("Range occupied", ascending=False)
