"""
Real-coded genetic algorithm for VECTRI parameter calibration.

Configured to match Dieng et al. (2026):
    n_ens = 80 ensemble members
    n_gen = 40 generations
    objective = minimise RMSE (simulated vs observed monthly incidence)
    sampling bounded by prior parameter uncertainty (NOT a free search)
  -> 80 x 40 = 3200 model evaluations

Operators: tournament selection, simulated binary crossover (SBX),
polynomial mutation, elitism. All operators respect the per-parameter bounds,
which is what distinguishes this from a free parameter search.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


@dataclass
class GAConfig:
    n_ens: int = 80             # population size  (paper: 80)
    n_gen: int = 40             # generations      (paper: 40)
    tournament_k: int = 3
    p_crossover: float = 0.9
    eta_crossover: float = 15.0  # SBX distribution index
    p_mutation: float | None = None   # default 1/n_params
    eta_mutation: float = 20.0        # polynomial mutation index
    n_elite: int = 2
    seed: int = 20260119

    def evaluations(self) -> int:
        return self.n_ens * self.n_gen


@dataclass
class GAResult:
    best_genome: np.ndarray
    best_fitness: float
    history_best: list[float]
    history_mean: list[float]
    final_population: np.ndarray
    final_fitness: np.ndarray
    config: GAConfig
    n_evaluations: int
    archive_genomes: np.ndarray | None = None   # every genome ever evaluated
    archive_fitness: np.ndarray | None = None

    def to_json(self, path: str | Path) -> None:
        payload = {
            "best_genome": self.best_genome.tolist(),
            "best_fitness": self.best_fitness,
            "history_best": self.history_best,
            "history_mean": self.history_mean,
            "final_population": self.final_population.tolist(),
            "final_fitness": self.final_fitness.tolist(),
            "config": asdict(self.config),
            "n_evaluations": self.n_evaluations,
        }
        Path(path).write_text(json.dumps(payload, indent=2))


def _tournament(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    """Pick the best of k random contenders. Lower fitness is better."""
    contenders = rng.integers(0, len(fitness), size=k)
    return int(contenders[np.argmin(fitness[contenders])])


def _sbx(p1: np.ndarray, p2: np.ndarray, lower: np.ndarray, upper: np.ndarray,
         eta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulated binary crossover, bounds-respecting."""
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() > 0.5:
            continue
        if abs(p1[i] - p2[i]) < 1e-14:
            continue
        x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        lo, hi = lower[i], upper[i]
        rand = rng.random()

        beta = 1.0 + (2.0 * (x1 - lo) / (x2 - x1))
        alpha = 2.0 - beta ** -(eta + 1.0)
        betaq = ((rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha
                 else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0)))
        cc1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + (2.0 * (hi - x2) / (x2 - x1))
        alpha = 2.0 - beta ** -(eta + 1.0)
        betaq = ((rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha
                 else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0)))
        cc2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        c1[i] = float(np.clip(cc1, lo, hi))
        c2[i] = float(np.clip(cc2, lo, hi))
    return c1, c2


def _polynomial_mutation(x: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                         p_mut: float, eta: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Polynomial mutation, bounds-respecting."""
    y = x.copy()
    for i in range(len(y)):
        if rng.random() > p_mut:
            continue
        lo, hi = lower[i], upper[i]
        if hi - lo < 1e-14:
            continue
        delta1 = (y[i] - lo) / (hi - lo)
        delta2 = (hi - y[i]) / (hi - lo)
        rand = rng.random()
        mut_pow = 1.0 / (eta + 1.0)
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1.0)
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1.0)
            deltaq = 1.0 - val ** mut_pow
        y[i] = float(np.clip(y[i] + deltaq * (hi - lo), lo, hi))
    return y


def run_ga(
    fitness_fn: Callable[[np.ndarray], float],
    lower: Sequence[float],
    upper: Sequence[float],
    config: GAConfig | None = None,
    seed_genomes: Sequence[Sequence[float]] | None = None,
    verbose: bool = True,
    progress_every: int = 1,
) -> GAResult:
    """
    Minimise `fitness_fn` over the bounded parameter space.

    seed_genomes: optional starting individuals injected into generation 0.
    Seed the literature defaults here so the GA can never do worse than the
    prior - this is cheap insurance and makes the run reproducible.
    """
    cfg = config or GAConfig()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n_params = len(lower)
    if len(upper) != n_params:
        raise ValueError("lower and upper must have equal length")
    if np.any(lower >= upper):
        bad = np.where(lower >= upper)[0]
        raise ValueError(f"invalid bounds at indices {bad.tolist()}")

    p_mut = cfg.p_mutation if cfg.p_mutation is not None else 1.0 / n_params
    rng = np.random.default_rng(cfg.seed)

    # --- generation 0: Latin-hypercube-ish uniform sampling within bounds --
    pop = rng.uniform(lower, upper, size=(cfg.n_ens, n_params))
    if seed_genomes:
        for j, g in enumerate(seed_genomes):
            if j >= cfg.n_ens:
                break
            pop[j] = np.clip(np.asarray(g, dtype=float), lower, upper)

    fitness = np.array([fitness_fn(ind) for ind in pop], dtype=float)
    n_eval = cfg.n_ens
    archive_g: list[np.ndarray] = [ind.copy() for ind in pop]
    archive_f: list[float] = list(fitness)

    history_best: list[float] = [float(np.nanmin(fitness))]
    history_mean: list[float] = [float(np.nanmean(fitness[np.isfinite(fitness)]))]
    if verbose:
        print(f"  gen  0/{cfg.n_gen}  best={history_best[-1]:12.4f}  "
              f"mean={history_mean[-1]:12.4f}")

    for gen in range(1, cfg.n_gen + 1):
        order = np.argsort(fitness)
        new_pop = [pop[i].copy() for i in order[: cfg.n_elite]]   # elitism

        while len(new_pop) < cfg.n_ens:
            i1 = _tournament(fitness, cfg.tournament_k, rng)
            i2 = _tournament(fitness, cfg.tournament_k, rng)
            c1, c2 = pop[i1].copy(), pop[i2].copy()
            if rng.random() < cfg.p_crossover:
                c1, c2 = _sbx(c1, c2, lower, upper, cfg.eta_crossover, rng)
            c1 = _polynomial_mutation(c1, lower, upper, p_mut, cfg.eta_mutation, rng)
            c2 = _polynomial_mutation(c2, lower, upper, p_mut, cfg.eta_mutation, rng)
            new_pop.append(c1)
            if len(new_pop) < cfg.n_ens:
                new_pop.append(c2)

        pop = np.array(new_pop[: cfg.n_ens])
        # elites keep their score; only the children need evaluating
        child_fitness = np.array([fitness_fn(ind) for ind in pop[cfg.n_elite:]],
                                 dtype=float)
        fitness = np.concatenate([fitness[order[: cfg.n_elite]], child_fitness])
        n_eval += cfg.n_ens - cfg.n_elite
        archive_g.extend(ind.copy() for ind in pop[cfg.n_elite:])
        archive_f.extend(child_fitness)

        history_best.append(float(np.nanmin(fitness)))
        finite = fitness[np.isfinite(fitness)]
        history_mean.append(float(np.nanmean(finite)) if finite.size else float("inf"))
        if verbose and (gen % progress_every == 0 or gen == cfg.n_gen):
            print(f"  gen {gen:2d}/{cfg.n_gen}  best={history_best[-1]:12.4f}  "
                  f"mean={history_mean[-1]:12.4f}")

    best_idx = int(np.nanargmin(fitness))
    return GAResult(
        best_genome=pop[best_idx].copy(),
        best_fitness=float(fitness[best_idx]),
        history_best=history_best,
        history_mean=history_mean,
        final_population=pop.copy(),
        final_fitness=fitness.copy(),
        config=cfg,
        n_evaluations=n_eval,
        archive_genomes=np.array(archive_g),
        archive_fitness=np.array(archive_f, dtype=float),
    )
