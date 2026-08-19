#!/usr/bin/env python3
"""
Rebuild Table 1 of Dieng et al. (2026) end to end.

    # 1. Dry run on synthetic data (works today, no restricted data needed)
    python scripts/run_calibration.py --synthetic --quick

    # 2. Full paper protocol (80 x 40 = 3200 evaluations) on synthetic data
    python scripts/run_calibration.py --synthetic

    # 3. Real run once you have WRF/WRF-Hydro forcing + KEMRI-CDC observations
    python scripts/run_calibration.py \
        --forcing data/forcing.csv \
        --obs data/observations.csv \
        --holdout 2019 2020 2021 2022

    # 4. Real VECTRI binary instead of the reduced surrogate
    python scripts/run_calibration.py --forcing ... --obs ... \
        --vectri-exe /opt/vectri/bin/vectri \
        --vectri-namelist config/namelist.vectri.template
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectri_calib.calibrate import calibrate, itn_experiment      # noqa: E402
from vectri_calib.data import (load_forcing, load_observations,   # noqa: E402
                               synthetic_siaya)
from vectri_calib.ga import GAConfig                              # noqa: E402
from vectri_calib.model import ReducedVectri, VectriRunner        # noqa: E402
from vectri_calib.calibrate import make_fitness                    # noqa: E402
from vectri_calib.parameters import LOWER, UPPER                   # noqa: E402
from vectri_calib.report import (build_table1, compensating_pairs,  # noqa: E402
                                 convergence_frame,
                                 identifiability_from_scan,
                                 identifiability_report,
                                 monte_carlo_behavioural, to_markdown)
from vectri_calib.sensitivity import (oat_sensitivity,            # noqa: E402
                                      temperature_dominance_check)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("data")
    src.add_argument("--synthetic", action="store_true",
                     help="use the synthetic Siaya testbed instead of real files")
    src.add_argument("--forcing", type=Path, help="daily forcing CSV")
    src.add_argument("--obs", type=Path, help="monthly observations CSV")

    mdl = ap.add_argument_group("model")
    mdl.add_argument("--vectri-exe", type=Path, help="path to the real VECTRI binary")
    mdl.add_argument("--vectri-namelist", type=Path, help="namelist template with @symbol@ tokens")
    mdl.add_argument("--population", type=float, default=60_000.0)
    mdl.add_argument("--area-km2", type=float, default=220.0)

    ga = ap.add_argument_group("genetic algorithm")
    ga.add_argument("--n-ens", type=int, default=80, help="ensemble members (paper: 80)")
    ga.add_argument("--n-gen", type=int, default=40, help="generations (paper: 40)")
    ga.add_argument("--seed", type=int, default=20260119)
    ga.add_argument("--quick", action="store_true",
                    help="tiny GA (12 x 6) for a fast smoke test")

    ev = ap.add_argument_group("evaluation")
    ev.add_argument("--holdout", type=int, nargs="*", default=[],
                    help="years withheld from calibration entirely")
    ev.add_argument("--folds", type=int, default=4)
    ev.add_argument("--skip-sensitivity", action="store_true")
    ev.add_argument("--identifiability", type=int, default=0, metavar="N",
                    help="Monte-Carlo identifiability scan with N independent "
                         "samples (extra model runs; 0 = skip). A GA cannot "
                         "measure this itself - see report.monte_carlo_behavioural")

    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    if args.synthetic:
        print("[data] synthetic Siaya testbed (pseudo-obs from published best-fit values)")
        forcing, obs = synthetic_siaya(seed=args.seed)
    else:
        if not args.forcing or not args.obs:
            print("ERROR: supply --forcing and --obs, or use --synthetic", file=sys.stderr)
            return 2
        print(f"[data] forcing      : {args.forcing}")
        print(f"[data] observations : {args.obs}")
        forcing = load_forcing(args.forcing)
        obs = load_observations(args.obs)

    print(f"[data] {len(forcing)} days, {len(obs)} observed months "
          f"({obs.index.min()} -> {obs.index.max()})")

    # --------------------------------------------------------------- model
    if args.vectri_exe:
        if not args.vectri_namelist:
            print("ERROR: --vectri-exe requires --vectri-namelist", file=sys.stderr)
            return 2
        print(f"[model] real VECTRI: {args.vectri_exe}")
        model = VectriRunner(args.vectri_exe, args.vectri_namelist)
    else:
        print("[model] ReducedVectri surrogate "
              "(NOT publishable as VECTRI results - see model.py)")
        model = ReducedVectri(population=args.population,
                              cell_area_m2=args.area_km2 * 1e6)

    # ------------------------------------------------------------------ GA
    cfg = (GAConfig(n_ens=12, n_gen=6, seed=args.seed) if args.quick
           else GAConfig(n_ens=args.n_ens, n_gen=args.n_gen, seed=args.seed))
    print(f"\n[ga] {cfg.n_ens} members x {cfg.n_gen} generations "
          f"= {cfg.evaluations()} evaluations")

    result = calibrate(
        model, forcing, obs,
        config=cfg,
        holdout_years=tuple(args.holdout),
        n_folds=args.folds,
        verbose=True,
    )

    print(f"\n[fit] train RMSE {result.train_metrics['rmse']:.2f} | "
          f"MAE {result.train_metrics['mae']:.2f} | "
          f"bias {result.train_metrics['bias']:+.2f} | "
          f"r {result.train_metrics['pearson_r']:.3f}")
    if result.test_metrics:
        print(f"[fit] HOLDOUT RMSE {result.test_metrics['rmse']:.2f} | "
              f"MAE {result.test_metrics['mae']:.2f} | "
              f"r {result.test_metrics['pearson_r']:.3f}")
        ratio = result.test_metrics["rmse"] / max(result.train_metrics["rmse"], 1e-9)
        verdict = "OK" if ratio < 1.5 else "OVERFITTING - holdout much worse than train"
        print(f"[fit] holdout/train RMSE ratio {ratio:.2f}  -> {verdict}")

    # -------------------------------------------------------------- Table 1
    table = build_table1(
        result.best_params,
        ensemble=result.ga.archive_genomes,
        ensemble_fitness=result.ga.archive_fitness,
    )
    table.to_csv(args.outdir / "table1_rebuilt.csv", index=False)
    to_markdown(table, args.outdir / "table1_rebuilt.md",
                title="Table 1 (rebuilt) - VECTRI parameters for Siaya")
    print(f"\n[out] {args.outdir / 'table1_rebuilt.csv'}")
    print(f"[out] {args.outdir / 'table1_rebuilt.md'}")

    if args.identifiability > 0:
        print(f"\n[ident] Monte-Carlo scan, {args.identifiability} independent samples")
        all_years = tuple(sorted(set(int(y) for y in obs.index.year)))
        train_years = tuple(y for y in all_years if y not in set(args.holdout))
        scan_fn = make_fitness(model, forcing, obs, train_years)
        behav, _ = monte_carlo_behavioural(
            scan_fn, LOWER, UPPER, n_samples=args.identifiability, seed=args.seed)
        ident = identifiability_from_scan(behav, result.best_params)
    else:
        ident = identifiability_report(table)
        print("\n[ident] from GA archive - UNDERSTATES uncertainty. "
              "Use --identifiability N for a real scan.")
    ident.to_csv(args.outdir / "identifiability.csv", index=False)
    n_poor = int((ident["Verdict"] == "POORLY CONSTRAINED").sum())
    print(f"[out] identifiability.csv  ({n_poor}/{len(ident)} poorly constrained)")

    pairs = compensating_pairs(result.ga.archive_genomes, result.ga.archive_fitness)
    if not pairs.empty:
        pairs.to_csv(args.outdir / "compensating_pairs.csv", index=False)
        print(f"[out] compensating_pairs.csv ({len(pairs)} pairs |r| >= 0.5)")

    convergence_frame(result.ga.history_best, result.ga.history_mean) \
        .to_csv(args.outdir / "ga_convergence.csv", index=False)
    result.ga.to_json(args.outdir / "ga_result.json")

    pd.DataFrame(result.fold_metrics).to_csv(args.outdir / "cv_folds.csv", index=False)

    # --------------------------------------------------- ITN experiment
    print("\n[itn] with vs without ITN coverage")
    itn = itn_experiment(model, forcing, result.best_params, obs=obs)
    for k, v in itn.items():
        print(f"       {k:34s} {v:12.2f}" if isinstance(v, float) else f"       {k}: {v}")
    (args.outdir / "itn_experiment.json").write_text(json.dumps(itn, indent=2))

    print("\n       paper targets: ~58% EIR reduction, ~41% incidence error "
          "reduction, +/-100-150 cases/month")

    # ------------------------------------------------------- sensitivity
    if not args.skip_sensitivity:
        print("\n[sens] one-at-a-time sensitivity around the optimum")
        sens = oat_sensitivity(model, forcing, result.best_params, verbose=False)
        sens.to_csv(args.outdir / "sensitivity_oat.csv", index=False)
        print(sens[["rank", "symbol", "group", "pct_change"]].head(8).to_string(index=False))

        check = temperature_dominance_check(sens)
        print(f"\n[sens] temperature-related in top {check['top_n']}: "
              f"{check['n_temperature_related_in_top']}  -> "
              f"{'MATCHES paper claim' if check['matches_paper_claim'] else 'DOES NOT match paper claim'}")
        (args.outdir / "sensitivity_check.json").write_text(json.dumps(check, indent=2, default=str))

    print(f"\nDone. Artefacts in {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
