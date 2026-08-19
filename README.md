# Rebuilding Table 1 of Dieng et al. (2026)

A complete, runnable harness for reproducing the genetic-algorithm calibration
that produced **Table 1** of:

> Dieng, M.D.B. *et al.* (2026). *High resolution physically based modelling
> reveals malaria incidence reduction by vector control measures.*
> **Scientific Reports 16:1288.** https://doi.org/10.1038/s41598-025-33539-w

---

## What Table 1 actually is

**It is not a dataset you download. It is a model configuration table.**

| Column | Type | Direction |
|---|---|---|
| `Default` | Literature prior from entomology studies (refs 27, 50–55) | **INPUT** |
| `Best fit` | Genetic-algorithm result, calibrated against KEMRI-CDC data | **OUTPUT** |

The `Best fit` column is itself a **research finding** of the paper. Cite the
`Default` column as literature, the `Best fit` column as the paper's result.

---

## The 10-step protocol

| # | Step | Where |
|---|---|---|
| 1 | Install VECTRI, confirm defaults match the shipped namelist | `vectri_calib/parameters.py` |
| 2 | Add the ITN compartment (Eqs. 1–5) | `vectri_calib/model.py` |
| 3 | Build WRF (1 km) + WRF-Hydro (50 m) forcing; validate to MAPE ≈ 15% | `vectri_calib/data.py` |
| 4 | Obtain the KEMRI-CDC calibration target (**restricted**) | see *Data access* |
| 5 | Define uncertainty bounds for all 22 parameters | `parameters.py` (`lower`/`upper`) |
| 6 | Run the GA: 80 members × 40 generations = 3,200 evaluations | `vectri_calib/ga.py` |
| 7 | Hold out years **before** starting; blocked-year cross-validation | `vectri_calib/objective.py` |
| 8 | Run sensitivity analysis; expect temperature parameters to dominate | `vectri_calib/sensitivity.py` |
| 9 | Validate the with/without-ITN contrast | `calibrate.itn_experiment` |
| 10 | Report the whole vector + identifiability + equifinality | `vectri_calib/report.py` |

---

## Quick start

```bash
pip install numpy pandas pyyaml pytest

# 60-second smoke test - no restricted data needed
python scripts/run_calibration.py --synthetic --quick

# Full paper protocol (3200 evaluations) on the synthetic testbed
python scripts/run_calibration.py --synthetic --holdout 2019 2020 2021 2022

# Real run, once you have forcing + observations
python scripts/run_calibration.py \
    --forcing data/forcing.csv --obs data/observations.csv \
    --holdout 2019 2020 2021 2022

# Real VECTRI binary instead of the reduced surrogate
python scripts/run_calibration.py --forcing ... --obs ... \
    --vectri-exe /opt/vectri/bin/vectri \
    --vectri-namelist config/namelist.vectri.template

pytest tests/ -q
```

---

## Two models, and which to use

| | `ReducedVectri` | `VectriRunner` |
|---|---|---|
| What | ~200-line reimplementation of the VECTRI core | Adapter for the real ICTP binary |
| Needs | nothing | VECTRI source + namelist template |
| Use for | building/debugging the pipeline, teaching | **publication** |

**`ReducedVectri` output is NOT publishable as VECTRI results.** It exists so
the GA, cross-validation, sensitivity and reporting machinery can be verified
end to end before you have the restricted inputs. It reproduces the paper's
*EIR* behaviour (11.3 → 4.9, a 57% reduction vs. their 58%) but **not** the
41% incidence reduction — that requires real VECTRI and real data. Tuning the
surrogate to match that number would be fitting the toy to the answer.

---

## Input data contract

`forcing.csv` — daily:

| column | units | source in the paper |
|---|---|---|
| `date` | YYYY-MM-DD | — |
| `tas` | °C | WRF d02, 1 km |
| `pr` | mm/day | WRF d02, 1 km |
| `waterfrac` | 0–1 | WRF-Hydro, 50 m routing |
| `itn_cover` | 0–1 | KEMRI-CDC, monthly, forward-filled |

`observations.csv` — monthly: `month` (YYYY-MM), `cases`.

`validate_forcing()` rejects the mistakes that silently ruin a calibration:
Kelvin instead of °C, percent instead of fraction, missing days, negative
rainfall, out-of-range water fraction.

---

## Data access

| Dataset | Access |
|---|---|
| CHIRPS | https://www.chc.ucsb.edu/data/chirps |
| CHIRTS | https://data.chc.ucsb.edu/products/CHIRTSdaily/v1.0/ |
| ERA5 | https://cds.climate.copernicus.eu/ |
| Sentinel-1 | https://dataspace.copernicus.eu/ |
| ESA CCI soil moisture | https://esa-soilmoisture-cci.org/ |
| GBD / GHDx | https://ghdx.healthdata.org/gbd-2021 |
| **KEMRI-CDC PBIDS cases + ITN** | **RESTRICTED** — Ethics Committee, Dr Stephen Munga (Smunga@kemri.org) |
| VECTRI | https://wiki.ictp.it/e-lib/vectri (contact A.M. Tompkins, ICTP) |

Step 4 is the only hard blocker. Without it you can run the model but cannot
produce a `Best fit` column.

---

## Bounds are a scientific choice

The paper states the GA samples *"within the bounds of their previously
estimated uncertainty"* rather than doing a free search — but **it does not
publish those bounds.** The ranges in `parameters.py` are reconstructed to
contain both the published default and the published best fit while staying
biologically plausible per refs 50–55. **They are not the authors' exact
ranges.** Document any change you make; a reviewer will ask.

---

## What this harness adds over the published table

The original reports point estimates only. This adds:

- **`Elite p05` / `p95`** — spread across the best-performing 25% of the final
  GA population.
- **`Constrained`** — how much of the search range good solutions still span.
  Near 0 means the data does **not** constrain that parameter and its "best
  fit" is close to arbitrary. See `identifiability.csv`.
- **`compensating_pairs.csv`** — equifinality detection. The published table
  shows clear trade-offs (`rbiteratio` ↓ 0.6→0.37 while `rpthost2vect_I`
  ↑ 0.2→0.84 and `rbitehighrisk` ↑ 5→18; `rbiocapacity` ↓ 300→134 while
  `neggmn` ↑ 80→119 and `rlarv_flushmin` ↑ 0.4→0.9). Strongly correlated
  pairs are **only meaningful together** — never quote one alone.

---

## Unit errors in the published Table 1

Corrected in `parameters.py` (`UNIT_NOTES`) and surfaced in the markdown report:

| Symbol | Published | Correct |
|---|---|---|
| `rtsporo` | days | **°C** (it is a temperature threshold) |
| `dgono`, `dsporo` | °C·day⁻¹ | **°C·day** (degree-days are a product) |
| `rbiteratio` | days | **day⁻¹** |
| `rimmune_loss_tau` | – | **days** |
| `rimmune_gain_eira` | – | **infectious bites person⁻¹ year⁻¹** |
| `rhostclear` | – | **days** (verify against VECTRI source) |

Do not copy units from the published table into your own methods section.

---

## Validation targets

| Quantity | Paper |
|---|---|
| Mean monthly deviation | ±100–150 cases |
| EIR reduction from ITN | ~58% (10–20 → ~5 bites/person/year) |
| Incidence error reduction | ~41% |
| WRF rainfall/temperature MAPE | ~15% |
| Dominant sensitivity | Temperature parameters (groups G, H, thermal E) |

---

## Layout

```
vectri_calib/
  parameters.py   22 parameters: defaults, bounds, corrected units, sources
  model.py        VectriRunner (real) + ReducedVectri (surrogate)
  data.py         loaders, validation, synthetic Siaya testbed
  ga.py           real-coded GA: tournament, SBX, polynomial mutation, elitism
  objective.py    RMSE/MAE/bias/r + blocked-year CV folds
  calibrate.py    driver + ITN experiment
  sensitivity.py  OAT + Morris screening + temperature-dominance check
  report.py       Table 1 builder, identifiability, equifinality
scripts/run_calibration.py
tests/test_pipeline.py
```

## Caveats

- Best-fit values are valid **as a set**. Do not transplant single rows.
- Check `identifiability.csv` before quoting any individual value.
- The paper reports no confidence intervals for these parameters. For real
  uncertainty quantification use MCMC/ABC rather than a single GA point estimate.
