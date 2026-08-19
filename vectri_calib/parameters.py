"""
The 22 VECTRI parameters calibrated in Dieng et al. (2026), Sci Rep 16:1288.

This module is the single source of truth for:
  * parameter order (the genome layout used by the GA),
  * literature "default" values (the priors, = Table 1 "Default" column),
  * physically-justified search bounds for the GA,
  * units (CORRECTED - see UNIT_NOTES; the published table has errors),
  * provenance references.

Reference key (Dieng et al. 2026 numbering):
  27 Tompkins & Ermert (2013)      Malar J 12:65     - VECTRI base model, source of all defaults
  50 Ermert et al. (2011)          Malar J 10:35     - Liverpool Malaria Model literature review
  51 Parihar et al. (2024)         Int J Biometeorol - VECTRI parameter sensitivity + ranges
  52 Munga et al. (2007)           J Med Entomol     - larval survivorship, western Kenya (field)
  53 Paaijmans et al. (2007)       PLoS ONE 2:e1146  - larval loss to rainfall (field)
  54 Christiansen-Jucht (2014)     Parasit Vectors   - thermal limits (lab)
  55 Detinova (1962)               WHO monograph     - degree-day cycle formulations
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Parameter:
    """One calibratable VECTRI parameter."""

    symbol: str
    definition: str
    default: float          # Table 1 "Default" column - the literature prior
    lower: float            # GA search bound, low
    upper: float            # GA search bound, high
    unit: str               # CORRECTED unit
    published_unit: str     # exactly as printed in Table 1 (may be wrong)
    sources: tuple[int, ...]
    group: str
    integer: bool = False   # round to integer when writing the namelist
    published_bestfit: float | None = None   # Dieng et al. Table 1, for comparison only

    def __post_init__(self) -> None:
        if not self.lower <= self.default <= self.upper:
            raise ValueError(
                f"{self.symbol}: default {self.default} outside bounds "
                f"[{self.lower}, {self.upper}]"
            )
        if self.lower >= self.upper:
            raise ValueError(f"{self.symbol}: lower bound must be < upper bound")


# ---------------------------------------------------------------------------
# Functional groups (see guide Part 3). Grouping matters for interpretation and
# for the compensating-pair diagnostics in report.py.
# ---------------------------------------------------------------------------
GROUPS = {
    "A_fecundity": "Vector fecundity",
    "B_biting": "Biting and transmission probabilities (drives EIR)",
    "C_immunity": "Human immunity and parasite clearance",
    "D_habitat": "Larval habitat and hydrology (WRF-Hydro coupling point)",
    "E_larvsurv": "Larval survival: predation and thermal",
    "F_devtime": "Aquatic development timing",
    "G_gono": "Gonotrophic cycle (degree-day)",
    "H_sporo": "Sporogonic cycle (degree-day)",
}


# ---------------------------------------------------------------------------
# THE 22 PARAMETERS.
#
# BOUNDS RATIONALE - read this before you change anything.
#
# Dieng et al. state the GA samples "within the bounds of their previously
# estimated uncertainty" rather than doing a free parameter search. The paper
# does NOT publish those bounds, so the ranges below are reconstructed to be
# (a) wide enough to contain both the published default and the published
# best fit, and (b) bounded by physical/biological plausibility from refs
# 50-55. They are a defensible starting point, NOT the authors' exact ranges.
# Document any change you make - the bounds are a scientific choice and a
# reviewer will ask about them.
# ---------------------------------------------------------------------------
PARAMETERS: tuple[Parameter, ...] = (
    # --- Group A: fecundity ------------------------------------------------
    Parameter(
        symbol="neggmn",
        definition="Average number of laid eggs per batch that result in female vectors",
        default=80.0, lower=40.0, upper=200.0,
        unit="eggs", published_unit="eggs",
        sources=(27, 50, 51), group="A_fecundity", integer=True,
        published_bestfit=119.0,
    ),
    # --- Group B: biting and transmission ----------------------------------
    Parameter(
        symbol="rbiteratio",
        definition="Biting rate",
        default=0.6, lower=0.10, upper=1.00,
        unit="day^-1", published_unit="days",
        sources=(27, 50), group="B_biting",
        published_bestfit=0.37,
    ),
    Parameter(
        symbol="rpthost2vect_I",
        definition="Probability of transmission from infected host to vector",
        default=0.2, lower=0.02, upper=0.95,
        unit="-", published_unit="-",
        sources=(27, 50), group="B_biting",
        published_bestfit=0.84,
    ),
    Parameter(
        symbol="rpthost2vect_R",
        definition="Probability of transmission from recovered (R) host to vector",
        default=0.04, lower=0.005, upper=0.60,
        unit="-", published_unit="-",
        sources=(27,), group="B_biting",
        published_bestfit=0.37,
    ),
    Parameter(
        symbol="rptvect2host",
        definition="Probability of transmission from vector to host",
        default=0.3, lower=0.05, upper=0.95,
        unit="-", published_unit="-",
        sources=(27, 50), group="B_biting",
        published_bestfit=0.60,
    ),
    Parameter(
        symbol="rbitehighrisk",
        definition="Ratio of rate of bites for high risk to low risk hosts",
        default=5.0, lower=1.0, upper=25.0,
        unit="-", published_unit="-",
        sources=(27,), group="B_biting",
        published_bestfit=18.0,
    ),
    # --- Group C: immunity and clearance -----------------------------------
    Parameter(
        symbol="rhostclear",
        definition="Clearance rate (timescale) for non-immune adults",
        default=30.0, lower=10.0, upper=120.0,
        unit="days", published_unit="-",
        sources=(27,), group="C_immunity",
        published_bestfit=58.0,
    ),
    Parameter(
        symbol="rimmune_gain_eira",
        definition="Annual EIR required to gain full immunity",
        default=100.0, lower=20.0, upper=400.0,
        unit="infectious bites person^-1 year^-1", published_unit="-",
        sources=(27,), group="C_immunity",
        published_bestfit=220.0,
    ),
    Parameter(
        symbol="rimmune_loss_tau",
        definition="e-folding timescale for immunity loss",
        default=365.0, lower=120.0, upper=1000.0,
        unit="days", published_unit="-",
        sources=(27,), group="C_immunity",
        published_bestfit=309.0,
    ),
    # --- Group D: habitat and hydrology ------------------------------------
    Parameter(
        symbol="rrainfall_factor",
        definition="Rainfall scaling factor for breeding-site water",
        default=1.0, lower=0.30, upper=2.00,
        unit="-", published_unit="-",
        sources=(27, 51), group="D_habitat",
        published_bestfit=0.9,
    ),
    Parameter(
        symbol="rbiocapacity",
        definition="Maximum larval biomass per m2 of suitable water body",
        default=300.0, lower=50.0, upper=600.0,
        unit="m^-2", published_unit="-",
        sources=(27,), group="D_habitat",
        published_bestfit=134.0,
    ),
    Parameter(
        symbol="rlarv_flushmin",
        definition="Minimal daily larval survival (L1) rate after intense rainfall",
        default=0.4, lower=0.10, upper=0.99,
        unit="-", published_unit="-",
        sources=(27, 51, 52), group="D_habitat",
        published_bestfit=0.9,
    ),
    Parameter(
        symbol="rlarv_flushtau",
        definition="e-folding factor for larval decay from flushing by rainfall",
        default=20.0, lower=5.0, upper=60.0,
        unit="mm day^-1", published_unit="mm.day^-1",
        sources=(27, 51, 53), group="D_habitat",
        published_bestfit=32.0,
    ),
    # --- Group E: larval survival ------------------------------------------
    Parameter(
        symbol="rlarvsurv",
        definition="Base daily survival rate due to predation events",
        default=0.98, lower=0.80, upper=0.999,
        unit="day^-1", published_unit="-",
        sources=(27,), group="E_larvsurv",
        published_bestfit=0.9,
    ),
    Parameter(
        symbol="rlarv_tmax",
        definition="Maximum temperature for larvae survival",
        default=37.0, lower=32.0, upper=42.0,
        unit="degC", published_unit="degC",
        sources=(27, 51, 54), group="E_larvsurv",
        published_bestfit=36.0,
    ),
    Parameter(
        symbol="rlarv_tmin",
        definition="Minimum temperature for larvae survival",
        default=12.0, lower=8.0, upper=18.0,
        unit="degC", published_unit="degC",
        sources=(27, 51, 54), group="E_larvsurv",
        published_bestfit=15.0,
    ),
    # --- Group F: development timing ---------------------------------------
    Parameter(
        symbol="rlarv_eggtime",
        definition="Time for egg hatching",
        default=1.0, lower=0.5, upper=3.0,
        unit="days", published_unit="days",
        sources=(27, 51, 55), group="F_devtime",
        published_bestfit=0.8,
    ),
    Parameter(
        symbol="rlarv_pupaetime",
        definition="Time for pupal stages",
        default=1.0, lower=0.5, upper=3.0,
        unit="days", published_unit="days",
        sources=(27, 55), group="F_devtime",
        published_bestfit=1.4,
    ),
    # --- Group G: gonotrophic cycle ----------------------------------------
    Parameter(
        symbol="rtgono",
        definition="Threshold temperature for gonotrophic cycle (egg development in vector)",
        default=7.0, lower=5.0, upper=18.0,
        unit="degC", published_unit="degC",
        sources=(27, 51), group="G_gono",
        published_bestfit=13.0,
    ),
    Parameter(
        symbol="dgono",
        definition="Degree-days to complete a full gonotrophic cycle",
        default=37.0, lower=20.0, upper=60.0,
        unit="degC day", published_unit="degC.day^-1",
        sources=(27, 51, 55), group="G_gono",
        published_bestfit=39.0,
    ),
    # --- Group H: sporogonic cycle -----------------------------------------
    Parameter(
        symbol="rtsporo",
        definition="Minimum temperature threshold for sporogonic cycle (parasite development in vector)",
        default=18.0, lower=12.0, upper=22.0,
        unit="degC", published_unit="days",   # <-- published unit is WRONG
        sources=(27, 50, 51), group="H_sporo",
        published_bestfit=15.0,
    ),
    Parameter(
        symbol="dsporo",
        definition="Degree-days to complete a full sporogonic cycle",
        default=111.0, lower=60.0, upper=160.0,
        unit="degC day", published_unit="degC.day^-1",
        sources=(27, 51, 55), group="H_sporo",
        published_bestfit=87.0,
    ),
)

N_PARAMS = len(PARAMETERS)
SYMBOLS = tuple(p.symbol for p in PARAMETERS)
DEFAULTS = tuple(p.default for p in PARAMETERS)
LOWER = tuple(p.lower for p in PARAMETERS)
UPPER = tuple(p.upper for p in PARAMETERS)
BY_SYMBOL = {p.symbol: p for p in PARAMETERS}

# Unit errors found in the published Table 1. Surfaced by report.py so anyone
# reusing the table is warned before copying units into their own methods.
UNIT_NOTES = {
    "rtsporo": "Published as 'days' but it is a temperature threshold. Correct unit: degC.",
    "dgono": "Published as 'degC.day^-1'. Degree-days are a product, not a rate. Correct: degC day.",
    "dsporo": "Published as 'degC.day^-1'. Degree-days are a product, not a rate. Correct: degC day.",
    "rbiteratio": "Published as 'days'. A rate, not a duration. Correct: day^-1.",
    "rimmune_loss_tau": "Published as unitless; 365 -> 309 is plainly a timescale. Correct: days.",
    "rimmune_gain_eira": "Published as unitless. Correct: infectious bites person^-1 year^-1.",
    "rhostclear": "Published as unitless; values 30 -> 58 read as a timescale. Verify against VECTRI source.",
}


def assert_integrity() -> None:
    """Fail loudly if the parameter table has drifted from the paper."""
    if N_PARAMS != 22:
        raise AssertionError(f"Expected 22 parameters (Dieng et al.), found {N_PARAMS}")
    if len(set(SYMBOLS)) != N_PARAMS:
        raise AssertionError("Duplicate parameter symbols")
    for p in PARAMETERS:
        if p.published_bestfit is not None and not (p.lower <= p.published_bestfit <= p.upper):
            raise AssertionError(
                f"{p.symbol}: published best fit {p.published_bestfit} lies outside "
                f"search bounds [{p.lower}, {p.upper}] - the GA could never find it"
            )
        if p.group not in GROUPS:
            raise AssertionError(f"{p.symbol}: unknown group {p.group}")


assert_integrity()
