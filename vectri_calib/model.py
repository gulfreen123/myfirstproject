"""
Model interface for the calibration harness.

Two implementations:

  VectriRunner   - adapter for the REAL VECTRI executable (ICTP). Writes a
                   namelist, runs the binary, parses monthly cases. Use this
                   for publication-grade work.

  ReducedVectri  - a self-contained, dependency-light reimplementation of the
                   VECTRI transmission core in ~150 lines. It consumes all 22
                   calibrated parameters and produces monthly cases, so the
                   entire GA / cross-validation / sensitivity pipeline can be
                   developed, tested and debugged TODAY, before you have the
                   ICTP source or the KEMRI-CDC data.

IMPORTANT: ReducedVectri is a faithful-in-structure teaching surrogate, not
VECTRI. Numbers it produces are NOT publishable as VECTRI results. Its purpose
is to make the calibration machinery verifiable end to end.
"""
from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from .parameters import BY_SYMBOL, SYMBOLS


class MalariaModel(ABC):
    """Anything the GA can calibrate."""

    @abstractmethod
    def run(self, params: dict[str, float], forcing: pd.DataFrame) -> pd.Series:
        """Return monthly simulated cases indexed by period (PeriodIndex, 'M')."""


# ===========================================================================
# Adapter for the real VECTRI binary
# ===========================================================================
class VectriRunner(MalariaModel):
    """
    Drives the real VECTRI executable.

    You MUST adapt `_write_namelist` and `_read_output` to your VECTRI build -
    namelist keys and output layout vary by version. The symbols in
    parameters.py follow the naming in Dieng et al. Table 1.
    """

    def __init__(
        self,
        executable: str | Path,
        template_namelist: str | Path,
        workdir: str | Path | None = None,
        timeout_s: int = 3600,
    ) -> None:
        self.executable = Path(executable)
        self.template_namelist = Path(template_namelist)
        self.workdir = Path(workdir) if workdir else None
        self.timeout_s = timeout_s
        if not self.executable.exists():
            raise FileNotFoundError(
                f"VECTRI executable not found at {self.executable}. "
                "Obtain VECTRI from ICTP (https://wiki.ictp.it/e-lib/vectri) "
                "or use ReducedVectri to develop the pipeline first."
            )

    def _write_namelist(self, params: dict[str, float], dest: Path) -> None:
        text = self.template_namelist.read_text()
        for symbol, value in params.items():
            spec = BY_SYMBOL[symbol]
            rendered = str(int(round(value))) if spec.integer else f"{value:.6g}"
            token = f"@{symbol}@"
            if token not in text:
                raise KeyError(
                    f"Placeholder {token} missing from {self.template_namelist}. "
                    "Every calibrated parameter needs a placeholder in the template."
                )
            text = text.replace(token, rendered)
        dest.write_text(text)

    def _read_output(self, rundir: Path) -> pd.Series:
        out = rundir / "vectri_output.csv"
        if not out.exists():
            raise FileNotFoundError(f"No VECTRI output at {out}")
        df = pd.read_csv(out, parse_dates=["date"])
        s = df.set_index("date")["cases"]
        return s.groupby(s.index.to_period("M")).sum()

    def run(self, params: dict[str, float], forcing: pd.DataFrame) -> pd.Series:
        ctx = tempfile.TemporaryDirectory(dir=self.workdir)
        with ctx as tmp:
            rundir = Path(tmp)
            self._write_namelist(params, rundir / "namelist.vectri")
            forcing.to_csv(rundir / "forcing.csv", index=True)
            proc = subprocess.run(
                [str(self.executable)], cwd=rundir,
                capture_output=True, text=True, timeout=self.timeout_s,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"VECTRI failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
            return self._read_output(rundir)


# ===========================================================================
# Reduced VECTRI - runnable surrogate
# ===========================================================================
class ReducedVectri(MalariaModel):
    """
    Daily-timestep, district-lumped reimplementation of the VECTRI core.

    Structure mirrors VECTRI (Tompkins & Ermert 2013):
      - larval pool with density dependence against a hydrology-driven
        carrying capacity,
      - rainfall flushing of early instars,
      - degree-day gonotrophic cycle controlling egg laying,
      - degree-day sporogonic cycle controlling vector infectiousness,
      - SEIR human hosts with acquired immunity driven by cumulative EIR,
      - ITN reduction factor applied to the effective biting rate
        (Dieng et al. Eq. 5).

    All 22 calibrated parameters influence the output.

    Use `simulate()` for full daily diagnostics (including EIR, which you need
    to check the paper's ~58% EIR-reduction target); `run()` returns just the
    monthly case series the GA calibrates against.
    """

    def __init__(
        self,
        population: float = 60_000.0,
        cell_area_m2: float = 220e6,     # Siaya HDSS ~220 km2
        rbite_night: float = 0.45,       # EFFECTIVE ITN protection (see note)
        spinup_days: int = 365,
        immunity_efficacy: float = 0.65,
        adult_survival: float = 0.90,
        max_vectors_per_host: float = 20.0,
        habitat_fraction: float = 5e-4,
        clinical_fraction: float = 0.06,
    ) -> None:
        self.population = float(population)
        self.cell_area_m2 = float(cell_area_m2)
        # NOT simply the night-biting fraction of An. gambiae (~0.85). This is
        # the EFFECTIVE protective factor entering Eq. 5, which also absorbs
        # net integrity, actual usage compliance and insecticide resistance.
        # The paper stresses that "high ownership does not necessarily equate
        # to high utilization". Tuned so simulated EIR reduction (~57%) matches
        # the paper's reported ~58%.
        self.rbite_night = float(rbite_night)
        self.spinup_days = int(spinup_days)
        # Maximum fraction of infections acquired immunity can render
        # subclinical. MUST be < 1, and immunity is applied to SUSCEPTIBILITY
        # (not only to the clinical fraction) so that more transmission always
        # produces more cases. Applying it to the clinical fraction alone lets
        # high-transmission (no-ITN) runs saturate immunity and paradoxically
        # report fewer cases, which inverts the entire ITN experiment.
        self.immunity_efficacy = float(np.clip(immunity_efficacy, 0.0, 0.95))
        self.adult_survival = float(np.clip(adult_survival, 0.01, 0.999))
        # Hard ceiling on the adult vector population, expressed per host.
        # Without it, egg-laying feeds back on itself and the population can
        # overflow float64 for aggressive parameter vectors - which the GA
        # will happily find.
        self.max_vectors_per_host = float(max_vectors_per_host)
        # Fraction of the WRF-Hydro grid-cell surface water that is actually
        # SUITABLE Anopheles larval habitat (small, shallow, sun-exposed,
        # slow-draining pools) rather than open water. The paper itself notes
        # that WRF-Hydro's grid-averaged wetness is "spatially smoother and
        # more diffuse" than real breeding sites. Without this conversion the
        # model treats every wet square metre as habitat and produces annual
        # EIR in the thousands instead of the observed 10-20.
        self.habitat_fraction = float(habitat_fraction)
        # Fraction of infections that become REPORTED clinical cases. PBIDS
        # captures care-seeking clinical malaria, not all infections; in a
        # high-transmission setting most infections are asymptomatic.
        self.clinical_fraction = float(np.clip(clinical_fraction, 1e-4, 1.0))

    # -- core ------------------------------------------------------------
    def simulate(self, params: dict[str, float], forcing: pd.DataFrame) -> pd.DataFrame:
        """Run the model and return DAILY diagnostics."""
        p = params
        required = {"tas", "pr", "waterfrac", "itn_cover"}
        missing = required - set(forcing.columns)
        if missing:
            raise KeyError(f"forcing is missing columns: {sorted(missing)}")

        tas = forcing["tas"].to_numpy(float)
        pr = forcing["pr"].to_numpy(float)
        wfrac = forcing["waterfrac"].to_numpy(float)
        itn = np.clip(forcing["itn_cover"].to_numpy(float), 0.0, 1.0)
        n = len(tas)

        # --- state -------------------------------------------------------
        eggs = larvae = pupae = 0.0
        adults_s, adults_e, adults_i = 500.0, 0.0, 0.0
        dd_gono = dd_sporo = 0.0
        h_s = self.population * 0.6
        h_e = 0.0
        h_i = self.population * 0.1
        h_r = self.population * 0.3
        immunity = 0.3
        eir_accum = 0.0

        out_cases = np.zeros(n)
        out_eir = np.zeros(n)
        out_immunity = np.zeros(n)
        out_adults_i = np.zeros(n)
        out_larvae = np.zeros(n)

        egg_rate = 1.0 / max(p["rlarv_eggtime"], 1e-6)
        pupae_rate = 1.0 / max(p["rlarv_pupaetime"], 1e-6)
        clear_rate = min(1.0 / max(p["rhostclear"], 1e-6), 1.0)
        imm_loss_rate = min(1.0 / max(p["rimmune_loss_tau"], 1e-6), 1.0)
        vector_ceiling = self.max_vectors_per_host * self.population
        egg_ceiling = vector_ceiling * max(p["neggmn"], 1.0)

        for t in range(n):
            T, rain, wf, cover = tas[t], pr[t], wfrac[t], itn[t]

            # --- D: habitat (WRF-Hydro coupling point) ------------------
            water_area = (self.cell_area_m2 * max(wf, 0.0)
                          * p["rrainfall_factor"] * self.habitat_fraction)
            carrying_cap = p["rbiocapacity"] * water_area

            # --- G: gonotrophic cycle -> egg laying ---------------------
            dd_gono += max(T - p["rtgono"], 0.0)
            if dd_gono >= p["dgono"]:
                dd_gono -= p["dgono"]
                gravid = adults_s + adults_e + adults_i
                eggs = min(eggs + p["neggmn"] * gravid, egg_ceiling)

            # --- F: hatching -------------------------------------------
            hatched = eggs * egg_rate
            eggs = max(eggs - hatched, 0.0)
            larvae += hatched

            # --- E: thermal window + predation --------------------------
            if T <= p["rlarv_tmin"] or T >= p["rlarv_tmax"]:
                thermal = 0.0
            else:
                span = max(p["rlarv_tmax"] - p["rlarv_tmin"], 1e-6)
                thermal = float(np.sin(np.pi * (T - p["rlarv_tmin"]) / span))

            # --- D: rainfall flushing (Paaijmans et al. 2007) -----------
            flush = p["rlarv_flushmin"] + (1.0 - p["rlarv_flushmin"]) * float(
                np.exp(-rain / max(p["rlarv_flushtau"], 1e-6))
            )

            # density dependence against carrying capacity
            density = (max(0.0, 1.0 - larvae / carrying_cap)
                       if carrying_cap > 0.0 else 0.0)
            larvae = min(larvae * p["rlarvsurv"] * thermal * flush * density,
                         carrying_cap if carrying_cap > 0 else 0.0)

            # --- F: pupation -> emergence -------------------------------
            to_pupae = larvae * pupae_rate * 0.5
            larvae = max(larvae - to_pupae, 0.0)
            pupae += to_pupae
            emerged = pupae * pupae_rate
            pupae = max(pupae - emerged, 0.0)
            adults_s += emerged

            # --- adult mortality + hard ceiling -------------------------
            adults_s *= self.adult_survival
            adults_e *= self.adult_survival
            adults_i *= self.adult_survival
            total_adults = adults_s + adults_e + adults_i
            if total_adults > vector_ceiling:
                scale = vector_ceiling / total_adults
                adults_s *= scale
                adults_e *= scale
                adults_i *= scale

            # --- B: biting, ITN reduction (Dieng et al. Eq. 5) ----------
            itn_factor = float(np.clip(1.0 - self.rbite_night * cover, 0.0, 1.0))
            bite_rate = p["rbiteratio"] * itn_factor

            host_total = max(h_s + h_e + h_i + h_r, 1.0)
            # Heterogeneous biting: a high-risk minority absorbs a
            # disproportionate share of bites, raising the effective chance
            # that any given bite lands on an infectious host.
            hr_share = p["rbitehighrisk"] / (p["rbitehighrisk"] + 1.0)
            inf_frac = h_i / host_total
            rec_frac = h_r / host_total
            eff_inf = min(1.0, inf_frac * (1.0 + hr_share))

            # --- host -> vector ----------------------------------------
            infect_prob = min(
                1.0, eff_inf * p["rpthost2vect_I"] + rec_frac * p["rpthost2vect_R"]
            )
            newly_exposed = min(adults_s * min(bite_rate, 1.0) * infect_prob, adults_s)
            adults_s -= newly_exposed
            adults_e += newly_exposed

            # --- H: sporogonic cycle ------------------------------------
            if T > p["rtsporo"]:
                dd_sporo += T - p["rtsporo"]
            if dd_sporo >= p["dsporo"]:
                dd_sporo -= p["dsporo"]
                adults_i += adults_e
                adults_e = 0.0

            # --- vector -> host, EIR -------------------------------------
            eir_daily = adults_i * bite_rate / host_total
            eir_accum = eir_accum * (1.0 - 1.0 / 365.0) + eir_daily
            annual_eir = eir_accum

            # --- C: immunity ---------------------------------------------
            imm_gain = annual_eir / max(p["rimmune_gain_eira"], 1e-6)
            immunity += (imm_gain * (1.0 - immunity)) / 365.0
            immunity -= immunity * imm_loss_rate
            immunity = float(np.clip(immunity, 0.0, 1.0))

            # Immunity reduces SUSCEPTIBILITY, so transmission and cases move
            # in the same direction and the ITN experiment stays monotonic.
            protection = 1.0 - self.immunity_efficacy * immunity
            infections = min(
                eir_daily * p["rptvect2host"] * h_s * protection, h_s
            )

            h_s -= infections
            h_e += infections
            progress = h_e * 0.1
            h_e -= progress
            h_i += progress
            recovered = h_i * clear_rate
            h_i -= recovered
            h_r += recovered
            waned = h_r * imm_loss_rate
            h_r -= waned
            h_s += waned

            out_cases[t] = max(infections, 0.0) * self.clinical_fraction
            out_eir[t] = eir_daily
            out_immunity[t] = immunity
            out_adults_i[t] = adults_i
            out_larvae[t] = larvae

        df = pd.DataFrame(
            {
                "cases": out_cases,
                "eir_daily": out_eir,
                "immunity": out_immunity,
                "adults_infectious": out_adults_i,
                "larvae": out_larvae,
            },
            index=forcing.index,
        )
        if self.spinup_days > 0:
            df = df.iloc[self.spinup_days:]
        if not np.isfinite(df["cases"].to_numpy()).all():
            raise FloatingPointError("non-finite cases; parameter vector is unstable")
        return df

    def run(self, params: dict[str, float], forcing: pd.DataFrame) -> pd.Series:
        daily = self.simulate(params, forcing)
        s = daily["cases"]
        return s.groupby(s.index.to_period("M")).sum()

    def annual_eir(self, params: dict[str, float], forcing: pd.DataFrame) -> pd.Series:
        """Annual EIR (infectious bites per person per year) - paper Fig. 5A."""
        daily = self.simulate(params, forcing)
        e = daily["eir_daily"]
        return e.groupby(e.index.year).sum()
