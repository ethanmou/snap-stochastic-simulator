"""Call Center #2 baseline parameters from the SNAP queueing paper."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fluid_steady_state import solve_fluid_steady_state
from light_simulation import LightState
from stochastic_simulation import SimulationParams


MODEL_DAY_MINUTES = 540.0


@dataclass(frozen=True)
class PaperCC2Baseline:
    """Raw and derived baseline values for Call Center #2."""

    model_day_minutes: float = MODEL_DAY_MINUTES
    c: int = 52
    lam: float = 623.3
    aht_minutes: float = 21.9
    p_plus: float = 0.5
    deltaB: float = 1 / 128.5
    deltaS: float = 3.0
    deltaL: float = 1 / 9
    thetaA: float = 4.0
    thetaS: float = 3.0
    thetaL: float = 3.0
    gamma: float = 1 / 260
    horizon: float = 50.0
    warmup: float = 2.0
    n_replications: int = 100
    seed: int = 20260628

    @property
    def mu(self) -> float:
        """Total service completion rate per model day."""

        return self.model_day_minutes / self.aht_minutes

    @property
    def mu_plus(self) -> float:
        """Successful service completion rate per model day."""

        return self.p_plus * self.mu

    @property
    def mu_minus(self) -> float:
        """Unsuccessful service completion rate per model day."""

        return (1.0 - self.p_plus) * self.mu

    def make_params(self, *, seed: int | None = None, **overrides) -> SimulationParams:
        """Build SimulationParams, allowing scenario overrides."""

        values = {
            "T": self.horizon,
            "warmup": self.warmup,
            "c": self.c,
            "lam": self.lam,
            "mu_plus": self.mu_plus,
            "mu_minus": self.mu_minus,
            "thetaA": self.thetaA,
            "thetaS": self.thetaS,
            "thetaL": self.thetaL,
            "deltaB": self.deltaB,
            "deltaS": self.deltaS,
            "deltaL": self.deltaL,
            "gamma": self.gamma,
            "q0": 0,
            "b0": 0,
            "rs0": 0,
            "rl0": 0,
            "seed": self.seed if seed is None else seed,
        }
        values.update(overrides)
        return SimulationParams(**values)

    def with_aht(self, aht_minutes: float):
        """Return a config copy with a different average handle time."""

        return replace(self, aht_minutes=aht_minutes)


BASELINE = PaperCC2Baseline()


def make_baseline_params(seed: int | None = None, **overrides) -> SimulationParams:
    """Return CC2 baseline SimulationParams."""

    return BASELINE.make_params(seed=seed, **overrides)


def zero_initial_state() -> LightState:
    """Return the all-zero aggregate initial state."""

    return LightState(0, 0, 0, 0)


def fluid_initial_state(params: SimulationParams | None = None) -> LightState:
    """Return the fluid steady state rounded to integer aggregate counts."""

    params = make_baseline_params() if params is None else params
    steady = solve_fluid_steady_state(params)
    return LightState(
        int(round(steady.q_bar)),
        int(round(steady.b_bar)),
        int(round(steady.rS_bar)),
        int(round(steady.rL_bar)),
    )


def default_initial_state(params: SimulationParams | None = None) -> LightState:
    """Default weekly experiments start near the fluid steady state."""

    return fluid_initial_state(params)
