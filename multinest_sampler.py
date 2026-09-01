"""
multinest_sampler.py
====================
Object-oriented MultiNest design-space characterisation.

REFERENCE IMPLEMENTATION
------------------------
Farhan Feroz's original MultiNest source: https://github.com/farhanferoz/MultiNest (MultiNest v3.12)
the MultiNest papers (arXiv:0704.3704, arXiv:0809.3437 & arXiv:1306.2144) 

Class hierarchy
---------------
DesignSpace                       – physical <-> unit-hypercube mapping,
                                    one (lo, hi) pair per design variable

UncertaintyDistribution           – distribution of uncertain process params
    GaussianUncertainty           – theta ~ N(mu, sigma) or N(mu_vec, Sigma)
    WeightedScenarios

BaseModel  (abstract)             – common interface for all model types
    ProcessModel                  – white-box: equation(d, theta), vectorised
    BlackBoxModel                 – black-box: simulator(d, theta) -> s, one run
                                    per uncertainty scenario

FeasibilityEstimator              – owns the feasibility criterion
                                    (feas_criterion in {"P","VaR","CVaR"}):
                                    P, or the (conditional) value-at-risk of
                                    the worst violation G = max_j g_j at
                                    alpha = 1 - alpha_star. g <= 0 is FEASIBLE
CriterionDisplay                  – the single display system all tables /
                                    colours / plots defer to

Ellipsoid                         – single ellipsoid geometry
EllipsoidalDecomposition          – Algorithm 1 (recursive split) + union sampler
MultiNestSampler                  – main NS loop with mode separation, run in
                                    the unit hypercube via DesignSpace. Stops
                                    once every live point satisfies
                                    ``P >= alpha_star``; no evidence, weights
                                    or prior-mass tracking — the only volumes
                                    used are the ellipsoids' own
SamplerResult                     – immutable run output: dead / live /
                                    rejected points with their P values,
                                    plus run counters
Visualizer                        – all matplotlib code (2-D slices)
RunFrame / RunRecorder            – read-only per-event snapshots, captured
                                    via MultiNestSampler.run(frame_callback=…)
save_run_gif                      – recorded frames -> animated GIF
                                    (headless-safe; SECTION 12)
RunPlayer                         – interactive replay of a recorded run
                                    (Prev/Next/Auto, slider, arrow keys)
"""

from __future__ import annotations

import warnings
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from scipy.special import gamma

warnings.filterwarnings("ignore")


# ============================================================
# SECTION 1 — STEP-THROUGH / VERBOSE DEBUG INFRASTRUCTURE
# ============================================================
#
# Each algorithmic decision (one ellipsoid fit, one cluster reassignment,
# one accept/reject test, one live-point eviction) goes through `_step()`,
# which prints the step name and the values behind it, optionally draws the
# current state, and pauses so you can read them.

# Off by default so an importing script stays headless; `_run_example`
# turns them on for viz_mode='step'.
VERBOSE           = False  # print step names + values
STEP_MODE         = False  # pause with input() after each step

# Seconds of silence the seed population is allowed before it starts
# reporting progress. Nothing prints below this, so a fast white-box model
# stays quiet; a slow simulator announces itself. Set to inf to silence it.
SEED_PROGRESS_AFTER = 2.0
# Directory to also save each step's plot as a numbered PNG (for headless runs).
SAVE_STEP_FRAMES: Optional[str] = None

_STEP_FIG = None
_STEP_AX  = None
_step_frame_counter = 0


def _get_step_figure():
    """
    Lazily create — once — the persistent figure/axes used for
    step-by-step visualization, so every call to `_visualize_state`
    updates the SAME window in place instead of spawning a new one for
    every single step (which would be unusable over hundreds of steps).
    """
    global _STEP_FIG, _STEP_AX
    if _STEP_FIG is None:
        plt.ion()
        _STEP_FIG, _STEP_AX = plt.subplots(figsize=(5.6, 5.6))
        _STEP_FIG.canvas.manager.set_window_title("MultiNest — step-by-step")
    return _STEP_FIG, _STEP_AX


def _step(label: str, visualize=None, **values) -> None:
    """
    Print one algorithmic step: its label, every named value/boolean
    passed in `values` (printed in insertion order), an optional
    visualization, and then pause for the user if STEP_MODE is set.
    """
    if not VERBOSE:
        return
    print(f"\n  ┌─ STEP: {label}")
    for k, v in values.items():
        if isinstance(v, np.ndarray):
            with np.printoptions(precision=5, suppress=True, linewidth=120):
                print(f"  │   {k} =\n{v}")
        else:
            print(f"  │   {k} = {v}")
    print("  └─")
    if visualize is not None:
        visualize()
    if STEP_MODE:
        input("      [Press Enter to continue to next step] ")


def _visualize_state(
    points:      np.ndarray,
    ellipsoids   = None,
    highlight:   Optional[np.ndarray] = None,
    highlight2:  Optional[np.ndarray] = None,
    title:       str = "",
    label1:      str = "highlight",
    label2:      str = "highlight2",
) -> None:
    """
    Draw the current live-point cloud (projected onto its first two
    coordinates), any ellipsoids attached to it (also projected onto
    their first two coordinates), and up to two highlighted points
    (e.g. the point about to be evicted, or a freshly drawn candidate)
    into the persistent step-by-step window, then refresh it.

    This is a 2-D *projection* of whatever-dimensional problem is
    running — for D > 2 problems only axes 0 and 1 are shown, exactly
    like ``Visualizer`` elsewhere in this file.

    Reads ``Ellipsoid`` fields (``mu``, ``L``, ``C``, ``f``) — the class
    itself is defined later, in SECTION 7.
    """
    global _step_frame_counter

    fig, ax = _get_step_figure()
    ax.clear()

    pts = np.atleast_2d(points)
    ax.scatter(pts[:, 0], pts[:, 1], s=18, c="steelblue",
               alpha=0.7, label="points", zorder=2)

    if ellipsoids:
        theta  = np.linspace(0, 2 * np.pi, 200)
        circle = np.column_stack([np.cos(theta), np.sin(theta)])
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(ellipsoids), 1)))
        for e, col in zip(ellipsoids, colors):
            # Regularised shape, not the raw covariance e.C — see
            # _unit_ellipsoid_to_physical_patch for why (raw C draws a
            # needle where the sampler uses a circle).
            C_reg = (e.L @ e.L.T) if e.L is not None else e.C
            C2  = C_reg[:2, :2]
            mu2 = e.mu[:2]
            vals, vecs = np.linalg.eigh(C2)
            vals = np.maximum(vals, 1e-300)
            axes_len = np.sqrt(vals * e.f)
            ell_pts  = circle @ np.diag(axes_len) @ vecs.T + mu2
            ax.plot(ell_pts[:, 0], ell_pts[:, 1], lw=1.8, color=col)

    if highlight is not None:
        h = np.atleast_2d(highlight)
        ax.scatter(h[:, 0], h[:, 1], s=90, c="red", marker="x",
                   linewidths=2.5, label=label1, zorder=3)
    if highlight2 is not None:
        h2 = np.atleast_2d(highlight2)
        ax.scatter(h2[:, 0], h2[:, 1], s=90, c="black", marker="*",
                   linewidths=1.0, label=label2, zorder=3)

    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    if SAVE_STEP_FRAMES:
        os.makedirs(SAVE_STEP_FRAMES, exist_ok=True)
        _step_frame_counter += 1
        fig.savefig(
            os.path.join(SAVE_STEP_FRAMES, f"step_{_step_frame_counter:05d}.png"),
            dpi=110, bbox_inches="tight",
        )

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


# ============================================================
# SECTION 2 — DESIGN SPACE  (physical <-> unit hypercube)
# ============================================================

class DesignSpace:
    """
    Represent a bounded physical design domain and map points to and from
    the unit hypercube.

    The sampler operates in unit-hypercube coordinates for clustering,
    ellipsoid fitting, and sampling, while model evaluations use physical
    design coordinates. A uniform prior is assumed over the bounded domain.

    Parameters
    ----------
    bounds : sequence of (lo_i, hi_i) pairs, one per design variable.
             E.g. ``[(-1.0, 1.0), (0.0, 5000.0)]`` for a 2-D space
    names  : optional list of variable names, used only for repr /
             error messages.
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        names:  Optional[List[str]] = None,
    ) -> None:
        b = np.atleast_2d(np.asarray(bounds, dtype=float))
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError(
                "bounds must be a sequence of (lo, hi) pairs, e.g. "
                "[(-1.0, 1.0), (0.0, 5000.0)]."
            )

        self.lo = b[:, 0]
        self.hi = b[:, 1]
        self.D  = b.shape[0]

        if np.any(self.hi <= self.lo):
            bad = np.where(self.hi <= self.lo)[0]
            raise ValueError(
                f"hi must be > lo for every dimension; violated at "
                f"index(es) {bad.tolist()} "
                f"(lo={self.lo[bad].tolist()}, hi={self.hi[bad].tolist()})."
            )

        if names is not None and len(names) != self.D:
            raise ValueError(
                f"names has length {len(names)} but bounds implies "
                f"D={self.D} dimensions."
            )
        self.names = names

    # ----------------------------------------------------------
    @property
    def bounds(self) -> np.ndarray:
        """ 
        Return the lower and upper bounds of the design variables.
        
        Returns 
        ------- 
        np.ndarray : 
            Array of shape ``(D, 2)``, where each row contains the lower 
            and upper bound of one design variable. 
        """
        return np.column_stack([self.lo, self.hi])

    # ----------------------------------------------------------
    def to_unit(self, d: np.ndarray) -> np.ndarray:
        """
        Map physical design coordinates to the unit hypercube. 
        
        Parameters 
        ----------
        d : np.ndarray 
            Physical design point of shape ``(D,)`` or batch of points 
            of shape ``(N, D)``. 

        Returns 
        ------- 
        np.ndarray : 
            Unit-hypercube coordinates with the same shape as ``d``.
        """
        return (np.asarray(d, dtype=float) - self.lo) / (self.hi - self.lo)

    def to_physical(self, u: np.ndarray) -> np.ndarray:
        """
        Map unit-hypercube coordinates to physical design coordinates. 
        
        Parameters 
        ---------- 
        u : np.ndarray 
            Unit-hypercube point of shape ``(D,)`` or batch of points 
            of shape ``(N, D)``. 
        
        Returns 
        ------- 
        np.ndarray :  
            Physical design coordinates with the same shape as ``u``.
        """
        return self.lo + np.asarray(u, dtype=float) * (self.hi - self.lo)

    # ----------------------------------------------------------
    def __repr__(self) -> str:
        if self.names:
            pairs = ", ".join(
                f"{n}=[{lo:g}, {hi:g}]"
                for n, lo, hi in zip(self.names, self.lo, self.hi)
            )
        else:
            pairs = ", ".join(
                f"[{lo:g}, {hi:g}]" for lo, hi in zip(self.lo, self.hi)
            )
        return f"DesignSpace(D={self.D}, {pairs})"


# ============================================================
# SECTION 3 — UNCERTAINTY DISTRIBUTIONS
# ============================================================

class UncertaintyDistribution(ABC):
    """
    Base class for uncertainty distributions.
    """

    @abstractmethod
    def get_samples_and_weights(
        self, N_theta: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return uncertainty samples and their probability weights.

        Parameters
        ----------
        N_theta : int 
            Requested number of uncertainty samples.  

        Returns
        -------
        theta_samples : np.ndarray
            Samples of shape ``(N_theta,)`` for a scalar uncertain parameter or 
            ``(N_theta, n_parameters)`` for multiple uncertain parameters. 
        weights : np.ndarray
            Non-negative sample weights of shape ``(N_theta,)`` that sum to one.
        """

    def n_scenarios(self, N_theta: int) -> int:
        """
        Return the number of scenarios used in one estimator evaluation.

        By default, the distribution returns the requested ``N_theta``
        samples, so one estimate evaluates the model ``N_theta`` times.

        Subclasses that use a fixed scenario set and ignore ``N_theta``
        should override this method.

        This method must not draw samples or modify the random-number
        generator state.

        Parameters
        ----------
        N_theta : int
            Number of uncertainty samples requested by the estimator.

        Returns
        -------
        int
            Number of scenarios, and therefore model runs, per estimate.
        """
        return int(N_theta)


class GaussianUncertainty(UncertaintyDistribution):
    """
    Represent Gaussian uncertainty for one or more parameters.
    Fresh samples are drawn on each call. Equal probability weights 
    are assigned to all samples.

    Parameters
    ----------
    mu : float or array-like 
        Mean of the uncertain parameter or parameters. 
    sigma : float, optional 
        Standard deviation for a single uncertain parameter. 
    cov : array-like, optional 
        Covariance matrix for multiple uncertain parameters.
    """

    def __init__(
        self,
        mu:    Union[float, np.ndarray],
        sigma: Optional[float]      = None,
        cov:   Optional[np.ndarray] = None,
    ) -> None:
        self.mu = np.atleast_1d(np.asarray(mu, dtype=float))

        if sigma is not None and cov is not None:
            raise ValueError(
                "Provide either sigma (single parameter) "
                "or cov (multiple parameters), not both."
            )

        if self.mu.size == 1:
            if sigma is None:
                raise ValueError(
                    "sigma is required for a single uncertain parameter. "
                    "Example: GaussianUncertainty(mu=1.0, sigma=0.3)"
                )
            if sigma <= 0:
                raise ValueError(f"sigma must be > 0, got {sigma}")
            self.sigma         = float(sigma)
            self.cov           = None
            self._multivariate = False
        else:
            if cov is None:
                raise ValueError(
                    "cov is required for multiple uncertain parameters. "
                    "Example: GaussianUncertainty(mu=[1.0, 0.5], "
                    "cov=[[0.1, 0.0], [0.0, 0.2]])"
                )
            self.cov           = np.asarray(cov, dtype=float)
            self.sigma         = None
            self._multivariate = True

    def get_samples_and_weights(
        self, N_theta: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw Gaussian uncertainty samples with equal probability weights. 
        
        Parameters 
        ---------- 
        N_theta : int 
            Number of samples to draw. 
        
        Returns 
        ------- 
        theta_samples : np.ndarray 
            Samples of shape ``(N_theta,)`` for a single uncertain parameter or 
            ``(N_theta, n_parameters)`` for multiple parameters.
         weights : np.ndarray 
            Equal weights of shape ``(N_theta,)``, each equal to ``1 / N_theta``.
        """
        if self._multivariate:
            theta = np.random.multivariate_normal(self.mu, self.cov,
                                                   size=N_theta)
        else:
            theta = np.random.normal(self.mu.item(), self.sigma, N_theta)

        weights = np.full(N_theta, 1.0 / N_theta)
        return theta, weights

    def __repr__(self) -> str:
        if self._multivariate:
            return (f"GaussianUncertainty(mu={self.mu.tolist()}, "
                    f"cov={self.cov.tolist()})")
        else:
            return f"GaussianUncertainty(mu={self.mu.item()}, sigma={self.sigma})"


class WeightedScenarios(UncertaintyDistribution):
    """
    Represent a fixed set of weighted uncertainty scenarios.

    The same scenarios and weights are returned for every design point.
    The ``N_theta`` argument passed to ``get_samples_and_weights`` is
    ignored.

    Parameters
    ----------
    theta_samples : np.ndarray
        Fixed uncertainty scenarios of shape ``(N,)`` or
        ``(N, n_parameters)``.
    weights : np.ndarray
        Non-negative scenario weights of shape ``(N,)``.
    normalise : bool, optional
        If ``True``, divide the weights by their sum. If ``False``,
        the weights must already sum to one.
    """

    def __init__(
        self,
        theta_samples: np.ndarray,
        weights:       np.ndarray,
        normalise:     bool = False,
    ) -> None:
        self.theta_samples = np.asarray(theta_samples)
        w = np.asarray(weights, dtype=float)

        if np.any(w < 0):
            raise ValueError(
                "All weights must be non-negative. "
                f"Found {np.sum(w < 0)} negative value(s)."
            )
        if np.all(w == 0):
            raise ValueError(
                "All weights are zero; cannot form a probability measure."
            )
        total = w.sum()
        if normalise:
            self.weights = w / total
        else:
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(
                    f"Weights sum to {total:.6f}, not 1.0. "
                    "Pass normalise=True to normalise automatically, "
                    "or pre-normalise before passing.\n"
                    "  MCMC samples      — weights are already 1/N, "
                    "no action needed\n"
                    "  Importance sampl. — pass normalise=True\n"
                    "  Nested sampling   — pass normalise=True"
                )
            self.weights = w

    def get_samples_and_weights(
        self, N_theta: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the fixed scenario set.  N_theta is ignored.

        Returns
        -------
        theta_samples : (N,) or (N, n_params)   the fixed θ_j
        weights       : (N,)                    the fixed ω_j

        Return the fixed uncertainty scenarios and their weights. 
        Parameters 
        ---------- 
        N_theta : int 
            Ignored. The complete fixed scenario set is always returned. 
        
        Returns ------- 
        theta_samples : np.ndarray 
            Fixed uncertainty scenarios of shape ``(N,)`` or ``(N, n_parameters)``. 
        weights : np.ndarray 
            Fixed scenario weights of shape ``(N,)``
        """
        return self.theta_samples, self.weights

    def n_scenarios(self, N_theta: int) -> int:
        """
        Return the size of the fixed scenario set.

        ``N_theta`` is ignored because all stored scenarios are always used.
        """
        return int(self.theta_samples.shape[0])

    def __repr__(self) -> str:
        N = len(self.theta_samples)
        return (f"WeightedScenarios(N={N}, "
                f"weight_range=[{self.weights.min():.3g}, "
                f"{self.weights.max():.3g}])")


# ============================================================
# SECTION 4 — BASE MODEL + WHITE-BOX + BLACK-BOX
# ============================================================

class BaseModel(ABC):
    """
    Base class for process-model interfaces. 
    
    Parameters 
    ---------- 
    constraints : list of tuple(float, float) 
        Lower and upper bounds for each model output. 
    name : str, optional 
        Name used in reports and visualisations.
    """
    def __init__(
        self,
        constraints: List[Tuple[float, float]],
        name:        str = "Model",
    ) -> None:
        self.constraints = constraints
        self.name        = name
        self.uncertainty: UncertaintyDistribution # Set by concrete subclasses.


    def make_estimator(
        self,
        uncertainty:    UncertaintyDistribution,
        N_theta:        int = 0,
        feas_criterion: str = "VaR",
    ) -> "FeasibilityEstimator":
        """
        Create a feasibility estimator for the model.

        Parameters
        ----------
        uncertainty : UncertaintyDistribution
            Representation of parameter uncertainty.
        N_theta : int, optional
            Number of uncertainty samples per evaluation. Ignored for
            ``WeightedScenarios``.
        feas_criterion : {"P", "VaR", "CVaR"}, optional
            Feasibility criterion used by the estimator.

        Returns
        -------
        FeasibilityEstimator
            Configured feasibility estimator.
        """
        if isinstance(uncertainty, GaussianUncertainty) and N_theta <= 0:
            raise ValueError(
                f"N_theta must be > 0 for GaussianUncertainty, got {N_theta}. "
                "Example: model.make_estimator(uncertainty, N_theta=100)"
            )
        return FeasibilityEstimator(
            model          = self,
            uncertainty    = uncertainty,
            N_theta        = N_theta,
            feas_criterion = feas_criterion,
        )

    def mc_probability_grid(
        self,
        grid_points: np.ndarray,
        N_theta:     int = 500,
    ) -> np.ndarray:
        """
        Estimate feasibility probabilities over a set of design points. 
        
        Parameters 
        ---------- 
        grid_points : np.ndarray
            Design points of shape ``(n_points, n_design)``. 
        N_theta : int, optional 
            Number of uncertainty samples used for each design point. 
            
        Returns 
        ------- 
        np.ndarray 
            Estimated feasibility probabilities of shape ``(n_points,)``.
        """
        return self.mc_criterion_grid(grid_points, N_theta=N_theta,
                                      feas_criterion="P")

    def mc_criterion_grid(
        self,
        grid_points:    np.ndarray,
        N_theta:        int   = 500,
        feas_criterion: str   = "VaR",
        alpha_star:     float = 0.95,
    ) -> np.ndarray:
        """
        Estimate a feasibility criterion over multiple design points. 
        
        Parameters 
        ---------- 
        grid_points : np.ndarray 
            Design points of shape ``(n_points, n_design)``. 
        N_theta : int, optional 
            Number of uncertainty samples per design point. 
        feas_criterion : {"P", "VaR", "CVaR"}, optional 
            Criterion to evaluate. 
        alpha_star : float, optional 
            Reliability level used in the VaR and CVaR calculations. 
            
        Returns 
        ------- 
        np.ndarray
            Raw criterion values of shape ``(n_points,)``.
        """
        estimator = self.make_estimator(
            uncertainty    = self.uncertainty,
            N_theta        = N_theta,
            feas_criterion = feas_criterion,
        )
        return np.array([estimator.criterion_value(d, alpha_star)
                         for d in grid_points])


# ------------------------------------------------------------------
class ProcessModel(BaseModel):
    """
    Represent a vectorised white-box process model.

    Parameters
    ----------
    equation : callable
        Function ``equation(d, theta)`` that evaluates a single design
        point over a batch of uncertainty scenarios. It returns an array 
        of shape (N_theta,) for a single model output or (N_theta, n_outputs) 
        for multiple model outputs.
    uncertainty : UncertaintyDistribution
        Distribution of the uncertain parameters.
    constraints : list of tuple(float, float)
        Lower and upper bounds for each model output.
    name : str, optional
        Model name used in reports and visualisations.
    """

    def __init__(
        self,
        equation:    Callable,
        uncertainty: UncertaintyDistribution,
        constraints: List[Tuple[float, float]],
        name:        str = "ProcessModel",
    ) -> None:
        super().__init__(constraints, name)
        self.equation    = equation
        self.uncertainty = uncertainty

    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ProcessModel(name='{self.name}', "
            f"uncertainty={self.uncertainty!r}, "
            f"constraints={self.constraints})"
        )


# ------------------------------------------------------------------
class BlackBoxModel(BaseModel):
    r"""
    Model wrapper for external simulators such as Aspen HYSYS, gPROMS
    and compiled process simulators.

    The simulator evaluates one design point under one uncertainty
    scenario at a time:

        simulator(d, theta)

    ``d`` contains the design variables selected by the sampler.

    ``theta`` contains the uncertain parameters for one scenario:

    - a scalar when there is one uncertain parameter;
    - a one-dimensional array when there are multiple uncertain
      parameters.

    For multiple uncertain parameters, the order of the values in
    ``theta`` must match the column order used by the uncertainty
    distribution.

    The simulator is called once for each uncertainty scenario.
    Therefore, one feasibility estimate requires:

        uncertainty.n_scenarios(N_theta)

    simulator evaluations. For slow external simulators, the number of
    scenarios should be selected carefully. A small fixed
    ``WeightedScenarios`` set may be more practical than a large number
    of Monte Carlo samples.

    Parameters
    ----------
    simulator : callable
        Function with the form:

            simulator(d, theta) -> float or array-like

        ``d`` is one design point with shape ``(n_design,)``.

        ``theta`` is one uncertainty scenario, represented by either a
        scalar or a one-dimensional array.

        The returned outputs must follow the same order as
        ``constraints``.

    uncertainty : UncertaintyDistribution
        Distribution or fixed scenario set representing one or more
        uncertain process parameters.

    constraints : list of tuple(float, float)
        Lower and upper bounds for each simulator output.

        Provide one ``(lb, ub)`` pair for each returned output, in the
        same order as the simulator outputs. Use ``-np.inf`` or
        ``np.inf`` for one-sided constraints.

    name : str, default="BlackBoxModel"
        Model name used in reports and diagnostic output.

    on_failure : {"raise", "infeasible"}, default="raise"
        Behaviour when the simulator raises an exception, for example
        because of non-convergence, a timeout or a COM error.

        ``"raise"``
            Propagate the exception and stop the complete run.

        ``"infeasible"``
            Store NaN outputs for the failed scenario and continue.
            The scenario is then treated as infeasible by the P, VaR
            and CVaR calculations. Failures are counted in
            ``n_failures``.

    Example
    -------
    Two uncertain parameters, so ``theta`` is one row to unpack:

    >>> unc = WeightedScenarios(
    ...     theta_samples=np.array([[0.9, 310.0],     # activity, feed T
    ...                             [1.0, 320.0],
    ...                             [1.1, 330.0]]),
    ...     weights=np.array([0.25, 0.5, 0.25]))
    >>>
    >>> def run_hysys(d, theta):
    ...     activity, feed_T = theta
    ...     out = run_case(pressure=d[0], temperature=d[1], ratio=d[2],
    ...                    catalyst_activity=activity, feed_T=feed_T)
    ...     return np.array([out["methanol_kg_h"], out["C_efficiency"]])
    >>>
    >>> model = BlackBoxModel(
    ...     simulator   = run_hysys,
    ...     uncertainty = unc,
    ...     constraints = [(1000.0, np.inf), (20.0, np.inf)],
    ...     name        = "HYSYS methanol synthesis",
    ...     on_failure  = "infeasible",
    ... )

    With a single uncertain parameter, ``theta_samples`` is 1-D
    (``np.array([0.9, 1.0, 1.1])``) and ``theta`` arrives as a float.
    """

    ON_FAILURE = ("raise", "infeasible")

    def __init__(
        self,
        simulator:      Callable,
        uncertainty:    UncertaintyDistribution,
        constraints:    Optional[List[Tuple[float, float]]] = None,
        name:           str = "BlackBoxModel",
        on_failure:     str = "raise",
    ) -> None:
        if constraints is None:
            raise ValueError(
                "constraints is required: one (lb, ub) pair per model "
                "output, in the order the simulator returns them. "
                "Example: constraints=[(1000.0, np.inf)]")
        if on_failure not in self.ON_FAILURE:
            raise ValueError(
                f"on_failure must be one of {self.ON_FAILURE}, "
                f"got {on_failure!r}")

        super().__init__(constraints, name)
        self.simulator      = simulator
        self.uncertainty    = uncertainty
        self.on_failure     = on_failure

        # Number of simulator calls that raised and were recorded as
        # infeasible. Reporting only; never read by the algorithm.
        self.n_failures     = 0

    def __repr__(self) -> str:
        return (
            f"BlackBoxModel(name='{self.name}', "
            f"uncertainty={self.uncertainty!r}, "
            f"constraints={self.constraints}, "
            f"on_failure='{self.on_failure}', "
            f"n_failures={self.n_failures})"
        )



# ============================================================
# SECTION 5 — FEASIBILITY ESTIMATOR
# ============================================================

class FeasibilityEstimator:
    """
    Evaluate design feasibility under parameter uncertainty.

    The estimator supports feasibility probability (``P``), value at
    risk (``VaR``), and conditional value at risk (``CVaR``).

    Parameters
    ----------
    model : BaseModel
        Process model to evaluate.
    uncertainty : UncertaintyDistribution
        Distribution or weighted scenarios representing uncertainty.
    N_theta : int
        Number of uncertainty scenarios per evaluation. Ignored for
        ``WeightedScenarios``.
    feas_criterion : {"P", "VaR", "CVaR"}, optional
        Feasibility criterion used to evaluate and rank design points.              
    """

    CRITERIA = ("P", "VaR", "CVaR")

    def __init__(
        self,
        model:          BaseModel,
        uncertainty:    UncertaintyDistribution,
        N_theta:        int,
        feas_criterion: str = "VaR",
    ) -> None:
        if feas_criterion not in self.CRITERIA:
            raise ValueError(
                f"feas_criterion must be one of {self.CRITERIA}, "
                f"got {feas_criterion!r}")
        self.model          = model
        self.uncertainty    = uncertainty
        self.N_theta        = N_theta
        self.feas_criterion = feas_criterion

    # ----------------------------------------------------------
    # SCENARIO EVALUATION
    # ---------------------------------------------------------
    def _s_values_batch(self, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate one design point over all uncertainty scenarios. 
        
        Parameters 
        ---------- 
        d : np.ndarray 
            Single design point of shape ``(n_design,)``. 
        
        Returns 
        ------- 
        s_values : np.ndarray 
            Model outputs of shape ``(N_theta,)`` for one output or 
            ``(N_theta, n_outputs)`` for multiple outputs. 
        weights : np.ndarray
            Scenario weights of shape ``(N_theta,)``.
        """
        theta_samples, weights = self.uncertainty.get_samples_and_weights(
            self.N_theta
        )

        if isinstance(self.model, ProcessModel):
            # White-box: one vectorised equation call over all scenarios
            s_values = self.model.equation(d, theta_samples)
        else:
            # Black-box: one simulator run per uncertainty scenario.
            # A scenario whose run raises is recorded as all-NaN when
            # on_failure="infeasible"; violation_matrix() turns that into
            # +inf violations and _all_feasible() into False, so it counts
            # as maximally infeasible under every criterion.
            n_out    = max(len(self.model.constraints), 1)
            outputs  = []
            for theta_j in theta_samples:
                # One scenario: a float when a single parameter is
                # uncertain, a 1-D (n_params,) row when several are.
                th = np.asarray(theta_j, dtype=float)
                th = float(th) if th.size == 1 else np.ravel(th)
                try:
                    out = self.model.simulator(d, th)
                    outputs.append(
                        np.atleast_1d(np.asarray(out, dtype=float)))
                except Exception:
                    if self.model.on_failure == "raise":
                        raise
                    self.model.n_failures += 1
                    outputs.append(np.full(n_out, np.nan))
            s_values = np.array(outputs)
            if s_values.shape[1] == 1:
                s_values = s_values.squeeze(axis=1)

        return s_values, weights

    def _all_feasible(self, s_values: np.ndarray) -> np.ndarray:
        """
        Identify scenarios that satisfy all model-output constraints.

        Parameters 
        ---------- 
        s_values : np.ndarray 
            Model outputs of shape ``(N_theta,)`` for one output or 
            ``(N_theta, n_outputs)`` for multiple outputs. 
            
        Returns 
        ------- 
        np.ndarray 
            Boolean mask of shape ``(N,)`` indicating feasible scenarios.
        """
        feasible = np.ones(s_values.shape[0], dtype=bool)

        if s_values.ndim == 1:
            lb, ub    = self.model.constraints[0]
            feasible &= (s_values >= lb) & (s_values <= ub)
        else:
            for i, (lb, ub) in enumerate(self.model.constraints):
                feasible &= (s_values[:, i] >= lb) & (s_values[:, i] <= ub)

        return feasible

    # ----------------------------------------------------------
    # FEASIBILITY CRITERIA
    # ----------------------------------------------------------
    def estimate_probability(self, d: np.ndarray) -> float: 
        """
        Estimate the feasibility probability of a design point. 
        
        Parameters 
        ---------- 
        d : np.ndarray 
            Single design point in physical coordinates with shape ``(n_design,)``. 
            
        Returns 
        ------- 
        float 
            Weighted feasibility probability in the interval ``[0, 1]``.
        """
        s_values, weights = self._s_values_batch(d)
        feasible = self._all_feasible(s_values)
        return float(np.sum(weights[feasible]))

    
    def constraint_violations(self, s_row: np.ndarray) -> np.ndarray:
        """
        Convert model outputs into constraint-violation values.

        Each finite lower bound contributes ``lb - s``, and each finite
        upper bound contributes ``s - ub``. A constraint is satisfied when
        its violation is non-positive.

        Parameters
        ----------
        s_row : np.ndarray
            Model outputs for one uncertainty scenario, with shape
            ``(n_outputs,)``.

        Returns
        -------
        np.ndarray
            Constraint violations, one for each finite bound. Non-finite
            model outputs produce infinite violations.
        """
        g = []
        for i, (lb, ub) in enumerate(self.model.constraints):
            s_i = s_row[i]
            if not np.isfinite(s_i):
                if np.isfinite(lb): g.append(np.inf)
                if np.isfinite(ub): g.append(np.inf)
                continue
            if np.isfinite(lb): g.append(lb - s_i)
            if np.isfinite(ub): g.append(s_i - ub)
        return np.asarray(g, dtype=float)

    
    def violation_matrix(self, s_values: np.ndarray) -> np.ndarray:
        """
        Convert model outputs into constraint violations for all scenarios.

        Each finite lower bound contributes ``lb - s``, and each finite
        upper bound contributes ``s - ub``. Non-finite model outputs produce
        infinite violations.

        Parameters
        ----------
        s_values : np.ndarray
            Model outputs of shape ``(N_theta,)`` for one output or
            ``(N_theta, n_outputs)`` for multiple outputs.

        Returns
        -------
        np.ndarray
            Constraint-violation matrix of shape ``(N_theta, n_violations)``,
            with one column for each finite constraint bound.
        """
        S = s_values if s_values.ndim == 2 else s_values[:, None]
        cols = []
        for i, (lb, ub) in enumerate(self.model.constraints):
            s_i = S[:, i]
            bad = ~np.isfinite(s_i)          
            if np.isfinite(lb):
                cols.append(np.where(bad, np.inf, lb - s_i))
            if np.isfinite(ub):
                cols.append(np.where(bad, np.inf, s_i - ub))
        if not cols:
            return np.zeros((S.shape[0], 0), dtype=float)
        return np.column_stack(cols)

    def worst_violation(self, s_values: np.ndarray) -> np.ndarray:
        """
        Return the worst constraint violation for each uncertainty scenario.

        The worst violation is the maximum violation across all finite
        constraint bounds. A scenario is feasible when its worst violation
        is non-positive.

        Parameters
        ----------
        s_values : np.ndarray
            Model outputs of shape ``(N_theta,)`` for one output or
            ``(N_theta, n_outputs)`` for multiple outputs.

        Returns
        -------
        np.ndarray
            Worst violation for each scenario, with shape ``(N_theta,)``.
            If no finite bounds are defined, all values are ``-np.inf``.
        """
        Gm = self.violation_matrix(s_values)
        if Gm.shape[1] == 0:
            return np.full(Gm.shape[0], -np.inf)
        return np.max(Gm, axis=1)

    @staticmethod
    def _var_cvar(G: np.ndarray, w: np.ndarray,
                  alpha_star: float) -> Tuple[float, float]:
        """
        Compute VaR and CVaR for a discrete distribution of violations.

        Scenario probabilities are specified by ``w``. VaR is evaluated at
        confidence level ``alpha_star``, and CVaR measures risk in the
        corresponding upper tail.

        Parameters
        ----------
        G : np.ndarray
            Worst violation for each uncertainty scenario, with shape
            ``(N_theta,)``.
        w : np.ndarray
            Scenario probabilities of shape ``(N_theta,)``.
        alpha_star : float
            VaR and CVaR confidence level.

        Returns
        -------
        var : float
            ``VaR_alpha_star`` of the scenario violations.
        cvar : float
            ``CVaR_alpha_star`` of the scenario violations.

        Notes
        -----
        When ``alpha_star = 1``, both measures reduce to the worst-case
        violation. Non-finite VaR values are returned unchanged for both
        measures.
        """
        order = np.argsort(-G, kind="stable")     # worst violation first
        Gs, ws = G[order], w[order]

        tail = 1.0 - alpha_star

        if tail <= 0.0:
            worst = float(Gs[0])
            return worst, worst

        cum   = np.cumsum(ws)
        over  = np.nonzero(cum > tail)[0]
        k     = int(over[0]) if over.size else int(Gs.size - 1)
        var   = float(Gs[k])

        if not np.isfinite(var):
            return var, var
        excess = np.maximum(Gs - var, 0.0)
        cvar   = var + float(np.sum(ws * excess)) / tail
        return var, cvar

    def criterion_value(self, d: np.ndarray, alpha_star: float) -> float:
        """
        Evaluate the selected feasibility criterion at one design point.

        Parameters
        ----------
        d : np.ndarray
            Single design point in physical coordinates, with shape
            ``(n_design,)``.
        alpha_star : float
            Target reliability level for ``P`` and confidence level for
            ``VaR`` and ``CVaR``.

        Returns
        -------
        float
            Selected raw criterion value. 
        """
        s_values, weights = self._s_values_batch(d)

        if self.feas_criterion == "P":
            return float(np.sum(weights[self._all_feasible(s_values)]))

        G          = self.worst_violation(s_values)
        var, cvar  = self._var_cvar(G, weights, alpha_star)
        return var if self.feas_criterion == "VaR" else cvar

    # ----------------------------------------------------------
    # SAMPLER INTERFACE
    # ----------------------------------------------------------
    def merit_threshold(self, alpha_star: float) -> float:
        """
        Return the design-space membership threshold in merit space.

        Parameters
        ----------
        alpha_star : float
            Target reliability level for the probability criterion.

        Returns
        -------
        float
            ``alpha_star`` for ``P`` and zero for ``VaR`` or ``CVaR``.
        """
        return alpha_star if self.feas_criterion == "P" else 0.0

    def merit(self, d: np.ndarray, alpha_star: float) -> float:
        """
        Evaluate the higher-is-better merit used by the sampler.

        The probability criterion is returned unchanged, while VaR and CVaR
        are negated so that larger values are better for every criterion.

        Parameters
        ----------
        d : np.ndarray
            Single design point in physical coordinates, with shape
            ``(n_design,)``.
        alpha_star : float
            Target reliability level for ``P`` and confidence level for
            ``VaR`` and ``CVaR``.

        Returns
        -------
        float
            Merit value. Design-space membership is determined by
            ``merit >= merit_threshold(alpha_star)``.
        """
        v = self.criterion_value(d, alpha_star)
        return v if self.feas_criterion == "P" else -v

    def merit_and_P(self, d: np.ndarray,
                    alpha_star: float) -> Tuple[float, float]:
        """
        Evaluate the sampler merit and feasibility probability together.

        Model outputs are evaluated once. The same scenario results are used
        to calculate both the selected merit and the feasibility probability.

        Parameters
        ----------
        d : np.ndarray
            Single design point in physical coordinates, with shape
            ``(n_design,)``.
        alpha_star : float
            Target reliability level for ``P`` and confidence level for
            ``VaR`` and ``CVaR``.

        Returns
        -------
        merit : float
            Higher-is-better sampler merit. This is ``P`` for the probability
            criterion and the negative VaR or CVaR value for the risk
            criteria.
        probability : float
            Feasibility probability of the design point.
        """
        s_values, weights = self._s_values_batch(d)
        return self.merit_and_P_from_s(s_values, weights, alpha_star)


    def merit_and_P_from_s(self, s_values: np.ndarray,
                           weights: np.ndarray,
                           alpha_star: float) -> Tuple[float, float]:
        """
        Turn already-evaluated scenario outputs into merit and probability.

        Split out of ``merit_and_P`` so that a subclass which obtains
        ``s_values`` some other way -- from a pool of external simulator
        processes, say -- reuses the criterion arithmetic verbatim instead
        of reimplementing it and drifting out of step.

        Parameters
        ----------
        s_values : np.ndarray
            Model outputs of shape ``(N_theta,)`` for one output or
            ``(N_theta, n_outputs)`` for several. Non-finite entries are
            read as maximally infeasible, exactly as in the serial path.
        weights : np.ndarray
            Scenario weights of shape ``(N_theta,)``.
        alpha_star : float
            Target reliability level for ``P``, confidence level for ``VaR``
            and ``CVaR``.

        Returns
        -------
        merit : float
            Higher-is-better sampler merit.
        probability : float
            Feasibility probability of the design point.
        """
        P = float(np.sum(weights[self._all_feasible(s_values)]))
        if self.feas_criterion == "P":
            return P, P
        G         = self.worst_violation(s_values)
        var, cvar = self._var_cvar(G, weights, alpha_star)
        rho       = var if self.feas_criterion == "VaR" else cvar
        return -rho, P


    def batch_merit_and_P(self, points: np.ndarray,
                          alpha_star: float) -> List[Tuple[float, float]]:
        """
        Evaluate several design points, returning one (merit, P) pair each.

        The default implementation is a plain loop, so it is exactly what
        the sampler did before this hook existed. It is a separate method
        only so a subclass can evaluate the points CONCURRENTLY without
        touching the sampler.

        Only worth overriding for a slow external simulator. A white-box
        ``ProcessModel`` already evaluates every scenario in one vectorised
        call, and dispatching those to other processes would cost far more
        than it saves.

        Parameters
        ----------
        points : np.ndarray
            Design points in physical coordinates, shape ``(n, n_design)``.
        alpha_star : float
            Passed through to ``merit_and_P``.

        Returns
        -------
        list of (float, float)
            One ``(merit, probability)`` pair per row of ``points``, in the
            same order.
        """
        total   = len(points)
        started = time.perf_counter()
        pairs   = []
        loud    = False

        # SEED_PROGRESS_AFTER = inf silences this block entirely, which is what
        # its own comment promises; a batch driver that prints one line per run
        # sets it and does not want three more from in here.
        announce = np.isfinite(SEED_PROGRESS_AFTER)

        # Said up front so nobody is left watching a blank screen wondering
        # whether the thing started. The running count below only kicks in
        # once the population is demonstrably slow -- a white-box equation
        # finishes the lot in milliseconds and would just be noise.
        if announce:
            print(f"  Initialising live points: {total} to evaluate",
                  flush=True)

        for i, d in enumerate(points):
            pairs.append(self.merit_and_P(np.asarray(d, dtype=float),
                                          alpha_star))
            elapsed = time.perf_counter() - started
            loud = loud or elapsed > SEED_PROGRESS_AFTER

            if loud:
                done = i + 1
                left = elapsed / done * (total - done)
                print(f"    {done}/{total}   {elapsed:.0f} s elapsed, "
                      f"~{left:.0f} s left", flush=True)

        if announce:
            print(f"  Live points ready in "
                  f"{time.perf_counter() - started:.1f} s\n", flush=True)
        return pairs


    @property
    def n_constraints(self) -> int:
        """
        Return the number of finite constraint bounds.

        Each finite lower or upper bound contributes one constraint. The
        result equals the number of columns in ``violation_matrix``.

        Returns
        -------
        int
            Total number of finite lower and upper bounds.
        """
        return sum(int(np.isfinite(lb)) + int(np.isfinite(ub))
                   for (lb, ub) in self.model.constraints)


# ============================================================
# SECTION 6 — CRITERION DISPLAY ADAPTER
# ============================================================

class CriterionDisplay:
    """
    Define how feasibility results are converted, classified and displayed.

    A single instance handles the selected criterion: ``"P"``, ``"VaR"``
    or ``"CVaR"``. It provides the common rules used by reliability tables,
    point colours, contour lines, shading, labels and colourbars.

    The sampler stores merit values, where larger is always better. Displayed
    values use the criterion's original form:

    - ``P``: feasibility probability, where larger is better.
    - ``VaR`` and ``CVaR``: violation risk, where values at or below zero
      are feasible.

    ``to_merit`` and ``from_merit`` convert between these two forms.

    Probability bands use fixed reliability ranges. VaR/CVaR use the absolute
    feasibility boundary ``risk <= 0``; colours among infeasible points are
    population-relative because violation values depend on the units and scale
    of the constraints.
    """

    # Probability levels used for contour lines. These are the band EDGES of
    # ``reliability_table`` below, so every contour drawn on a figure is a row
    # boundary in the table and the two can be read against each other. They
    # are also the ranges of Table 2 in Kusumo et al. (2020). The top level is
    # 0.95 rather than 0.90 because that is alpha*, i.e. the design-space
    # boundary itself -- a contour set that stops at 0.90 never draws it.
    P_ALPHAS       = [0.25, 0.50, 0.70, 0.95]

    # Colours corresponding to the probability contour levels.
    P_ALPHA_COLORS = ["blue", "green", "orange", "red"]

    # Probability thresholds and colours used to shade nested reliability regions.
    P_SHADING      = [(0.25, "lightyellow"), (0.50, "lightgreen"),
                      (0.70, "steelblue"),   (0.95, "navy")]

    def __init__(self, feas_criterion: str = "VaR",
                 alpha_star: float = 0.95) -> None:
        """
        Configure the display rules for one feasibility criterion.

        Parameters
        ----------
        feas_criterion : str
            Criterion to display: ``"P"``, ``"VaR"`` or ``"CVaR"``.

        alpha_star : float
            Target reliability or confidence level.
        """
        if feas_criterion not in FeasibilityEstimator.CRITERIA:
            raise ValueError(
                f"feas_criterion must be one of "
                f"{FeasibilityEstimator.CRITERIA}, got {feas_criterion!r}")
        self.criterion  = feas_criterion
        self.alpha_star = float(alpha_star)
        self.tail_probability =  1 - self.alpha_star

    def __repr__(self) -> str:
        return (f"CriterionDisplay({self.criterion!r}, "
                f"alpha_star={self.alpha_star:g}, tail_probability={self.tail_probability:g})")

    # ------------------------------------------------------------------
    # CRITERION INFORMATION
    # ------------------------------------------------------------------
    @property
    def is_probability(self) -> bool:
        """Return True when the selected criterion is feasibility probability."""
        return self.criterion == "P"

    @property
    def symbol(self) -> str:
        """Return the short mathematical symbol used in tables and plots.

        Bare "VaR" / "CVaR", with no "[G]" argument. The risk measure is
        taken over the constraint margin and nothing else, in every table
        and every figure this file writes, so naming G buys no
        disambiguation -- it only adds two characters to a symbol that
        appears in column headers, axis labels and one-line log rows.
        ``value_label`` is the fully-set form used on colourbars.
        """
        if self.is_probability:
            return "P"
        return self.criterion

    @property
    def value_label(self) -> str:
        """Return the criterion label used on axes and colourbars.

        Set in maths, with the confidence level as a NUMERIC subscript --
        VaR_0.95 rather than VaR_{alpha*} plus a parenthetical saying what
        alpha* is -- and the design vector d in bold, since it is a vector.
        Written as a function of d alone: G and theta are integrated out by
        the criterion, so naming them on the colourbar is noise.
        """
        if self.is_probability:
            return r"$P(\mathrm{feasible} \mid \mathbf{d})$"
        return (rf"$\mathrm{{{self.criterion}}}_{{{self.alpha_star:g}}}"
                rf"(\mathbf{{d}})$")

    @property
    def threshold_value(self) -> float:
        """Return the criterion cutoff defining design-space membership."""
        return self.alpha_star if self.is_probability else 0.0

    @property
    def threshold_label(self) -> str:
        """Return the short label for the design-space membership cutoff."""
        return "α*" if self.is_probability else "0"

    # ------------------------------------------------------------------
    # CRITERION-VALUE AND MERIT CONVERSION
    # ------------------------------------------------------------------
    def to_merit(self, v):
        """Convert criterion values to the sampler's higher-is-better merit."""
        return v if self.is_probability else -np.asarray(v, dtype=float)

    def from_merit(self, m):
        """ Convert sampler merits back to criterion values."""
        return m if self.is_probability else -np.asarray(m, dtype=float)

    def feasible(self, v) -> np.ndarray:
        """    
        Return a Boolean mask indicating design-space membership.

        A point belongs to the design space when ``P >= alpha_star`` for the
        probability criterion, or when ``VaR <= 0`` or ``CVaR <= 0`` for the
        risk criteria.
        """
        v = np.asarray(v, dtype=float)
        return (v >= self.alpha_star) if self.is_probability else (v <= 0.0)

    # ---- colours (ONE band system, see class docstring) -----------
    def merit_colors(self, merits) -> List[str]:
        """Band colours from MERIT values. The single entry point every
        renderer uses; ``point_colors`` just converts first."""
        m = np.asarray(merits, dtype=float)
        if self.is_probability:
            return [_band_color(float(p)) for p in m]
        # rho <= 0 <=> merit >= 0: population-relative map with the
        # threshold at 0.
        return _relative_band_colors(m, 0.0)

    def point_colors(self, values) -> List[str]:
        """Band colours from CRITERION VALUES."""
        return self.merit_colors(self.to_merit(np.asarray(values, float)))

    # ---- field rendering ------------------------------------------
    @staticmethod
    def _finite_limits(grid: np.ndarray) -> Tuple[float, float]:
        f = np.asarray(grid, dtype=float)
        f = f[np.isfinite(f)]
        if f.size == 0:
            return 0.0, 1.0
        lo, hi = float(f.min()), float(f.max())
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def sanitise_field(self, grid: np.ndarray) -> np.ndarray:
        """Plot-safe copy: ±inf (failed simulations -> rho = +inf) clipped
        to the finite range, so contourf/pcolormesh don't choke. For "P"
        this is a no-op — a probability is always finite."""
        g = np.asarray(grid, dtype=float)
        if self.is_probability:
            return g
        lo, hi = self._finite_limits(g)
        return np.clip(np.nan_to_num(g, nan=hi, posinf=hi, neginf=lo), lo, hi)

    def grid_vlim(self, grid: np.ndarray) -> Tuple[float, float]:
        """(vmin, vmax) for the greyscale backdrop in the GIF / player.
        (0, 1) for "P" — bit-identical to the hardcoded values the frame
        renderer used before."""
        if self.is_probability:
            return 0.0, 1.0
        return self._finite_limits(grid)

    def field_kwargs(self, grid: np.ndarray) -> dict:
        """kwargs for ``contourf`` of the ground-truth field.

        "P"        : the original RdYlGn over a fixed [0,1], 21 levels.
        VaR / CVaR : RdYlGn reversed (low violation = green = feasible)
                     and pivoted on rho = 0 with a TwoSlopeNorm, so the
                     colour change IS the design-space boundary rather
                     than landing at some arbitrary mid-range value.
                     Falls back to a plain linear map when the field does
                     not straddle 0 (TwoSlopeNorm requires vmin < 0 < vmax).
        """
        if self.is_probability:
            return dict(levels=np.linspace(0.0, 1.0, 21), cmap="RdYlGn")
        lo, hi = self._finite_limits(grid)
        kw = dict(levels=np.linspace(lo, hi, 21), cmap="RdYlGn_r")
        if lo < 0.0 < hi:
            kw["norm"] = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
        return kw

    def contour_levels(self) -> List[Tuple[float, str, str]]:
        """[(level, colour, legend label)] — the iso-lines drawn over the
        field. For "P" these are the four reliability contours the file
        always drew; for a risk measure there is exactly one meaningful
        iso-line, rho = 0."""
        if self.is_probability:
            return [(a, c, f"P = {a}")
                    for a, c in zip(self.P_ALPHAS, self.P_ALPHA_COLORS)]
        return [(0.0, "black", f"{self.symbol} = 0  (DS boundary)")]

    def nested_shading(self, grid: np.ndarray
                       ) -> List[Tuple[np.ndarray, str, str]]:
        """[(mask, colour, label)] for the filled 'feasible region' panel.
        Nested reliability sets for "P"; the single set rho <= 0 for a
        risk measure, which has no nested analogue that isn't unit-
        dependent."""
        g = np.asarray(grid, dtype=float)
        if self.is_probability:
            return [((g >= a), c, f"P ≥ {a}")
                    for a, c in self.P_SHADING]
        return [(self.feasible(g), "steelblue", f"{self.symbol} ≤ 0")]

    # ------------------------------------------------------------------
    # CRITERION TABLES
    # ------------------------------------------------------------------
    def table(self, values: np.ndarray,
              count_column: str = "Nested Sampling"):
        """
        Count points within ranges of the selected criterion.

        For ``P``, points are grouped into fixed feasibility-probability
        ranges.

        For ``VaR`` and ``CVaR``, values at or below zero form the feasible
        group. Positive violations are grouped relative to the largest finite
        positive violation in the supplied population.

        The returned values are point counts, not design-space volume
        estimates.
        """
        V = np.asarray(values, dtype=float)

        if self.is_probability:
            # Fixed probability ranges used in the reliability table.
            bins = [(0.95, 1.00), (0.70, 0.95), (0.50, 0.70),
                    (0.25, 0.50), (0.00, 0.25)]
            labels = ["0.95 ≤ P", "0.70 ≤ P < 0.95", "0.50 ≤ P < 0.70",
                      "0.25 ≤ P < 0.50", "P < 0.25"]
            counts = []
            for (lo, hi), _lab in zip(bins, labels):
                if hi == 1.00:
                    mask = V >= lo
                elif lo == 0.00:
                    mask = V < hi
                else:
                    mask = (V >= lo) & (V < hi)
                counts.append(int(np.sum(mask)))
            table = pd.DataFrame({"Reliability range": labels,
                                  count_column: counts})
            table.loc[len(table)] = ["Total", len(V)]
            return table

        sym = self.symbol

        # Largest finite positive violation in the supplied population.
        pos = V[np.isfinite(V) & (V > 0.0)]
        vmx = float(pos.max()) if pos.size else 1.0

        labels = [
            f"     {sym} ≤ 0    (feasible)",
            f"0 < {sym}/{sym}_max ≤ 0.25",
            f"0.25 < {sym}/{sym}_max ≤ 0.50",
            f"0.50 < {sym}/{sym}_max ≤ 0.75",
            f"{sym}/{sym}_max > 0.75",
        ]

        # Express each criterion value relative to the largest finite
        # positive violation in the supplied population.
        with np.errstate(divide="ignore", invalid="ignore"):
            u = V / vmx

        # Treat undefined values as belonging to the worst range.
        u = np.where(np.isnan(u), np.inf, u)

        counts = [
            int(np.sum(V <= 0.0)),
            int(np.sum((u > 0.0) & (u <= 0.25))),
            int(np.sum((u > 0.25) & (u <= 0.50))),
            int(np.sum((u > 0.50) & (u <= 0.75))),
            int(np.sum(u > 0.75)),         
        ]

        table = pd.DataFrame({f"{sym} range ({sym}_max = {vmx:.4g})": labels,
                              count_column: counts})

        table.loc[len(table)] = ["Total", int(V.size)]
        return table


# ============================================================
# SECTION 7 — ELLIPSOID
# ============================================================

@dataclass
class Ellipsoid:
    """
    Represent an enlarged ellipsoid fitted to a set of points.

    The ellipsoid is defined as

    ``(x - mu).T @ inv(kfac * eff * C) @ (x - mu) <= 1``.

    The enlargement is stored as two factors. ``kfac`` enlarges the
    covariance ellipsoid enough to contain its assigned points, while
    ``eff`` applies any additional enlargement required to satisfy a
    target-volume floor. Their product is available through ``f``.

    Parameters
    ----------
    mu : np.ndarray
        Ellipsoid centre, with shape ``(n_dimensions,)``.
    C : np.ndarray
        Raw covariance matrix defining the ellipsoid shape, with shape
        ``(n_dimensions, n_dimensions)``.
    kfac : float
        Point-enclosing enlargement factor.
    eff : float
        Additional volume-floor enlargement factor.
    V : float
        Volume of the enlarged ellipsoid.
    points : np.ndarray
        Points assigned to the ellipsoid, with shape
        ``(n_points, n_dimensions)``.
    V_raw : float, optional
        Ellipsoid volume before applying ``kfac`` and ``eff``.
    L : np.ndarray, optional
        Matrix square root of the regularised covariance, satisfying
        ``L @ L.T = C_reg``.
    A_inv : np.ndarray, optional
        Inverse enlarged-shape matrix used for containment and
        Mahalanobis-distance calculations.
    C_inv : np.ndarray, optional
        Inverse regularised covariance before enlargement.
    det_C : float, optional
        Determinant of the regularised covariance before enlargement.
    rows : np.ndarray, optional
        Row indices of ``points`` in the parent live-point array, stored
        in the same order as ``points``.
    """

    mu:     np.ndarray
    C:      np.ndarray
    kfac:   float
    eff:    float
    V:      float
    points: np.ndarray

    # Geometry calculated during ellipsoid fitting.
    V_raw:  float       = None
    L:      np.ndarray  = None
    A_inv:  np.ndarray  = None
    C_inv:  np.ndarray  = None
    det_C:  float       = None

    # Original positions of the ellipsoid points in the mode's full
    # live-point array. `rows[i]` identifies the live point stored as
    # `points[i]`, allowing that point to be removed or updated correctly.
    rows:   np.ndarray  = None

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def f(self) -> float:
        """
        Return the combined ellipsoid enlargement factor.

        Returns
        -------
        float
            Product of the point-enclosing and volume-floor factors,
            ``kfac * eff``.
        """
        return self.kfac * self.eff

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls, points: np.ndarray, true_volume: float, e: float = 1,
    ) -> "Ellipsoid":
        """
        Fit and enlarge an ellipsoid around a set of points.

        The sample covariance determines the ellipsoid orientation and
        relative axis lengths. Its eigenvalues are regularised before the
        inverse, matrix square root, determinant, and volume are calculated.

        The factor ``kfac`` expands the ellipsoid sufficiently to contain all
        assigned points. The factor ``eff`` applies any additional enlargement
        required to reach the minimum volume ``true_volume / e``.

        Parameters
        ----------
        points : np.ndarray
            Points used to fit the ellipsoid, with shape
            ``(n_points, n_dimensions)``.
        true_volume : float
            Reference volume assigned to the point set.
        e : float, optional
            Positive divisor used to define the minimum ellipsoid volume as
            ``true_volume / e``.

        Returns
        -------
        Ellipsoid
            Fitted ellipsoid containing the supplied points and satisfying the
            requested minimum-volume condition.

        Notes
        -----
        ``C`` stores the raw sample covariance. ``C_inv``, ``L``, ``det_C``,
        and the ellipsoid volume are calculated from its regularised
        eigenvalues.
        """
        N, D = points.shape
        mu   = np.mean(points, axis=0)

        # A one-point set has no estimable covariance. Return a degenerate
        # zero-volume ellipsoid instead. [CalcEllProp (L346-352)]
        if N <= 1:
            zeros = np.zeros((D, D))
            _step(
                "Ellipsoid.fit — degenerate (npt<=1)",
                N_points=N, D=D,
            )
            return cls(mu=mu, C=zeros, kfac=0.0, eff=1.0, V=0.0,
                       points=points, V_raw=0.0, L=zeros,
                       A_inv=zeros, C_inv=zeros, det_C=0.0)

        # Estimate the raw covariance using the unbiased N - 1 denominator. 
        # [calc_covmat (utils1.f90 L93-114)]
        C = np.cov(points.T, bias=False)
        if C.ndim == 0:
            C = C.reshape(1, 1)

        # Obtain eigenvalues and eigenvectors of the symmetric covariance.
        # np.linalg.eigh returns the eigenvalues in ascending order.
        # [Diagonalize (utils1.f90 L17-74)]
        eigval, evec = np.linalg.eigh(C)

        # Replace an unusable eigendecomposition (NaN/inf) with the identity geometry.
        # [Diagonalize, L62-72]
        broken_eig = bool(np.any(np.isnan(eigval)) or np.any(eigval > 1e300))
        if broken_eig:
            eigval = np.ones(D)
            evec   = np.eye(D)

        # Replace non-positive eigenvalues using the next larger direction. 
        # [CalcEllProp, L362-364]
        for i in range(D - 1):
            if eigval[i] <= 0.0:
                eigval[: i + 1] = eigval[i + 1] / 2.0

        # When there are too few points to determine every direction, assign
        # the unconstrained directions the smallest constrained eigenvalue.
        # [CalcEllProp, L366-370]
        underdetermined = N < D + 1
        if underdetermined:
            k = D + 1 - N
            eigval[:k] = eigval[k]

        # Construct geometric quantities from the regularised eigenvalues
        det_C = float(np.prod(eigval))
        C_inv = evec @ np.diag(1.0 / eigval) @ evec.T
        L     = evec @ np.diag(np.sqrt(eigval))   

        # Volume of the covariance ellipsoid before applying enlargement.
        unit_ball = (np.pi ** (D / 2.0)) / gamma(D / 2.0 + 1.0)
        V_raw     = unit_ball * np.sqrt(max(det_C, 1e-300))

        # Squared Mahalanobis distance of every point from the centre under
        # the regularised, unenlarged covariance.
        diff  = points - mu
        mahal = np.einsum("ij,jk,ik->i", diff, C_inv, diff)

        # Enlarge the ellipsoid enough to contain the most distant point.
        kfac = max(float(np.max(mahal)), 1e-300)

        # [CalcEllProp L378-390]
        # Calculate the volume after point-enclosing enlargement.
        vol_at_kfac = V_raw * (kfac ** (D / 2.0))

        # Apply an additional factor only when the current volume is below
        # the required minimum volume.
        target      = true_volume / e
        if target > 0.0 and target > vol_at_kfac:
            eff = (target / vol_at_kfac) ** (2.0 / D)
            V   = target
        else:
            eff = 1.0
            V   = vol_at_kfac

        A_inv = C_inv / (kfac * eff)

        _step(
            "Ellipsoid.fit",
            N_points=N, D=D,
            mu=mu, C_raw=C,
            eigenvalues_regularized=eigval,
            eigen_decomposition_was_broken_nan_inf=broken_eig,
            fewer_points_than_D_plus_1=underdetermined,
            det_C=det_C, V_raw=V_raw,
            mahalanobis_per_point=mahal,
            kfac=kfac, eff=eff,
            true_volume=true_volume,
            V_final=V,
        )

        return cls(mu=mu, C=C, kfac=kfac, eff=eff, V=V, points=points,
                  V_raw=V_raw, L=L, A_inv=A_inv, C_inv=C_inv, det_C=det_C)

    # ------------------------------------------------------------------
    # Geometry and sampling
    # ------------------------------------------------------------------
    def contains(self, d: np.ndarray) -> bool:
        """
        Check whether a point lies inside the enlarged ellipsoid.

        Parameters
        ----------
        d : np.ndarray
            Point to test, with shape ``(n_dimensions,)``.

        Returns
        -------
        bool
            ``True`` when the point satisfies the ellipsoid equation,
            otherwise ``False``.
        """
        diff  = d - self.mu
        A_inv = self.A_inv if self.A_inv is not None else np.linalg.inv(self.f * self.C)
        return float(diff @ A_inv @ diff) <= 1.0


    def sample(self) -> np.ndarray:
        """
        Draw one point uniformly from the enlarged ellipsoid.

        A random direction and radius are first sampled uniformly from the
        unit ball. The unit-ball point is then stretched and rotated according
        to the covariance matrix, enlarged by ``sqrt(kfac * eff)``, and shifted
        to the ellipsoid centre ``mu``.

        Returns
        -------
        np.ndarray
            Sampled point with shape ``(n_dimensions,)``.

        [``genPtInEll`` (utils1.f90 L529-547), ``genPtInSpheroid`` (L277-293)]
        """
        D = len(self.mu)

        # Draw a uniformly distributed direction on the unit sphere.
        z = np.random.normal(0, 1, size=D)
        z = z / np.linalg.norm(z)

        # Draw the radius required for uniform sampling inside a D-ball.
        r = np.random.uniform(0, 1) ** (1.0 / D)

        # Obtain a matrix factor satisfying L @ L.T = C.
        L = self.L if self.L is not None else np.linalg.cholesky(self.C)

        # Transform the unit-ball sample into the enlarged ellipsoid.
        return self.mu + np.sqrt(self.f) * L @ (r * z)

    # ------------------------------------------------------------------
    # Incremental updates
    # ------------------------------------------------------------------
    def evolve_on_reject(
        self, evicted_point: np.ndarray, remaining_points: np.ndarray,
        target_volume: float, remaining_rows: np.ndarray = None,
    ) -> "Ellipsoid":
        """
        Update the ellipsoid after one assigned point is removed.

        The ellipsoid centre and covariance geometry are kept unchanged.
        Only the point-enclosing factor ``kfac``, volume-floor factor
        ``eff``, and volume ``V`` may be updated.

        If the removed point defined the ellipsoid boundary, ``kfac`` is
        recomputed from the remaining points. Otherwise, the existing
        enlargement factors are retained. The volume floor is then applied
        using ``target_volume``.

        Parameters
        ----------
        evicted_point : np.ndarray
            Removed point, with shape ``(n_dimensions,)``.
        remaining_points : np.ndarray
            Points still assigned to the ellipsoid, with shape
            ``(n_remaining, n_dimensions)``.
        target_volume : float
            Minimum required volume after the point is removed.
        remaining_rows : np.ndarray, optional
            Parent live-point row indices corresponding to
            ``remaining_points``.

        Returns
        -------
        Ellipsoid
            Updated ellipsoid containing the remaining points.

        [Faithful port of ``evolveEll(a_r=0, ...)`` (utils1.f90 L576-681,
        rejection branch L650-679), called from the main sampling loop at
        nested.F90 L2270-2271 every time a live point is evicted.]
        """
        D = len(self.mu)

        # A zero target volume or degenerate covariance gives a zero-volume
        # ellipsoid. [evolveEll top guard (utils1.f90 L605-610)]
        if target_volume == 0.0 or self.V_raw == 0.0:
            return Ellipsoid(mu=self.mu, C=self.C, kfac=0.0, eff=1.0, V=0.0,
                              points=remaining_points, V_raw=self.V_raw,
                              L=self.L, C_inv=self.C_inv, det_C=self.det_C,
                              A_inv=self.A_inv, rows=remaining_rows)

        # No remaining assigned points also gives a degenerate ellipsoid.
        # [evolveEll L651-656: npt==0]
        if remaining_points.shape[0] == 0:
            return Ellipsoid(mu=self.mu, C=self.C, kfac=0.0, eff=1.0, V=0.0,
                              points=remaining_points, V_raw=self.V_raw,
                              L=self.L, C_inv=self.C_inv, det_C=self.det_C,
                              A_inv=self.A_inv, rows=remaining_rows)

        # Squared Mahalanobis distance of the removed point under the
        # unenlarged regularised covariance.
        diff_evicted = evicted_point - self.mu
        new_kfac_candidate = float(diff_evicted @ self.C_inv @ diff_evicted)

        # Recompute kfac only when the removed point defined the previous
        # boundary.
        if np.isclose(new_kfac_candidate, self.kfac, rtol=1e-8, atol=1e-12):
            diff_rem = remaining_points - self.mu
            mahal_rem = np.einsum("ij,jk,ik->i", diff_rem, self.C_inv, diff_rem)
            kfac = max(float(np.max(mahal_rem)), 1e-300)
            eff  = 1.0
            V    = self.V_raw * (kfac ** (D / 2.0))
        else:
            # Removing an interior point does not change the ellipsoid
            # boundary, so retain the existing geometry.
            kfac, eff, V = self.kfac, self.eff, self.V

        # Re-apply the volume floor against the (now-shrunk) target volume 
        # for this cluster. [evolveEll L674-678]
        if V < target_volume:
            eff = (target_volume / max(V, 1e-300)) ** (2.0 / D)
            V   = target_volume

        # Inverse of the updated enlarged shape matrix.
        A_inv = self.C_inv / (kfac * eff) if kfac * eff > 0 else self.A_inv
        return Ellipsoid(mu=self.mu, C=self.C, kfac=kfac, eff=eff, V=V,
                          points=remaining_points, V_raw=self.V_raw, L=self.L,
                          A_inv=A_inv, C_inv=self.C_inv, det_C=self.det_C,
                          rows=remaining_rows)


    def evolve_on_insert(
        self, new_point: np.ndarray, updated_points: np.ndarray,
        target_volume: float, updated_rows: np.ndarray = None,
    ) -> "Ellipsoid":
        """
        Update the ellipsoid after inserting a replacement live point.

        The centre and covariance geometry remain unchanged. Although the new
        point was sampled from the ellipsoid, it may lie outside the region
        defined by ``kfac`` alone because the full sampling boundary is defined
        by ``kfac * eff``. The ellipsoid may also have shrunk after the evicted
        point was removed.

        If necessary, ``kfac`` is increased to enclose the new point. Existing
        enlargement stored in ``eff`` is reduced first so that the total
        enlargement ``kfac * eff`` remains unchanged where possible. The
        minimum-volume requirement is then reapplied using ``target_volume``.

        Parameters
        ----------
        new_point : np.ndarray
            Inserted live point, with shape ``(n_dimensions,)``.
        updated_points : np.ndarray
            Points assigned to the ellipsoid after insertion, with shape
            ``(n_updated, n_dimensions)``.
        target_volume : float
            Minimum required ellipsoid volume after insertion.
        updated_rows : np.ndarray, optional
            Parent live-point row indices corresponding to
            ``updated_points``.

        Returns
        -------
        Ellipsoid
            Updated ellipsoid containing the inserted point.

        [Faithful port of ``evolveEll(a_r=1, ...)`` (utils1.f90 L614-649),
        called from nested.F90 L2291. As with ``evolve_on_reject``,
        mean/covariance are NOT refit -- only kfac/eff/V move.]
        """
        D = len(self.mu)

        # A zero target volume or degenerate covariance gives a zero-volume
        # ellipsoid. [evolveEll top guard (utils1.f90 L605-610)]
        if target_volume == 0.0 or self.V_raw == 0.0:
            return Ellipsoid(mu=self.mu, C=self.C, kfac=0.0, eff=1.0, V=0.0,
                              points=updated_points, V_raw=self.V_raw,
                              L=self.L, C_inv=self.C_inv, det_C=self.det_C,
                              A_inv=self.A_inv, rows=updated_rows)

        diff_new = new_point - self.mu
        # kfac required by the new point.
        new_kfac_candidate = float(diff_new @ self.C_inv @ diff_new)

        kfac, eff, V = self.kfac, self.eff, self.V

        # Increase kfac only when the new point lies beyond the current
        # ellipsoid boundary. [evolveEll L621-629]
        if new_kfac_candidate > kfac:
            # Reduce eff as kfac increases so that their product, and hence
            # the ellipsoid volume, remains unchanged where possible.
            eff  = eff * kfac / new_kfac_candidate
            kfac = new_kfac_candidate

            # eff cannot be smaller than one. If preserving the previous
            # volume would require eff < 1, use the volume defined by kfac.
            if eff < 1.0:
                eff = 1.0
                V   = self.V_raw * (kfac ** (D / 2.0))

        # Re-apply the volume floor. [evolveEll L633-648]
        # Increase eff when the current volume is below the required minimum.
        if target_volume > V:
            eff = eff * (target_volume / max(V, 1e-300)) ** (2.0 / D)
            V   = target_volume
        else:
            # Check whether kfac alone satisfies the minimum-volume requirement.
            V = self.V_raw * (kfac ** (D / 2.0))
            if V < target_volume:
                eff = max(1.0, (target_volume / max(V, 1e-300)) ** (2.0 / D))
                V   = target_volume
            else:
                eff = 1.0

        # Inverse of the updated shape matrix.
        A_inv = self.C_inv / (kfac * eff) if kfac * eff > 0 else self.A_inv
        return Ellipsoid(mu=self.mu, C=self.C, kfac=kfac, eff=eff, V=V,
                          points=updated_points, V_raw=self.V_raw, L=self.L,
                          A_inv=A_inv, C_inv=self.C_inv, det_C=self.det_C,
                          rows=updated_rows)


    def decay_untouched(self, shrink: float) -> "Ellipsoid":
        """
        Shrink an ellipsoid not involved in the current point replacement.

        Because no point is removed from or inserted into this ellipsoid,
        its centre, covariance geometry, assigned points, and ``kfac`` remain
        unchanged. Only the additional enlargement ``eff`` is reduced.

        The reduction is chosen so that, unless ``eff`` reaches its lower
        bound of one, the ellipsoid volume is multiplied by ``shrink``.
        Decay stops at ``eff = 1``, preventing the ellipsoid from becoming
        smaller than the region required to contain its assigned points.

        Parameters
        ----------
        shrink : float
            Per-iteration prior-volume shrink factor, usually
            ``exp(-1 / n_live)``.

        Returns
        -------
        Ellipsoid
            Ellipsoid with reduced ``eff`` and volume, or the current object
            when no further decay is possible.

        # The reference checks kfac of the insertion ellipsoid here. This is
        # likely an i/j indexing error, since the decay is applied to the
        # untouched ellipsoid. This implementation therefore checks its own
        # kfac instead.
        [nested.F90 L2313-2317]
        """
        D = len(self.mu)
        if self.eff > 1.0 and self.V > 0.0 and self.kfac > 0.0:
            old_eff = self.eff
            # For new_eff > 1, this gives new_V = old_V * shrink. If new_eff is
            # clipped to 1, the ellipsoid stops at its kfac-defined minimum size,
            # so new_V may be greater than old_V * shrink.
            new_eff = max(1.0, self.eff * (shrink ** (2.0 / D)))
            ratio   = new_eff / old_eff
            new_V   = self.V * (ratio ** (D / 2.0))
            A_inv   = self.C_inv / (self.kfac * new_eff)
            return Ellipsoid(mu=self.mu, C=self.C, kfac=self.kfac,
                              eff=new_eff, V=new_V, points=self.points,
                              V_raw=self.V_raw, L=self.L, A_inv=A_inv,
                              C_inv=self.C_inv, det_C=self.det_C,
                              rows=self.rows)
        return self


# ============================================================
# SECTION 8 — ELLIPSOIDAL DECOMPOSITION  (Algorithm 1)
# ============================================================

def _kmeans_seed_random(points: np.ndarray, k: int) -> np.ndarray:
    """
    Select distinct input points as initial k-means centroids.

    The centroids are sampled uniformly without replacement, matching the
    initialization rule used by the reference MultiNest ``kmeans3``
    routine. [kmeans3 L259-279]

    Parameters
    ----------
    points : np.ndarray
        Input points with shape ``(n_points, n_dimensions)``.
    k : int
        Number of initial centroids to select.

    Returns
    -------
    np.ndarray
        Selected centroids with shape ``(k, n_dimensions)``.
    """
    # Uniformly choose k distinct input points as the initial centroids for
    # Lloyd's algorithm.
    return points[np.random.choice(points.shape[0], size=k,
                                   replace=False)].astype(float).copy()


def _kmeans_seed_pp(points: np.ndarray, k: int) -> np.ndarray:
    """
    Select initial k-means centroids using k-means++.
    (Arthur & Vassilvitskii 2007)

    The first centroid is selected uniformly from the input points. Each
    subsequent centroid is sampled with probability proportional to its
    squared distance from the nearest centroid already selected. This
    favours points that are far from the existing centroids.

    Parameters
    ----------
    points : np.ndarray
        Input points with shape ``(n_points, n_dimensions)``.
    k : int
        Number of initial centroids to select.

    Returns
    -------
    np.ndarray
        Initial centroids with shape ``(k, n_dimensions)``.

    Notes
    -----
    K-means++ is an optional extension and is not used by the reference
    MultiNest implementation.
    """
    N = points.shape[0]
     
    # Select the first centroid uniformly from the input points.
    centres = [points[np.random.randint(N)].astype(float)]

    for _ in range(1, k):
        # For each point, compute its squared distance to the nearest
        # centroid selected so far.
        d2 = np.min(np.array([np.sum((points - c) ** 2, axis=1)
                              for c in centres]), axis=0)
        
        total_distance  = float(d2.sum())
        
        # Fall back to uniform selection when the distance-based
        # probabilities cannot be formed.
        if not np.isfinite(total_distance) or total_distance  <= 0.0:
            centres.append(points[np.random.randint(N)].astype(float))
        else:
            # Points farther from the existing centroids receive a higher
            # probability of being selected.
            probabilities = d2 / total_distance
            centres.append(points[np.random.choice(N, p=probabilities)]
                           .astype(float))
    return np.array(centres)


def _kmeans_lloyd(points: np.ndarray, means: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run Lloyd's k-means algorithm until cluster assignments stop changing.

    Each iteration assigns every point to its nearest centroid using
    squared Euclidean distance, then recomputes each non-empty centroid
    as the mean of its assigned points. The procedure has no explicit
    iteration limit, matching the reference MultiNest implementation.
    [kmeans3 L284-362]

    Parameters
    ----------
    points : np.ndarray
        Input points with shape ``(n_points, n_dimensions)``.
    means : np.ndarray
        Initial centroids with shape ``(n_clusters, n_dimensions)``.
        The array is updated in place during the iterations.

    Returns
    -------
    labels : np.ndarray
        Cluster label of each point, with shape ``(n_points,)``.
    means : np.ndarray
        Final cluster centroids with shape
        ``(n_clusters, n_dimensions)``.
    wcss : float
        Within-cluster sum of squared distances from each point to its
        assigned centroid.
    """
    k = means.shape[0]
    
    # Store the current and previous cluster assignments. Initialising
    # old_cluster to -1 ensures that at least one iteration is performed.
    cluster     = np.zeros(points.shape[0], dtype=int)
    old_cluster = np.full(points.shape[0], -1, dtype=int)

    while True:
        # Compute the squared Euclidean distance from every point to every
        # centroid. The resulting array has shape (n_points, n_clusters).
        dists   = np.sum((points[:, None, :] - means[None, :, :]) ** 2, axis=2)
        
        # Assign each point to its nearest centroid.
        cluster = np.argmin(dists, axis=1)

        # Stop when the cluster assignments no longer change.
        if np.array_equal(cluster, old_cluster):
            break
        old_cluster = cluster.copy()

        # Recompute each non-empty centroid from its assigned points.
        for i in range(k):
            if np.sum(cluster == i) == 0:
                continue
            means[i] = points[cluster == i].mean(axis=0)
    
    # Compute the within-cluster sum of squares for the converged
    # clustering.
    w = float(sum(np.sum((points[cluster == i] - means[i]) ** 2)
                  for i in range(k) if np.any(cluster == i)))
    return cluster, means, w


def _kmeans3(points: np.ndarray, min_pt: int,
             restarts: int = 1, init: str = "random") -> np.ndarray:
    """
   Partition the input points into two clusters using k-means.

    Each run initializes two centroids, applies Lloyd's algorithm until
    convergence, and records the within-cluster sum of squares. When
    multiple restarts are requested, the labelling with the lowest
    within-cluster sum of squares is retained.

    After selecting the best labelling, any cluster containing fewer than
    ``min_pt`` points is expanded by transferring the nearest available
    points from the other cluster. This minimum-size correction is applied
    once, following the original MultiNest ``kmeans3`` routine.

    Parameters
    ----------
    points : np.ndarray
        Input points with shape ``(n_points, n_dimensions)``.
    min_pt : int
        Minimum number of points required in each cluster.
    restarts : int, optional
        Number of independent centroid initializations. A value of one
        reproduces the single-run behaviour of the reference implementation.
    init : {"random", "kmeans++"}, optional
        Centroid initialization method. ``"random"`` selects two distinct
        input points uniformly, while ``"kmeans++"`` uses distance-weighted
        initialization.

    Returns
    -------
    np.ndarray
        Integer cluster labels with shape ``(n_points,)``. Each label is
        either ``0`` or ``1``.

    Notes
    -----
    The reference MultiNest implementation uses one random initialization
    and no restarts. The ``restarts`` and ``"kmeans++"`` options are
    extensions to that behaviour.
    """
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}")
    if init not in ("random", "kmeans++"):
        raise ValueError(
            f"init must be 'random' or 'kmeans++', got {init!r}")

    k = 2

    # Select the requested centroid-initialization rule.
    seed_fn = _kmeans_seed_random if init == "random" else _kmeans_seed_pp

    # Run k-means several times and retain the solution with the smallest
    # within-cluster sum of squares.
    best_w, cluster, means = np.inf, None, None
    for _ in range(restarts):
        initial_means = seed_fn(points, k)
        cl, mu, w = _kmeans_lloyd(points, initial_means)
        if w < best_w:
            best_w, cluster, means = w, cl, mu
        
    # Count how many points are currently assigned to each cluster
    cluster_sizes = np.array([np.sum(cluster == i) for i in range(k)])

    # Enforce the minimum cluster size once. An undersized cluster receives
    # the nearest points that the other cluster can give up while still
    # retaining at least min_pt points. [kmeans3 L307-341]
    for i in range(k):
        if cluster_sizes[i] >= min_pt:
            continue

        shortfall = min_pt - cluster_sizes[i]
        other = 1 - i

        # The donor cluster cannot give away points when it is already at
        # the minimum permitted size.
        if cluster_sizes[other] <= min_pt:
            continue
        donor_idx = np.where(cluster == other)[0]
        n_steal = min(shortfall, cluster_sizes[other] - min_pt)
        
        # Rank donor points by squared Euclidean distance to the centroid
        # of the undersized cluster.
        dists_to_needy = np.sum((points[donor_idx] - means[i]) ** 2, axis=1)
        steal_idx = donor_idx[np.argsort(dists_to_needy)[:n_steal]]

        # Transfer the selected points to the undersized cluster.
        cluster[steal_idx] = i
        cluster_sizes[i] += n_steal
        cluster_sizes[other] -= n_steal

    # Recompute the centroids after the minimum-size correction to preserve
    # the reference routine's final state, although only the labels are returned.
    for i in range(k):
        if np.sum(cluster == i) > 0:
            means[i] = points[cluster == i].mean(axis=0)

    return cluster


def _anderson_darling(pt: np.ndarray, delta_mean: np.ndarray) -> bool:
    """
    Test whether candidate sub-clusters can still be represented as one
    Gaussian cluster.

    Based on MultiNest's ``AndersonDarling`` routine
    (xmeans_clstr.f90 L1482-1548). The points are projected onto the
    direction connecting the two candidate cluster means, standardized,
    and tested for normality using the Anderson-Darling statistic.

    If the projected data are consistent with a normal distribution,
    they can still be treated as one Gaussian cluster and the function
    returns ``True``. If normality is rejected, the data show evidence
    of separate structure and the function returns ``False``.

    The reference critical value ``1.8692`` corresponds to
    ``alpha = 0.0001``.
    """
    n_points = pt.shape[0]

    # Project each multidimensional point onto the direction connecting
    # the two candidate cluster centres. This reduces the clustering
    # structure to a one-dimensional set of projected values.
    direction_norm = float(np.sqrt(np.sum(delta_mean ** 2)))
    projections = (pt @ delta_mean) / direction_norm

    # Standardize the one-dimensional projections so that their mean is
    # approximately zero and their spread is normalized before comparing
    # them with a standard normal distribution.
    projection_mean = float(np.mean(projections))
    # Compute the spread of the projected values using the correction
    # adopted in the reference implementation (xmeans_clstr.f90 L1507).
    projection_std = float(np.sqrt(max(np.mean(projections ** 2) - projection_mean ** 2, 0.0))
                           * n_points / (n_points - 1.0))
    standardized_projections = (projections - projection_mean) / projection_std

    # Anderson-Darling is evaluated using the ordered observations
    standardized_projections = np.sort(standardized_projections)

    # Reverse the ordered values for the upper-tail contribution to the
    # Anderson-Darling statistic.
    reversed_projections = standardized_projections[::-1]

    # Evaluate the standard-normal cumulative distribution function
    # Phi(z) for each standardized projection. Values outside [-5, 5]
    # are approximated directly by 0 or 1, following the reference code
    normal_cdf = np.where(standardized_projections > 5.0, 1.0,
                          np.where(standardized_projections < -5.0, 0.0,
                                   _norm_cdf(standardized_projections)))
    reversed_normal_cdf = np.where(reversed_projections > 5.0, 1.0,
                                   np.where(reversed_projections < -5.0, 0.0,
                                            _norm_cdf(reversed_projections)))

    # Compute the Anderson-Darling normality statistic.
    # Larger values indicate a stronger departure from a single
    # Gaussian distribution.
    ranks = np.arange(1, n_points + 1)
    ad_statistic = np.sum((2.0 * ranks - 1.0)
                          * (np.log(normal_cdf) + np.log(1.0 - reversed_normal_cdf)))
    ad_statistic = ad_statistic / (-1.0 * n_points) - n_points
    # Apply the finite-sample correction used in the reference code.
    ad_statistic = ad_statistic * (1.0 + 4.0 / n_points - 25.0 / (n_points ** 2))

    # A statistic above 1.8692 rejects normality:
    #   True  -> normality is not rejected -> keep one G-means cluster.
    #   False -> normality is rejected     -> allow the proposed split.
    return not (ad_statistic > 1.8692)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the standard normal cumulative distribution function.
    Returns, for each value in ``x``, the probability that a standard
    normal random variable is less than or equal to that value. This
    corresponds to MultiNest's ``stNormalCDF`` routine.
    """
    from scipy.special import ndtr
    # ndtr(x) evaluates Phi(x), the CDF of a standard normal N(0, 1).
    return ndtr(x)


def _gmeans(points: np.ndarray, min_pt: int, max_clusters: int) -> List[np.ndarray]:
    """
    Partition points into candidate clusters using G-means.

    Based on MultiNest's ``Gmeans`` routine
    (xmeans_clstr.f90 L1142-1219), called through ``doGmeans``
    (L285-324). The recursive splitting procedure is implemented in
    ``_gmeans_recurse``.

    Returns
    -------
    list of np.ndarray
        Arrays containing the row indices of ``points`` belonging to
        each final G-means cluster.
    """
    N = points.shape[0]
    leaves: List[np.ndarray] = []
    # Start from all points as one cluster and recursively split it.
    _gmeans_recurse(points, np.arange(N), min_pt, max_clusters, [1], leaves)

    # ``leaves`` stores the final partition produced by G-means.
    # Each element is an array of row indices into ``points`` identifying
    # all points that belong to one final cluster.
    return leaves


def _gmeans_recurse(
    points: np.ndarray, idx: np.ndarray, min_pt: int, max_clusters: int,
    n_cls_counter: List[int], leaves: List[np.ndarray],
) -> None:
    """
    Test recursively whether one G-means cluster should be split.

    The current cluster is first divided into two using ``_kmeans3``.
    The proposed split is accepted only if both child clusters contain
    at least ``min_pt`` points and the Anderson-Darling test rejects
    normality after projecting the points along the direction between
    the two child-cluster means.  In other words, the projected data must
    not look like a single unimodal Gaussian cluster, providing evidence
    that the two-way split represents genuinely separate structure.

    If the split is accepted, the same procedure is applied recursively
    to both children. Otherwise, the current cluster is stored as a
    final G-means cluster.
    """
    N = idx.shape[0]

    # ``flag=True`` means: stop splitting this cluster.
    # A split is impossible if there are fewer than ``2 * min_pt``
    # points, because two valid children could not both contain
    # ``min_pt`` points. Splitting also stops once ``max_clusters`` has
    # been reached (Gmeans, xmeans_clstr.f90 L1169).
    flag = (N < 2 * min_pt) or (n_cls_counter[0] == max_clusters)

    if not flag:
        # Try to divide the current cluster into two using the same
        # k-means routine used elsewhere in the ellipsoidal decomposition.
        labels = _kmeans3(points[idx], min_pt)
        n0 = int(np.sum(labels == 0))
        n1 = N - n0
        # Reject the split if either child contains too few points
        if n0 < min_pt or n1 < min_pt:
            flag = True

    if not flag:
        # Convert the local k-means labels back to indices in ``points``.
        idx0, idx1 = idx[labels == 0], idx[labels == 1]

        # Direction joining the two child-cluster centres.
        mean0 = points[idx0].mean(axis=0)
        mean1 = points[idx1].mean(axis=0)
        delta_mean = mean0 - mean1

        # Test whether the points projected along this direction are
        # consistent with a single Gaussian cluster.
        # ``_anderson_darling`` returns True when normality is NOT
        # rejected, so ``flag=True`` means that the proposed split
        # should be rejected.
        flag = _anderson_darling(points[idx], delta_mean)

    if not flag:
        # The split is accepted. One cluster has become two, so the
        # total cluster count increases by one.
        n_cls_counter[0] += 1
        # Test each accepted child independently for further splitting.
        _gmeans_recurse(points, idx0, min_pt, max_clusters, n_cls_counter, leaves)
        _gmeans_recurse(points, idx1, min_pt, max_clusters, n_cls_counter, leaves)
    else:
        # The cluster cannot or should not be split further, so store
        # its point indices as one final G-means cluster.
        leaves.append(idx)


def _ellipsoids_overlap(e1: "Ellipsoid", e2: "Ellipsoid") -> bool:
    """
    Test whether two enlarged ellipsoids overlap.

    Based on MultiNest's ``ellIntersect`` routine
    (nested.F90 L2640-2718). The two ellipsoids are written in quadratic
    form and compared using an eigenvalue-based separation test.

    For mode isolation, each point-enclosing ellipsoid is enlarged by a
    factor of ``1.5 * kfac`` (nested.F90 L3043), rather than using the
    usual sampling enlargement ``kfac * eff``.

    The function returns ``False`` when the eigenvalue test identifies
    the ellipsoids as disjoint, and ``True`` otherwise.
    """
    D = e1.mu.shape[0]

    # Express the first ellipsoid in its principal-axis basis. The
    # reference implementation receives these as arguments
    # [nested.F90 L2645], supplied by the caller [nested.F90 L3041, L3045].
    eval1, evec1 = np.linalg.eigh(e1.C)

    # Mode-isolation enlargement used by the reference implementation.
    # The point-count-dependent factor on L3042 is immediately overwritten
    # by the fixed 1.5 on L3043. [nested.F90 L3042-3043]
    ef1 = e1.kfac * 1.5
    ef2 = e2.kfac * 1.5

    # Handle degenerate ellipsoids with zero enlargement separately.
    # [nested.F90 L2657-2666] The two point-in-ellipsoid checks below
    # follow MultiNest's ``ptIn1Ell`` routine [utils1.f90 L1036-1056].
    if ef1 == 0.0 and ef2 == 0.0:
        return False
    if ef1 == 0.0:
        # Treat e1 as a point and check whether its centre lies inside e2.
        diff = e1.mu - e2.mu
        return bool(diff @ (e2.C_inv / ef2) @ diff <= 1.0)
    if ef2 == 0.0:
        # Treat e2 as a point and check whether its centre lies inside e1.
        diff = e2.mu - e1.mu
        return bool(diff @ (e1.C_inv / ef1) @ diff <= 1.0)

    # Relative displacement between the two ellipsoid centres.
    # [nested.F90 L2671]
    delta_mean = e1.mu - e2.mu

    # Build the quadratic-form matrices used by the ellipsoid
    # intersection test. The additional row/column represents the
    # linear and constant terms arising from the centre displacement.
    # [nested.F90 L2668-2672]
    matA = np.zeros((D + 1, D + 1))
    matB = np.zeros((D + 1, D + 1))

    # First ellipsoid, expressed in its principal-axis representation.
    # [nested.F90 L2675, L2681]
    matA[:D, :D] = np.diag(eval1)
    matA[D, D]   = -1.0 / ef1

    # Second ellipsoid, including its position relative to the first.
    # [nested.F90 L2676-2678, L2682]
    matB[:D, :D] = e2.C_inv
    matB[D, :D]  = -delta_mean @ e2.C_inv
    matB[:D, D]  = matB[D, :D]
    matB[D, D]   = float(np.sum(matB[D, :D] * (-delta_mean))) - ef2

    # Rotate the second quadratic form into the principal-axis basis
    # of the first ellipsoid. [nested.F90 L2683-2690]
    matR = np.zeros((D + 1, D + 1))
    matR[:D, :D] = evec1.T
    matR[D, D]   = 1.0

    matB_rot  = matR @ matB @ matR.T

    # Apply the eigenvalue-based separation test used by ellIntersect.
    # The reference implementation calls LAPACK's DGEEV here.
    # [nested.F90 L2691-2697]
    matAinvB  = matA @ matB_rot
    eigvals   = np.linalg.eigvals(matAinvB)

    # Complex eigenvalues are treated as indicating overlap.
    # [nested.F90 L2701-2704]
    if np.any(np.abs(eigvals.imag) > 1e-9):
        return True

    # Two or more negative real eigenvalues indicate that the
    # ellipsoids are disjoint; otherwise they are considered overlapping.
    # [nested.F90 L2705-2716]
    k = int(np.sum(eigvals.real < 0.0))
    return k < 2


def _isolate_modes(points: np.ndarray, min_pt: int) -> List[np.ndarray]:
    """
    Identify separate sub-modes within one existing mode.

    Based on MultiNest's ``isolateModes2`` routine
    [nested.F90 L2855-3229]. G-means is first used to obtain candidate
    clusters. A bounding ellipsoid is fitted to each cluster, and
    clusters whose ellipsoids overlap are grouped together.

    A candidate group is retained as a separate mode only if it contains
    at least ``2 * (D + 1)`` points (nested.F90 L3122) and is more
    localized than the parent mode in at least one dimension
    (nested.F90 L3148-3151).

    Groups that do not satisfy these conditions remain part of the
    unsplit parent mode.

    Returns
    -------
    list of np.ndarray
        Arrays containing the indices of points assigned to each
        identified mode. If no valid split is found, a single array
        containing all point indices is returned.
    """
    # Obtain candidate clusters using G-means, capped at the maximum
    # allowed by the minimum cluster size.
    max_clusters = max(1, points.shape[0] // min_pt)
    leaves = _gmeans(points, min_pt, max_clusters)
    k = len(leaves)
    if k <= 1:
        return [np.arange(points.shape[0])]

    # Fit a bounding ellipsoid to each candidate cluster.
    bounding = [Ellipsoid.fit(points[idx], true_volume=0.0) for idx in leaves]

    # Determine which cluster ellipsoids overlap.
    overlap = np.ones((k, k), dtype=bool)
    for a in range(k):
        for b in range(a + 1, k):
            ov = _ellipsoids_overlap(bounding[a], bounding[b])
            overlap[a, b] = overlap[b, a] = ov

    # Group clusters whose bounding ellipsoids overlap
    # (grouping logic in isolateModes2, nested.F90 L2971-3080).
    # ``groups`` stores connected groups of G-means clusters.
    # Each element contains the indices of G-means clusters whose
    # bounding ellipsoids are connected through overlap.
    visited: List[bool] = [False] * k
    groups: List[List[int]] = []
    for start in range(k):
        if visited[start]:
            continue
        stack, comp = [start], []
        visited[start] = True
        while stack:
            j = stack.pop()
            comp.append(j)
            for n in range(k):
                if not visited[n] and overlap[j, n]:
                    visited[n] = True
                    stack.append(n)
        groups.append(comp)

    # No separated groups were found.
    if len(groups) <= 1:
        return [np.arange(points.shape[0])]

    # Minimum number of points required for a separate mode (nested.F90 L3122).
    D = points.shape[1]
    min_group_pts = 2 * (D + 1)  

    # Population standard deviation of the complete parent mode
    # (mStdErr in nested.F90 L3000-3007).
    parent_std = points.std(axis=0, ddof=0)

    accepted:  List[np.ndarray] = []
    leftover_parts: List[np.ndarray] = []
    for g in groups:
        # Collect all points belonging to this group of overlapping
        # G-means clusters.
        idx_g = np.concatenate([leaves[j] for j in g])

        # A group with too few points cannot form a separate mode.
        if idx_g.shape[0] < min_group_pts:
            leftover_parts.append(idx_g)
            continue

        # Population standard deviation of this candidate group
        # (lStdErr in nested.F90 L3092-3113).
        group_std  = points[idx_g].std(axis=0, ddof=0)
        localized  = bool(np.any(group_std < parent_std))
        if localized:
            accepted.append(idx_g)
        else:
            leftover_parts.append(idx_g)

    # No candidate qualified as a separate mode.
    if not accepted:
        return [np.arange(points.shape[0])]

    result = list(accepted)
    # Keep rejected candidate groups together as the remaining
    # part of the parent mode.
    if leftover_parts:
        leftover = np.concatenate(leftover_parts)
        if leftover.shape[0] > 0:
            result.append(leftover)

    # Require at least two sufficiently populated groups for a split
    if len(result) <= 1 or any(r.shape[0] < min_group_pts for r in result):
        return [np.arange(points.shape[0])]

    return result


def _delF(
    ndim: int, n1: int, n2: int,
    mdis1: float, mdis2: float,
    detcov1: float, detcov2: float,
    f1: float, f2: float,
) -> float:
    """
    Estimate whether moving one point from cluster 1 to cluster 2 improves
    the two-cluster grouping.

    The calculation follows the ``delF`` function in MultiNest's
    ``kmeans_clstr.f90``.

    Parameters
    ----------
    ndim : int
        Number of dimensions.

    n1 : int
        Number of points in cluster 1 before removing the candidate point.

    n2 : int
        Number of points in cluster 2 before adding the candidate point.

    mdis1 : float
        Mahalanobis distance of the candidate point from ellipsoid 1.

    mdis2 : float
        Mahalanobis distance of the same point from ellipsoid 2.

    detcov1 : float
        Determinant of the regularised covariance matrix of ellipsoid 1.

    detcov2 : float
        Determinant of the regularised covariance matrix of ellipsoid 2.

    f1 : float
        Total enlargement factor of ellipsoid 1

    f2 : float
        Total enlargement factor of ellipsoid 2

    Returns
    -------
    float
        Estimated change produced by moving the point from cluster 1 to
        cluster 2. A negative value means that the move improves the
        two-cluster grouping.

    References
    ----------
    MultiNest v3.12, ``kmeans_clstr.f90``, ``delF``, lines 1455–1476.
    Lu, Choi, Wang and Kim, Computer Graphics Forum, 26(3), 2007,
    as cited by the MultiNest source.
    
    """
    # Estimated change in cluster 1 after removing the point.
    term1 = (
        (((n1 / (n1 - 1.0)) ** 3) * (1.0 - mdis1 / (n1 - 1.0)) - 1.0)
        * np.sqrt(max(detcov1, 0.0) * (f1 ** ndim))
        / (((n1 / (n1 - 1.0)) ** 1.5)
           * np.sqrt(max(1.0 - mdis1 / (n1 - 1.0), 0.0)) + 1.0)
    )

    # Estimated change in cluster 2 after adding the point.
    term2 = (
        (((n2 / (n2 + 1.0)) ** 3) * (1.0 + mdis2 / (n2 + 1.0)) - 1.0)
        * np.sqrt(max(detcov2, 0.0) * (f2 ** ndim))
        / (((n2 / (n2 + 1.0)) ** 1.5)
           * np.sqrt(1.0 + mdis2 / (n2 + 1.0)) + 1.0)
    )

    # Combine the effects on both clusters.
    return (term1 + term2) / (n1 + n2)


class EllipsoidalDecomposition:
    """
    Construct a recursive multi-ellipsoidal decomposition.

    The decomposition follows Algorithm 1 of MultiNest and operates
    entirely in the unit hypercube. [``kmeans_clstr.f90``
    and ``xmeans_clstr.f90``]

    Each candidate cluster is represented by an ``Ellipsoid``. Recursive
    splitting continues while it improves the decomposition and satisfies
    the minimum cluster-size and maximum ellipsoid-count constraints.

    Clusters smaller than ``min_pt`` are prevented through the k-means
    size correction and the decomposition stopping conditions. Covariance
    singularity is handled separately by the eigenvalue regularization in
    ``Ellipsoid.fit``.

    Parameters
    ----------
    D : int
        Dimension of the point sets this decomposition will be fitted to.
        Required because the default ``min_pt`` is dimension-scaled.
    domain_bounds : tuple of float, optional
        Lower and upper sampling bounds. The default represents
        the unit hypercube.
    min_pt : int, optional
        Minimum number of points allowed in an independent cluster. When
        omitted, ``2 * (D + 1)`` is used.

        An ellipsoid in ``D`` dimensions needs at least ``D + 1`` points for a
        non-singular sample covariance, and ``2 * (D + 1)`` is the smallest
        value that leaves margin above that bound. It is also the minimum
        population ``_isolate_modes`` already requires before a group is
        tracked as a separate mode (``min_group_pts``), so the same value
        keeps the decomposition and mode separation consistent.

        The reference MultiNest implementation uses 2 and lets the eigenvalue
        regularisation in ``Ellipsoid.fit`` supply the directions the points
        do not constrain.

    max_ellipsoids : int, optional
        Maximum number of ellipsoid leaves. When omitted, the exact upper
        bound implied by ``min_pt``, ``floor(n_points / min_pt)``, is used
        during fitting.
    kmeans_restarts : int, optional
        Number of independent two-cluster k-means runs performed for each
        proposed split.
    kmeans_init : {"random", "kmeans++"}, optional
        Centroid initialization method used by k-means.
    em_mode : {"none", "multinest", "bugfix", "paper"}, optional
        EM refinement strategy applied to every proposed two-cluster
        split; see ``_em_refinement``. The default is ``"bugfix"``.

    Notes
    -----
    The combination ``kmeans_restarts=1`` and ``kmeans_init="random"``
    reproduces the clustering setup used by the reference implementation.
    """

    # ------------------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------------------
    def __init__(
        self,
        D:              int,
        domain_bounds:  Tuple[float, float] = (0.0, 1.0),
        min_pt:         Optional[int] = None,
        max_ellipsoids: Optional[int] = None,
        kmeans_restarts: int = 1,
        kmeans_init:     str = "random",
        em_mode:         str = "bugfix",
    ) -> None:
        if em_mode not in ("none", "multinest", "bugfix", "paper"):
            raise ValueError(
                "em_mode must be one of (none, multinest, bugfix, paper), "
                f"got {em_mode!r}")

        self.domain_bounds          = domain_bounds
        self.D                      = int(D)

        # D + 1 points are the minimum for a non-singular covariance in D
        # dimensions; 2*(D+1) is the smallest value leaving margin above that.
        self.min_pt                 = (2 * (self.D + 1) if min_pt is None
                                       else int(min_pt))

        self.kmeans_restarts        = kmeans_restarts
        self.kmeans_init            = kmeans_init
        self.em_mode                = em_mode
        self.max_ellipsoids         = max_ellipsoids
        self._n_leaves              = 1

        # Ellipsoid leaves produced by the current decomposition.
        self.ellipsoids: List[Ellipsoid] = []
        
        # --- DECOMPOSITION TRACING (pure observation; None = off) ----
        # When a list is attached here, _algorithm1 appends one dict per
        # recursion node (preorder), recording every number behind every
        # decision it takes. Attached/detached by the sampler around
        # _maybe_refit when trace_decomposition is active; never consumes
        # RNG, never alters control flow.
        self.trace_sink: Optional[list] = None
        self._trace_depth = 0

        # point_owner[i] stores the ellipsoid that owns live-point row i.
        # Updated by fit() and used by evolve_step() when replacing points.
        self.point_owner: np.ndarray = np.empty(0, dtype=int)

    # ------------------------------------------------------------------
    # DECOMPOSITION INTERFACE
    # ------------------------------------------------------------------
    def fit(self, points: np.ndarray, true_volume: float) -> None:
        """
         Fit a recursive ellipsoidal decomposition to the supplied points.

        Algorithm 1 recursively splits the point set into clusters and fits one
        ellipsoid to each final cluster. The method also records the ownership
        relationship between live-point rows and ellipsoids so that later
        incremental updates can identify which ellipsoid must be modified.

        Parameters
        ----------
        points : np.ndarray
            Live points in unit-hypercube coordinates, with shape
            ``(n_points, n_dimensions)``.
        true_volume : float
            Reference volume represented by the complete point set.

        Notes
        -----
        After fitting:

        - ``self.ellipsoids`` stores the final ellipsoid leaves.
        - ``self.point_owner[i]`` stores the ellipsoid that owns live-point row
        ``i``.
        - ``self.ellipsoids[eid].rows`` stores the live-point rows owned by
        ellipsoid ``eid``.
        """
        N = points.shape[0]

        # Reset the leaf count for the new decomposition.
        self._n_leaves = 1

        # Limit the number of final ellipsoids. If no explicit limit is given,
        # use the maximum allowed by the minimum cluster size.
        self._leaf_cap = (self.max_ellipsoids if self.max_ellipsoids
                           is not None else max(1, N // self.min_pt))
        
        # Reset decomposition-tracing counters.
        self._trace_depth = 0
        self._trace_node_counter = 0

        # Run Algorithm 1.
        # idx_lists[eid] contains the original row indices assigned to
        # ellipsoids[eid].
        ellipsoids, idx_lists = self._algorithm1(points, true_volume,
                                                  np.arange(N))
        self.ellipsoids = ellipsoids

        # point_owner[i] stores the ellipsoid that owns live-point row i
        owner = np.full(N, -1, dtype=int)
        for eid, idx in enumerate(idx_lists):
            owner[idx] = eid
            # Keep each ellipsoid's local point ordering aligned with its
            # corresponding live-point row indices.
            self.ellipsoids[eid].rows = np.asarray(idx, dtype=int)

        # Used by evolve_step() to locate the ellipsoid affected by a
        # live-point replacement.
        self.point_owner = owner

    def compute_F(self, true_volume: float) -> float:
        """
        Compute the ellipsoidal decomposition factor ``F(S)``.

        ``F(S)`` is the ratio between the total volume of all fitted
        ellipsoids and the reference volume represented by the point set.

        Parameters
        ----------
        true_volume : float
            Reference volume of the current point set ``S``.

        Returns
        -------
        float
            The ratio ``sum(V_i) / true_volume``. Returns infinity when
            ``true_volume`` is numerically zero.
        """
        total = sum(e.V for e in self.ellipsoids)
        return total / true_volume if true_volume > 1e-300 else np.inf

    # ------------------------------------------------------------------
    # INCREMENTAL UPDATE
    # ------------------------------------------------------------------
    def evolve_step(
        self,
        evicted_row:    int,
        evicted_point:  np.ndarray,
        new_point:      np.ndarray,
        chosen_ellipsoid_id: int,
        N_total:        int,
        X_next:         float,
    ) -> None:
        """
        Incrementally update the ellipsoidal decomposition after replacing one
        live point.

        Instead of performing a full re-decomposition, this method updates only
        the ellipsoids affected by the replacement:

        - The ellipsoid containing the evicted point is updated after removal.
        - The selected ellipsoid is updated after inserting the new point.
        - All unaffected ellipsoids apply only the standard volume-floor decay.

        The ellipsoid centres and covariance matrices are not fully refitted here.
        A complete clustering and refit are performed separately when required.

        Parameters
        ----------
        evicted_row : int
            Row index of the replaced point in the main live-point array.
        evicted_point : np.ndarray
            Point removed from the live set, with shape ``(n_dimensions,)``.
        new_point : np.ndarray
            Replacement point, with shape ``(n_dimensions,)``.
        chosen_ellipsoid_id : int
            Index of the ellipsoid from which the replacement point was sampled.
            The new point is assigned to this ellipsoid.
        N_total : int
            Total number of live points.
        X_next : float
            Estimated remaining prior volume after the replacement.

        Notes
        -----
        This follows the point-replacement update in ``nested.F90`` rather than
        refitting and re-clustering the complete live-point set after every
        iteration. [``nested.F90`` L2200-2295]
        """
        # Ellipsoid the evicted point currently belongs to.
        q = int(self.point_owner[evicted_row])          

        # Ellipsoid that will receive the replacement point.
        i = int(chosen_ellipsoid_id)

        # Standard nested-sampling shrink factor. [nested.F90 L1780]                   
        shrink = np.exp(-1.0 / N_total)                

        # Work on copies so the current decomposition remains unchanged until
        # the complete replacement update has finished.
        new_ellipsoids = list(self.ellipsoids)
        new_owner      = self.point_owner.copy()

        # ----------------------------------------------------------
        # Remove the evicted point from ellipsoid q. [evolveEll a_r=0]
        # ----------------------------------------------------------
        eq = new_ellipsoids[q]

        # Find the point's local position using its global live-point row.
        pos_in_cluster = int(np.where(eq.rows == evicted_row)[0][0])

        # Remove the point and its corresponding row index together.
        remaining_mask = np.ones(eq.points.shape[0], dtype=bool)
        remaining_mask[pos_in_cluster] = False
        remaining_pts  = eq.points[remaining_mask]
        remaining_rows = eq.rows[remaining_mask]

        # Assign the remaining prior volume in proportion to the number of
        # live points still owned by this ellipsoid.
        n_q_after = remaining_pts.shape[0]
        target_q  = (n_q_after / N_total) * X_next
        new_ellipsoids[q] = eq.evolve_on_reject(
            evicted_point, remaining_pts, target_q,
            remaining_rows=remaining_rows)

        # Mark the live-point row as temporarily unassigned.
        new_owner[evicted_row] = -1

        # ----------------------------------------------------------
        # Insert the replacement point into ellipsoid i. [evolveEll a_r=1]
        # ----------------------------------------------------------
        ei = new_ellipsoids[i]

        # Append the point and its global row index in the same order.
        updated_pts  = np.vstack([ei.points, new_point[None, :]])
        updated_rows = np.append(ei.rows, evicted_row)

        # Assign the target volume in proportion to the updated cluster size.
        n_i_after = updated_pts.shape[0]
        target_i  = (n_i_after / N_total) * X_next
        new_ellipsoids[i] = ei.evolve_on_insert(
            new_point, updated_pts, target_i, updated_rows=updated_rows)

        # Record the new owner of the replaced live-point row. 
        new_owner[evicted_row] = i

        # ----------------------------------------------------------
        # Apply volume-floor decay to all unaffected ellipsoids. [nested.F90 L2313-2317]
        # ---------------------------------------------------------- 
        for j, ej in enumerate(new_ellipsoids):
            if j == q or j == i:
                continue
            new_ellipsoids[j] = ej.decay_untouched(shrink)

        # Store the updated decomposition and ownership mapping.
        self.ellipsoids  = new_ellipsoids
        self.point_owner = new_owner

    # ------------------------------------------------------------------
    # UNION SAMPLING
    # ------------------------------------------------------------------
    def sample_from_union(self) -> Tuple[np.ndarray, int, int]:
        """
        Draw a point uniformly from the union of the current ellipsoids.

        An ellipsoid is first selected with probability proportional to its
        volume. A point is then sampled uniformly from that ellipsoid. Because
        ellipsoids may overlap, a point lying inside ``n_e`` ellipsoids is accepted
        with probability ``1 / n_e``. This overlap correction ensures uniform
        sampling over the ellipsoidal union.

        Candidates outside ``domain_bounds`` are rejected and the complete
        sampling procedure is restarted.

        Returns
        -------
        d_star : np.ndarray
            Accepted point, with shape ``(n_dimensions,)``.
        chosen_ellipsoid_id : int
            Index of the ellipsoid from which the accepted point was sampled.
            This is used by ``evolve_step()`` to assign ownership of the new
            live point.
        n_attempts : int
            Number of candidate draws required before acceptance.

        Notes
        -----
        The method uses the MultiNest overlap correction

        ``P(accept | d_star) = 1 / n_e``,

        where ``n_e`` is the number of ellipsoids containing ``d_star``.

        Unlike the reference implementation, this method applies the overlap
        and domain checks before evaluating the candidate's feasibility merit.
        This preserves the accepted-point distribution while avoiding unnecessary
        model evaluations.

        When a point lies outside the bounded domain, the ellipsoid is selected
        again rather than resampling repeatedly from the same ellipsoid. Therefore,
        the resulting proposal is uniform over the intersection of the ellipsoidal
        union and the bounded domain.
        """
        # Select ellipsoids with probability proportional to their volumes.
        volumes    = np.array([e.V for e in self.ellipsoids])
        probs      = volumes / volumes.sum()

        # Sampling bounds, normally the unit hypercube.
        lo, hi     = self.domain_bounds
        n_attempts = 0

        while True:
            n_attempts += 1

            # Select one ellipsoid according to p_k = V_k / sum_j(V_j).
            k        = np.random.choice(len(self.ellipsoids), p=probs)

            # Draw uniformly from the selected ellipsoid.
            d_star   = self.ellipsoids[k].sample()

            # Count how many ellipsoids contain the sampled point.
            n_e      = sum(1 for e in self.ellipsoids if e.contains(d_star))

            # Correct for overlapping ellipsoids. A point covered by multiple
            # ellipsoids would otherwise be proposed too frequently.
            u_accept = np.random.uniform()
            accept_overlap = u_accept < 1.0 / n_e

            # Reject candidates outside the allowed sampling domain.
            in_domain      = bool(np.all(d_star >= lo) and np.all(d_star <= hi))

            accepted       = accept_overlap and in_domain

            _step(
                f"sample_from_union — attempt {n_attempts}",
                ellipsoid_chosen_k=int(k), p_k=float(probs[k]),
                d_star=d_star,
                n_e_overlapping_ellipsoids=n_e,
                u_draw_for_1_over_ne_test=float(u_accept),
                accept_threshold_1_over_ne=1.0 / n_e,
                overlap_accept_bool=bool(accept_overlap),
                in_domain_bool=in_domain,
                FINAL_ACCEPT_bool=accepted,
                visualize=lambda: _visualize_state(
                    np.vstack([e.points for e in self.ellipsoids]),
                    self.ellipsoids, highlight=d_star,
                    title=f"sample_from_union attempt {n_attempts} "
                          f"(accept={accepted})"),
            )

            if accepted:
                return d_star, int(k), n_attempts

    # ------------------------------------------------------------------
    # RECURSIVE DECOMPOSITION
    # ------------------------------------------------------------------
    def _algorithm1(
        self, points: np.ndarray, true_volume: float, idx: np.ndarray,
        _parent_id: int = -1,
    ) -> Tuple[List[Ellipsoid], List[np.ndarray]]:
        """
        Recursively construct a multi-ellipsoidal decomposition of a point set.

        The method implements Algorithm 1 of Feroz, Hobson & Bridges (2009).
        It first fits one ellipsoid around the complete point set, proposes a
        binary split using k-means and EM refinement, and accepts the split when
        it improves the ellipsoidal representation or when the parent ellipsoid
        is too large relative to the represented prior volume.

        Accepted subsets are processed recursively until no further valid or
        beneficial split can be made.

        Parameters
        ----------
        points : np.ndarray
            Points in the current subset, with shape
            ``(n_points, n_dimensions)``.
        true_volume : float
            Prior volume represented by the current point subset.
        idx : np.ndarray
            Original row indices of ``points`` in the complete live-point array
            passed to ``fit()``. These indices are carried through recursion so
            each final ellipsoid can be linked back to its live-point rows.

        Returns
        -------
        ellipsoids : list of Ellipsoid
            Final leaf ellipsoids produced for the current point subset.
        idx_lists : list of np.ndarray
            ``idx_lists[i]`` contains the original live-point rows owned by
            ``ellipsoids[i]``.

        Notes
        -----
        The ``idx`` bookkeeping is not part of the mathematical Algorithm 1.
        It is required by this implementation so that ``fit()`` can construct
        ``point_owner`` and ``evolve_step()`` can later update the correct
        ellipsoid during a live-point replacement.
        """
        N, D = points.shape

        # Fit one bounding ellipsoid around the complete current subset S.
        single = Ellipsoid.fit(points, true_volume)

        _step(
            "Algorithm 1 (steps 1-2) — bounding ellipsoid of S",
            N_points_in_S=N, D=D,
            V_single=single.V, true_volume_S=true_volume,
            visualize=lambda: _visualize_state(
                points, [single],
                title=f"Single bounding ellipsoid over {N} points"),
        )

        # Stop when the parent ellipsoid is already within 1% of the target
        # volume. In this case, splitting offers little meaningful reduction
        # in unused ellipsoidal volume.
        # [makeDino, xmeans_clstr.f90 L2393-2394] 
        already_tight = abs((single.V - true_volume) / true_volume) < 0.01


        _tn = None
        if self.trace_sink is not None:
            _my_id = self._trace_node_counter
            self._trace_node_counter += 1
            _tn = dict(node_id=_my_id, parent_id=int(_parent_id),
                       depth=self._trace_depth, N=int(N),
                       V_single=float(single.V), target=float(true_volume),
                       eff_single=float(single.eff),
                       already_tight=bool(already_tight),
                       cannot_split=None, n_leaves=self._n_leaves,
                       leaf_cap=self._leaf_cap,
                       kmeans_sizes=None, em_sizes=None, straggler=None,
                       VE1=None, VE2=None,
                       condition_A=None, condition_B=None, split=None,
                       decision="", ell_single=single, ell_children=None,
                       # geometry for animation (unit-cube coords):
                       points=np.asarray(points).copy(),
                       idx=np.asarray(idx).copy(),
                       labels=None)          # filled in at the split
            self.trace_sink.append(_tn)
        else:
            _my_id = -1
        _step(
            "Algorithm 1 early-stop (real MultiNest: xmeans_clstr.f90 "
            "makeDino L2393-2394, 'abs((vol-pVol)/pVol)<0.01')",
            V_single=single.V, true_volume=true_volume,
            relative_volume_gap=abs((single.V - true_volume) / true_volume),
            DECISION_already_tight_fit=already_tight,
        )

        if already_tight:
            if _tn is not None:
                _tn["decision"] = "KEEP: already tight (|V-tgt|/tgt < 1%)"

            # Return the current ellipsoid and all rows belonging to it.  
            return [single], [idx]

        # A binary split requires enough points for two valid clusters.
        # Also stop when the global maximum number of leaves has been reached.
        # [Dmeans L1132-1135 and makeDino L2393, L2459]
        cannot_split = (N < 2 * self.min_pt
                        or self._n_leaves >= self._leaf_cap)
        _step(
            "Algorithm 1 pre-check (real MultiNest: kmeans_clstr.f90 "
            "Dmeans L1131-1135 / xmeans_clstr.f90 makeDino L2393, "
            "incl. the nCls>=maxClstr leaf cap)",
            N_points_in_S=N, min_pt=self.min_pt,
            threshold_2x_min_pt=2 * self.min_pt,
            n_leaves_so_far=self._n_leaves, leaf_cap=self._leaf_cap,
            DECISION_cannot_split=cannot_split,
        )

        if cannot_split:
            if _tn is not None:
                _tn["cannot_split"] = True
                _tn["decision"] = (
                    f"KEEP: cannot split (N={N} < 2*min_pt={2*self.min_pt}"
                    f" or leaves {self._n_leaves} >= cap {self._leaf_cap})")

            # Return the current ellipsoid and all rows belonging to it.
            return [single], [idx]

        # Propose an initial binary partition using Euclidean k-means.
        labels = self._kmeans_split(points)
        n0_split, n1_split = int(np.sum(labels == 0)), int(np.sum(labels == 1))

        if _tn is not None:
            _tn["cannot_split"] = False
            _tn["kmeans_sizes"] = (n0_split, n1_split)
        _step(
            "Algorithm 1 (step 3) — k-means K=2 initial split (post top-up)",
            n_in_cluster_0=n0_split, n_in_cluster_1=n1_split,
        )

        # Refine the proposed partition using the weighted Mahalanobis
        # assignment rule from MultiNest.(EM refinement)
        labels, sub_ells = self._em_refinement(
            points, labels, true_volume, father_volume=single.V,
            em_mode=self.em_mode)

        # Divide points and original live-point rows using the same labels.
        n0, n1   = int(np.sum(labels == 0)), int(np.sum(labels == 1))
        p0, p1   = points[labels == 0], points[labels == 1]
        idx0, idx1 = idx[labels == 0], idx[labels == 1]

        # Allocate the parent prior volume in proportion to cluster size.
        VS0, VS1 = (n0 / N) * true_volume, (n1 / N) * true_volume

        # EM may leave one cluster with fewer than min_pt points.
        # In a binary split, merge it back by rejecting the split.
        # [xmeans_clstr.f90, Dinosaur, L2119-2152]
        straggler = n0 < self.min_pt or n1 < self.min_pt

        if _tn is not None:
            _tn["em_sizes"]  = (n0, n1)
            _tn["straggler"] = bool(straggler)
        _step(
            "Algorithm 1 post-EM check (real MultiNest: xmeans_clstr.f90 "
            "Dinosaur, 'sort out clusters with less than min_pt points', "
            "L2119-2152)",
            n_in_final_cluster_0=n0, n_in_final_cluster_1=n1,
            min_pt=self.min_pt,
            DECISION_straggler_cluster_merge_back=straggler,
        )

        if straggler:
            if _tn is not None:
                _tn["decision"] = (f"KEEP: post-EM straggler "
                                   f"({n0}/{n1} vs min_pt={self.min_pt})")
            # Return the current ellipsoid and all rows belonging to it.
            return [single], [idx]

        # Accept the split when:
        # A. The two child ellipsoids use less total volume than the parent; or
        # B. The parent ellipsoid is more than twice the represented volume.
        condition_B_factor = 2
        condition_A = (sub_ells[0].V + sub_ells[1].V) < single.V
        condition_B = single.V > condition_B_factor * true_volume
        split_decision = condition_A or condition_B

        if _tn is not None:
            _tn["labels"] = np.asarray(labels).copy()
            _tn.update(VE1=float(sub_ells[0].V), VE2=float(sub_ells[1].V),
                       condition_A=bool(condition_A),
                       condition_B=bool(condition_B),
                       split=bool(split_decision),
                       ell_children=list(sub_ells),
                       decision=(
                           "SPLIT" if split_decision else "KEEP") +
                           f": VE1+VE2={sub_ells[0].V + sub_ells[1].V:.4g} "
                           f"{'<' if condition_A else '>='} VE={single.V:.4g}"
                           f" (A={condition_A}); VE "
                           f"{'>' if condition_B else '<='} 2*tgt="
                           f"{2 * true_volume:.4g} (B={condition_B})")

        _step(
            "Algorithm 1 (step 14) — split-or-keep decision",
            V_E1=sub_ells[0].V, V_E2=sub_ells[1].V,
            V_E1_plus_V_E2=sub_ells[0].V + sub_ells[1].V,
            V_E_single=single.V,
            n_in_final_cluster_0=n0, n_in_final_cluster_1=n1,
            condition_A_sum_less_than_single=condition_A,
            condition_B_factor=condition_B_factor,
            condition_B_single_gt_factor_times_true_volume=condition_B,
            DECISION_split_S=split_decision,
            visualize=lambda: _visualize_state(
                points, sub_ells,
                title=f"Candidate split (accept split = {split_decision})"),
        )

        if not split_decision:
            # The proposed split provides no meaningful improvement.
            return [single], [idx]

        # One leaf becomes two leaves after an accepted binary split.
        self._n_leaves += 1

        self._trace_depth += 1

        # Recursively decompose both child subsets while preserving their
        # original live-point row indices.
        ell0, idxl0 = self._algorithm1(p0, VS0, idx0, _parent_id=_my_id)
        ell1, idxl1 = self._algorithm1(p1, VS1, idx1, _parent_id=_my_id)
        self._trace_depth -= 1

        # Combine the leaf ellipsoids and ownership rows from both branches
        return ell0 + ell1, idxl0 + idxl1

    # ------------------------------------------------------------------
    # CLUSTERING AND SPLIT REFINEMENT
    # ------------------------------------------------------------------
    def _kmeans_split(self, points: np.ndarray) -> np.ndarray:
        """
        Partition the supplied point set into two clusters using k-means. 
        
        The imported ``_kmeans3`` function applies the configured minimum 
        cluster size, number of restarts, and centroid initialisation method.
        [kmeans_clstr.f90 L198-365]

        Parameters
        ----------
        points : np.ndarray
            Points to cluster, with shape ``(n_points, n_dimensions)``.

        Returns
        -------
        np.ndarray
            Integer cluster labels of shape ``(n_points,)``. Each entry is
            either ``0`` or ``1``.
         """

        # Split the point set into two clusters while enforcing min_pt.
        return _kmeans3(points, self.min_pt,
                        restarts=self.kmeans_restarts,
                        init=self.kmeans_init)


    def _min_pt_kill(self, labels: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Apply the ``min_pt`` cluster-removal rule for a two-cluster partition.

        This method reproduces the ``min_pt`` kill block from MultiNest's
        ``Dmeans`` routine in ``kmeans_clstr.f90`` (lines 1327–1347),
        specialised to the case of two clusters.

        When either cluster contains fewer than ``self.min_pt`` points, all
        points assigned to that undersized cluster are moved to the other
        cluster. The resulting one-cluster partition indicates that the
        proposed split has collapsed and should be rejected by the caller.

        Parameters
        ----------
        labels : np.ndarray
            One-dimensional array of cluster labels. Each entry must be either
            ``0`` or ``1``.

        Returns
        -------
        updated_labels : np.ndarray
            The original labels when both clusters satisfy ``min_pt``.
            Otherwise, an array in which every point is assigned to the
            surviving cluster.

        collapsed : bool
            ``True`` if an undersized cluster was removed and the split
            collapsed. ``False`` if both clusters satisfy ``min_pt``.
        """
        for j in (0, 1):
            if int(np.sum(labels == j)) < self.min_pt:
                return np.full_like(labels, 1 - j), True
        return labels, False
    

    def _em_refinement(
        self,
        points:        np.ndarray,
        labels:        np.ndarray,
        true_volume:   float,
        father_volume: float = np.inf,
        em_mode:       Optional[str] = None,
        paper_max_iter: int = 100000,
    ) -> Tuple[np.ndarray, List["Ellipsoid"]]:
        """
        Refine an initial two-cluster k-means partition.

        The incoming partition is produced by the Euclidean two-cluster k-means
        step. This method optionally refines that partition using one of four
        control-flow variants:

        ``"none"``
            Do not refine the k-means partition. Fit two ellipsoids and return
            them immediately.

        ``"multinest"``
            Reproduce the control flow of the MultiNest v3.12 ``Dmeans`` routine
            as written. If the h(u) reassignment moves any point, the reassigned
            partition is discarded and the last saved partition is returned.

        ``"bugfix"``
            Follow the apparent intended ``Dmeans`` behaviour. Retain h(u)
            reassignments, apply the ``min_pt`` cluster-removal rule, refit the
            ellipsoids, and continue through the shared save logic.

        ``"paper"``
            Follow the iterative h(u) refinement described in the MultiNest
            paper. Refit the ellipsoids after each reassignment until a fixed
            point is reached or ``paper_max_iter`` is exceeded.

        Parameters
        ----------
        points : np.ndarray
            Point cloud to partition, with shape ``(N, D)``.

        labels : np.ndarray
            Initial k-means cluster assignments, with shape ``(N,)``.
            Each entry must be either ``0`` or ``1``.

        true_volume : float
            Prior-volume target assigned to the parent point set. This
            corresponds to ``pVol`` in the Fortran ``Dmeans`` routine.

        father_volume : float, optional
            Volume of the single ellipsoid fitted to all points. This corresponds
            to ``fVol`` in ``Dmeans`` and is used by the early-return condition
            in the ``"multinest"`` and ``"bugfix"`` modes.

            The default is positive infinity, which makes the
            ``best_total_volume < father_volume`` condition automatically true.

        em_mode : {"none", "multinest", "bugfix", "paper"}, optional
            Refinement strategy used for this call. If ``None``, the method uses
            ``self.em_mode``, which was configured when the decomposition was
            created. An explicit value overrides that setting for this call only
            and does not modify ``self.em_mode``. This is useful for testing or
            comparing different refinement strategies on the same initial
            partition.

        paper_max_iter : int, optional
            Maximum number of h(u) refinement passes in ``"paper"`` mode.

            This limit is used only in ``"paper"`` mode because that iterative
            procedure is not guaranteed to reach a fixed point. The
            ``"multinest"`` and ``"bugfix"`` modes use the reference
            ``count == 20`` stopping rule instead.

        Returns
        -------
        refined_labels : np.ndarray
            Final or best saved cluster assignments, with shape ``(N,)``.

        ellipsoids : list of Ellipsoid
            Two ellipsoids fitted to the returned partition
        """
        # Fall back to the decomposition's configured mode when no per-call mode
        # is supplied. Any supplied mode affects only this refinement call.
        if em_mode is None:
            em_mode = self.em_mode
        if em_mode not in ("none", "multinest", "bugfix", "paper"):
            raise ValueError(f"em_mode must be one of (none, multinest, bugfix, paper) got {em_mode!r}")
    
        N, D = points.shape
    
        def _fit_pair(lbls):
            """
            Fit one ellipsoid to each cluster in the current partition.
            Each cluster receives a fraction of ``true_volume`` proportional
            to its number of assigned points.
            """
            p0, p1   = points[lbls == 0], points[lbls == 1]
            n0, n1   = p0.shape[0], p1.shape[0]
            VS0, VS1 = (n0 / N) * true_volume, (n1 / N) * true_volume
            return Ellipsoid.fit(p0, VS0), Ellipsoid.fit(p1, VS1)
    
        def _h_pass(e0, e1, n0, n1):
            """Perform one h(u)-based reassignment pass.
            The assignment score for cluster i is proportional to
            ellipsoid_volume * Mahalanobis_distance / cluster_size``.
            [Dmeans L1287-1288] """
            diff0, diff1 = points - e0.mu, points - e1.mu
            d0 = np.einsum("ij,jk,ik->i", diff0, e0.A_inv, diff0)
            d1 = np.einsum("ij,jk,ik->i", diff1, e1.A_inv, diff1)
            h0 = e0.V * d0 / max(n0, 1)
            h1 = e1.V * d1 / max(n1, 1)
            return np.where(h0 <= h1, 0, 1), d0, d1, h0, h1
    
        # Fit the initial k-means partition and save it as the first candidate.
        # [Dmeans L1250-1264]
        e0, e1           = _fit_pair(labels)
        best_labels      = labels.copy()
        best_e0, best_e1 = e0, e1
        best_total_V     = e0.V + e1.V
    
        # ══ mode "none" ═══════════════════════════════════════════════════
        if em_mode == "none":
            _step("EM refinement — mode 'none': k-means split used as-is",
                n0=int(np.sum(labels == 0)), n1=int(np.sum(labels == 1)),
                V_E0=e0.V, V_E1=e1.V)
            return best_labels, [best_e0, best_e1]
    
        # ══ mode "paper" ══════════════════════════════════════════════════
        if em_mode == "paper":
            converged = False
            for it in range(paper_max_iter):
                n0, n1 = int(np.sum(labels == 0)), int(np.sum(labels == 1))
                new_labels, _, _, _, _ = _h_pass(e0, e1, n0, n1)

                # No point changes cluster, so the grouping has settled.
                if np.array_equal(new_labels, labels):
                    converged = True
                    break

                n0n = int(np.sum(new_labels == 0))
                n1n = int(np.sum(new_labels == 1))
                if n0n < self.min_pt or n1n < self.min_pt:
                    # Refining further would push a cluster below min_pt. Keep
                    # the last valid partition rather than collapsing the split;
                    # the split itself is still legitimate, we just stop here.
                    _step("EM refinement — 'paper': stopped, next pass would "
                        "breach min_pt", iteration=it, n0=n0n, n1=n1n,
                        min_pt=self.min_pt)
                    converged = True
                    break

                labels = new_labels
                e0, e1 = _fit_pair(labels)

                # Save the valid grouping with the smallest total ellipsoid volume so far.
                # This will be returned if no stable grouping is found within paper_max_iter.
                if e0.V + e1.V < best_total_V:
                    best_total_V     = e0.V + e1.V
                    best_labels      = labels.copy()
                    best_e0, best_e1 = e0, e1

            if converged:
                e0, e1 = _fit_pair(labels)
                return labels.copy(), [e0, e1]

            # Cycled or ran out of budget: fall back to the lowest-volume
            # iterate seen instead of the final iterate.
            print(f"  WARNING: EM refinement — 'paper': converged=False, "
                  f"paper_max_iter={paper_max_iter} reached without a fixed "
                  f"point; returning lowest-volume iterate "
                  f"(best_total_V={best_total_V:.6g}).")

            _step("EM refinement — 'paper': paper_max_iter reached without a "
                "fixed point, returning lowest-volume iterate",
                paper_max_iter=paper_max_iter, best_total_V=best_total_V)

            return best_labels, [best_e0, best_e1]
    
        # ══ modes "multinest" and "bugfix" ════════════════════════════════
        # Both run the Dmeans inner loop (L1281-1440).The two modes differ 
        # only in how they handle an h(u) reassignment step that changes
        # at least one cluster label.
        count   = 1                    
        it      = 0
        _RUNAWAY = 100_000              
        while True:
            it += 1
            if it > _RUNAWAY:
                raise RuntimeError(
                    f"Dmeans inner loop ran {it} passes without terminating. "
                    f"(em_mode={em_mode!r}, N={N}, D={D})")
            n0, n1 = int(np.sum(labels == 0)), int(np.sum(labels == 1))
            new_labels, d0, d1, h0, h1 = _h_pass(e0, e1, n0, n1)
            n_reassigned = int(np.sum(new_labels != labels))
            
            if n_reassigned > 0:
                # h(u) reassigned one or more points.
                # ── L1328-1349 ────────────────────────────────────────────
                if em_mode == "multinest":
                    # In the MultiNest v3.12 Fortran , the loop stops as soon as h(u) 
                    # changes any point's cluster. Because the new grouping is not saved 
                    # before the loop stops, those changes are ignored and the previous 
                    # best grouping is returned.
                    _step("EM refinement — 'multinest': h(u) pass moved points, "
                        "Dmeans L1348 exits and discards the result",
                        iteration=it, n_points_discarded=n_reassigned)
                    return best_labels, [best_e0, best_e1]
    
                # In bugfix mode, retain the new assignments and apply the
                # reference min_pt cluster-removal rule before refitting.
                # [L1329-1347]
                labels, collapsed = self._min_pt_kill(new_labels)
                if collapsed:
                    _step("EM refinement — 'bugfix': min_pt kill collapsed the "
                        "split", iteration=it, min_pt=self.min_pt)
                    return labels, [best_e0, best_e1]
            else:
                # h(u) is stable. Try one delF boundary-point transfer.
                #  The donor-side min_pt protection is implemented inside
                # _delF_boundary_swap. [L1351-1389]
                swapped_labels, boundary_flag = self._delF_boundary_swap(
                    D, labels, e0, e1, d0, d1, n0, n1)
                if not boundary_flag:
                    # No valid boundary swap exists, so the refinement has
                    # converged.
                    break                        
                labels = swapped_labels
    
            # ── shared refit / save block, L1391-1436 ─────────────────────
            count += 1

            # MultiNest stops after 20 consecutive passes without saving a
            # strictly better partition.
            if count == 20:                      
                break

            e0, e1 = _fit_pair(labels)
            new_total_V = e0.V + e1.V
            if new_total_V < best_total_V: 
                # A strict improvement resets the consecutive non-improvement
                # counter and replaces the saved output partition.      
                count            = 0
                best_total_V     = new_total_V
                best_labels      = labels.copy()
                best_e0, best_e1 = e0, e1

                # Dmeans early-return rule [L1429-1431]:
                # 1. the two-child decomposition must be smaller than the
                # parent ellipsoid, and
                # 2. its total volume must be within 1% of true_volume.
                if (best_total_V < father_volume
                        and abs(best_total_V - true_volume) / true_volume < 0.01):
                    _step("EM refinement — early return (Dmeans L1429-1431)",
                        best_total_V=best_total_V,
                        father_volume=father_volume, true_volume=true_volume)
                    return best_labels, [best_e0, best_e1]
    
        return best_labels, [best_e0, best_e1]


    def _delF_boundary_swap(
        self, ndim: int, labels: np.ndarray,
        e0: "Ellipsoid", e1: "Ellipsoid",
        d0: np.ndarray, d1: np.ndarray,
        n0: int, n1: int,
    ) -> Tuple[np.ndarray, bool]:
        """
        Try to move one boundary point from one cluster to the other.

        This method follows the boundary-point exchange step in MultiNest's
        ``Dmeans`` routine. It is called only when the h(u) reassignment step
        does not change any cluster labels.

        For each cluster, the method selects the point that is farthest from
        its own ellipsoid centre according to its Mahalanobis distance. It then
        uses ``_delF`` to estimate whether moving that point to the other cluster
        improves the two-cluster grouping.

        The first beneficial move is accepted.

        Parameters
        ----------
        ndim : int
            Number of dimensions.

        labels : np.ndarray
            Current cluster labels, with shape ``(N,)``. Each entry is either
            ``0`` or ``1``.

        ellipsoid_0 : Ellipsoid
            Ellipsoid fitted to cluster 0.

        ellipsoid_1 : Ellipsoid
            Ellipsoid fitted to cluster 1.

        distance_0 : np.ndarray
            Mahalanobis distance of every point from ellipsoid 0.

        distance_1 : np.ndarray
            Mahalanobis distance of every point from ellipsoid 1.

        n_0 : int
            Number of points currently assigned to cluster 0.

        n_1 : int
            Number of points currently assigned to cluster 1.

        Returns
        -------
        updated_labels : np.ndarray
            Cluster labels after the first beneficial move. If no beneficial
            move is found, the original labels are returned.

        swapped : bool
            ``True`` if one point was moved to the other cluster.
            ``False`` if no beneficial move was found.

        References
        ----------
        MultiNest v3.12, ``kmeans_clstr.f90``, ``Dmeans``,
        boundary-point exchange, lines 1352–1389.

        MultiNest v3.12, ``kmeans_clstr.f90``, ``delF``,
        lines 1455–1476.
        """
        distances = [d0, d1]
        ellipsoids = [e0, e1]
        cluster_sizes = [n0, n1]

        # A point cannot be transferred if either cluster is already empty.
        if cluster_sizes[0] == 0 or cluster_sizes[1] == 0:
            return labels, False

        for donor_cluster in (0, 1):
            receiver_cluster = 1 - donor_cluster

            # Do not remove a point from a cluster that has already reached
            # the minimum allowed size.
            if cluster_sizes[donor_cluster] == self.min_pt:
                continue

            donor_point_indices = np.where(labels == donor_cluster)[0]

            if donor_point_indices.size == 0:
                continue

            # Select the point farthest from its current ellipsoid centre.
            boundary_point = donor_point_indices[np.argmax(
                    distances[donor_cluster][donor_point_indices])]

            delta_f = _delF(
                ndim,
                cluster_sizes[donor_cluster],
                cluster_sizes[receiver_cluster],
                float(distances[donor_cluster][boundary_point]),
                float(distances[receiver_cluster][boundary_point]),
                ellipsoids[donor_cluster].det_C,
                ellipsoids[receiver_cluster].det_C,
                ellipsoids[donor_cluster].f,
                ellipsoids[receiver_cluster].f,
            )

            # A negative delF value means that moving the point improves
            # the two-cluster grouping.
            if delta_f < 0.0:
                updated_labels = labels.copy()
                updated_labels[boundary_point] = receiver_cluster

                _step(
                    "EM boundary delF swap "
                    "(Dmeans L1352-1389, delF L1455-1476)",
                    donor_cluster=donor_cluster,
                    receiver_cluster=receiver_cluster,
                    boundary_point_row=int(boundary_point),
                    delF_value=delta_f,
                )

                return updated_labels, True

        # No beneficial point transfer was found.
        return labels, False


# ============================================================
# SECTION 9 — DECOMPOSITION DEBUGGING INSTRUMENTS
# ============================================================
#
# Two tools for "are these overlapping / nested / bloated ellipsoids
# expected, or a bug?":
#
#   * analyze_decomposition_frames(frames) — post-hoc on an existing
#     RunRecorder recording; per frame gives coverage, overlap
#     multiplicity, floor domination and staleness.
#   * MultiNestSampler(trace_decomposition={sweeps}) — in-run; at the
#     chosen sweeps captures the refit-gate decision and the full
#     Algorithm-1 recursion tree with every inequality evaluated.
#
# READING THE NUMBERS:
#   uncovered_frac  > 0 right after an adopted refit -> BUG; growing
#                   between refits -> expected staleness (evolve_step
#                   only touches the two ellipsoids a replacement hits).
#   mean_mult       ~1-3 on curved sets -> expected chain tiling; 1/n_e
#                   keeps sampling unbiased, so it costs only efficiency.
#   floor_frac      -> 1.0 while X lags the level set -> expected bloat,
#                   and explains missing refits (the gate's eff-cancel
#                   clause suppresses them precisely then).
#   accepted=True with new_total_V > old_total_V              -> BUG.
#   split=True with condition_A=False and condition_B=False   -> BUG.
# ============================================================

def _membership_multiplicity(U: np.ndarray,
                             ells: List["Ellipsoid"]) -> np.ndarray:
    """(N,) how many of ``ells`` contain each unit-cube point in U.
    Vectorised Mahalanobis per ellipsoid; same test as
    ``Ellipsoid.contains`` (A_inv when present, else inv(f*C))."""
    N = U.shape[0]
    mult = np.zeros(N, dtype=int)
    for e in ells:
        A_inv = (e.A_inv if e.A_inv is not None
                 else np.linalg.inv(e.f * e.C))
        diff = U - e.mu
        mult += (np.einsum("ij,jk,ik->i", diff, A_inv, diff) <= 1.0)
    return mult


def analyze_decomposition_frames(frames: List["RunFrame"],
                                 every: int = 1):
    """Per-frame decomposition health table from an EXISTING recording.

    Parameters
    ----------
    frames : the RunRecorder frame list (or any subset).
    every  : analyse every k-th frame (the metrics are O(N_L * n_ell)
             per frame; 1 is fine for typical recordings).

    Returns a DataFrame with one row per analysed frame:
      sweep, event, n_modes, n_ell                   — identification
      sum_V           total union volume (unit-cube units; overlap
                      double-counted, exactly the quantity the refit
                      gate compares against the target)
      mean_mult /     live-point overlap multiplicity: how many
      max_mult        ellipsoids contain each live point. The sampler's
                      1/n_e thinning makes overlap unbiased, so this is
                      an EFFICIENCY gauge, not a correctness one.
      uncovered /     live points inside NO ellipsoid. Nonzero right
      uncovered_frac  after an ADOPTED refit = bug; growing during a
                      Refit=no stretch = expected staleness (evolve_step
                      only updates the two ellipsoids a replacement
                      touches; see Section 6).
      floor_frac      share of ellipsoids with eff > 1.00001, i.e.
                      volume-floor-inflated rather than point-shaped.
                      ~1.0 is the bloat regime AND suppresses refits
                      (the gate's eff-cancel clause).
      min_leaf_n /    current point count per ellipsoid (from .rows);
      max_leaf_n      min < min_pt after a refit = bug.
      since_refit     frames since the last "refit" event (a staleness
                      clock; mode-resolved refit attribution is not
                      recorded in frames, so this is an approximation
                      when several modes refit independently).
    """
    rows = []
    last_refit_frame = None
    for k, fr in enumerate(frames):
        if fr.event == "refit":
            last_refit_frame = k
        if k % every:
            continue
        ells = [e for mode in fr.mode_ellipsoids for e in mode]
        if not ells:
            continue
        U    = fr.design_space.to_unit(fr.live_points)
        mult = _membership_multiplicity(U, ells)
        leaf_n = [len(e.rows) for e in ells if e.rows is not None]
        rows.append(dict(
            frame          = k,
            sweep          = fr.sweep,
            event          = fr.event,
            n_modes        = len(fr.mode_ellipsoids),
            n_ell          = len(ells),
            sum_V          = float(sum(e.V for e in ells)),
            mean_mult      = float(mult.mean()),
            max_mult       = int(mult.max()),
            uncovered      = int(np.sum(mult == 0)),
            uncovered_frac = float(np.mean(mult == 0)),
            floor_frac     = float(np.mean([e.eff > 1.00001 for e in ells])),
            min_leaf_n     = (int(min(leaf_n)) if leaf_n else -1),
            max_leaf_n     = (int(max(leaf_n)) if leaf_n else -1),
            since_refit    = (k - last_refit_frame
                              if last_refit_frame is not None else -1),
        ))
    return pd.DataFrame(rows)


@dataclass
class DecompositionTrace:
    """One traced ``_maybe_refit`` call: the gate decision plus (when a
    fit actually ran) the full Algorithm-1 recursion, node by node."""
    sweep:      int
    mode_label: int
    X:          float
    target:     float          # X/ef, the union-volume floor handed to fit
    gate:       dict           # _maybe_refit's info dict (gate internals)
    nodes:      List[dict]     # preorder Algorithm-1 nodes; [] if no fit

    def summary(self) -> str:
        g = self.gate
        L = [f"--- decomposition trace | sweep {self.sweep} "
             f"mode {self.mode_label} | X={self.X:.4g} "
             f"target=X/ef={self.target:.4g} ---"]
        L.append(
            f"  gate: F_S={g['F_S']:.4g} vs thr={g['F_threshold']:g} | "
            f"pred_F_S={g['predicted_F_S']:.4g} | "
            f"periodic {g['sweeps_since_refit']}/{g['nsc']}"
            f"{' DUE' if g['periodic_due'] else ''}"
            f"{' | EFF-CANCELLED (all ellipsoids floor-inflated)' if g['eff_cancelled'] else ''}"
            f"{' | WARMUP-FORCED' if g['warmup_forced'] else ''}"
            f" -> {'REFIT' if g['need_refit'] else 'no refit'}")
        if g["need_refit"]:
            L.append(
                f"  attempt: n_ell {g['n_ell_old']} -> {g['n_ell_new']} | "
                f"V {g['old_total_V'] if g['old_total_V'] is not None else float('nan'):.4g}"
                f" -> {g['new_total_V']:.4g} | "
                f"{'ADOPTED (nsc-10)' if g['accepted'] else 'REVERTED, kept old (nsc+10)'}"
                f"{' [forced: first fit]' if g['forced'] else ''}")
        for nd in self.nodes:
            pad = "    " + "  " * nd["depth"]
            head = (f"{pad}[d{nd['depth']}] N={nd['N']} "
                    f"V_E={nd['V_single']:.4g} tgt={nd['target']:.4g} "
                    f"eff={nd['eff_single']:.3g}")
            if nd["kmeans_sizes"] is not None:
                head += (f" | kmeans {nd['kmeans_sizes'][0]}/"
                         f"{nd['kmeans_sizes'][1]}")
            if nd["em_sizes"] is not None:
                head += f" -> EM {nd['em_sizes'][0]}/{nd['em_sizes'][1]}"
            L.append(head)
            L.append(f"{pad}     {nd['decision']}")
        if not self.nodes and g["need_refit"]:
            L.append("    (no Algorithm-1 nodes captured?)")
        return "\n".join(L)

    def plot(self, save_path: Optional[str] = None,
             points: Optional[np.ndarray] = None,
             dims: Tuple[int, int] = (0, 1)):
        """Draw the recursion: every node's bounding ellipsoid, coloured
        by depth (dashed = internal/kept-single candidates, solid =
        final leaves), in UNIT-CUBE coordinates, over ``points`` (the
        live subset that was fitted, if provided)."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        # identity "design space" so _unit_ellipsoid_to_patch draws u-space
        D = (points.shape[1] if points is not None
             else self.nodes[0]["ell_single"].mu.shape[0])
        ds_id = DesignSpace([(0.0, 1.0)] * D)
        cmap  = plt.get_cmap("viridis")
        maxd  = max((nd["depth"] for nd in self.nodes), default=0)
        for nd in self.nodes:
            col  = cmap(nd["depth"] / max(maxd, 1))
            leaf = not nd.get("split")
            ax.add_patch(_unit_ellipsoid_to_physical_patch(
                nd["ell_single"], ds_id, dims=dims, fill=False,
                edgecolor=col, lw=2.0 if leaf else 1.0,
                linestyle="-" if leaf else "--", alpha=0.9))
        if points is not None:
            ax.scatter(points[:, dims[0]], points[:, dims[1]],
                       s=8, c="k", alpha=0.6, zorder=3)
        ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.15, 1.15)
        ax.set_aspect("equal")
        ax.set_title(f"Algorithm-1 recursion — sweep "
                     f"{self.sweep} mode {self.mode_label}\n"
                     f"solid = final leaf, dashed = internal candidate, "
                     f"colour = depth", fontsize=10)
        if save_path:
            fig.savefig(save_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"Trace plot saved: {save_path}")
        return fig

    # ----------------------------------------------------------
    # STEP-BY-STEP RECURSION ANIMATION
    # ----------------------------------------------------------
    # The trace's ``nodes`` list is already a PREORDER walk of the
    # recursion (parent appended before its two children — see
    # _algorithm1), and every node now carries its own points, original
    # row indices, and the k-means/EM split labels. That is everything
    # needed to replay the recursion as it happened: at each node show
    # its point subset, the parent bounding ellipsoid, then (if it split)
    # recolour those points by cluster and draw the two children.
    #
    # Frames are emitted so a split node becomes three: ARRIVE (parent
    # ellipsoid over its points), SPLIT (points recoloured, both children
    # drawn dashed), and — carried by the child nodes themselves —
    # DESCEND. A kept node emits one KEEP frame. Persisted context (all
    # ancestor + previously-finalised leaf ellipsoids, faint) is drawn
    # under every frame so you always see where the current subset sits
    # in the whole decomposition.

    def _color_for(self, node_id: int):
        import matplotlib.pyplot as plt
        # stable, well-separated colour per node id
        cmap = plt.get_cmap("tab20")
        return cmap((node_id * 7) % 20)

    def recursion_frames(self) -> List[dict]:
        """Build the ordered frame list for the animation. Each frame is
        a dict the renderer understands; returned separately so callers
        can inspect / re-render without recomputing."""
        if not self.nodes:
            return []
        by_id = {nd["node_id"]: nd for nd in self.nodes}
        # a node is a final LEAF iff it did not split
        leaf_ids = {nd["node_id"] for nd in self.nodes if not nd.get("split")}
        finalised_leaves: List[dict] = []      # accrues as we walk
        ancestors_of: dict = {}                # node_id -> [ancestor ells]

        frames: List[dict] = []
        for nd in self.nodes:                  # preorder
            pid = nd["parent_id"]
            anc = list(ancestors_of.get(pid, []))
            ancestors_of[nd["node_id"]] = anc + [nd["ell_single"]]

            base = dict(node=nd,
                        context_leaves=list(finalised_leaves),
                        context_anc=anc)

            # Frame A: arrive — parent bounding ellipsoid over its points
            frames.append(dict(kind="arrive", **base))

            if nd.get("split"):
                # Frame B: the split — points recoloured by cluster,
                # both candidate children shown
                frames.append(dict(kind="split", **base))
            else:
                # Frame K: kept as a leaf
                frames.append(dict(kind="keep", **base))
                finalised_leaves.append(nd)

        # Final frame: the whole decomposition, all leaves solid
        frames.append(dict(kind="final",
                           node=None,
                           context_leaves=[by_id[i] for i in
                                           sorted(leaf_ids)],
                           context_anc=[]))
        return frames

    def _render_recursion_frame(self, ax, fr: dict, dims: Tuple[int, int]):
        import numpy as _np
        i, j = dims
        D = self.nodes[0]["ell_single"].mu.shape[0]
        ds_id = DesignSpace([(0.0, 1.0)] * D)
        ax.clear()
        ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.15, 1.15)
        ax.set_aspect("equal")
        ax.set_facecolor("#0d0d0d")

        # persistent context: ancestors (faint grey) + finalised leaves
        for e in fr["context_anc"]:
            ax.add_patch(_unit_ellipsoid_to_physical_patch(
                e, ds_id, dims=dims, fill=False, edgecolor="#555555",
                lw=0.8, linestyle=":", alpha=0.6))
        for lf in fr["context_leaves"]:
            ax.add_patch(_unit_ellipsoid_to_physical_patch(
                lf["ell_single"], ds_id, dims=dims, fill=False,
                edgecolor=self._color_for(lf["node_id"]),
                lw=1.6, linestyle="-", alpha=0.7))
            P = lf["points"]
            ax.scatter(P[:, i], P[:, j], s=10,
                       color=self._color_for(lf["node_id"]),
                       alpha=0.5, zorder=2)

        kind = fr["kind"]
        if kind == "final":
            ax.set_title(f"Algorithm-1 recursion — sweep "
                         f"{self.sweep} mode {self.mode_label}\n"
                         f"FINAL: {len(fr['context_leaves'])} leaf "
                         f"ellipsoid(s)", fontsize=10, color="w")
            ax.tick_params(colors="w")
            return

        nd  = fr["node"]
        P   = nd["points"]
        col = self._color_for(nd["node_id"])

        if kind == "arrive":
            # points of this subset in the node's own colour, parent ell
            ax.scatter(P[:, i], P[:, j], s=14, color=col, alpha=0.9, zorder=3)
            ax.add_patch(_unit_ellipsoid_to_physical_patch(
                nd["ell_single"], ds_id, dims=dims, fill=False,
                edgecolor=col, lw=2.2, linestyle="-", alpha=0.95))
            title = (f"depth {nd['depth']} · node {nd['node_id']} · "
                     f"N={nd['N']} · V_E={nd['V_single']:.3g} "
                     f"eff={nd['eff_single']:.2g}")
            sub = "bounding ellipsoid of this subset"
        elif kind == "keep":
            ax.scatter(P[:, i], P[:, j], s=14, color=col, alpha=0.9, zorder=3)
            ax.add_patch(_unit_ellipsoid_to_physical_patch(
                nd["ell_single"], ds_id, dims=dims, fill=False,
                edgecolor=col, lw=2.6, linestyle="-", alpha=1.0))
            title = f"depth {nd['depth']} · node {nd['node_id']} · LEAF"
            sub = nd["decision"]
        else:  # split
            lbl = nd["labels"]
            c0, c1 = self._color_for(nd["node_id"] * 100 + 1), \
                     self._color_for(nd["node_id"] * 100 + 2)
            ax.scatter(P[lbl == 0][:, i], P[lbl == 0][:, j], s=16,
                       color=c0, alpha=0.95, zorder=3, label="cluster 0")
            ax.scatter(P[lbl == 1][:, i], P[lbl == 1][:, j], s=16,
                       color=c1, alpha=0.95, zorder=3, label="cluster 1")
            for e, c in zip(nd["ell_children"], (c0, c1)):
                ax.add_patch(_unit_ellipsoid_to_physical_patch(
                    e, ds_id, dims=dims, fill=False, edgecolor=c,
                    lw=2.2, linestyle="--", alpha=0.95))
            # parent, faded, for reference
            ax.add_patch(_unit_ellipsoid_to_physical_patch(
                nd["ell_single"], ds_id, dims=dims, fill=False,
                edgecolor=col, lw=1.2, linestyle="-", alpha=0.4))
            km = nd["kmeans_sizes"]; em = nd["em_sizes"]
            title = (f"depth {nd['depth']} · node {nd['node_id']} · SPLIT "
                     f"k-means {km[0]}/{km[1]} → EM {em[0]}/{em[1]}")
            sub = nd["decision"]

        ax.set_title(title, fontsize=9, color="w")
        ax.text(0.5, -0.14, sub, transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color="#bbbbbb", wrap=True)
        ax.tick_params(colors="w")

    def animate_recursion(self, save_path: str,
                          dims: Tuple[int, int] = (0, 1),
                          fps: float = 1.2,
                          hold_final: int = 3):
        """Render the recursion as an animated GIF, one step at a time:
        parent ellipsoid → split (points recoloured by cluster, children
        drawn) → descend into each child, in the exact order the
        recursion happened. Ancestor ellipsoids and already-finalised
        leaves persist faintly for context.

        Parameters
        ----------
        save_path  : output .gif path.
        dims       : which 2 unit-cube axes to project onto.
        fps        : frames per second.
        hold_final : extra copies of the final frame (a pause at the end).

        Requires pillow (matplotlib's PillowWriter). Returns the list of
        frame dicts (also available via ``recursion_frames()``).
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        frames = self.recursion_frames()
        if not frames:
            print("No recursion nodes to animate (was this sweep traced, "
                  "and did a fit actually run?).")
            return frames
        seq = frames + [frames[-1]] * max(0, hold_final)

        fig, ax = plt.subplots(figsize=(7.2, 7.6))
        fig.patch.set_facecolor("#0d0d0d")

        def _draw(k):
            self._render_recursion_frame(ax, seq[k], dims)
            return []

        anim = FuncAnimation(fig, _draw, frames=len(seq), blit=False)
        anim.save(save_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"Recursion animation saved: {save_path} "
              f"({len(frames)} steps, {len(self.nodes)} nodes)")
        return frames

    def plot_recursion_steps(self, save_path: Optional[str] = None,
                             dims: Tuple[int, int] = (0, 1),
                             ncols: int = 4):
        """Static contact-sheet alternative to ``animate_recursion``:
        the same ordered steps laid out as a grid of panels, for when a
        GIF is inconvenient (papers, quick scan)."""
        import matplotlib.pyplot as plt
        frames = self.recursion_frames()
        if not frames:
            print("No recursion nodes to plot.")
            return None
        n = len(frames)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.2 * ncols, 3.4 * nrows))
        fig.patch.set_facecolor("#0d0d0d")
        axes = np.atleast_1d(axes).ravel()
        for k, fr in enumerate(frames):
            self._render_recursion_frame(axes[k], fr, dims)
        for k in range(n, len(axes)):
            axes[k].axis("off")
        fig.suptitle(f"Algorithm-1 recursion steps — sweep "
                     f"{self.sweep} mode {self.mode_label}",
                     color="w", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        if save_path:
            fig.savefig(save_path, dpi=110, bbox_inches="tight",
                        facecolor="#0d0d0d")
            plt.close(fig)
            print(f"Recursion contact sheet saved: {save_path}")
        return fig


# ============================================================
# SECTION 10 — NESTED SAMPLER  (modes, run loop, result)
# ============================================================

@dataclass
class _Mode:
    """
    Store the state of one independently tracked sampling mode.

    Each mode owns:

    - a subset of the global live-point population,
    - its own remaining-volume value ``X``,
    - its own ellipsoidal decomposition,
    - its own adaptive refit schedule.

    ``idx`` contains row indices into the global ``live_points_u`` and
    ``live_merit`` arrays. When a mode splits, its positions are divided 
    between the new child modes.

    ``label`` is a stable identifier assigned when the mode is created.
    Parent labels are retired after a split, while each child receives a
    new label.

    Evidence and information quantities from the original MultiNest
    implementation, such as ``ic_Z`` and ``ic_info``, are not stored
    because this implementation does not compute evidence or
    nested-sampling integration weights.
    """
    # Rows in the global live-point arrays that belong to this mode.
    idx:            np.ndarray

    # Stable mode identifier assigned once at creation: 1, 2, 3, ...
    # New labels are assigned to the initial mode and to every child
    # created by a split. Labels are not reused, so points evicted before
    # a split remain associated with the parent mode that owned them.
    label:          int   = 0         


    # Mode-specific remaining-volume value, corresponding to ``ic_vnow``.
    # It is used only for ellipsoid sizing and refit decisions:
    #     target volume = X / ef
    #     F(S) = total ellipsoid volume / target volume
    # It is not reported as a design-space volume estimate and is not
    # used by the stopping criterion.
    X:              float = 1.0      

    # True when this mode should no longer be processed.
    done:           bool  = False     

    # Human-readable explanation of why the mode stopped.
    # Used only for reporting, not for algorithmic decisions
    done_reason:    str   = ""        

    # Ellipsoidal decomposition used to sample replacement points.
    decomposition:  "EllipsoidalDecomposition" = None

    # Global replacement count at which this mode was last refitted.
    last_refit_at:  int   = 0

    # History of the mode's F(S) values after decomposition checks.
    fs_history:     List[float] = field(default_factory=list)
    
    # Mode-specific adaptive interval between periodic refit checks.
    # The interval decreases when refitting improves F(S), causing more
    # frequent checks, and increases when refitting does not improve it.
    # It is initialised from the sampler's ``nsc_def`` setting.
    nsc:            int   = None     


@dataclass
class SamplerResult:
    """
    Store the output of MultiNestSampler.run.

    All point coordinates are stored in physical units. Live, dead and
    rejected points are stored together with their sampler merit and
    feasibility probability. Matching row indices refer to the same point.

    For feas_criterion="P", merit equals probability. For "VaR" and
    "CVaR", merit stores the negative risk value.
    """

    # Number of live points maintained by the sampler.
    N_L:               int

    # ------------------------------------------------------------------
    # SAMPLED POINTS
    # ------------------------------------------------------------------
    live_points:       np.ndarray
    live_merit:        np.ndarray
    live_probs:        np.ndarray

    dead_points:       np.ndarray
    dead_merit:        np.ndarray
    dead_probs:        np.ndarray

    rejected_points:   np.ndarray
    rejected_merit:    np.ndarray
    rejected_probs:    np.ndarray

    # Number of accepted replacements. Normally this equals the number of
    # dead points because each eviction is followed by one accepted
    # replacement.
    n_replacements:          int

    # Number of candidate evaluations:
    # accepted candidates + rejected candidates.
    # The initial N_L evaluations are excluded.
    n_candidate_estimates:   int

    # Number of model runs used for one merit/probability estimate.
    # Equal to N_theta for a sampling-based uncertainty, but NOT in general:
    # it is ``uncertainty.n_scenarios(N_theta)``, which a WeightedScenarios
    # overrides with its own fixed scenario count.
    model_runs_per_estimate: int = 1

    # Criterion used to drive sampling: "P", "VaR" or "CVaR".
    feas_criterion: str = "VaR"

    # Target reliability for P, or confidence level for VaR/CVaR.
    alpha_star: float = 0.95

    # Mode label of each final live point.
    # Labels may not start from 1 or be consecutive because old mode labels
    # are not reused after a mode splits.
    live_mode_ids: np.ndarray = None

    # ------------------------------------------------------------------
    # DECOMPOSITION DIAGNOSTICS
    # ------------------------------------------------------------------
    # Algorithm-1 and refit-gate traces recorded when decomposition tracing
    # was enabled. None when trace_decomposition was disabled.
    decomp_traces: Optional[List["DecompositionTrace"]] = None

    # ------------------------------------------------------------------
    # TERMINATION
    # ------------------------------------------------------------------
    # Why the run stopped. ``"converged"`` when every live point satisfied
    # the merit threshold; ``"modes_exhausted"`` when every mode was done
    # (frozen or structurally exhausted) while some live points were still
    # below it. This is the same distinction the RESULT block prints as
    # CONVERGED / STOPPED.
    termination_reason: str = "converged"

    # Live points still below the merit threshold when the run stopped.
    # Zero for a converged run; the certified region is incomplete when
    # this is positive.
    n_uncertified_live: int = 0

    # ----------------------------------------------------------
    @property
    def converged(self) -> bool:
        """True when the run left no live point below the threshold."""
        return self.n_uncertified_live == 0

    # ----------------------------------------------------------
    @property
    def total_model_runs(self) -> int:
        """Total number of model runs used by the run: the initial live
        population plus every candidate, each costing one estimate."""
        return ((self.N_L + self.n_candidate_estimates)
                * self.model_runs_per_estimate)

    # ----------------------------------------------------------
    def all_points_and_merits(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Combine dead, rejected and live points for visualisation.

        Returns
        -------
        points : np.ndarray
            All sampled points in physical coordinates.

        merits : np.ndarray
            Sampler merit corresponding to each point.

        roles : list of str
            Role of each point: ``"dead"``, ``"rejected"`` or ``"live"``.
            The names match the ``dead_*`` / ``_rejected_*`` /
            ``_live_*`` attributes each group is taken from.
        """
        dead_points = np.asarray(self.dead_points)
        rejected_points = np.asarray(self.rejected_points)
        live_points = np.asarray(self.live_points)

        points = np.vstack([
            dead_points,
            rejected_points,
            live_points,
        ])

        merits = np.concatenate([
            self.dead_merit,
            self.rejected_merit,
            self.live_merit,
        ])

        roles = (
            ["dead"] * len(dead_points)
            + ["rejected"] * len(rejected_points)
            + ["live"] * len(live_points)
        )

        return points, merits, roles

    def merits(self, scope: str = "all") -> np.ndarray:
        """
        Return merit values for the selected point group.

        Parameters
        ----------
        scope : {"all", "ns", "final_live"}, optional
            Points to include:

            - ``"all"``: dead, rejected and final live points.
            - ``"ns"``: dead and final live points.
            - ``"final_live"``: only the final live points.

        Returns
        -------
        np.ndarray
            Merit values for the selected points.
        """
        valid_scopes = ("all", "ns", "final_live")

        if scope not in valid_scopes:
            raise ValueError(f"scope must be one of {valid_scopes}, got {scope!r}")

        live_merit = np.asarray(self.live_merit, dtype=float)
        if scope == "final_live":
            return live_merit

        dead_merit = np.asarray(self.dead_merit, dtype=float)
        if scope == "ns":
            return np.concatenate([dead_merit, live_merit])

        rejected_merit = np.asarray(self.rejected_merit, dtype=float)
        return np.concatenate([dead_merit, rejected_merit, live_merit])

    def feas_probabilities(self, scope: str = "all") -> np.ndarray:
        """
        Return feasibility probabilities for the selected point group.

        Parameters
        ----------
        scope : {"all", "ns", "final_live"}, optional
            Points to include:

            - ``"all"``: dead, rejected and final live points.
            - ``"ns"``: dead and final live points.
            - ``"final_live"``: only the final live points.

        Returns
        -------
        np.ndarray
            Feasibility probability of each selected point.

        Notes
        -----
        When ``feas_criterion == "P"``, these values are identical to the
        corresponding merit values.
        """
        valid_scopes = ("all", "ns", "final_live")

        if scope not in valid_scopes:
            raise ValueError(f"scope must be one of {valid_scopes}, got {scope!r}")

        # For the probability criterion, merit is already P(d).
        if self.feas_criterion == "P":
            return self.merits(scope)

        live_probs = np.asarray(self.live_probs,dtype=float)
        if scope == "final_live":
            return live_probs

        dead_probs = np.asarray(self.dead_probs, dtype=float)
        if scope == "ns":
            return np.concatenate([dead_probs, live_probs])

        rejected_probs = np.asarray(self.rejected_probs, dtype=float)
        return np.concatenate([dead_probs, rejected_probs, live_probs])

    @property
    def display(self) -> "CriterionDisplay":
        """ 
        Return a ``CriterionDisplay`` configured for this run.

        The returned object uses ``feas_criterion`` and ``alpha_star`` to define
        how merit values are converted, classified in tables and displayed in
        plots, colours, labels and contour boundaries.
        """
        return CriterionDisplay(self.feas_criterion, self.alpha_star)

    def criterion_values(self, scope: str = "all") -> np.ndarray:
        """Convert stored merits to the selected criterion's original values."""
        return self.display.from_merit(self.merits(scope))

    def reliability_table(self, scope: str = "all"):
        """
        Count sampled points within ranges of the selected criterion.

        For ``P``, points are grouped into fixed feasibility-probability
        ranges. For ``VaR`` and ``CVaR``, values at or below zero are feasible,
        while positive violations are grouped relative to the largest observed
        positive violation.

        Parameters
        ----------
        scope : {"all", "ns", "final_live"}, optional
            Points to include:

            - ``"all"``: dead, rejected and final live points.
            - ``"ns"``: dead and final live points.
            - ``"final_live"``: final live points only.

        Returns
        -------
        pandas.DataFrame
            Number of points in each criterion range.

        Notes
        -----
        The table contains raw point counts, not design-space volume estimates.
        """
        return self.display.table(self.criterion_values(scope))

    def reliability_table_P(self, scope: str = "all"):
        """
        Count points within fixed feasibility-probability ranges.

        This table is available for every driving criterion. For ``P`` runs,
        it matches ``reliability_table``. For ``VaR`` and ``CVaR`` runs, it
        provides an additional view based on the recorded ``P(d)`` values.
        """
        return CriterionDisplay("P", self.alpha_star).table(
            self.feas_probabilities(scope))

    def mode_table(self):
        """Final LIVE points per mode.

        Modes are numbered 1..M FOR DISPLAY, in ascending label order --
        the same order the run's own final mode summary prints, so
        "Mode 1" names the same mode in both. The internal stable
        ``_Mode.label`` keeps its own column: it is what ``live_mode_ids``
        actually stores and what the run log prints, and it is generally
        NOT 1..M, since a label is handed out once per mode ever created
        and the ones retired at splits never come back.

        Live points only, and deliberately so: a dead point belongs to
        whatever mode evicted it, and a split retires the parent's label,
        so dead-point counts by label would describe the mode TREE and
        would not partition the final modes.
        Live points do partition exactly -- the counts sum to N_L.

        Returns
        -------
        pandas.DataFrame  (empty if live_mode_ids was not recorded)
        """
        cols = ["Mode", "Label", "Final live points"]
        if self.live_mode_ids is None:
            return pd.DataFrame({c: [] for c in cols})
        lab, cnt = np.unique(np.asarray(self.live_mode_ids), return_counts=True)
        table = pd.DataFrame({
            "Mode": [f"Mode {i}" for i in range(1, len(lab) + 1)],
            "Label": [int(l) for l in lab],
            "Final live points": [int(c) for c in cnt],
        })
        table.loc[len(table)] = ["Total", "", int(np.sum(cnt))]
        return table



class _ProgressLog:
    """
    LOGGING ONLY — holds no algorithmic state, draws no random numbers,
    and every method is a no-op when disabled.

    One table, one row per mode, closed off per sweep by aggregate rows:

        sweep  iter  mode   n  feas  left  feas%  {sym}  ell  tries  elapsed  note
          530   598     1  167    71    96  42.5%  .3086    6      1     0:05
                599     3  167    40   127  24.0%  .2698    5      1     0:05
                     done  166   166     0  100.%  -.002    —      —           frozen [2]@402
                    Σ 2/3  500   277   223  55.4%  .3086   11      2     0:05

    Reading rules:

      * ``sweep`` prints only when it changes, so one sweep = one block.
      * ``iter`` counts ACCEPTED replacements, one per mode row; rejected
        candidates are counted by ``tries`` instead (1 = first draw).
      * every column is scoped to its own row — mode row to that mode,
        ``done`` row to all finished modes pooled, ``Σ`` row to the whole
        live set. ``n = feas + left`` holds everywhere, and Σ is the exact
        column-wise aggregate (sum, worst-of for the criterion, ratio for
        feas%).
      * population columns (n, feas, left, feas%, criterion) stay valid
        for a finished mode; machinery columns (ell, tries) print "—",
        since a done mode is never refit or sampled from again.
      * ``left`` is the sole termination quantity: the run stops when the
        Σ row hits 0, and ``left == 0`` on a mode row is its freeze
        condition. On the ``done`` row a non-zero ``left`` means an
        *exhausted* mode (plateau / < D+1 points) holds uncertified
        points the run can never fix — it will end on a warning.
      * splits, finishes and warnings print as ``·`` event lines between
        blocks and refresh the column header.

    THROTTLING is per SWEEP, never per row — ``begin_sweep`` decides once
    and a printed sweep always shows its COMPLETE block, so the Σ row
    never aggregates rows the reader cannot see. ``log_every`` takes:

      ``1``      every sweep (default)
      ``n > 1``  every n-th sweep, whole block
      ``"2%"``   whenever ``feas%`` advanced by >= 2 points since the last
                 printed sweep. Gives a roughly fixed block count (~50)
                 whatever the run length, which an iteration cadence
                 cannot since the sweep count is unknown up front.
                 ``feas`` never decreases, so pacing cannot oscillate.
      ``0``/``None``  nothing at all

    ``heartbeat`` bounds the gap in ``"x%"`` mode: if progress stalls, a
    block prints anyway every ``heartbeat`` sweeps.
    """

    def __init__(self, symbol: str, n_live: int, threshold_text: str,
                 log_every: Union[int, str, None] = 1,
                 heartbeat: int = 100) -> None:
        self.sym            = symbol
        self.n_live         = max(int(n_live), 1)
        self.threshold_text = threshold_text   # e.g. "VaR[G] <= 0"
        self.log_every      = log_every
        self.heartbeat      = max(int(heartbeat), 1)
        self.enabled        = log_every is not None and log_every != 0
        # cadence: either every n-th sweep, or every `pct` points of feas%
        self.pct: Optional[float] = None
        self.every                = 1
        if isinstance(log_every, str) and log_every.strip().endswith("%"):
            self.pct = max(float(log_every.strip()[:-1]), 0.0)
        elif self.enabled:
            try:
                self.every = max(int(log_every), 1)
            except (TypeError, ValueError):
                raise ValueError(
                    "log_every must be an int (print every n-th sweep), a "
                    "percentage string such as '2%' (print whenever feas% "
                    "advances that much), or 0/None to silence the table; "
                    f"got {log_every!r}") from None
        self.printing       = False   # set by begin_sweep()
        self._shown         = False   # did THIS sweep put anything on screen
        self._tail: Optional[dict] = None   # last sweep, for close_table
        self._t0            = time.perf_counter()
        self._last_sweep: Optional[int] = None  # for the blank sweep cell
        self._last_block    = 0       # last sweep whose block was printed
        self._last_pct: Optional[float] = None
        self._need_hdr      = True
        # per-sweep accumulators, reset by begin_sweep()
        self._acted         = 0    # mode turns taken this sweep
        self._rows          = 0    # mode rows actually printed
        self._tries         = 0
        self._ell           = 0
        self._cols = [
            ("sweep", 8), ("repl", 7), ("mode", 6), ("live", 7), ("feas", 7),
            # the criterion column must fit a signed 4-significant-digit
            # value such as "-0.0007005" (10 chars) without touching the
            # column to its left
            ("left", 7), ("feas%", 8), (symbol, max(12, len(symbol) + 4)),
            ("ell", 6), ("tries", 7), ("elapsed", 9),
        ]
        self.width = sum(w for _, w in self._cols)

    # ---------------------------------------------------- helpers
    @staticmethod
    def _hms(sec: Optional[float]) -> str:
        """Compact h:mm:ss / m:ss, or '—' when unknown."""
        if sec is None or not np.isfinite(sec) or sec < 0:
            return "—"
        sec  = int(round(sec))
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    @staticmethod
    def _hms_long(sec: Optional[float]) -> str:
        """Unambiguous 00h 00m 00s 000ms, for the closing summary."""
        if sec is None or not np.isfinite(sec) or sec < 0:
            return "—"
        h, r = divmod(int(sec), 3600)
        m, s = divmod(r, 60)
        ms   = int((sec - int(sec)) * 1000)
        return f"{h:02d}h {m:02d}m {s:02d}s {ms:03d}ms"

    @staticmethod
    def _pct(a: int, b: int) -> str:
        return f"{100.0 * a / b:.1f}%" if b > 0 else "—"

    @staticmethod
    def _val(v: Optional[float]) -> str:
        return "—" if v is None or not np.isfinite(v) else f"{v:.4g}"

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def begin_sweep(self, sweep: int, feas: int) -> bool:
        """Settle, BEFORE the sweep's first row, whether this sweep gets
        printed, and reset the per-sweep accumulators. Deciding here
        rather than per row is what keeps the Σ row an aggregate of rows
        the reader can actually see."""
        self._acted = self._rows = self._tries = self._ell = 0
        self._shown = False
        if not self.enabled:
            self.printing = False
        elif self.pct is None:
            self.printing = (sweep % self.every == 0)
        else:
            pct = 100.0 * feas / self.n_live
            self.printing = (self._last_pct is None
                             or pct - self._last_pct >= self.pct
                             or sweep - self._last_block >= self.heartbeat)
            if self.printing:
                self._last_pct = pct
        if self.printing:
            self._last_block = sweep
        return self.printing

    def _line(self, cells: List[str], note: str = "") -> None:
        s = "  " + "".join(f"{c:>{w}}"
                           for c, (_, w) in zip(cells, self._cols))
        if note:
            s += "  " + note
        print(s.rstrip())

    # ---------------------------------------------------- blocks
    def block(self, title: str, fields, rule: str = "═") -> None:
        """A titled key/value block (run banner, closing summary)."""
        if not self.enabled:
            return
        print()
        print("  " + rule * self.width)
        print(f"  {title}")
        print("  " + rule * self.width)
        for k, v in fields:
            # An empty key continues the previous field: the value stays
            # in the value column so the block reads as two columns.
            print(f"    {k:<20}: {v}" if k else f"    {'':<20}  {v}")
        print("  " + rule * self.width)

    def legend(self) -> None:
        """Printed ONCE per run, above the first header: with a single
        header and a table hundreds of rows long, this is what keeps a
        saved console log self-describing."""
        if not self.enabled:
            return
        sym = self.sym
        print()
        print("  Progress table — one row per accepted replacement, "
              "blocked by sweep:")
        for k, v in (
            ("sweep",  "sweep counter; blank = same sweep as the row above"),
            ("repl",   "accepted replacements so far, = dead points"),
            ("mode",   "mode label · 'done' = finished modes pooled · "
                       "'Σ a/t' = all modes, a of t active"),
            ("live",   "live points in this row's scope (= feas + left)"),
            ("feas",   f"of those, certified: {self.threshold_text}"),
            ("left",   "still uncertified; the run stops at 0 on the Σ row"),
            ("feas%",  "feas / live"),
            (sym,      "worst live point here; on a mode row, the point "
                       "this step evicted"),
            ("ell",    "ellipsoids covering this mode ('—' once done)"),
            ("tries",  "candidates drawn for this replacement"),
            ("elapsed", "wall time since the run started"),
            ("note",   "why a mode stopped, and at which sweep"),
        ):
            print(f"    {k:<9} {v}")

    def header(self) -> None:
        if not self.enabled:
            return
        print()
        print("  " + "".join(f"{h:>{w}}" for h, w in self._cols) + "  note")
        print("  " + "─" * (self.width + 6))
        self._need_hdr   = False
        self._last_sweep = None     # first row after a header repeats it

    def event(self, text: str) -> None:
        """A one-off happening (mode split, mode finished, warning),
        printed between blocks."""
        if not self.enabled:
            return
        print(f"  · {text}")
        self._need_hdr = True

    # ---------------------------------------------------- rows
    def mode_row(self, *, sweep: int, repl: int, label: int, live: int,
                 feas: int, left: int, worst: float, ell: int,
                 tries: int) -> None:
        """One accepted replacement, scoped to the mode that made it.
        Silent unless ``begin_sweep`` selected this sweep — and then
        EVERY mode row of the sweep is printed, so the Σ row below
        aggregates exactly what is on screen."""
        # Accumulate unconditionally: a sweep the cadence skipped can
        # still need its Σ row if it turns out to be the last one.
        self._acted += 1
        self._tries += tries
        self._ell   += ell
        if not self.printing:
            return
        self._rows  += 1
        if self._need_hdr:
            self.header()
        sw = "" if sweep == self._last_sweep else f"{sweep:d}"
        self._last_sweep = sweep
        self._line([sw, f"{repl:d}", f"{label:d}", f"{live:d}", f"{feas:d}",
                    f"{left:d}", self._pct(feas, live), self._val(worst),
                    f"{ell:d}", f"{tries:d}", self._hms(self.elapsed())])
        self._shown = True

    def sweep_end(self, *, sweep: int, n_modes: int, n_active: int,
                  done: Optional[dict], total: dict) -> None:
        """Closes the sweep block with the pooled 'done' row (only when
        some mode has finished) and the Σ row. Skipped entirely for a
        sweep in which a single mode acted and nothing has finished —
        there the Σ row would copy the mode row above it verbatim."""
        acted, rows = self._acted, self._rows
        tries, ell  = self._tries, self._ell
        self._acted = self._rows = self._tries = self._ell = 0
        if not self.enabled:
            return

        # Remember this sweep unconditionally: if it turns out to be the
        # last one and the cadence skipped it, ``close_table`` replays it
        # so the table always ends on the sweep the run actually stopped
        # on, whichever way it stopped.
        self._tail = dict(sweep=sweep, n_modes=n_modes, n_active=n_active,
                          total=total, ell=ell, tries=tries, acted=acted)
        if not self.printing:
            return
        if rows < 2 and done is None:
            return

        if self._need_hdr:
            self.header()
        if done is not None:
            self._line(["", "", "done", f"{done['n']:d}", f"{done['feas']:d}",
                        f"{done['left']:d}", self._pct(done['feas'], done['n']),
                        self._val(done['worst']), "—", "—", ""],
                       note=done.get("note", ""))
        # The sweep number sits on the block's first mode row; only a
        # block with no mode row at all (every active mode finished at
        # its turn) needs the Σ row to carry it.
        self._line([f"{sweep:d}" if rows == 0 else "", "",
                    f"Σ {n_active}/{n_modes}",
                    f"{total['n']:d}", f"{total['feas']:d}",
                    f"{total['left']:d}",
                    self._pct(total['feas'], total['n']),
                    self._val(total['worst']), f"{ell:d}", f"{tries:d}",
                    self._hms(self.elapsed())])
        self._last_sweep = None
        self._shown      = True

    def close_table(self) -> None:
        """Called once when the loop exits. If the cadence happened to
        skip the final sweep, replay it as a single Σ row carrying its
        own sweep number — so the table's last line is ALWAYS the state
        the run stopped at, whether it converged, hit the all-modes-done
        warning, or ended on a sweep in which every remaining mode
        finished. A no-op when the final sweep was printed normally
        (always the case at ``log_every=1``). ``ell``/``tries`` read "—"
        when no mode took a turn: nothing was covered or drawn, and 0
        would read as a measurement rather than as "not applicable"."""
        t = self._tail
        if not self.enabled or self._shown or t is None:
            return
        if self._need_hdr:
            self.header()
        total = t["total"]
        self._line([f"{t['sweep']:d}", "", f"Σ {t['n_active']}/{t['n_modes']}",
                    f"{total['n']:d}", f"{total['feas']:d}",
                    f"{total['left']:d}",
                    self._pct(total['feas'], total['n']),
                    self._val(total['worst']),
                    f"{t['ell']:d}" if t["acted"] else "—",
                    f"{t['tries']:d}" if t["acted"] else "—",
                    self._hms(self.elapsed())])
        self._last_sweep = None
        self._shown      = True


class MultiNestSampler:
    """
    MultiNest-inspired sampler for probabilistic design-space
    characterisation.

    The sampler maintains a population of live design points and repeatedly
    replaces the point with the lowest feasibility merit by drawing from a
    union of ellipsoids fitted to the surviving live points. The ellipsoidal
    decomposition follows the main ideas of MultiNest, but the objective and
    stopping rule have been adapted from Bayesian evidence estimation to
    design-space certification.

    Coordinate systems
    ------------------
    Clustering, ellipsoid fitting, mode separation, and constrained sampling
    are performed in the unit hypercube ``[0, 1]^D``. The
    :class:`DesignSpace` object maps between unit coordinates and physical
    design coordinates.

    User-facing points, including live, dead, and rejected points, are stored
    in physical coordinates. Likewise, the feasibility estimator is always
    evaluated using physical design coordinates. User-defined process models
    therefore do not need to know about the internal unit-hypercube
    representation.

    Merit convention
    ----------------
    The sampler uses a single higher-is-better merit convention:

    - ``P``:
      ``merit(d) = P(feasible | d)``, with membership threshold
      ``alpha_star``.

    - ``VaR`` or ``CVaR``:
      ``merit(d) = -risk(d)``, with membership threshold ``0``.

    Consequently, eviction, candidate acceptance, per-mode freezing, and
    termination can all be written using the same comparisons:

    ``candidate_merit > current_minimum_merit``

    and

    ``merit >= merit_threshold``.

    Ellipsoidal decomposition and refitting
    ---------------------------------------
    Let ``X`` denote the internal remaining-volume variable associated with a
    mode and let ``ef`` be the sampling-efficiency parameter. The target total
    ellipsoid volume is proportional to

    ``X / ef``.

    Thus, decreasing ``ef`` increases the target ellipsoid volume, usually
    improving geometric coverage at the cost of lower rejection-sampling
    efficiency.

    For a decomposition ``S``, define

    ``F(S) = total_ellipsoid_volume / target_volume``.

    A full repartitioning is normally considered when ``F(S)`` exceeds
    ``F_threshold`` and either the current value is worse than its recent
    history or the periodic refit interval has elapsed. During the initial
    history warm-up, however, the periodic condition may trigger a refit
    even when ``F(S) <= F_threshold``.

    Termination
    -----------
    This implementation does not use Bayesian evidence, nested-sampling
    quadrature weights, or a remaining-evidence tolerance.

    Under normal termination, the run ends when every active live point
    satisfies the selected criterion:

    - ``P >= alpha_star`` for probability;
    - ``VaR <= 0`` for value at risk;
    - ``CVaR <= 0`` for conditional value at risk.

    A structural fallback may also stop a mode when no valid replacement can
    be generated, for example because too few points remain or because the
    live merit values have collapsed to an unrecoverable plateau.

    Important differences from MultiNest
    ------------------------------------
    This class uses MultiNest's ellipsoidal decomposition and constrained
    sampling ideas, but it is not a complete reimplementation of MultiNest.

    The main differences are:

    1. Bayesian evidence, posterior weights, Importance Nested Sampling, and
    evidence-based convergence are not implemented.

    2. The stopping rule is replaced by a design-space certification rule:
    the run terminates when all active live points satisfy the selected
    feasibility criterion.

    3. Ellipsoidal sampling is used from the beginning. The reference
    implementation initially samples from the full prior and switches to
    ellipsoidal sampling through ``eswitch`` when appropriate.

    4. Modes are decomposed in the global unit hypercube. The reference
    implementation's additional per-mode bounding-box rescaling is not
    applied.

    5. New modes start with an empty ellipsoidal decomposition and are fitted
    once when first processed. Unlike the reference implementation's forced
    rebuild path, this implementation does not perform repeated fitting
    attempts with progressively relaxed volume conditions.

    6. Candidate generation and model evaluation are sequential. The
    reference implementation can use MPI to evaluate multiple candidates
    in parallel.

    7. ``freeze_satisfied_modes`` applies a design-space membership criterion
    rather than MultiNest's evidence-based per-mode completion criterion.

    8. Multiple k-means restarts and k-means++ initialisation are optional
    extensions and are not part of the reference clustering procedure.

    Parameters
    ----------
    estimator : FeasibilityEstimator
        Evaluates the selected feasibility criterion and exposes a
        higher-is-better sampler merit.

    design_space : DesignSpace
        Defines the physical bounds of the design variables and provides the
        physical-to-unit and unit-to-physical coordinate transformations.

    N_L : int, default=500
        Total number of live points.

    alpha_star : float, default=0.95
        Target reliability for the probability criterion and confidence level
        used when evaluating VaR or CVaR.

    F_threshold : float, default=1.1
        Minimum decomposition-volume ratio ``F(S)`` required before a full
        repartitioning may be triggered.

        In the reference-style adaptive trigger, this is a necessary but not
        sufficient condition: the current ratio must also exceed its recent
        prediction, or the periodic refit condition must be due.

    fs_history_len : int, default=4
        Number of recent candidate decomposition ratios used to estimate the
        expected or predicted value of ``F(S)``.

        This corresponds to the role of ``neVol`` in the reference Fortran
        implementation.

    nsc_def : int, default=50
        Periodic component of the adaptive refit trigger. It prevents a mode
        from indefinitely avoiding reconsideration of its decomposition.

        This parameter does not override ``F_threshold``.

    min_pt : int, optional
        Minimum number of points retained in each child cluster produced by
        the recursive decomposition. Defaults to ``2 * (D + 1)``, taken from
        ``design_space.D``; pass an explicit value to override it.

        In ``D`` dimensions, at least ``D + 1`` points in general position are
        required for the sample covariance to be full rank. If
        ``min_pt < D + 1``, some final clusters contain too few points to
        determine the ellipsoid shape in every direction, and in those
        directions the shape is supplied by the covariance regularisation in
        ``Ellipsoid.fit`` rather than inferred from the data. ``2 * (D + 1)``
        is the smallest value leaving margin above that bound, and it matches
        the minimum population ``_isolate_modes`` already requires before a
        group is tracked as a separate mode.

        The reference implementation uses 2, which is below ``D + 1`` in every
        dimension and relies on that regularisation throughout. A sensitivity
        check over ``{2, D+1, 2(D+1), 4(D+1)}`` found sampling efficiency and
        coverage insensitive to the choice across that range, so the default
        is set on the covariance-rank argument rather than tuned
        (``min_pt_sensitivity.py``).

    ef : float, default=0.8
        Sampling-efficiency parameter. The target ellipsoid volume is
        proportional to ``X / ef`` [nested.F90 L1362, L1366].

        Smaller values produce larger ellipsoids and typically more robust
        coverage, but lower candidate-acceptance efficiency.

        The default follows the reference implementation's recommendation
        of 0.8 for parameter estimation, as opposed to 0.3 for evidence
        evaluation [MultiNest v3.12 README L156-157]. Feroz, Hobson &
        Bridges (2009), Section 8, states the 0.3 figure; the 0.8 figure
        appears only in the README.

    multimodal : bool, default=True
        Enable periodic separation of disconnected groups of ellipsoids into
        independently tracked modes.

    max_modes : int, default=100
        Maximum number of simultaneously tracked modes. Once this many modes
        exist, no further separation is attempted. It does not constrain the
        clustering used inside a single mode.

    freeze_satisfied_modes : bool, default=True
        Stop evolving a mode once all of its live points satisfy the selected
        design-space membership criterion.

        Freezing avoids spending further model evaluations on an already
        certified mode while other modes continue evolving. It does not alter
        the global membership condition used for normal termination.

    trace_decomposition : None, bool, or iterable of int, optional
        Controls collection of decomposition traces.

        - ``None`` or ``False``: tracing disabled.
        - ``True``: trace every refit attempt.
        - iterable of integers: trace only the specified sweep numbers.

        Tracing is observational and should not consume random numbers or
        modify sampler decisions.

    kmeans_restarts : int, default=1
        Number of independent initialisations used for each two-cluster
        k-means split. The result with the lowest within-cluster sum of
        squares is retained.

        ``1`` reproduces the single-start structure of the reference
        clustering routine.

    kmeans_init : {"random", "kmeans++"}, default="random"
        Initialisation method used for each k-means run.

        ``"random"`` selects distinct data points uniformly, matching the
        reference-style initialisation. ``"kmeans++"`` uses distance-weighted
        seeding and is an optional extension.

    em_mode : {"none", "multinest", "bugfix", "paper"}, default="bugfix"
        Refinement strategy applied after the initial two-cluster k-means
        partition.

        ``"none"`` keeps the k-means partition unchanged.

        ``"multinest"`` follows the control flow of the MultiNest v3.12
        ``Dmeans`` routine as written.

        ``"bugfix"`` retains h(u)-based point reassignments and follows the
        apparent intended behaviour of ``Dmeans``.

        ``"paper"`` repeatedly applies the h(u)-based refinement described
        in the MultiNest paper until the partition stabilises.

    References
    ----------
    Feroz, F. and Hobson, M. P. (2008).
        "Multimodal nested sampling: an efficient and robust alternative to
        Markov Chain Monte Carlo methods for astronomical data analyses."
        Monthly Notices of the Royal Astronomical Society, 384, 449-463.
        doi:10.1111/j.1365-2966.2007.12353.x
        arXiv:0704.3704.

    Feroz, F., Hobson, M. P. and Bridges, M. (2009).
        "MultiNest: an efficient and robust Bayesian inference tool for
        cosmology and particle physics."
        Monthly Notices of the Royal Astronomical Society, 398, 1601-1614.
        doi:10.1111/j.1365-2966.2009.14548.x
        arXiv:0809.3437.

    Feroz, F., Hobson, M. P., Cameron, E. and Pettitt, A. N. (2019).
        "Importance Nested Sampling and the MultiNest Algorithm."
        The Open Journal of Astrophysics, 2.
        doi:10.21105/astro.1306.2144
        arXiv:1306.2144.

    MultiNest v3.12 reference implementation:
        https://github.com/farhanferoz/MultiNest

        Relevant Fortran sources include ``nested.F90``,
        ``xmeans_clstr.f90``, and ``utils1.f90``.

    Arthur, D. and Vassilvitskii, S. (2007).
        "k-means++: The Advantages of Careful Seeding."
        Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete
        Algorithms, 1027-1035.
    """

    def __init__(
        self,
        estimator:      FeasibilityEstimator,
        design_space:   DesignSpace,
        N_L:            int   = 500,
        alpha_star:     float = 0.95,
        F_threshold:    float = 1.1,
        fs_history_len: int   = 4,
        nsc_def:        int   = 50,
        min_pt:         Optional[int] = None,
        ef:             float = 0.8,
        multimodal:     bool  = True,
        max_modes:      int   = 100,
        freeze_satisfied_modes: bool = True,
        trace_decomposition: Optional[object] = None,
        kmeans_restarts: int  = 1,
        kmeans_init:     str  = "random",
        em_mode:         str  = "bugfix",
    ) -> None:

        self.estimator      = estimator
        self.design_space   = design_space
        self.N_L             = N_L
        self.alpha_star      = alpha_star
        self.F_threshold     = F_threshold
        self.fs_history_len  = fs_history_len
        self.nsc_def         = nsc_def

        # Dimension-scaled default: D+1 points are the minimum for a
        # non-singular covariance in D dimensions, and 2*(D+1) is the smallest
        # value leaving margin above it. The reference behaviour is min_pt=2.
        self.min_pt          = (2 * (design_space.D + 1) if min_pt is None
                                else min_pt)
        self.ef              = ef
        self.multimodal      = multimodal
        self.max_modes       = max_modes
        self.freeze_satisfied_modes = freeze_satisfied_modes

        self.feas_criterion = getattr(estimator, "feas_criterion", "VaR")
        self.merit_thres    = estimator.merit_threshold(alpha_star)
        self.display        = CriterionDisplay(self.feas_criterion, alpha_star)

        if kmeans_restarts < 1:
            raise ValueError(
                f"kmeans_restarts must be >= 1, got {kmeans_restarts}")
        if kmeans_init not in ("random", "kmeans++"):
            raise ValueError("kmeans_init must be 'random' or "
                             f"'kmeans++', got {kmeans_init!r}")
        self.kmeans_restarts = kmeans_restarts
        self.kmeans_init     = kmeans_init

        if em_mode not in ("none", "multinest", "bugfix", "paper"):
            raise ValueError(
                "em_mode must be one of (none, multinest, bugfix, paper), "
                f"got {em_mode!r}")
        self.em_mode = em_mode

        # ------------------------------------------------------------
        # DECOMPOSITION TRACING (see Section 5b). 
        if trace_decomposition is None or trace_decomposition is False:
            self.trace_decomposition = None
        elif trace_decomposition is True:
            self.trace_decomposition = True
        else:
            self.trace_decomposition = set(trace_decomposition)
        self.decomp_traces: List[DecompositionTrace] = []

    # ----------------------------------------------------------
    def _trace_active(self, sweep: int) -> bool:
        td = self.trace_decomposition
        return td is True or (td is not None and sweep in td)

    def _traced_refit(self, m: "_Mode", X_i_target: float,
                      points: np.ndarray, sweep: int) -> dict:
        """``_maybe_refit`` plus optional capture. When tracing is off
        for this sweep it IS ``_maybe_refit`` — same object states, same
        RNG, same returned dict."""
        if not self._trace_active(sweep):
            return self._maybe_refit(m, X_i_target, points, sweep)
        m.decomposition.trace_sink = []
        try:
            info = self._maybe_refit(m, X_i_target, points, sweep)
        finally:
            nodes = m.decomposition.trace_sink or []
            m.decomposition.trace_sink = None
        tr = DecompositionTrace(sweep=sweep,
                                mode_label=m.label, X=m.X,
                                target=X_i_target, gate=info, nodes=nodes)
        self.decomp_traces.append(tr)
        print(tr.summary())
        return info

    def _maybe_refit(self, m: "_Mode", X_i_target: float,
                     points: np.ndarray, sweep: int) -> dict:
        """
        Decide whether mode ``m`` requires a new ellipsoidal decomposition.

        A refit is normally attempted when the current decomposition ratio

            F(S) = total ellipsoid volume / target volume

        exceeds ``F_threshold`` and either:

        - ``F(S)`` is worse than its recent refit history, or
        - the mode-specific periodic refit interval has elapsed.

        A mode with no ellipsoids is always fitted on first use. During the
        initial history-building phase, the periodic condition may also trigger
        a refit before the rolling ``F(S)`` history is full.

        When a candidate decomposition is produced, it replaces the current
        decomposition only if it has a smaller total volume. An initial
        decomposition is always accepted because no previous decomposition
        exists.

        The refit trigger reduces the computational cost of repeatedly running
        Algorithm 1, including recursive k-means and partition refinement. It
        does not reduce the number of process-model evaluations.

        Parameters
        ----------
        m : _Mode
            Mode whose decomposition is being considered.

        X_i_target : float
            Target total ellipsoid volume for the mode, normally ``m.X / ef``.

        points : np.ndarray
            Current live points belonging to the mode, in unit-hypercube
            coordinates.

        sweep : int
            Current sampler sweep number.

        Returns
        -------
        dict
            Diagnostic information describing the refit decision, candidate
            decomposition, and adoption result.

        Notes
        -----
        The adaptive trigger and adoption logic follow the corresponding
        MultiNest v3.12 routines in ``nested.F90`` and ``xmeans_clstr.f90``.
        """
        # Current decomposition ratio. An empty decomposition is assigned
        # infinity so that its initial fit is always requested.
        F_S = (m.decomposition.compute_F(X_i_target)
               if m.decomposition.ellipsoids else np.inf)
        
        # Estimate the expected F(S) from the most recent refit attempts.
        # ``fs_history`` is stored newest-first; only the newest
        # ``fs_history_len`` entries contribute to this prediction.
        # [nested.F90 L1373-1385]
        recent = m.fs_history[: self.fs_history_len]
        predicted_F_S = float(np.mean(recent)) if recent else 1.0

        # Each mode maintains its own adaptive refit interval ``m.nsc``.
        # A refit is due once that many sweeps have passed since the mode's
        # previous refit attempt.
        # [nested.F90 L1347, L1388-1389]
        periodic_due = (sweep - m.last_refit_at) >= m.nsc

        # Outside the history warm-up case below, a populated decomposition is
        # reconsidered only when F(S) exceeds the threshold and is either
        # worsening relative to recent attempts or periodically due.
        trending_worse_or_due = (predicted_F_S < F_S) or periodic_due
        need_refit = (
            len(m.decomposition.ellipsoids) == 0
            or (F_S > self.F_threshold and trending_worse_or_due)
        )

        # Cancel the refit when every ellipsoid is already enlarged by its
        # target-volume floor. In that case, repartitioning cannot reduce the
        # total volume through a tighter fit to the points.
        # [nested.F90 L1391-1396]
        eff_cancelled = False
        if (need_refit and m.decomposition.ellipsoids
                and all(e.eff > 1.00001
                        for e in m.decomposition.ellipsoids)):
            need_refit = False
            eff_cancelled = True

        # While the F(S) history is still being filled, perform a refit whenever
        # the mode-specific periodic interval is reached, even if the normal
        # F(S) trigger is not satisfied. This provides additional candidate
        # F(S) values for the rolling history.
        #
        # This check comes after the efficiency-based cancellation, so it may
        # re-enable a refit that was cancelled above, matching the reference
        # implementation.
        # [nested.F90 L1398]
        warmup_forced = False
        if (not need_refit
                and len(m.fs_history) < 2 * self.fs_history_len
                and periodic_due
                and m.decomposition.ellipsoids):
            need_refit = True
            warmup_forced = True

        info = dict(need_refit=need_refit, forced=False, F_S=F_S,
                    old_total_V=None, new_total_V=None, accepted=None,
                    # --- gate internals (trace/diagnostics; additive) ---
                    predicted_F_S=predicted_F_S, periodic_due=periodic_due,
                    nsc=m.nsc, sweeps_since_refit=sweep - m.last_refit_at,
                    eff_cancelled=eff_cancelled, warmup_forced=warmup_forced,
                    n_ell_old=len(m.decomposition.ellipsoids),
                    F_threshold=self.F_threshold)

        if not need_refit:
            return info

        # ``forced`` means that the mode currently has no valid decomposition.
        # This occurs for the initial mode and for newly separated child modes.
        # It does not mean that a previous fit raised an error or failed.
        forced         = len(m.decomposition.ellipsoids) == 0

        # Preserve the current decomposition so it can be restored if the
        # candidate does not improve the total ellipsoid volume.
        # [nested.F90 L1417-L1420]
        old_ellipsoids = m.decomposition.ellipsoids
        old_owner      = m.decomposition.point_owner
        old_total_V    = (sum(e.V for e in old_ellipsoids)
                          if old_ellipsoids else np.inf)

        # Construct a candidate decomposition using Algorithm 1. A re-decomposition 
        # is an attempt and does not automatically replace the current one.
        # [nested.F90 L1459-L1517]
        m.decomposition.fit(points, X_i_target)
        new_total_V = sum(e.V for e in m.decomposition.ellipsoids)

        # Record the refit attempt regardless of whether the candidate is later
        # adopted. This corresponds to the reference's ``eswitchff`` update.
        # [nested.F90 L1426]
        m.last_refit_at = sweep   

        # Record the candidate F(S), including candidates that are later rejected.
        # The buffer stores at most ``2 * fs_history_len`` entries, newest first.
        # Only the newest ``fs_history_len`` entries are used above to calculate
        # ``predicted_F_S``.
        # [nested.F90 L841; xmeans_clstr.f90 L2112-2114, L2308-2310]
        F_S_candidate = (new_total_V / X_i_target
                         if X_i_target > 1e-300 else np.inf)
        m.fs_history.insert(0, F_S_candidate)
        del m.fs_history[2 * self.fs_history_len:]

        # The first decomposition is always accepted. Later candidates are
        # accepted only when they reduce the total ellipsoid volume.
        accepted_refit = forced or (new_total_V < old_total_V)
        if accepted_refit:
            m.nsc = max(1, m.nsc - 10)          
        else:
            # Restore the previous decomposition and reduce the frequency of
            # future refit attempts for this mode.
            # [nested.F90 L1509-1511]
            m.decomposition.ellipsoids  = old_ellipsoids
            m.decomposition.point_owner = old_owner
            m.nsc = m.nsc + 10

        info.update(forced=forced, old_total_V=old_total_V,
                    new_total_V=new_total_V, accepted=accepted_refit,
                    n_ell_new=len(m.decomposition.ellipsoids),
                    F_S_candidate=F_S_candidate)
        return info

    # ----------------------------------------------------------
    def run(
        self,
        seed:           Optional[int] = None,
        frame_callback: Optional[Callable[["RunFrame"], None]] = None,
        log_every:      Union[int, str, None] = 1,
        log_heartbeat:  int = 100,
    ) -> SamplerResult:
        """
        Execute the nested-sampling loop with mode separation.

        This is a trimmed, object-oriented port of ``clusteredNest`` from
        ``nested.F90`` (L639-2597). The implementation
        follows MultiNest's source-level multi-mode structure.

        Multi-mode operation
        --------------------
        Each active ``_Mode`` represents an independent nested-sampling
        sub-run with:

        - its own subset of live points,
        - its own remaining-volume estimate ``X``,
        - its own ``EllipsoidalDecomposition``.

        The mode-specific ``X`` corresponds to MultiNest's ``ic_vnow``. In
        this implementation it is retained only as the target used when
        sizing and refitting ellipsoids. The reference implementation's
        evidence and information accumulators, such as ``ic_Z`` and
        ``ic_info``, are intentionally omitted.

        One outer sweep processes every active mode once. For each mode, the
        lowest-merit live point is evicted and replaced by a point satisfying
        that mode's current likelihood or merit constraint.

        The remaining-volume estimate of a mode is updated using its own
        number of live points:

        ``X <- X * exp(-1 / n_mode)``

        Therefore, the shrinkage factor is based on the mode-local live-point
        count rather than the global number of live points.

        Two iteration counters are maintained:

        - ``sweep`` counts outer sweeps;
        - ``n_replacements`` counts accepted replacements across all modes.

        With one active mode, both counters advance together. With multiple
        active modes, one sweep may produce several replacements.

        Every 15 sweeps, active modes are checked for further separation by
        ``_isolate_modes``. A confirmed split divides the parent's ``X``
        among its children in proportion to their live-point counts.

        Progress logging
        ----------------
        Progress is displayed as one row per mode:

        ``sweep  repl  mode  live  feas  left  feas%  criterion  ell  tries
        elapsed  note``

        The columns have the following meanings:

        - ``sweep``: outer-sweep index;
        - ``repl``: cumulative number of accepted replacements;
        - ``live``: current number of live points in the mode;
        - ``feas``: live points satisfying the final feasibility threshold;
        - ``left``: live points that do not yet satisfy the threshold;
        - ``ell``: number of ellipsoids in the mode decomposition;
        - ``tries``: candidate draws required for the accepted replacement.

        For every mode,

        ``live = feas + left``.

        A completed printed sweep includes a ``Σ`` row containing the exact
        column-wise totals across all active and completed modes. The run
        terminates when the total ``left`` value reaches zero.

        Logging is controlled by ``log_every``:

        - ``1``: print every sweep;
        - integer greater than 1: print every n-th sweep;
        - percentage string such as ``"2%"``: print when the total feasible
        percentage advances by that amount;
        - ``0`` or ``None``: disable regular progress output.

        In percentage mode, ``log_heartbeat`` forces a progress block after
        the specified number of sweeps even when the feasible percentage has
        not advanced.

        Logging is cosmetic and does not affect random-number generation or
        sampler state. Runs with the same ``seed`` therefore produce the same
        result regardless of ``log_every``.

        Parameters
        ----------
        seed : int, optional
            Random seed used for reproducibility.

        frame_callback : callable, optional
            Read-only observer called with a ``RunFrame`` at notable events,
            including ``"init"``, ``"candidate"``, ``"accepted"``,
            ``"refit"``, ``"mode_split"``, ``"mode_frozen"`` and
            ``"converged"``.

            The callback is used by ``RunRecorder``, ``save_run_gif`` and
            ``RunPlayer``. It must not mutate sampler state or draw random
            numbers. When it is ``None``, no frame snapshots are constructed.

        log_every : int, str or None, default=100
            Progress-table output frequency.

        log_heartbeat : int, default=100
            Maximum number of silent sweeps in percentage-based logging mode.
            Ignored for other logging modes.

        Returns
        -------
        SamplerResult
            Result containing the final live points, evicted points,
            rejected candidates, criterion values and run counters.

        """
        if seed is not None:
            np.random.seed(seed)

        D = self.design_space.D

        # Logging only. Constructed before the initial live-point evaluations
        # so that elapsed time includes the complete computational run.
        _plog = _ProgressLog(
            self.display.symbol, self.N_L,
            threshold_text=(
                f"{self.display.symbol} "
                f"{'>=' if self.display.is_probability else '<='} "
                f"{self.display.threshold_value:g}"),
            log_every=log_every, heartbeat=log_heartbeat)

        # Draw the initial live-point population uniformly in the unit
        # hypercube, then map it to physical design coordinates for model
        # evaluation.
        live_points_u    = np.random.uniform(0.0, 1.0, size=(self.N_L, D))
        live_points_phys = self.design_space.to_physical(live_points_u)

        # "K": how many model runs ONE estimate() call costs.
        _n_theta = int(getattr(self.estimator, "N_theta", 1) or 1)
        _unc     = getattr(self.estimator, "uncertainty", None)
        K = int(_unc.n_scenarios(_n_theta)) if _unc is not None else _n_theta

        # Describe the condition required for all live points at termination.
        _term = (f"P >= {self.alpha_star:g}" if self.display.is_probability
                 else f"{self.display.symbol} <= 0  (alpha = "
                      f"{self.display.tail_probability:g})")

        # Settings first, before anything slow happens: on a black-box model
        # the seed evaluation below can run for a quarter of an hour, and
        # whoever is watching should be able to read what is running while
        # they wait rather than after.
        _plog.block(
            "NESTED SAMPLING",
            [("Design space", f"{self.design_space!r}"),
             ("Dimension", f"D = {D}"),
             ("Live points", f"{self.N_L}"),
             ("Uncertainty", f"{self.estimator.uncertainty!r}"),
             ("", f"K = {K} model runs per estimate"
                  + ("" if K == _n_theta
                     else f"  (N_theta = {_n_theta} requested, ignored "
                          f"by this uncertainty)")),
             ("Criterion", f"{self.feas_criterion} — feasible when "
                           f"{self.display.symbol} "
                           f"{'>=' if self.display.is_probability else '<='} "
                           f"{self.display.threshold_value:g}"),
             ("Mode separation", (f"on, every 15 sweeps, max "
                                  f"{self.max_modes} modes"
                                  if self.multimodal else "off")),
             ("Ellipsoids", f"ef = {self.ef}, min_pt = {self.min_pt}"),
             ("Termination", f"all live points {_term}")],
        )

        # The seed population is the one place in the run where evaluations
        # are independent of each other -- everything after depends on the
        # verdict of the point before -- so it is the only place an
        # estimator can fan out without changing the algorithm.
        _seed_pairs      = self.estimator.batch_merit_and_P(
                               live_points_phys, self.alpha_star)
        live_merit       = np.array([mp[0] for mp in _seed_pairs])
        live_probs       = np.array([mp[1] for mp in _seed_pairs])
        
        # Final design-space membership threshold expressed in merit space:
        #  alpha_star for P
        #  0 for negated VaR or CVaR
        M_thr            = self.merit_thres   

        dead_points:    List[np.ndarray] = []
        dead_merit:     List[float]      = []   
        dead_probs:     List[float]      = []   

        rejected_pts:   List[np.ndarray] = []
        rejected_merit: List[float]      = []
        rejected_probs: List[float]      = []

        # Number of replacement candidates evaluated so far, including both
        # accepted and rejected candidates. Initial live-point evaluations are excluded.
        n_candidate_evals = 0

        # Outer-sweep counter, corresponding to MultiNest's ``ff``.
        sweep             = 0  

        # Number of accepted replacements across all modes, corresponding to
        # MultiNest's ``globff``. This is also the number of dead points.
        n_replacements    = 0   

        # Counter used to give each mode a unique number.
        # It starts at 0 and increases whenever a new mode is created.              
        _mode_label_counter = [0]

        def _next_mode_label() -> int:
            """Increase the mode counter and return the new mode number."""
            _mode_label_counter[0] += 1
            return _mode_label_counter[0]

        def _fresh_mode(idx: np.ndarray, X: float) -> _Mode:
            """    
            Create a new active mode for a subset of the live-point population.

            The mode owns the live-point rows specified by ``idx`` and receives
            the remaining-volume target ``X``. Its ellipsoidal decomposition is
            initially empty; the main sampling loop performs the first fit when
            the mode is processed.

            This applies both to the initial mode and to child modes created
            after mode separation.

            Parameters
            ----------
            idx : np.ndarray
                Row indices of the global live-point arrays belonging to the mode.

            X : float
                Mode-specific remaining-volume value used as the target when
                sizing and refitting its ellipsoidal decomposition.

            Returns
            -------
            _Mode
                A newly initialised active mode with a unique stable label and
                an empty ellipsoidal decomposition.
            """
            return _Mode(
                label=_next_mode_label(),
                idx=np.asarray(idx), X=float(X), done=False,
                decomposition=EllipsoidalDecomposition(
                    D=self.design_space.D,
                    domain_bounds=(0.0, 1.0), min_pt=self.min_pt,
                    kmeans_restarts=self.kmeans_restarts,
                    kmeans_init=self.kmeans_init,
                    em_mode=self.em_mode),
                last_refit_at=0, fs_history=[], nsc=self.nsc_def,
            )

        # Start with one mode containing all N_L live points.
        # Its initial remaining-volume value is X = 1.0.
        modes: List[_Mode] = [_fresh_mode(np.arange(self.N_L), 1.0)]

        def _emit(event: str,
                  worst_point=None, worst_merit=None,
                  candidate=None, candidate_merit=None, candidate_ok=None,
                  F_S=None, note: str = "") -> None:
            """READ-ONLY observer hook (see ``frame_callback`` in the
            docstring). Draws no random numbers, mutates nothing;
            builds a ``RunFrame`` snapshot only when a callback is
            actually installed, so the hook-less path costs one
            ``if`` per event."""
            if frame_callback is None:
                return
            frame_callback(RunFrame(
                event            = event,
                sweep            = sweep,
                n_replacements   = n_replacements,
                n_candidate_evals = n_candidate_evals,
                live_points      = live_points_phys.copy(),
                live_merit       = live_merit.copy(),
                dead_ref         = dead_points,
                n_dead           = len(dead_points),
                rejected_ref     = rejected_pts,
                rejected_merit_ref = rejected_merit,
                n_rejected       = len(rejected_pts),
                worst_point      = None if worst_point is None else np.array(worst_point),
                worst_merit      = worst_merit,
                candidate        = None if candidate is None else np.array(candidate),
                candidate_merit  = candidate_merit,
                candidate_ok     = candidate_ok,
                mode_ellipsoids  = [list(mm.decomposition.ellipsoids) for mm in modes],
                mode_done        = [mm.done for mm in modes],
                mode_ids         = [mm.label for mm in modes],
                F_S              = F_S,
                below_alpha      = int(np.sum(live_merit < M_thr)),
                alpha_star       = self.alpha_star,
                design_space     = self.design_space,
                note             = note,
                feas_criterion   = self.feas_criterion,
                merit_threshold  = M_thr,
            ))


        # Store the sweep at which each mode became done.
        _done_at: dict = {}

        def _done_note() -> str:
            """
            Return a short summary showing which modes stopped,
            why they stopped, and at which sweep.
            """
            groups: dict = {}
            for mm in modes:
                if not mm.done:
                    continue
                kind = ("frozen" if mm.done_reason.startswith("frozen")
                        else "exhausted")
                groups.setdefault(kind, []).append(
                    (mm.label, _done_at.get(mm.label)))
            parts = []
            for kind in ("frozen", "exhausted"):
                items = sorted(groups.get(kind, []))
                if not items:
                    continue
                shown = " ".join(
                    f"[{lab}]@{sw if sw is not None else '?'}"
                    for lab, sw in items[:6])
                if len(items) > 6:
                    shown += f" +{len(items) - 6} more"
                parts.append(f"{kind} {shown}")
            return "  ·  ".join(parts)

        def _report_mode_done(m: _Mode) -> None:
            """
            Log when a mode becomes done.

            The log includes the sweep, mode label, stopping reason,
            number of live points, criterion range, and change in the
            number of active modes.
            """
            _done_at[m.label] = sweep
            active_now = sum(1 for mm in modes if not mm.done)
            n          = int(m.idx.shape[0])
            sym = self.display.symbol
            if n > 0:
                sub   = self.display.from_merit(live_merit[m.idx])
                p_str = (f"{sym}_min={float(np.min(sub)):.3g} | "
                         f"{sym}_max={float(np.max(sub)):.3g}")
            else:
                p_str = f"{sym}_min=n/a | {sym}_max=n/a"
            _plog.event(
                f"sweep {sweep}: mode [label {m.label}] done — "
                f"{m.done_reason} | n={n} | {p_str} | "
                f"active {active_now + 1} -> {active_now}")

        
        def _mode_hilike_lowlike(m: _Mode):
            """Return the highest and lowest live-point merit values in a mode."""
            sub = live_merit[m.idx]
            return float(np.max(sub)), float(np.min(sub))
        
        def _update_mode_done(m: _Mode) -> None:
            """
            Mark a mode as done when it cannot continue sampling safely.

            A mode is stopped for one of two structural reasons:

            1. It has fewer than ``D + 1`` live points, so a full-dimensional
            ellipsoid cannot be fitted.
            2. All live points have the same merit value, so the strict
            replacement condition ``candidate_merit > lowest_merit`` may
            never be satisfied.

            These checks are safety guards, not convergence tests. They do not
            use evidence, nested-sampling weights or remaining-volume estimates.

            For the probability criterion, the plateau test follows MultiNest's
            log-space likelihood comparison. For VaR and CVaR, only exact merit
            equality is checked because their merit values may be negative or
            infinite.
            """
            n = m.idx.shape[0]

            # At least D + 1 points are required to fit a D-dimensional ellipsoid.
            if n < D + 1:
                m.done = True
                m.done_reason = f"exhausted: only {n} < D+1 points left"
                return

            hilike, lowlike = _mode_hilike_lowlike(m)

            # VaR and CVaR can have signed or infinite merit values, so check only 
            # whether all live points have exactly the same merit value, including 
            # the all--inf case (exact degeneracy)
            if self.feas_criterion != "P":
                if hilike <= lowlike:
                    m.done = True
                    val = self.display.from_merit(np.asarray(hilike))
                    m.done_reason = (
                        f"exhausted: merit plateau (all points at "
                        f"{self.display.symbol}={float(val):.4g})")
                return

            # If every live point has P = 0, no strictly better candidate may be
            # reachable and the rejection loop could continue indefinitely.
            if hilike <= 0.0:
                m.done = True   
                m.done_reason = "exhausted: merit plateau (all points at P=0)"
                return

            # Compare the lowest and highest probabilities in log space
            # to detect whether the mode has reached a plateau.
            if lowlike > 0.0 and abs(np.log(lowlike) - np.log(hilike)) <= 0.0001:
                m.done = True
                m.done_reason = (f"exhausted: merit plateau "
                                 f"(all points at P={hilike:.4g})")
                return


        # Which of the two exits below the run took. Reported on the
        # result object so a caller can tell a converged run from a
        # stopped one without reading the log.
        _term_reason = "converged"

        def _all_done() -> bool:
            """
            Return True when the sampling run should stop.

            The run normally stops when every live point satisfies the selected
            merit threshold. It also stops as a safety fallback when all modes
            are marked done and no further live-point updates are possible.
            """
            nonlocal _term_reason

            # Normal termination: every live point satisfies the threshold.
            if not np.any(live_merit < M_thr):
                return True

            # Safety fallback: all modes have stopped, so live merits can no
            # longer change. Terminate with a warning if some points remain
            # below the threshold.
            if all(m.done for m in modes):
                _term_reason = "modes_exhausted"
                n_frozen = sum(
                    1 for m in modes
                    if m.done_reason.startswith("frozen"))
                _plog.event(
                    "WARNING: all modes done "
                    f"({n_frozen} frozen, "
                    f"{len(modes) - n_frozen} structurally exhausted — "
                    "plateau or too few points) before every live point "
                    f"reached {self.display.symbol} "
                    f"{'>=' if self.display.is_probability else '<='} "
                    f"{self.display.threshold_value:g}; terminating with "
                    f"{int(np.sum(live_merit < M_thr))} live point(s) "
                    "below the threshold (all of them in the exhausted "
                    "modes — frozen modes cannot hold sub-threshold "
                    "points).")
                return True
            
             # At least one active mode can still improve its live points.
            return False

        # Print explanations for the progress-table columns.
        _plog.legend()

        _emit("init")

        while not _all_done():
            sweep += 1   

            # Logging only: decide whether to print this complete sweep.
            _plog.begin_sweep(sweep, int(np.sum(live_merit >= M_thr)))
        

        # ------------------------------------------------------------------
        # Periodic mode separation
        # ------------------------------------------------------------------
        # In multimodal runs, each active mode is checked every 15 sweeps for
        # disconnected live-point groups, following MultiNest's mode-isolation
        # logic [nested.F90 L1196; isolateModes2, L2816-3190].
        #
        # If a mode splits, each child is tracked independently with its own
        # live-point subset, prior volume X, and ellipsoidal decomposition.
        # The parent's X is divided in proportion to child live-point counts:
        #
        #     X_child = X_parent * (n_child / n_parent)
        #
        # following nested.F90 L1259-1263. The total number of tracked modes is
        # limited by ``max_modes`` (MultiNest ``maxCls`` / ``maxmodes``).
        # Evidence/information bookkeeping from MultiNest is omitted here.
        # ------------------------------------------------------------------

            if (self.multimodal and sweep != 1 and sweep % 15 == 0
                    and len(modes) < self.max_modes):

                new_modes: List[_Mode] = []
                for m in modes:
                    # Finished modes, or modes containing too few live points to
                    # form two valid groups, are not tested for further separation.
                    if m.done or m.idx.shape[0] < 2 * self.min_pt:
                        new_modes.append(m)
                        continue

                    # Search for disconnected groups within the current mode.
                    # [nested.F90, isolateModes2, L2816-3190.]
                    groups = _isolate_modes(live_points_u[m.idx], self.min_pt)

                    # No distinct sub-modes were identified.
                    if len(groups) <= 1:
                        new_modes.append(m)
                        continue

                    n_parent = m.idx.shape[0]
                    child_sizes = []
                    for g in groups:
                        # Split the parent's remaining prior volume according to
                        # the child's share of the parent's live points.
                        # [nested.F90 L1259-1263]
                        frac = g.shape[0] / n_parent
                        child = _Mode(
                            label=_next_mode_label(),
                            idx=m.idx[g], X=m.X * frac,
                            done=False,
                            decomposition=EllipsoidalDecomposition(
                                D=self.design_space.D,
                                domain_bounds=(0.0, 1.0), min_pt=self.min_pt,
                                kmeans_restarts=self.kmeans_restarts,
                                kmeans_init=self.kmeans_init,
                                em_mode=self.em_mode),
                            last_refit_at=sweep, fs_history=[],
                            nsc=self.nsc_def,
                        )
                        new_modes.append(child)
                        child_sizes.append(g.shape[0])

                    _step(
                        "Mode separation (isolateModes2, nested.F90 "
                        "L2816-3190)",
                        parent_size=n_parent, n_new_modes=len(groups),
                        new_mode_sizes=child_sizes,
                    )
                split_happened = len(new_modes) != len(modes)
                if split_happened:
                    _plog.event(f"sweep {sweep}: mode separation "
                                f"{len(modes)} -> {len(new_modes)} modes "
                                f"(sizes "
                                f"{[int(mm.idx.shape[0]) for mm in new_modes]})")
                modes = new_modes
                if split_happened:
                    _emit("mode_split",
                          note=f"mode separation -> {len(modes)} modes")

            # [nested.F90 L1739]
            # Process each active mode once during this sweep.
            for m in modes:
                if m.done:
                    continue

                # Freeze the mode when all of its live points satisfy the final
                # feasibility threshold. Further replacements are unnecessary because
                # the mode has already been fully certified.
                # This check is performed before refitting to avoid rebuilding the
                # ellipsoidal decomposition of a mode that will no longer be sampled.
                if (self.freeze_satisfied_modes
                        and m.idx.shape[0] > 0
                        and float(np.min(live_merit[m.idx])) >= M_thr):
                    m.done = True

                    # Store a human-readable reason for the final log.
                    m.done_reason = (
                        f"frozen: all {m.idx.shape[0]} live points "
                        f"certified ({self.display.symbol} "
                        f"{'>=' if self.display.is_probability else '<='} "
                        f"{self.display.threshold_value:g})")
                    
                    # Record and report that the mode has stopped.
                    _report_mode_done(m)  

                    _emit("mode_frozen",
                          note=f"mode frozen: all points certified at "
                               f"{self.display.symbol} = "
                               f"{self.display.threshold_value:g}")

                    # Continue with the next mode
                    continue

                # --- VaR/CVaR pre-turn plateau guard -----------------
                # For VaR and CVaR, stop the mode before candidate sampling if all
                # live points have exactly the same merit. Otherwise, the strict
                # acceptance condition merit_star > merit_min may never be satisfied.
                if self.feas_criterion != "P":
                    _sub = live_merit[m.idx]
                    if float(np.max(_sub)) <= float(np.min(_sub)):
                        m.done = True

                        # Store a human-readable reason for the final log.
                        _val = float(self.display.from_merit(
                            np.asarray(np.max(_sub))))
                        m.done_reason = (
                            f"exhausted: merit plateau (all points at "
                            f"{self.display.symbol}={_val:.4g})")

                        # Record and report that the mode has stopped.
                        _report_mode_done(m)
                        _emit("mode_frozen",
                              note="merit plateau: no candidate can be "
                                   "strictly better")   

                        # Continue with the next mode
                        continue

                # Select the live point with the lowest merit in this mode.
                sub_idx    = m.idx
                sub_merit  = live_merit[sub_idx]
                local_min  = int(np.argmin(sub_merit))
                idx_min    = int(sub_idx[local_min])

                # Store the selected point in both unit-hypercube and physical coordinates.
                d_min_u    = live_points_u[idx_min].copy()
                d_min_phys = live_points_phys[idx_min].copy()
                merit_min      = float(live_merit[idx_min])

                # Update this mode's X value using its own number of live points.
                # The global number of live points N_L is not used here.
                # [nested.F90 L1780]
                n_mode = sub_idx.shape[0]
                shrink = np.exp(-1.0 / n_mode)
                X_next = m.X * shrink

                # Current target volume used for ellipsoid fitting and refit checks.
                X_i_target = m.X / self.ef

                # # Check whether this mode needs a new ellipsoidal decomposition.
                # [nested.F90 L1346-1517]. 
                _rf = self._traced_refit(m, X_i_target,
                                         live_points_u[sub_idx], sweep)
                F_S = _rf["F_S"]
                if _rf["need_refit"]:
                    forced         = _rf["forced"]
                    old_total_V    = _rf["old_total_V"]
                    new_total_V    = _rf["new_total_V"]
                    accepted_refit = _rf["accepted"]

                    _step(
                        f"Mode re-decomposition attempt (nested.F90 "
                        f"L1459-1517)",
                        forced_rebuild=forced,
                        old_total_volume=old_total_V,
                        new_total_volume=new_total_V,
                        ADOPTED=accepted_refit,
                        mode_nsc_after=m.nsc,
                    )
                    _emit("refit",
                          F_S=m.decomposition.compute_F(X_i_target),
                          note=("re-decomposition ADOPTED"
                                if accepted_refit else
                                "re-decomposition attempted, kept old"))

                # Draw candidates until one has a higher merit than the point being removed.
                n_reject_this_mode = 0
                while True:
                    # Sample a candidate from the union of this mode's ellipsoids.
                    d_star_u, k_star, _ = m.decomposition.sample_from_union()
                    
                    # Convert the candidate to physical coordinates and evaluate it.
                    d_star_phys = self.design_space.to_physical(d_star_u)
                    merit_star, prob_star = self.estimator.merit_and_P(
                        d_star_phys, self.alpha_star)
                    n_candidate_evals += 1
                    
                    # A candidate can replace the current worst point only if its merit
                    # is strictly higher.
                    accept_replacement = merit_star > merit_min

                    _step(
                        f"NS sweep {sweep} mode {m.label} — "
                        f"candidate replacement test ({self.feas_criterion})",
                        d_star_phys=d_star_phys,
                        merit_star=merit_star, merit_min=merit_min,
                        ACCEPT_merit_star_greater_than_merit_min=accept_replacement,
                        visualize=lambda: _visualize_state(
                            live_points_phys[sub_idx], m.decomposition.ellipsoids,
                            highlight=d_min_phys, highlight2=d_star_phys,
                            title=f"Sweep {sweep}: evict(x)={merit_min:.3f}  "
                                  f"candidate(*)={merit_star:.3f}",
                            label1="point to evict", label2="candidate"),
                    )

                    if accept_replacement:
                        n_replacements += 1

                        # Record the removed live point as a dead point.
                        dead_points.append(d_min_phys)
                        dead_merit.append(merit_min)
                        dead_probs.append(float(live_probs[idx_min]))

                        # Update the mode's ellipsoids after removing the old point and
                        # inserting the accepted candidate.
                        m.decomposition.evolve_step(
                            evicted_row          = local_min,
                            evicted_point        = d_min_u,
                            new_point            = d_star_u,
                            chosen_ellipsoid_id  = k_star,
                            N_total              = n_mode,
                            X_next               = X_next / self.ef,
                        )

                        # Replace the evicted point in the global live-point arrays.
                        live_points_u[idx_min]    = d_star_u
                        live_points_phys[idx_min] = d_star_phys
                        live_merit[idx_min]       = merit_star
                        live_probs[idx_min]       = prob_star
                        
                        # Replace the evicted point in the global live-point arrays.
                        m.X = X_next   

                        # Log the state of this mode after the replacement.
                        _sub_merit = live_merit[m.idx]
                        _m_n       = int(m.idx.shape[0])
                        _m_left    = int(np.sum(_sub_merit < M_thr))
                        _plog.mode_row(
                            sweep = sweep,
                            repl  = n_replacements,
                            label = m.label,
                            live  = _m_n,
                            feas  = _m_n - _m_left,
                            left  = _m_left,
                            worst = float(self.display.from_merit(
                                np.asarray(float(np.min(_sub_merit))))),
                            ell   = len(m.decomposition.ellipsoids),
                            # candidates drawn for this one replacement:
                            # the rejects plus the accepted draw itself.
                            tries = n_reject_this_mode + 1,
                        )
                        _emit("accepted",
                              worst_point=d_min_phys, worst_merit=merit_min,
                              candidate=d_star_phys, candidate_merit=merit_star,
                              candidate_ok=True,
                              F_S=m.decomposition.compute_F(m.X / self.ef))

                        # This mode has completed its replacement for the current sweep.
                        break

                    else:
                         # Store the unsuccessful candidate and draw another one.
                        n_reject_this_mode += 1
                        rejected_pts.append(d_star_phys.copy())
                        rejected_merit.append(merit_star)
                        rejected_probs.append(prob_star)

                        _emit("candidate",
                              worst_point=d_min_phys, worst_merit=merit_min,
                              candidate=d_star_phys, candidate_merit=merit_star,
                              candidate_ok=False, F_S=F_S)

                # After the successful replacement, check whether the mode has become
                # structurally unable to continue, for example because of a plateau.
                _was_done = m.done          
                _update_mode_done(m) 
                
                # Report only a new False -> True done transition.       
                if m.done and not _was_done:
                    _report_mode_done(m)

            # --- end-of-sweep aggregate rows (LOGGING ONLY) ----------
            # The Σ row is printed only when it says something the mode
            # rows above cannot: two or more modes acted, or a mode has
            # finished and so stopped producing rows. The pooled ``done``
            # row above it covers exactly those finished modes, so the
            # rows above Σ always add up.
            _done_modes = [mm for mm in modes if mm.done]
            _done_row   = None
            if _done_modes:
                _d_n    = int(sum(mm.idx.shape[0] for mm in _done_modes))
                _d_left = int(sum(int(np.sum(live_merit[mm.idx] < M_thr))
                                  for mm in _done_modes))
                _d_mins = [float(np.min(live_merit[mm.idx]))
                           for mm in _done_modes if mm.idx.shape[0] > 0]
                _done_row = dict(
                    n     = _d_n,
                    feas  = _d_n - _d_left,
                    left  = _d_left,
                    worst = (float(self.display.from_merit(
                        np.asarray(min(_d_mins)))) if _d_mins else None),
                    note  = _done_note(),
                )

            _t_left = int(np.sum(live_merit < M_thr))
            _plog.sweep_end(
                sweep    = sweep,
                n_modes  = len(modes),
                n_active = sum(1 for mm in modes if not mm.done),
                done     = _done_row,
                total    = dict(
                    n     = int(self.N_L),
                    feas  = int(self.N_L) - _t_left,
                    left  = _t_left,
                    worst = float(self.display.from_merit(
                        np.asarray(float(np.min(live_merit))))),
                ),
            )

        # LOGGING ONLY: make sure the table's last line is the sweep the
        # run actually stopped on, even if the cadence skipped it.
        _plog.close_table()

        rej_arr  = np.array(rejected_pts)  if rejected_pts  else np.empty((0, D))
        rej_merit_arr = np.array(rejected_merit) if rejected_merit else np.empty(0)

        _emit("converged", note="run finished")


        # Counter conventions used below:
        #   candidate evals = accepted + rejected proposals; the initial
        #                     N_L seed is NOT a proposal.
        #   sampling eff.   = accepted / candidate evals (seed excluded).
        #   model runs      = (seed + candidates) * K
        sto_dead  = len(dead_points)
        sto_cand  = n_candidate_evals                  # seed excluded
        sto_runs  = (self.N_L + n_candidate_evals) * K # seed included

        def _eff(acc, cand):
            return f"{100.0*acc/cand:.1f}%" if cand > 0 else "n/a"

        _n_left = int(np.sum(live_merit < M_thr))
        _worst  = float(self.display.from_merit(
            np.asarray(float(np.min(live_merit)))))

        _rows = [
            ("Termination", f"all live points {_term}"),
            ("", (f"reached — 0 uncertified live points"
                  if _n_left == 0 else
                  f"NOT reached — {_n_left} live point(s) still below "
                  f"the threshold")),
            ("Worst live point", f"{self.display.symbol} = {_worst:.4g}"),
            ("Sweeps", f"{sweep}"),
            ("Replacements", f"{n_replacements}"),
            ("Final modes", f"{len(modes)}"),
            ("Wall time", _ProgressLog._hms_long(_plog.elapsed())),
            ("Live points", f"{self.N_L}"),
            ("Dead points", f"{sto_dead}"),
            ("Candidate evals", f"{sto_cand}  (accepted {sto_dead}, "
                                f"rejected {sto_cand - sto_dead})"),
            ("Sampling eff.", f"{_eff(sto_dead, sto_cand)}  "
                              f"(accepted / candidate evals)"),
            ("Model runs", f"{sto_runs}  (K = {K} per eval × "
                           f"[{self.N_L} seed + {sto_cand} candidates])"),
        ]

        # Only BlackBoxModel keeps a failure count -- a white-box
        # ProcessModel has no simulator to fail -- so the row is left out
        # entirely rather than printed as a misleading zero.
        #
        # This number matters more than its size suggests: a failed run is
        # recorded as NaN and read as maximally infeasible, so it is
        # indistinguishable from a genuinely bad design point in every
        # other line of this summary. A large count means the certified
        # region may be shaped by solver trouble rather than by the
        # constraints.
        _n_fail = getattr(self.estimator.model, "n_failures", None)
        if _n_fail is not None:
            _share = (f"  ({100.0 * _n_fail / sto_runs:.1f}% of model runs, "
                      f"counted as infeasible)") if sto_runs else ""
            _rows.append(("Failed runs", f"{_n_fail}{_share}"))

        _plog.block(
            ("RESULT — CONVERGED" if _n_left == 0
             else "RESULT — STOPPED (see warning above)"),
            _rows,
        )


        # Build the final summary of live points in each remaining mode.
        _mode_rows = []
        for i, mm in enumerate(sorted(modes, key=lambda mo: mo.label),
                               start=1):
            _mode_rows.append(
                (f"Mode {i}",
                 f"{mm.idx.shape[0]:5d} live points  [label {mm.label}]"
                 + ("  [done]" if mm.done else "")))
        _mode_rows.append(
            ("Total",
             f"{sum(mm.idx.shape[0] for mm in modes):5d} live points"))
        _plog.block(
            "FINAL MODES  (LIVE points only — dead points keep their "
            "evicting mode's label,\n  which may have retired at a split, "
            "so they do not partition the final modes)",
            _mode_rows, rule="─")

        live_mode_ids = np.zeros(self.N_L, dtype=int)
        for mm in modes:
            live_mode_ids[mm.idx] = mm.label

        return SamplerResult(
            dead_points     = (np.array(dead_points) if dead_points
                                          else np.empty((0, D))),
            dead_merit      = np.array(dead_merit, dtype=float),
            dead_probs      = np.array(dead_probs, dtype=float),
            live_points     = live_points_phys,
            live_merit      = live_merit,
            live_probs      = live_probs,
            rejected_points = rej_arr,
            rejected_merit  = rej_merit_arr,
            rejected_probs  = np.array(rejected_probs, dtype=float),
            n_replacements        = n_replacements,
            n_candidate_estimates = n_candidate_evals,  # seed excluded
            N_L               = self.N_L,
            live_mode_ids = live_mode_ids,
            model_runs_per_estimate = K,
            feas_criterion      = self.feas_criterion,
            alpha_star          = self.alpha_star,
            decomp_traces       = (self.decomp_traces
                                   if self.decomp_traces else None),
            termination_reason  = _term_reason,
            n_uncertified_live  = _n_left,
        )
    
    

    

# ============================================================
# SECTION 11 — VISUALISER
# ============================================================

class Visualizer:
    """
    All plotting code, fully decoupled from computation.

    The ground-truth P grid is estimated by MC so the Visualizer
    works with any ``BaseModel`` subclass — ``ProcessModel`` (white-box)
    or ``BlackBoxModel`` (black-box) — with no analytical formula
    required.

    NOTE: this Visualizer always renders a 2-D slice/projection (it
    builds a 2-D meshgrid and calls the model with only 2 design
    variables varying).  For D > 2 problems, pass ``design_space`` and
    pick which two design variables to plot via ``dims``; the other
    D-2 design variables are *not* represented in the grid at all —
    this class only ever evaluates the model at exactly 2 coordinates
    per grid point, so it is only meaningful as-is for genuinely 2-D
    models (e.g. ``model_A`` / ``model_B`` in ``main()`` below, both
    of which take a 2-D ``d``). Visualising a genuine D>2 problem
    properly (e.g. marginal slices holding the other D-2 variables
    fixed) would need a different, dedicated method — out of scope
    here.

    Parameters
    ----------
    model         : BaseModel  (ProcessModel or BlackBoxModel)
    alpha_star    : float   target reliability for shading
    design_space  : DesignSpace, optional
                    If given, axis bounds are taken from
                    ``design_space.bounds[dims[0]]`` /
                    ``design_space.bounds[dims[1]]`` — allowing
                    different x/y ranges. Overrides ``prior_bounds``.
    dims          : (int, int)  which two design-variable indices to
                    plot when ``design_space`` is given (default (0, 1))
    prior_bounds  : (lo, hi)  fallback: single symmetric box for both
                    axes, used only if ``design_space`` is not given
                    (kept for backward compatibility with the simple
                    2-D toy-model usage in ``main()``)
    n_grid        : int     grid resolution per axis
    n_grid_theta  : int     MC samples per grid point for ground-truth
    title_suffix  : str     appended to every plot's suptitle
    """

    # Kept as class attributes for backward compatibility; the live
    # values now come from CriterionDisplay so "P" and the risk measures
    # cannot drift apart.
    ALPHAS       = CriterionDisplay.P_ALPHAS
    ALPHA_COLORS = CriterionDisplay.P_ALPHA_COLORS

    def __init__(
        self,
        model:          BaseModel,
        alpha_star:     float = 0.90,
        design_space:   Optional[DesignSpace] = None,
        dims:           Tuple[int, int] = (0, 1),
        prior_bounds:   Tuple[float, float] = (-1.0, 1.0),
        n_grid:         int = 60,
        n_grid_theta:   int = 300,
        title_suffix:   str = "",
        feas_criterion: str = "VaR",
    ) -> None:
        self.model          = model
        self.alpha_star     = alpha_star
        self.feas_criterion = feas_criterion
        self.disp           = CriterionDisplay(feas_criterion, alpha_star)
        self.title_suffix   = title_suffix or model.name

        if design_space is not None:
            i, j = dims
            self.bounds_x = tuple(design_space.bounds[i])
            self.bounds_y = tuple(design_space.bounds[j])
        else:
            self.bounds_x = prior_bounds
            self.bounds_y = prior_bounds

        print(f"Building {self.disp.symbol} grid ({n_grid}×{n_grid}, "
              f"{n_grid_theta} MC samples each) …")
        self._build_grid(n_grid, n_grid_theta)

    # ----------------------------------------------------------
    def _build_grid(self, n_grid: int, n_theta: int) -> None:
        """Ground-truth field for the SELECTED criterion, so the
        background always shows the same quantity the sampler was driven
        by rather than a probability the run never computed."""
        lo_x, hi_x = self.bounds_x
        lo_y, hi_y = self.bounds_y
        d1g     = np.linspace(lo_x, hi_x, n_grid)
        d2g     = np.linspace(lo_y, hi_y, n_grid)
        self.D1, self.D2 = np.meshgrid(d1g, d2g)

        flat         = np.column_stack([self.D1.ravel(), self.D2.ravel()])
        V_flat       = self.model.mc_criterion_grid(
            flat, N_theta=n_theta,
            feas_criterion=self.feas_criterion,
            alpha_star=self.alpha_star)
        self.V_grid  = self.disp.sanitise_field(V_flat.reshape(self.D1.shape))
        # ``P_grid`` is retained as an ALIAS so save_run_gif / RunPlayer
        # and any external caller keep working; it holds the criterion
        # field, which for feas_criterion="P" is literally the old P grid.
        self.P_grid  = self.V_grid

    @property
    def grid_vlim(self) -> Tuple[float, float]:
        """(vmin, vmax) for the GIF / player greyscale backdrop — pass to
        ``save_run_gif`` / ``RunPlayer``. (0, 1) for "P"."""
        return self.disp.grid_vlim(self.V_grid)

    # ----------------------------------------------------------
    def _point_color(self, v: float) -> str:
        """Band colour for ONE criterion value. Delegates to the single
        adaptive system (CriterionDisplay), so the PNG and the GIF band
        identically — they did not before; see CriterionDisplay's
        'KNOWN COSMETIC CHANGE' note."""
        return self.disp.point_colors(np.atleast_1d(v))[0]

    # ----------------------------------------------------------
    def plot_landscape(
        self, save_path: str = "landscape.png"
    ) -> None:
        """
        2-panel plot: full P landscape + cluster shading at alpha_star.
        Works for any user model.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # Panel 1: continuous criterion heatmap
        ax = axes[0]
        cf = ax.contourf(self.D1, self.D2, self.V_grid, alpha=0.9,
                         **self.disp.field_kwargs(self.V_grid))
        plt.colorbar(cf, ax=ax, label=self.disp.value_label)
        for level, color, _lab in self.disp.contour_levels():
            ax.contour(self.D1, self.D2, self.V_grid,
                       levels=[level], colors=[color], linewidths=2.0)
        ax.set_title(f"{self.disp.symbol} landscape\n(MC ground truth)",
                     fontsize=11)
        self._format_ax(ax, legend=True)

        # Panel 2: feasible-region shading (nested reliability sets for
        # "P"; the single set rho <= 0 for a risk measure)
        ax = axes[1]
        for mask, c, _lab in self.disp.nested_shading(self.V_grid):
            ax.contourf(self.D1, self.D2, mask.astype(float),
                        levels=[0.5, 1.5], colors=[c], alpha=0.5)
        for level, color, _lab in self.disp.contour_levels():
            ax.contour(self.D1, self.D2, self.V_grid,
                       levels=[level], colors=[color], linewidths=2.0)
        if self.disp.is_probability:
            ax.set_title(f"Feasible region at α* = {self.alpha_star}",
                         fontsize=11)
        else:
            ax.set_title(f"Design space: {self.disp.symbol} ≤ 0   "
                         f"(α* = {self.alpha_star:g}, "
                         f"α = {self.disp.tail_probability:g})",
                         fontsize=11)
        self._format_ax(ax, legend=False)

        plt.suptitle(self.title_suffix, fontsize=12)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {save_path}")

    # ----------------------------------------------------------
    def plot_multinest_result(
        self,
        result:    SamplerResult,
        save_path: str = "multinest_result.png",
        announce:  bool = True,
    ) -> None:
        """2-panel: NS sample scatter vs. MC ground truth.

        Reads the run's OWN criterion off ``result``, and warns rather
        than silently mis-colouring if it disagrees with the one this
        Visualizer built its grid for."""
        if result.feas_criterion != self.feas_criterion:
            print(f"  WARNING: this Visualizer's grid is "
                  f"{self.feas_criterion!r} but the run used "
                  f"{result.feas_criterion!r}; the scatter and the "
                  f"background show different quantities.")

        points, merits, _ = result.all_points_and_merits()
        colors            = result.display.merit_colors(merits)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: NS scatter
        ax = axes[0]
        ax.scatter(points[:, 0], points[:, 1],
                   c=colors, s=5, alpha=0.5, zorder=2)
        for level, color, _lab in self.disp.contour_levels():
            ax.contour(self.D1, self.D2, self.V_grid,
                       levels=[level], colors=[color], linewidths=2.0, zorder=3)
        ax.set_title(
            f"MultiNest result ({result.feas_criterion})\n"
            f"$N_L={result.N_L}$,  α*={self.alpha_star}",
            fontsize=11,
        )
        self._format_ax(ax)

        # Right: MC ground truth — the design space itself
        ax = axes[1]
        ax.contourf(self.D1, self.D2,
                    self.disp.feasible(self.V_grid).astype(float),
                    levels=[0.5, 1.5], colors=["steelblue"], alpha=0.4)
        for level, color, _lab in self.disp.contour_levels():
            ax.contour(self.D1, self.D2, self.V_grid,
                       levels=[level], colors=[color], linewidths=2.0)
        ax.set_title("MC ground truth\n(reference)", fontsize=11)
        self._format_ax(ax)

        # Point bands: the SAME five colours for every criterion (one
        # system), but their meaning differs — absolute reliability bins
        # for "P", population-relative progress towards rho = 0 for a
        # risk measure. See CriterionDisplay.
        if self.disp.is_probability:
            band_labels = ["P > 0.95", "0.70 < P ≤ 0.95", "0.50 < P ≤ 0.70",
                           "0.25 < P ≤ 0.50", "P ≤ 0.25"]
        else:
            sym = self.disp.symbol
            band_labels = [f"{sym} ≤ 0  (feasible)",
                           "closest to feasible", "…", "…",
                           "worst live violation"]
        legend_elements = [
            Patch(facecolor=c, label=l) for c, l in
            zip(["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"],
                band_labels)
        ] + [
            Line2D([0], [0], color=c, lw=2, label=lab)
            for _lv, c, lab in self.disp.contour_levels()
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.05))

        plt.suptitle(
            f"MultiNest design-space characterisation "
            f"({result.feas_criterion}) — {self.title_suffix}",
            fontsize=12,
        )
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if announce:
            print(f"Plot saved: {save_path}")

    # ----------------------------------------------------------
    def _format_ax(self, ax, legend: bool = False) -> None:
        lo_x, hi_x = self.bounds_x
        lo_y, hi_y = self.bounds_y
        ax.set_xlim(lo_x, hi_x); ax.set_ylim(lo_y, hi_y)
        ax.set_xlabel("$d_1$", fontsize=13)
        ax.set_ylabel("$d_2$", fontsize=13)
        ax.grid(True, alpha=0.3)
        if legend:
            handles = [
                Line2D([0], [0], color=c, lw=2.5, label=lab)
                for _lv, c, lab in self.disp.contour_levels()
            ]
            ax.legend(handles=handles, fontsize=10, loc="lower right")


# ============================================================
# SECTION 12 — RUN RECORDER, GIF EXPORT & INTERACTIVE FRAME PLAYER
#   (the "algorithm movie" — same idea as multinest_visualizer_v5.py,
#    re-implemented for THIS file's multi-mode architecture)
# ============================================================
#
# multinest_visualizer_v5.py achieved zero-drift fidelity by keeping a
# line-for-line generator COPY of the (old, single-mode) run() loop and
# yielding FrameState after every candidate. That approach cannot be
# reused here: run() is now multi-mode and any copied-out duplicate
# would silently rot the moment the algorithm changes. Instead, run()
# itself exposes a read-only ``frame_callback`` observer hook (see its
# docstring) that emits a ``RunFrame`` snapshot at every notable event.
# The classes below consume those snapshots:
#
#   RunRecorder   — collects every RunFrame from one run
#   save_run_gif  — renders recorded frames to an animated GIF (Pillow;
#                   works headless, no GUI backend needed)
#   RunPlayer     — interactive matplotlib player over the recorded
#                   frames: Prev / Next / Auto / slider / arrow keys
#                   (needs a display; use the GIF on a headless box)
#
# The rendering style deliberately mirrors multinest_visualizer_v5.py:
# ground-truth P(d) heatmap background, live points coloured by the
# same P bands, translucent evicted/rejected dead points, ellipsoid
# outlines (one colour PER MODE here, since this sampler tracks several
# independent modes), a gold ring on the point being evicted, and a
# green/red square on the current candidate.


@dataclass(eq=False)          # identity hash/eq: frames hold ndarrays
class RunFrame:
    """One read-only snapshot of the sampler state, emitted by
    ``MultiNestSampler.run(frame_callback=...)``.

    ``dead_ref``/``rejected_ref``/``rejected_merit_ref`` are SHARED
    references to the run's growing lists — slice them with
    ``n_dead``/``n_rejected`` (the lengths at emit time) rather than
    copying thousands of points into every frame.
    """

    event:           str                  # init|candidate|accepted|refit|mode_split|mode_frozen|converged
    sweep:           int                    # outer-sweep counter ("ff")
    n_replacements:  int                    # accepted replacements so far
    n_candidate_evals: int
    live_points:     np.ndarray             # (N_L, D) physical, copied
    live_merit:      np.ndarray             # (N_L,)   copied
    dead_ref:        List[np.ndarray]       # shared, slice with n_dead
    n_dead:          int
    rejected_ref:    List[np.ndarray]       # shared, slice with n_rejected
    rejected_merit_ref: List[float]
    n_rejected:      int
    worst_point:     Optional[np.ndarray]
    worst_merit:     Optional[float]
    candidate:       Optional[np.ndarray]
    candidate_merit: Optional[float]
    candidate_ok:    Optional[bool]
    mode_ellipsoids: List[List["Ellipsoid"]]   # unit-cube ellipsoids per mode
    mode_done:       List[bool]
    mode_ids:        List[int]   # _Mode.label per mode, stable for the
                                 # whole run (never id(mode) -- see
                                 # _Mode.label for why id() is unusable)
    F_S:             Optional[float]
    below_alpha:     int
    alpha_star:      float
    design_space:    "DesignSpace"
    note:            str = ""
    # ---- CRITERION SUPPORT (defaulted so a frame is
    # constructed exactly as before and old recordings still render) ----
    # ``live_merit``/``worst_merit``/``candidate_merit`` hold the sampler's
    # MERIT (higher-is-better). With feas_criterion="P" that IS the
    # probability and merit_threshold == alpha_star. With
    # "VaR"/"CVaR" the merit is the NEGATED risk measure and
    # merit_threshold is 0; ``below_alpha`` counts live points below
    # merit_threshold in every case. Renderers should go through
    # ``CriterionDisplay`` rather than reading these directly.
    feas_criterion:  str = "P"
    merit_threshold: Optional[float] = None


class RunRecorder:
    """Collects every RunFrame emitted during one ``run()``.

    Usage::

        rec              = RunRecorder()
        result           = sampler.run(seed=0, frame_callback=rec)
        frames           = rec.frames
    """

    def __init__(self) -> None:
        self.frames: List[RunFrame] = []

    def __call__(self, frame: RunFrame) -> None:
        self.frames.append(frame)


# ------------------------------------------------------------------
# rendering helpers (shared by the GIF exporter and the player)
# ------------------------------------------------------------------

_MODE_COLORS = ["#6ab0ff", "#f4b942", "#22c97a", "#e07be0",
                "#ff8c69", "#7fdbdb", "#c9c9c9", "#b0e07b"]


def _band_color(p: float) -> str:
    """Same P bands / palette as multinest_visualizer_v5.py."""
    if   p > 0.95: return "#d62728"
    elif p > 0.70: return "#ff7f0e"
    elif p > 0.50: return "#2ca02c"
    elif p > 0.25: return "#1f77b4"
    else:          return "#9467bd"


def _relative_band_colors(scores: np.ndarray, thr: Optional[float]) -> List[str]:
    """Colour points from an UNBOUNDED higher-is-better scalar.

    Used for the VaR/CVaR merit (= -risk), which is signed, unbounded
    below, and measured in the CONSTRAINT's own units — so there is no
    scale-free ABSOLUTE map onto ``_band_color``'s [0,1] bands. A map
    like p = thr/S silently inherits that scale: on a problem whose
    violations span ~1 every point lands in the red/orange bands (the
    palette never opens), and on one whose violations span ~1000 every
    point lands in purple.

    So the map is POPULATION-RELATIVE instead, recomputed per frame:

        p = 1                                  if S >= thr  (feasible)
        p = min((S - lo) / (thr - lo), 0.95)   otherwise

    with lo = the worst score in the population. So colour = this
    point's progress from the current worst point up to the feasibility
    threshold. Consequences worth knowing when reading a GIF: the scale
    is relative, so a point's colour can change while its value does not
    (the population moved), and the bands mean "who is lagging", not an
    absolute violation size.

    What IS absolute is the RED band: the 0.95 clip on the infeasible
    branch keeps it exclusive, so red means feasible and nothing else.
    Without the clip a point a hair below the threshold would also
    render red (p rounds to ~1) and the one reliable signal in the
    frame would be a lie.
    """
    sc = np.asarray(scores, dtype=float)
    if thr is None:
        return ["#9467bd"] * sc.size
    feasible = sc >= thr
    below    = sc[~feasible & np.isfinite(sc)]
    lo       = float(below.min()) if below.size else (thr - 1.0)
    span     = max(thr - lo, 1e-300)
    out = []
    for S in sc:
        if not np.isfinite(S):
            out.append("#9467bd")          # -inf: failed simulation
        elif S >= thr:
            out.append(_band_color(1.0))   # feasible
        else:
            # 0.95 clip, not 1.0: keeps the red band exclusive to
            # feasible points (see docstring).
            out.append(_band_color(float(np.clip((S - lo) / span, 0.0, 0.95))))
    return out


def _unit_ellipsoid_to_physical_patch(ell: "Ellipsoid",
                                      ds: "DesignSpace",
                                      dims: Tuple[int, int] = (0, 1),
                                      **kwargs):
    """
    Convert a UNIT-CUBE ellipsoid { u : (u-mu)^T (f C)^{-1} (u-mu) <= 1 }
    into a matplotlib Ellipse patch in PHYSICAL coordinates.

    The physical map is the affine d = lo + (hi-lo)*u, so the physical
    shape matrix is S C S with S = diag(hi-lo); on the plotted 2-D
    projection that is elementwise C2 * outer(s2, s2). (Same maths as
    ``_to_physical_ellipsoids`` in multinest_visualizer_v5.py, but the
    ellipsoids in THIS file live in the unit cube, so the scaling is
    applied here at render time instead of at fit time.)

    USES THE REGULARISED SHAPE, NOT ``ell.C``. ``Ellipsoid.C`` is the RAW
    covariance; the algorithm never uses it. Every quantity that decides
    where points actually go -- ``L`` (sample), ``C_inv``/``A_inv``
    (contains, Mahalanobis) -- is built from the EIGENVALUE-FLOORED
    covariance (Ellipsoid.fit, CalcEllProp L362-370). Drawing raw ``C``
    therefore draws an ellipse the sampler does not use: for a leaf whose
    raw covariance is rank-deficient (any leaf of N <= D points, e.g. 2
    points in 2-D) the raw eigenvalues are [~0, lambda], so the patch
    comes out with width ~ 0 -- the razor-thin "needle" ellipsoids
    visible in the GIF/player. The sampler was sampling a perfectly
    round circle at the same time. L L^T reconstructs exactly the
    floored covariance the algorithm uses, so the drawing now matches
    the object.
    """
    from matplotlib.patches import Ellipse as _MplEllipse
    i, j   = dims
    s2     = np.array([ds.hi[i] - ds.lo[i], ds.hi[j] - ds.lo[j]])
    mu_ph  = ds.to_physical(ell.mu)[[i, j]]
    C_reg  = (ell.L @ ell.L.T) if ell.L is not None else ell.C
    C2     = C_reg[np.ix_([i, j], [i, j])] * np.outer(s2, s2)
    vals, vecs = np.linalg.eigh(C2)
    vals   = np.maximum(vals, 0.0)
    # eigh is ascending: pair width with the vals[0] axis (vecs[:,0])
    # and height with vals[1], with the patch rotated to vecs[:,0].
    w      = 2.0 * np.sqrt(ell.f * vals[0])
    h      = 2.0 * np.sqrt(ell.f * vals[1])
    ang    = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return _MplEllipse(xy=mu_ph, width=w, height=h, angle=ang, **kwargs)


# ------------------------------------------------------------------
# THEMES
# ------------------------------------------------------------------
# "dark" is the original GIF/player look and stays the DEFAULT, so the
# movie and the interactive player render exactly as before.
#
# "paper" is the same panel -- same layers, same title block -- restyled
# for a still that goes into the write-up: white ground, no field behind
# the points, and a different rule for colouring the live points.
#
# HOW THE LIVE POINTS ARE COLOURED — ``point_style``, three choices:
#
#   "bands"   the five adaptive MERIT BANDS, the dark theme's own. WRONG
#             FOR A STILL: _relative_band_colors is POPULATION-RELATIVE
#             and recomputed per frame, so a point's colour is its
#             progress from the CURRENT worst live point up to the
#             threshold. Right for a movie, where the eye reads the
#             bands as motion (who is lagging, who caught up); wrong for
#             a series of stills, where the same colour then means a
#             different absolute value in every panel and a reader
#             looking at one panel alone cannot decode it at all.
#
#   "binary"  certified (criterion <= 0) against not, two colours.
#             Absolute and identical in every panel, needs no colourbar.
#             The safe minimum.
#
#   "value"   the criterion value itself, on a DIVERGING scale centred
#             on the feasibility threshold, in the same RdYlGn_r the
#             reference field figure uses. THE DEFAULT HERE, because
#             VaR/CVaR is measured in the constraint's own units and is
#             the same quantity in every panel of a problem: fix one
#             scale across the whole SERIES (``_series_point_norm``,
#             computed over all the selected frames, never per panel)
#             and the colour is as absolute as the binary split while
#             also showing the population's distribution shifting toward
#             feasible as the constraint tightens. Centring on the
#             threshold keeps the certified/not split readable too --
#             the colour flips at the boundary. This is only legible
#             because there is no field behind the points; drawn over
#             the VaR field it would be camouflage, every point sitting
#             on a background of its own colour.
#
# THE TWO SINGLE-POINT MARKERS — ``show_worst`` and ``show_candidate``,
# both ON in the dark theme and OFF in paper, and kept as SEPARATE
# switches because they are not the same kind of thing:
#
#   candidate  the point drawn at this instant, ringed green or red by
#              whether it was accepted. Pure sub-iteration transient: in
#              a movie it is the action, in a still it is whichever draw
#              the frame happened to be caught on, and a reader has no
#              way to tell that from something meaningful. Nothing to be
#              gained by turning it back on for a figure.
#
#   worst      the live point that is about to be evicted -- which IS
#              the nested constraint level the frame is sitting at, i.e.
#              exactly the quantity a "how the constraint tightens"
#              figure is about. Off by default only because it is one
#              more glyph; if either of the two is ever worth restoring
#              on a still, it is this one.
_FRAME_THEMES = {
    "dark": dict(
        surface="#111111", ink="#dddddd", axis_label="#cccccc",
        tick="#888888", spine="#555555", grid=None,
        field_cmap="Greys_r", field_alpha=0.55,
        dead="#e05252", dead_alpha=0.35,
        rejected="#888888", rejected_alpha=0.30,
        point_style="bands", point_size=16, point_edge=None,
        certified="#2a78d6", uncertified="#eb6834", uniform="#6ab0ff",
        worst="#f4b942", ok="#22c97a", bad="#e05252",
        show_worst=True, show_candidate=True,
        modes=_MODE_COLORS, ell_alpha=0.9, legend=False,
    ),
    "paper": dict(
        surface="#ffffff", ink="#0b0b0b", axis_label="#3c3c3c",
        tick="#6e6e6e", spine="#b4b4b4", grid="#ededed",
        field_cmap="Greys", field_alpha=0.35,
        dead="#d2d2cc", dead_alpha=0.55,
        rejected="#d2d2cc", rejected_alpha=0.40,
        point_style="value", point_size=22, point_edge="#00000022",
        # Used by "binary" / "uniform"; lifted from live_monitor.PALETTE
        # so the stills and the progress panel agree.
        certified="#2a78d6", uncertified="#eb6834", uniform="#2a78d6",
        worst="#b8860b", ok="#2e7d4f", bad="#c0392b",
        show_worst=False, show_candidate=False,
        # Deliberately avoids the RdYlGn_r point colours: an ellipsoid is
        # a different OBJECT, not another class of point, and sharing a
        # hue with the markers reads as "these two belong together".
        modes=["#1a1a1a", "#7a3ea8", "#2b4a8a", "#b02a7a",
               "#6d6a1f", "#0f7d6e", "#a34a1f", "#4a4a4a"],
        ell_alpha=1.0, legend=True,
    ),
}


def _series_point_norm(frames: List[RunFrame]):
    """One diverging colour scale for the live-point values of a WHOLE
    snapshot series, centred on the feasibility threshold.

    Shared across panels on purpose. Normalising each panel to its own
    population would put the extremes of the colourmap at both ends of
    every frame, so the last panel -- where every point is comfortably
    feasible -- would look exactly as alarming as the first. Fixing the
    scale over the selected frames is what makes "the constraint became
    more restrictive" visible as colour.

    ``TwoSlopeNorm`` rather than a plain linear range because the two
    sides are wildly unequal: early on the worst live point can violate
    by tens while the feasible ones sit a fraction below zero, and a
    linear scale would collapse every certified point into one colour.
    The two-slope form gives each side half the colourmap whatever its
    width, so the boundary always lands on the colourmap's midpoint.

    Returns None when there is nothing to normalise (e.g. every value
    equal), which the renderer reads as "fall back to per-frame".
    """
    disp   = CriterionDisplay(frames[0].feas_criterion, frames[0].alpha_star)
    vals   = np.concatenate([np.asarray(disp.from_merit(f.live_merit),
                                        dtype=float).ravel()
                             for f in frames])
    vals   = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    centre = float(disp.threshold_value)
    # A strictly-increasing (vmin, vcenter, vmax) is required. A series
    # that never crosses the threshold on one side -- all-feasible
    # converged frames, say -- would otherwise raise.
    span = max(float(vals.max()) - centre, centre - float(vals.min()), 1e-12)
    lo   = min(float(vals.min()), centre - 1e-3 * span)
    hi   = max(float(vals.max()), centre + 1e-3 * span)
    return TwoSlopeNorm(vmin=lo, vcenter=centre, vmax=hi)


def _two_slope_ticks(norm, per_side: int = 3):
    """Colourbar ticks for a ``TwoSlopeNorm``, chosen SIDE BY SIDE.

    A two-slope norm gives each side of the centre half the bar whatever
    the data widths are, but the default locator picks ticks from the
    combined data range and knows nothing about that. On a strongly
    asymmetric field the short side then gets no tick at all: the single
    ellipse spans VaR in about [-1.5, +24], the locator settles on
    multiples of five, and none of them land below zero -- so half the
    colourbar, the whole feasible half, carries no number.

    Ticking each side independently fixes it, and pins the centre (the
    feasibility threshold) as a labelled tick rather than leaving it to
    chance. Returns None for anything that is not a two-slope norm, which
    matplotlib reads as "use the default locator".
    """
    if norm is None or not isinstance(norm, TwoSlopeNorm):
        return None
    from matplotlib.ticker import MaxNLocator
    loc  = MaxNLocator(nbins=per_side, prune=None)
    below = [t for t in loc.tick_values(norm.vmin, norm.vcenter)
             if norm.vmin <= t < norm.vcenter]
    above = [t for t in loc.tick_values(norm.vcenter, norm.vmax)
             if norm.vcenter < t <= norm.vmax]
    ticks = sorted(below + [norm.vcenter] + above)
    # Drop anything that rounds onto the centre, so the threshold does not
    # get a near-duplicate neighbour crowding its label.
    span = max(norm.vmax - norm.vmin, 1e-300)
    return [t for t in ticks
            if t == norm.vcenter or abs(t - norm.vcenter) > 0.02 * span]


def _math_axis_label(name: str) -> str:
    """Set an indexed variable name as maths: ``"d1"`` -> ``"$d_1$"``.

    ``DesignSpace.names`` are plain identifiers because they also appear
    in reprs and error messages, where mathtext would be noise. An axis
    label is the one place the subscript belongs, and the reference field
    figure already writes d_1 / d_2, so the two figures should match.
    Anything not of the form letters-then-digits (``temperature``,
    ``recycle_fraction``) is left exactly as it is.
    """
    import re
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    return rf"${m.group(1)}_{{{m.group(2)}}}$" if m else name


def _frame_threshold(fr: RunFrame) -> float:
    """The merit cutoff separating certified from uncertified live
    points. Same rule as ``LiveMonitor._threshold``."""
    if fr.merit_threshold is not None:
        return float(fr.merit_threshold)
    return float(fr.alpha_star) if fr.feas_criterion == "P" else 0.0


def _render_run_frame(ax, fr: RunFrame,
                      P_grid=None, D1=None, D2=None,
                      dims: Tuple[int, int] = (0, 1),
                      show_rejected: bool = True,
                      grid_vlim: Optional[Tuple[float, float]] = None,
                      theme: str = "dark",
                      point_style: Optional[str] = None,
                      point_norm=None,
                      colorbar: bool = True,
                      show_worst: Optional[bool] = None,
                      show_candidate: Optional[bool] = None) -> None:
    """Draw ONE RunFrame onto ``ax`` (2-D projection onto ``dims``).

    ``P_grid`` is the ground-truth field drawn as the greyscale backdrop.
    It is whatever criterion the run used (``Visualizer.V_grid``), so
    ``grid_vlim`` supplies its (vmin, vmax): pass
    ``Visualizer.grid_vlim``, or leave it None for the historical
    probability range (0, 1). The name ``P_grid`` is kept for
    backward compatibility with existing callers.

    ``theme`` selects the palette: "dark" (default, the GIF/player look)
    or "paper" (white ground, criterion-valued points). ``point_style``
    overrides how the live points are coloured -- "bands", "binary",
    "value" or "uniform". See ``_FRAME_THEMES``.

    ``point_norm`` is the colour normalisation for ``point_style="value"``
    and should come from ``_series_point_norm`` over EVERY frame of the
    series, so the panels share one scale; None normalises this frame
    alone, which is only right for a single standalone panel.
    """
    try:
        th = _FRAME_THEMES[theme]
    except KeyError:
        raise ValueError(
            f"unknown theme {theme!r}; expected one of "
            f"{sorted(_FRAME_THEMES)}") from None

    i, j = dims
    ds   = fr.design_space
    ax.clear()
    ax.set_facecolor(th["surface"])
    if th["grid"]:
        ax.grid(True, color=th["grid"], linewidth=0.6)
        ax.set_axisbelow(True)

    if P_grid is not None:
        g_lo, g_hi = grid_vlim if grid_vlim is not None else (0.0, 1.0)
        ax.pcolormesh(D1, D2, P_grid, cmap=th["field_cmap"],
                      vmin=g_lo, vmax=g_hi, alpha=th["field_alpha"],
                      shading="auto", zorder=0)

    # dead — evicted (red translucent) and rejected (grey translucent).
    if fr.n_dead:
        dp = np.array(fr.dead_ref[: fr.n_dead])
        ax.scatter(dp[:, i], dp[:, j], s=9, c=th["dead"],
                   alpha=th["dead_alpha"], lw=0, zorder=1)
    if show_rejected and fr.n_rejected:
        rp = np.array(fr.rejected_ref[: fr.n_rejected])
        ax.scatter(rp[:, i], rp[:, j], s=7, c=th["rejected"],
                   alpha=th["rejected_alpha"], lw=0, zorder=1)

    # ellipsoids, one colour per mode
    for k, ells in enumerate(fr.mode_ellipsoids):
        col = th["modes"][k % len(th["modes"])]
        for e in ells:
            if e.V <= 0.0:
                continue
            patch = _unit_ellipsoid_to_physical_patch(
                e, ds, dims=dims, fill=False, lw=1.6,
                edgecolor=col, alpha=th["ell_alpha"], zorder=3)
            ax.add_patch(patch)
            mu_ph = ds.to_physical(e.mu)[[i, j]]
            ax.plot(*mu_ph, marker="+", ms=7, mew=1.6, c=col, zorder=3)

    _disp = CriterionDisplay(fr.feas_criterion, fr.alpha_star)
    style = point_style or th["point_style"]
    size  = th["point_size"]
    lx, ly = fr.live_points[:, i], fr.live_points[:, j]

    if style == "bands":
        # ONE adaptive band system for every criterion — see
        # CriterionDisplay. For "P" this is [_band_color(p) for p in ...],
        # i.e. exactly what this line used to be.
        ax.scatter(lx, ly, s=size, c=_disp.merit_colors(fr.live_merit),
                   lw=0, zorder=4)

    elif style == "value":
        # The criterion value itself, on the series-wide diverging scale.
        # Same colourmap as the ground-truth field figure, so green still
        # means feasible in both.
        vals = np.asarray(_disp.from_merit(fr.live_merit), dtype=float)
        cmap = "RdYlGn" if _disp.is_probability else "RdYlGn_r"
        norm = point_norm if point_norm is not None else _series_point_norm([fr])
        sc   = ax.scatter(lx, ly, c=vals, cmap=cmap, norm=norm, s=size,
                          lw=0.25, edgecolors=th["point_edge"], zorder=4)
        if colorbar:
            cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.03,
                                    ticks=_two_slope_ticks(norm))
            cb.set_label(_disp.value_label, color=th["axis_label"],
                         fontsize=10)
            cb.ax.tick_params(colors=th["tick"], labelsize=8)
            cb.outline.set_edgecolor(th["spine"])
            # The one level that has to be readable without decoding a
            # gradient: where the design space starts.
            if norm is not None:
                cb.ax.axhline(norm.vcenter, color=th["ink"], lw=1.2)

    elif style == "uniform":
        ax.scatter(lx, ly, s=size, c=th["uniform"], lw=0, zorder=4,
                   label=f"live points ({lx.size})")

    elif style == "binary":
        # Absolute, so it means the same thing in every panel of a
        # series. Same marker for both classes — the classes differ in
        # what they ARE, not in kind, and a second glyph shape implies a
        # second variable that is not there.
        thr  = _frame_threshold(fr)
        cert = np.asarray(fr.live_merit, dtype=float) >= thr
        if (~cert).any():
            ax.scatter(lx[~cert], ly[~cert], c=th["uncertified"],
                       marker="o", s=size, lw=0, zorder=4,
                       label=f"uncertified ({int((~cert).sum())})")
        if cert.any():
            ax.scatter(lx[cert], ly[cert], c=th["certified"],
                       marker="o", s=size, lw=0, zorder=5,
                       label=f"certified ({int(cert.sum())})")
    else:
        raise ValueError(
            f"unknown point_style {style!r}; expected one of "
            f"'bands', 'value', 'binary', 'uniform'")

    # Worst live point (ring) and current candidate (accepted/rejected
    # square). Both ON in the movie and OFF on a still — see the theme
    # table for why they are two switches and not one.
    _worst = th["show_worst"]     if show_worst     is None else show_worst
    _cand  = th["show_candidate"] if show_candidate is None else show_candidate
    if _worst and fr.worst_point is not None:
        ax.scatter([fr.worst_point[i]], [fr.worst_point[j]], s=110,
                   facecolors="none", edgecolors=th["worst"],
                   lw=2.0, zorder=6, label="worst live point")
    if _cand and fr.candidate is not None:
        c = th["ok"] if fr.candidate_ok else th["bad"]
        ax.scatter([fr.candidate[i]], [fr.candidate[j]], s=70,
                   marker="s", facecolors="none", edgecolors=c,
                   lw=2.0, zorder=6,
                   label="candidate " + ("accepted" if fr.candidate_ok
                                         else "rejected"))

    if th["legend"] and ax.get_legend_handles_labels()[0]:
        # BELOW the axes, not inside them. A legend box floating over the
        # panel hides whatever population happens to be under it, and the
        # corner it has to hide is different in every snapshot of a
        # series — so the panels stop being comparable for the sake of
        # two lines of text. Frameless, since it is outside now and has
        # nothing to be separated from.
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                        ncol=2, fontsize=9, frameon=False,
                        handletextpad=0.4, columnspacing=1.6)
        for text in leg.get_texts():
            text.set_color(th["axis_label"])

    # Real variable names when the DesignSpace carries them (T, P,
    # ratio...); d_1 / d_2, set as maths to match the reference figure,
    # only as the fallback.
    _names = getattr(ds, "names", None) or []
    _xlab  = _names[i] if i < len(_names) else f"d{i+1}"
    _ylab  = _names[j] if j < len(_names) else f"d{j+1}"
    ax.set_xlim(ds.lo[i], ds.hi[i])
    ax.set_ylim(ds.lo[j], ds.hi[j])
    ax.set_xlabel(_math_axis_label(_xlab), color=th["axis_label"])
    ax.set_ylabel(_math_axis_label(_ylab), color=th["axis_label"])
    ax.tick_params(colors=th["tick"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(th["spine"])

    n_ell = sum(len(e) for e in fr.mode_ellipsoids)
    head  = fr.event.upper()
    if _disp.is_probability:
        tail = f"P < \u03b1*={fr.alpha_star:g}: {fr.below_alpha}"
    else:
        tail = (f"uncertified ({_disp.symbol}>0): {fr.below_alpha}"
                f"   \u03b1={_disp.tail_probability:g}")
    info  = (f"{head:<18}  sweep {fr.sweep}  "
             f"iter {fr.n_replacements}  evals {fr.n_candidate_evals}\n"
             f"modes {len(fr.mode_ellipsoids)}  ellipsoids {n_ell}  "
             f"dead {fr.n_dead}  rejected {fr.n_rejected}\n"
             + tail
             + (f"   F(S) {fr.F_S:.2f}" if fr.F_S not in (None, np.inf) else "")
             + (f"\n{fr.note}" if fr.note else ""))
    ax.set_title(info, fontsize=8.5, color=th["ink"],
                 loc="left", family="monospace")


def _frame_order(f: RunFrame):
    """Chronological sort key over the recorded frames."""
    return (f.n_replacements, f.n_candidate_evals, f.sweep)


def _thin_frames(frames: List[RunFrame],
                 include_candidates: bool,
                 max_frames: int) -> List[RunFrame]:
    """Filter/downsample frames for the GIF: always keep init /
    refit / mode_split / converged; stride the (many) accepted (and
    optionally rejected-candidate) frames to fit ``max_frames``."""
    keep_events = {"init", "refit", "mode_split", "mode_frozen",
                   "converged", "accepted"}
    if include_candidates:
        keep_events.add("candidate")
    fl = [f for f in frames if f.event in keep_events]
    if len(fl) <= max_frames:
        return sorted(fl, key=_frame_order)
    must    = [f for f in fl if f.event in
               ("init", "refit", "mode_split", "mode_frozen",
                "converged")]
    bulk    = [f for f in fl if f.event in ("accepted", "candidate")]
    n_bulk  = max(max_frames - len(must), 1)
    stride  = max(1, int(np.ceil(len(bulk) / n_bulk)))
    thinned = bulk[::stride]
    out     = sorted(set(must) | set(thinned) | {fl[-1]},
                     key=_frame_order)
    return out


def _snapshot_stem(path: str) -> Tuple[str, str]:
    """Split a snapshot path template into (stem, extension), creating the
    directory. ``"figs/snap.png"`` -> ``("figs/snap", ".png")``."""
    stem, ext = os.path.splitext(path)
    ext       = ext or ".png"
    directory = os.path.dirname(stem)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return stem, ext


def _write_frame_png(fr: RunFrame, out: str,
                     P_grid=None, D1=None, D2=None,
                     grid_vlim: Optional[Tuple[float, float]] = None,
                     dims: Tuple[int, int] = (0, 1),
                     theme: str = "paper",
                     show_rejected: bool = True,
                     point_style: Optional[str] = None,
                     point_norm=None,
                     colorbar: bool = True,
                     show_worst: Optional[bool] = None,
                     show_candidate: Optional[bool] = None,
                     figsize: Tuple[float, float] = (6.0, 6.0),
                     dpi: int = 300) -> str:
    """Render ONE frame to its own figure and write it. The single place
    a still is produced, shared by ``save_frame_snapshots`` (post-hoc,
    from recorded frames) and ``SnapshotSaver`` (during the run)."""
    th      = _FRAME_THEMES[theme]
    surface = th["surface"]
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(surface)

    # Margins are set BEFORE the axes is created, because add_subplot
    # reads the figure's subplot params at creation and a colourbar then
    # takes ITS space out of the axes as positioned. Adjusting afterwards
    # moves the axes and leaves the colourbar where it was.
    #   top    : the 3-4 left-aligned monospace title lines
    #   bottom : the legend, which now sits under the axes
    #   right  : the colourbar and its tick labels, when there is one
    _cbar  = colorbar and (point_style or th["point_style"]) == "value"
    fig.subplots_adjust(top=0.82, left=0.13, bottom=0.15,
                        right=0.88 if _cbar else 0.97)

    ax = fig.add_subplot(111)
    _render_run_frame(ax, fr, P_grid=P_grid, D1=D1, D2=D2, dims=dims,
                      show_rejected=show_rejected, grid_vlim=grid_vlim,
                      theme=theme, point_style=point_style,
                      point_norm=point_norm, colorbar=colorbar,
                      show_worst=show_worst, show_candidate=show_candidate)
    fig.savefig(out, format=os.path.splitext(out)[1].lstrip(".") or "png",
                dpi=dpi, facecolor=surface)
    plt.close(fig)
    return out


def save_frame_snapshots(frames: List[RunFrame],
                         sweeps,
                         path: str = "snapshot.png",
                         P_grid=None, D1=None, D2=None,
                         grid_vlim: Optional[Tuple[float, float]] = None,
                         dims: Tuple[int, int] = (0, 1),
                         theme: str = "paper",
                         event: Optional[str] = None,
                         show_rejected: bool = True,
                         point_style: Optional[str] = None,
                         colorbar: bool = True,
                         show_worst: Optional[bool] = None,
                         show_candidate: Optional[bool] = None,
                         figsize: Tuple[float, float] = (6.0, 6.0),
                         dpi: int = 300) -> List[str]:
    """Write ONE PNG per requested sweep — the same panel the GIF and
    the player draw, on a white ground, at print resolution.

    This is the still-figure counterpart of ``save_run_gif``: nothing new
    is drawn and nothing is recomputed, it is ``_render_run_frame`` with
    ``theme="paper"`` pointed at chosen frames instead of all of them.
    Use it for the "selected snapshots of the live-point population and
    the associated ellipsoidal representation" figure rather than
    screenshotting the player, which gives a dark, screen-resolution
    image with whatever window furniture was on top of it.

    Parameters
    ----------
    frames : list of RunFrame
        From a ``RunRecorder``, or unpickled from
        ``LiveMonitor.frames_path``.
    sweeps : int, or iterable of int
        Which sweeps to write. A ``set`` is fine — output is written in
        ascending sweep order regardless. For each one the LAST frame of
        that sweep is used, i.e. the state the sweep ENDED in, so the
        ellipsoids are the ones the sweep finished with.
    path : str
        Template. ``"figs/snap.png"`` writes ``figs/snap_sweep12.png``,
        ``figs/snap_sweep121.png``, ... The directory is created.
    theme : {"paper", "dark"}
        "paper" by default — this function exists to produce the light
        version. Pass "dark" to match the player exactly.
    event : str, optional
        Restrict the choice to one event kind, e.g. ``"refit"`` to land
        on the frames where the decomposition was just rebuilt (the
        interesting instant for a proposal-geometry figure). Default
        None takes whatever frame ended the sweep.
    point_style : {"value", "binary", "uniform", "bands"}, optional
        How the live points are coloured; None takes the theme's own
        ("value" for paper). See ``_FRAME_THEMES``.
    colorbar : bool
        Only relevant to "value". The scale is shared by every panel, so
        one colourbar describes them all — set False on all but one when
        the panels sit side by side in the manuscript.
    show_worst, show_candidate : bool, optional
        The gold ring on the live point about to be evicted, and the
        green/red square on the point drawn this iteration. Both off in
        the paper theme; None takes the theme's setting. Of the two,
        only ``show_worst`` is worth restoring on a still — it marks the
        nested constraint level. See ``_FRAME_THEMES``.

    Returns
    -------
    list of str
        The paths written, in ascending sweep order.
    """
    if isinstance(sweeps, (int, np.integer)):
        sweeps = [int(sweeps)]
    wanted = sorted({int(s) for s in sweeps})
    if not wanted:
        raise ValueError("no sweeps requested")

    pool = [f for f in frames if event is None or f.event == event]
    if not pool:
        raise ValueError(
            f"no frames with event={event!r}; recorded events are "
            f"{sorted({f.event for f in frames})}")

    # Last frame of each requested sweep. _frame_order is the run's own
    # chronological key, so "last" means last in RUN order and not
    # merely last in list order.
    by_sweep = {}
    for f in sorted(pool, key=_frame_order):
        by_sweep[f.sweep] = f

    missing = [s for s in wanted if s not in by_sweep]
    if missing:
        have = sorted(by_sweep)
        raise ValueError(
            f"no frame for sweep(s) {missing}"
            + (f" with event={event!r}" if event else "")
            + f"; recorded sweeps run {have[0]}..{have[-1]} "
              f"({len(have)} distinct). Note LiveMonitor keeps only "
              f"KEEP_DEFAULT events, so a sweep with no kept event has "
              f"no frame.")

    stem, ext = _snapshot_stem(path)
    # ONE scale over the whole series, so a colour means the same value
    # in every panel — see _series_point_norm.
    norm      = _series_point_norm([by_sweep[s] for s in wanted])
    written   = []
    for s in wanted:
        fr  = by_sweep[s]
        out = _write_frame_png(fr, f"{stem}_sweep{s}{ext}",
                               P_grid=P_grid, D1=D1, D2=D2,
                               grid_vlim=grid_vlim, dims=dims, theme=theme,
                               show_rejected=show_rejected,
                               point_style=point_style, point_norm=norm,
                               colorbar=colorbar, show_worst=show_worst,
                               show_candidate=show_candidate,
                               figsize=figsize, dpi=dpi)
        written.append(out)
        print(f"  snapshot : sweep {s:>5}  {fr.event:<11} -> {out}")
    return written


class SnapshotSaver:
    """Frame callback that writes a still at each of a FIXED set of
    sweeps, chosen before the run starts.

    This is what ``sampler_kw['snapshots'] = {12, 121}`` builds. It is
    the cheap counterpart of ``RunRecorder``: that one keeps EVERY frame
    (each carrying its own copy of the live population, so a few hundred
    thousand candidate frames is gigabytes), while this keeps at most one
    frame per requested sweep and drops the rest on the floor. So it can
    be left on for any run, including a long one, at no cost.

    The catch is the obvious one: the sweeps have to be named up front,
    and how many sweeps a run takes is not known until it has taken
    them. So the workflow is run-once-then-pin --- run the example, read
    the final sweep count off the log, put the interesting sweeps in
    ``snapshots``, and run again with the SAME ``seed``, which replays
    the identical trajectory. A sweep that never happens is reported at
    the end rather than raised: a run that has already done the work
    should not be thrown away over a mis-typed figure number.

    Parameters
    ----------
    sweeps : iterable of int
        Sweeps to capture. For each, the LAST frame of that sweep is
        kept --- the state the sweep ended in.
    path : str
        Template, as in ``save_frame_snapshots``.
    event : str, optional
        Restrict to one event kind, e.g. ``"refit"``.
    background : bool
        Draw the criterion field behind the points. OFF by default. The
        field ``_run_example`` has on hand is the LANDSCAPE plot's, only
        ``n_grid=60`` — fine as orientation on screen, visibly blocky at
        300 dpi — and with ``point_style="value"`` it is worse than
        blocky: every point then sits on a background of its own colour
        and the population disappears into the field. The geometry is the
        subject here; the field belongs to the separate reference figure.
    point_style, colorbar, show_worst, show_candidate
        As in ``save_frame_snapshots``.
    """

    def __init__(self, sweeps, path: str = "snapshot.png",
                 event: Optional[str] = None,
                 dims: Tuple[int, int] = (0, 1),
                 theme: str = "paper",
                 show_rejected: bool = True,
                 background: bool = False,
                 point_style: Optional[str] = None,
                 colorbar: bool = True,
                 show_worst: Optional[bool] = None,
                 show_candidate: Optional[bool] = None,
                 figsize: Tuple[float, float] = (6.0, 6.0),
                 dpi: int = 300) -> None:
        if isinstance(sweeps, (int, np.integer)):
            sweeps = [int(sweeps)]
        self.wanted = sorted({int(s) for s in sweeps})
        if not self.wanted:
            raise ValueError("snapshots: no sweeps requested")
        self.path          = path
        self.event         = event
        self.dims          = dims
        self.theme         = theme
        self.show_rejected = show_rejected
        self.background    = background
        self.point_style   = point_style
        self.colorbar      = colorbar
        self.show_worst     = show_worst
        self.show_candidate = show_candidate
        self.figsize       = figsize
        self.dpi           = dpi

        self._keep: dict = {}      # sweep -> the latest frame seen for it
        self._max_sweep  = -1

    def __call__(self, frame: RunFrame) -> None:
        self._max_sweep = max(self._max_sweep, frame.sweep)
        if frame.sweep in self._wanted_set and (
                self.event is None or frame.event == self.event):
            # Overwrite: the last frame of the sweep wins. The frame
            # holds a COPY of the live points but only a REFERENCE to the
            # dead/rejected lists (sliced by the length recorded at emit
            # time), so holding onto it costs one population, not the
            # whole history.
            self._keep[frame.sweep] = frame

    @property
    def _wanted_set(self):
        # Built once, lazily: __call__ runs on every emitted frame and a
        # set membership test there should not rebuild the set.
        if not hasattr(self, "_ws"):
            self._ws = set(self.wanted)
        return self._ws

    def save(self, P_grid=None, D1=None, D2=None,
             grid_vlim: Optional[Tuple[float, float]] = None) -> List[str]:
        """Write the captured frames. Call after ``run()`` returns."""
        if not self.background:
            P_grid = D1 = D2 = None
        stem, ext = _snapshot_stem(self.path)
        got       = [self._keep[s] for s in self.wanted if s in self._keep]
        # ONE scale over the whole series — see _series_point_norm.
        norm      = _series_point_norm(got) if got else None
        written   = []
        for s in self.wanted:
            fr = self._keep.get(s)
            if fr is None:
                continue
            out = _write_frame_png(fr, f"{stem}_sweep{s}{ext}",
                                   P_grid=P_grid, D1=D1, D2=D2,
                                   grid_vlim=grid_vlim, dims=self.dims,
                                   theme=self.theme,
                                   show_rejected=self.show_rejected,
                                   point_style=self.point_style,
                                   point_norm=norm, colorbar=self.colorbar,
                                   show_worst=self.show_worst,
                                   show_candidate=self.show_candidate,
                                   figsize=self.figsize, dpi=self.dpi)
            written.append(out)
            print(f"  snapshot : sweep {s:>5}  {fr.event:<11} -> {out}")

        missed = [s for s in self.wanted if s not in self._keep]
        if missed:
            why = (f" with event={self.event!r}" if self.event else "")
            print(f"  snapshot : no frame for sweep(s) {missed}{why} — "
                  f"the run reached sweep {self._max_sweep}")
        return written


def save_run_gif(frames: List[RunFrame],
                 path: str,
                 P_grid=None, D1=None, D2=None,
                 grid_vlim: Optional[Tuple[float, float]] = None,
                 dims: Tuple[int, int] = (0, 1),
                 fps: float = 8.0,
                 max_frames: int = 300,
                 include_candidates: bool = False,
                 figsize: Tuple[float, float] = (6.4, 6.4),
                 dpi: int = 100,
                 title: str = "") -> str:
    """
    Render recorded RunFrames into an animated GIF (works headless).

    Parameters
    ----------
    frames             : list of RunFrame from a RunRecorder
    path               : output .gif path
    P_grid, D1, D2     : optional ground-truth heatmap background
                         (e.g. ``viz.P_grid, viz.D1, viz.D2`` from this
                         file's ``Visualizer``) — same background idea
                         as multinest_visualizer_v5.py's heatmap panel
    fps                : playback speed (last frame is held ~2 s)
    max_frames         : GIF frame budget; accepted/candidate frames
                         are strided down to fit, while init / refit /
                         mode_split / converged frames are always kept
    include_candidates : also render every REJECTED candidate draw as
                         its own frame (sub-iteration detail, like the
                         v5 visualizer's per-candidate stepping) —
                         much longer GIF
    """
    try:
        from PIL import Image
    except ImportError as exc:                       # pragma: no cover
        raise RuntimeError(
            "save_run_gif needs Pillow (pip install pillow)") from exc

    import matplotlib
    fl = _thin_frames(frames, include_candidates, max_frames)
    if not fl:
        raise ValueError("no frames to render")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#111111")
    ax  = fig.add_subplot(111)
    # _render_run_frame writes a 3-4 line, left-aligned ax title; without
    # reserving room it collides with the centred suptitle (the two
    # overlapped and were mutually unreadable in every frame).
    if title:
        fig.suptitle(title, color="#dddddd", fontsize=10, y=0.985)
        fig.subplots_adjust(top=0.80, left=0.12, right=0.97, bottom=0.09)
    else:
        fig.subplots_adjust(top=0.84, left=0.12, right=0.97, bottom=0.09)

    images = []
    for fr in fl:
        _render_run_frame(ax, fr, P_grid=P_grid, D1=D1, D2=D2, dims=dims,
                          grid_vlim=grid_vlim)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        images.append(Image.fromarray(buf[..., :3].copy()))
    plt.close(fig)

    dur           = [int(1000.0 / fps)] * len(images)
    dur[-1]       = 2000                     # hold the final frame
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=dur, loop=0, optimize=False)
    print(f"  GIF saved : {path}  "
          f"({len(images)} frames, {fps:g} fps)")
    return path


class RunPlayer:
    """
    Interactive matplotlib player over PRE-RECORDED RunFrames —
    the multi-mode counterpart of multinest_visualizer_v5.py's live
    window, but replaying a finished run so you can also step
    BACKWARDS. Controls: Prev / Next / Auto / Reset buttons, a frame
    slider, and \u2190 / \u2192 / space / home keys.

    Needs an interactive matplotlib backend (TkAgg/Qt5Agg…); on a
    headless machine use ``save_run_gif`` instead.
    """

    def __init__(self, frames: List[RunFrame],
                 P_grid=None, D1=None, D2=None,
                 grid_vlim: Optional[Tuple[float, float]] = None,
                 dims: Tuple[int, int] = (0, 1),
                 title: str = "MultiNest — run player",
                 auto_ms: int = 120) -> None:
        from matplotlib.widgets import Button, Slider
        self.frames = frames
        self.P_grid, self.D1, self.D2 = P_grid, D1, D2
        self.grid_vlim = grid_vlim
        self.dims   = dims
        self.idx    = 0
        self._auto  = False

        self.fig = plt.figure(figsize=(7.2, 7.8))
        self.fig.patch.set_facecolor("#111111")
        self.fig.canvas.manager.set_window_title(title)
        # top edge at 0.86, not 0.93: _render_run_frame writes a 3-4 line
        # left-aligned ax title that otherwise runs off the canvas.
        self.ax  = self.fig.add_axes([0.09, 0.16, 0.88, 0.70])

        def _btn(rect, label, cb):
            axb = self.fig.add_axes(rect)
            b   = Button(axb, label, color="#333333",
                         hovercolor="#555555")
            b.label.set_color("#dddddd")
            b.on_clicked(cb)
            return b

        self.b_prev  = _btn([0.09, 0.045, 0.12, 0.05], "\u25c0 Prev",  self._prev)
        self.b_next  = _btn([0.23, 0.045, 0.12, 0.05], "Next \u25b6",  self._next)
        self.b_auto  = _btn([0.37, 0.045, 0.12, 0.05], "Auto",          self._toggle_auto)
        self.b_reset = _btn([0.51, 0.045, 0.12, 0.05], "Reset",         self._reset)

        ax_sl = self.fig.add_axes([0.68, 0.055, 0.28, 0.03])
        self.slider = Slider(ax_sl, "", 0, max(len(frames) - 1, 1),
                             valinit=0, valstep=1, color="#6ab0ff")
        self.slider.on_changed(self._on_slider)

        self.timer = self.fig.canvas.new_timer(interval=auto_ms)
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    # ── navigation ────────────────────────────────────────────
    def _draw(self) -> None:
        _render_run_frame(self.ax, self.frames[self.idx],
                          P_grid=self.P_grid, D1=self.D1, D2=self.D2,
                          dims=self.dims, grid_vlim=self.grid_vlim)
        self.ax.text(0.99, 0.01,
                     f"frame {self.idx + 1}/{len(self.frames)}",
                     transform=self.ax.transAxes, ha="right",
                     va="bottom", fontsize=8, color="#aaaaaa")
        self.fig.canvas.draw_idle()

    def _set(self, i: int) -> None:
        self.idx = int(np.clip(i, 0, len(self.frames) - 1))
        if int(self.slider.val) != self.idx:
            self.slider.set_val(self.idx)   # re-enters via _on_slider
            return
        self._draw()

    def _prev(self, *_):  self._set(self.idx - 1)
    def _next(self, *_):  self._set(self.idx + 1)
    def _reset(self, *_): self._stop_auto(); self._set(0)
    def _on_slider(self, val): self.idx = int(val); self._draw()

    def _toggle_auto(self, *_):
        self._auto = not self._auto
        self.b_auto.label.set_text("Stop" if self._auto else "Auto")
        (self.timer.start if self._auto else self.timer.stop)()

    def _stop_auto(self):
        if self._auto:
            self._toggle_auto()

    def _tick(self):
        if self.idx >= len(self.frames) - 1:
            self._stop_auto()
            return
        self._set(self.idx + 1)

    def _on_key(self, ev):
        if   ev.key == "right": self._next()
        elif ev.key == "left":  self._prev()
        elif ev.key == " ":     self._toggle_auto()
        elif ev.key == "home":  self._reset()

    def show(self) -> None:
        plt.show()


# ============================================================
# SECTION 13 — EXAMPLE CATALOGUE  (mirrors multinest_visualizer_v5.py)
# ============================================================
#
# Seven self-contained white-box examples, identical to the ones used
# by the interactive visualizer.  Each entry is a plain dict with keys:
#   title        : str   — short label shown in the menu
#   description  : str   — one-line equation / constraint summary
#   equation     : callable  (d, theta) -> np.ndarray
#   uncertainty  : GaussianUncertainty
#   constraints  : list of (lb, ub)
#   design_space : DesignSpace
#   sampler_kw   : dict  {N_L, N_theta, alpha_star, F_threshold}
#
# Some sampler_kw keys are per-RUN choices rather than sampler options, so
# ``_run_example`` strips them before splatting the rest into
# ``MultiNestSampler``:
#   feas_criterion  : "P" | "VaR" | "CVaR" — goes to the ESTIMATOR
#   seed            : int                  — goes to ``run(seed=...)``
#   snapshots       : iterable of int      — sweeps to write as stills
#   snapshot_kwargs : dict                 — options for those stills
# Declare feas_criterion or seed and ``main()`` stops asking for it; leave
# it out and main() prompts — defaulting to "VaR", and to a freshly drawn
# seed that the run header echoes so the run can still be repeated
# afterwards.
#
# ``snapshots`` writes ``snapshots/<title>_sweep<N>.png`` for each named
# sweep: the frame player's panel, on a white ground at 300 dpi, for the
# figure showing how the proposal geometry evolves. See ``SnapshotSaver``
# — in particular, the sweeps must be named UP FRONT, so the workflow is
# to run once, read the final sweep count off the log, then pin the
# interesting sweeps and re-run with the same ``seed``. Options go in
# ``snapshot_kwargs``, e.g. ``dict(event="refit")`` to land only on the
# sweeps where the decomposition was rebuilt, or ``dict(theme="dark")``
# for the player's own colours.
# ──────────────────────────────────────────────────────────────────

# ── shared cluster centres (same as visualizer) ──────────────────
_C5A = np.array([-0.70,  0.70])
_C5B = np.array([ 0.70,  0.70])
_C5C = np.array([ 0.00, -0.75])

_C6A = np.array([-0.55,  0.55])
_C6B = np.array([ 0.55,  0.55])
_C6C = np.array([ 0.00, -0.60])

_C7NW = np.array([-0.60,  0.60])
_C7NE = np.array([ 0.60,  0.60])
_C7SW = np.array([-0.60, -0.60])
_C7SE = np.array([ 0.60, -0.60])

_C9A = np.array([-0.50, -0.50])
_C9B = np.array([ 0.50,  0.50])

_DS_STD = DesignSpace([(-1.0, 1.0), (-1.0, 1.0)], names=["d1", "d2"])


def _eq1(d, theta):
    return theta * (d[0]**2 - 0.3) + d[1]

def _eq2(d, theta):
    return theta * (d[0]**2 - d[1]**2) + 0.5 * d[1]

def _eq3(d, theta):
    return theta * (d[0]**2 - 0.5) + d[1] + 0.3 * d[0]

def _eq4(d, theta):
    return theta * d[0]**2 + d[1]

def _eq5(d, theta):
    bump = (np.exp(-9.0 * np.sum((d - _C5A)**2))
          + np.exp(-9.0 * np.sum((d - _C5B)**2))
          + np.exp(-9.0 * np.sum((d - _C5C)**2)))
    return theta * bump

def _eq6(d, theta):
    bump = (np.exp(-7.0 * np.sum((d - _C6A)**2))
          + np.exp(-7.0 * np.sum((d - _C6B)**2))
          + np.exp(-7.0 * np.sum((d - _C6C)**2)))
    return theta * bump

def _eq7(d, theta):
    bump = (np.exp(-6.0 * np.sum((d - _C7NW)**2))
          + np.exp(-6.0 * np.sum((d - _C7NE)**2))
          + np.exp(-6.0 * np.sum((d - _C7SW)**2))
          + np.exp(-6.0 * np.sum((d - _C7SE)**2)))
    return theta * bump

def _eq8(d, theta):
    a, b = 0.4, 0.2
    return theta * ((d[0]/a)**2 + (d[1]/b)**2)

def _eq9(d, theta):
    r = 0.35
    d1 = np.sum(((d - _C9A) / r)**2)
    d2 = np.sum(((d - _C9B) / r)**2)
    return theta * min(d1, d2)


# ------------------------------------------------------------------
# Fixed scenario sets, for the like-for-like comparison
# ------------------------------------------------------------------
# GaussianUncertainty redraws theta at every feasibility evaluation; the
# reference implementation discretises the Gaussian ONCE and reuses that
# set everywhere. The two therefore differ in the treatment of the
# scenarios as well as in the sampler, and a comparison that changes both
# at once attributes to one what may belong to the other. The four
# problems that enter the comparison are switched to a fixed set below,
# with the redrawing setting kept commented out one line above each, so
# either behaviour is a single edit away.
#
# The set is drawn from the SAME N(mu, sigma) and is the SAME size as the
# N_theta it replaces, so the per-evaluation cost is unchanged and the
# treatment is the only thing that differs.

# 11 because that is the seed the MAGNUS runs were made at
# (``scenario_seed`` in points_nsfeas_kusumo.npz and _banana.npz). The
# reference side builds its set with
#
#     np.random.default_rng(seed).normal(mu, sigma, nscen)
#
# in magnus_benchmark_2d.py, which is the same call as below, so at this
# seed the two sides do not merely draw from the same distribution --
# they see the SAME theta values, to the bit. Change it only together
# with the MAGNUS ``--seed``, or the comparison stops being paired.
SCENARIO_SEED = 11


def _fixed_scenarios(sigma, n, mu=1.0, seed=SCENARIO_SEED):
    """One Gaussian scenario set, drawn once and then held fixed.

    ``default_rng(seed)`` is a generator of its own rather than the global
    stream the sampler draws from, which is the point of it: two runs that
    differ only in ``sampler_kw["seed"]`` see identical scenarios, and the
    set does not shift with how many draws the sampler has already made.
    The scenario seed and the sampler seed are separate knobs because they
    answer separate questions.

    Weights are 1/n, so this is the same plain Monte Carlo estimator
    GaussianUncertainty forms -- only frozen.

    Passing ``seed`` explicitly gives a different set, which is how to
    measure the scenario-set dependence itself: repeat a run under a few
    seeds and the spread of the certified region is the part of the answer
    that belongs to the draw rather than to the design. That spread is
    invisible from inside any single fixed-set run, which is the one thing
    a fixed set cannot report about itself.
    """
    theta = np.random.default_rng(seed).normal(mu, sigma, n)
    return WeightedScenarios(theta_samples=theta,
                             weights=np.full(n, 1.0 / n))


EXAMPLES = [
    dict(
        title       = "1 · Arch",
        description = "s=θ(d₁²−0.30)+d₂  |  θ~N(1,1)  |  s∈[0.20,0.75]  |  single arch",
        equation    = _eq1,
        uncertainty = GaussianUncertainty(mu=1.0, sigma=1.0),
        constraints = [(0.20, 0.75)],
        design_space= _DS_STD,
        sampler_kw  = dict(
            N_L=700, N_theta=100, alpha_star=0.95,
            feas_criterion = "VaR",   
        ),
    ),
    dict(
        title       = "2 · Twin Wings",
        description = "s=θ(d₁²−d₂²)+0.5d₂  |  θ~N(1,0.4)  |  s∈[0.10,0.50]  |  two wings",
        equation    = _eq2,
        uncertainty = GaussianUncertainty(mu=1.0, sigma=0.4),
        constraints = [(0.10, 0.50)],
        design_space= _DS_STD,
        sampler_kw  = dict(
            N_L=700, N_theta=100, alpha_star=0.95,
            feas_criterion = "VaR",  
        ),
    ),
    dict(
        title       = "3 · Banana + Island",
        description = "s=θ(d₁²−0.5)+d₂+0.3d₁  |  θ~N(1,0.5) fixed K=100  |  s∈[0.00,0.40]  |  banana+island",
        equation    = _eq3,
        # uncertainty = GaussianUncertainty(mu=1.0, sigma=0.5),
        uncertainty = _fixed_scenarios(sigma=0.5, n=100),
        constraints = [(0.00, 0.40)],
        design_space= _DS_STD,
        sampler_kw  = dict(
            # N_theta is ignored by WeightedScenarios: the fixed set is
            # 100 scenarios, so the cost per evaluation is what it was.
            N_L=900, N_theta=100, alpha_star=0.95,
            feas_criterion = "VaR",  
            seed           = 1,
            #snapshots      = {1109, 2133},
            #snapshots      = {1057, 2199},  104 seed old
        ),
    ),
    dict(
        title       = "4 · Kusumo et al.",
        description = "s=θd₁²+d₂  |  θ~N(1,√0.3) fixed K=100  |  s∈[0.20,0.75]  |  paper Fig 1",
        equation    = _eq4,
        # uncertainty = GaussianUncertainty(mu=1.0, sigma=float(np.sqrt(0.3))),
        uncertainty = _fixed_scenarios(sigma=float(np.sqrt(0.3)), n=100),
        constraints = [(0.20, 0.75)],
        design_space= _DS_STD,
        sampler_kw  = dict(
            # N_theta is ignored by WeightedScenarios: the fixed set is
            # 100 scenarios, so the cost per evaluation is what it was.
            N_L=500, N_theta=100, alpha_star=0.95,
            feas_criterion = "VaR",  
            seed           = 11, 
            #trace_decomposition = {66, 75,135},
        ),
    ),
    dict(
        title       = "5 · 3 Clusters — Solid Discs",
        description = "s=θ·Σ exp(−9‖d−cₖ‖²)  |  θ~N(1,0.15)  |  s≥0.28  |  3 filled discs",
        equation    = _eq5,
        uncertainty = GaussianUncertainty(mu=1.0, sigma=0.15),
        constraints = [(0.28, np.inf)],
        design_space= _DS_STD,
        sampler_kw  = dict(
            N_L=700, N_theta=150, alpha_star=0.95,
            feas_criterion = "VaR",  
        ),
    ),
    dict(
        title       = "6 · 3 Clusters — Torus Rings",
        description = "s=θ·Σ exp(−7‖d−cₖ‖²)  |  θ~N(1,0.18) fixed K=100  |  s∈[0.30,0.75]  |  3 torus rings",
        equation    = _eq6,
        # uncertainty = GaussianUncertainty(mu=1.0, sigma=0.18),
        uncertainty = _fixed_scenarios(sigma=0.18, n=100),
        constraints = [(0.30, 0.75)],
        design_space= _DS_STD,
        sampler_kw  = dict(
            # N_theta is ignored by WeightedScenarios: the fixed set is
            # 100 scenarios, so the cost per evaluation is what it was.
            N_L=900, N_theta=100, alpha_star=0.95,
            feas_criterion = "VaR", 
            seed           = 98, 
            #trace_decomposition = {1058, 1180, 1195} 
            #snapshots      = {586, 1195}, 
            #snapshots      = {506, 1140}, old
            #trace_decomposition = {476, 630,945,1002,1109,1140,1262} old
        ),
    ),
    dict(
        title       = "7 · 4 Clusters — Solid Discs",
        description = "s=θ·Σ exp(−6‖d−cₖ‖²)  |  θ~N(1,0.25)  |  s≥0.35  |  4 filled discs",
        equation    = _eq7,
        uncertainty = GaussianUncertainty(mu=1.0, sigma=0.25),
        constraints = [(0.35, np.inf)],
        design_space= _DS_STD,
        sampler_kw  = dict(
            N_L=900, N_theta=150, alpha_star=0.95,
            feas_criterion = "VaR",  
        ),
    ),
    dict(
        title       = "8 · Single Ellipse",
        description = "s=θ((d₁/0.4)²+(d₂/0.2)²)  |  θ~N(1,0.45) fixed K=100  |  s≤3.30  |  single ellipse",
        equation    = _eq8,
        # uncertainty = GaussianUncertainty(mu=1.0, sigma=0.45),
        uncertainty = _fixed_scenarios(sigma=0.45, n=100),
        constraints = [(-np.inf, 3.3)],
        design_space= _DS_STD,
        sampler_kw  = dict(
            # N_theta is ignored by WeightedScenarios: the fixed set is
            # 100 scenarios, so the cost per evaluation is what it was.
            N_L=500, N_theta=100, alpha_star=0.95,
            feas_criterion = "VaR",  
            seed           = 11,
            snapshots      = {156, 921}
            #snapshots      = {142, 921}, old
        ),
    ),
    dict(
        title       = "9 · Two Separated Ellipses",
        description = "s=θ·min(‖(d−c₁)/r‖²,‖(d−c₂)/r‖²)  |  θ~N(1,0.15)  |  s∈[0.00,1.00]  |  two separated ellipses",
        equation    = _eq9,
        uncertainty = GaussianUncertainty(mu=1.0, sigma=0.15),
        constraints = [(0.0, 1.0)],
        design_space= _DS_STD,
        sampler_kw  = dict(
            N_L=400, N_theta=100, alpha_star=0.95,
            feas_criterion = "VaR",  
        ),
    ),
]


# ============================================================
# SECTION 14 — TEE LOGGER  (mirrors console output to a .txt file)
# ============================================================

import sys as _sys
import io as _io

class _TeeLogger:
    """
    Wraps sys.stdout so that every print() call is written both to
    the terminal AND to an in-memory buffer.  Call save(path) at the
    end to flush the buffer to a UTF-8 text file.
    """

    def __init__(self) -> None:
        self._terminal = _sys.stdout
        self._buf      = _io.StringIO()

    def write(self, message: str) -> None:
        self._terminal.write(message)
        self._buf.write(message)

    def flush(self) -> None:
        self._terminal.flush()
        self._buf.flush()

    def save(self, path: str) -> None:
        content = self._buf.getvalue()
        _sys.stdout = self._terminal
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Console log saved : {path}")

    def restore(self) -> None:
        _sys.stdout = self._terminal


# ============================================================
# SECTION 15 — INTERACTIVE EXAMPLE SELECTOR
# ============================================================

def _select_example() -> dict:
    """
    Print a numbered menu and return the chosen example dict.
    Pressing Enter without a number re-prints the menu.
    """
    sep = "─" * 62
    print(f"\n  {sep}")
    print("  MultiNest — Example Selector")
    print(f"  {sep}")
    for i, ex in enumerate(EXAMPLES, 1):
        name = ex['title']
        head, dot, tail = name.partition("·")
        if dot and head.strip().isdigit():
            name = tail.strip()
        print(f"  [{i}]  {name}")
    print(f"  {sep}")

    while True:
        raw = input(f"  Select example [1–{len(EXAMPLES)}] (or q to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            print("  Exiting.")
            raise SystemExit(0)
        if raw.isdigit() and 1 <= int(raw) <= len(EXAMPLES):
            return EXAMPLES[int(raw) - 1]
        print(f"  ✗  Please enter a number between 1 and {len(EXAMPLES)}.")


# ============================================================
# SECTION 16 — ENTRY POINT
# ============================================================

def _run_example(ex: dict, interactive: bool = False,
                 viz_mode: Optional[str] = None,
                 gif_kwargs: Optional[dict] = None,
                 feas_criterion: Optional[str] = None,
                 run_seed: Optional[int] = None) -> "SamplerResult":
    """
    Run a single example end-to-end:
      - landscape plot  →  landscape_<title>.png
      - MultiNest run   →  multinest_<title>.png
      - console log     →  results_<title>.txt
      - viz_mode='gif'  →  additionally multinest_run_<title>.gif
      - viz_mode='player' → additionally opens the interactive
                            frame player after the run

    Parameters
    ----------
    ex          : one entry from EXAMPLES
    interactive : back-compat alias for ``viz_mode='step'``
    viz_mode    : None/'plain' — straight batch run (old default);
                  'step'   — pause on Enter after every algorithmic
                             step, drawn live (old interactive mode;
                             VERBOSE=STEP_MODE=True);
                  'gif'    — record every event via the run()
                             ``frame_callback`` hook and save an
                             animated GIF of the whole run (headless-
                             safe, SECTION 12);
                  'player' — record the run, then open the
                             interactive ``RunPlayer`` window
                             (Prev/Next/Auto/slider) to replay it.
    gif_kwargs  : optional dict forwarded to ``save_run_gif`` in
                  'gif' mode (fps, max_frames, include_candidates, …).
    feas_criterion : {"P", "VaR", "CVaR"} or None. None -> whatever the
                  example declares in ``sampler_kw['feas_criterion']``,
                  else "VaR". Anything but "P" is suffixed onto every
                  output filename so runs of the same example under
                  different criteria don't overwrite each other.
    run_seed    : int passed to ``run(seed=...)``, or None -> whatever the
                  example declares in ``sampler_kw['seed']``, else a
                  freshly DRAWN seed, so repeated runs explore different
                  trajectories. Whatever is used is echoed in the run
                  header, which is what keeps a run reproducible after the
                  fact — essential for ``trace_decomposition``: to dissect
                  the sweeps you saw in the player, read the seed off the
                  log and pin it in ``sampler_kw['seed']``.

    Returns
    -------
    SamplerResult
        The full run result. ``main()`` / ``_run_all_examples`` capture
        it, so after the run you can reach e.g.
        ``result.decomp_traces`` (populated when ``sampler_kw`` set
        ``trace_decomposition``) for ``.summary()`` / ``.plot()``, or
        ``analyze_decomposition_frames(...)`` on the recorded frames.
    """
    global VERBOSE, STEP_MODE

    import sys as _sys
    import time as _time

    if viz_mode is None:
        viz_mode = 'step' if interactive else 'plain'
    if viz_mode not in ('plain', 'step', 'gif', 'player'):
        raise ValueError(f"unknown viz_mode {viz_mode!r}")
    interactive = (viz_mode == 'step')

    if viz_mode == 'step':
        # Use whatever GUI backend is available so the live step
        # window can actually be shown; do NOT force Agg here.
        VERBOSE, STEP_MODE = True, True
    elif viz_mode == 'player':
        # Needs a GUI backend for the player window, but the run
        # itself is silent (frames are recorded, not stepped).
        VERBOSE, STEP_MODE = False, False
    else:
        import matplotlib
        matplotlib.use('Agg')   # non-interactive: no window, no blocking
        VERBOSE, STEP_MODE = False, False

    kw           = ex['sampler_kw']
    alpha_star   = kw['alpha_star']
    # Criterion resolution order: explicit argument (main()'s prompt) >
    # whatever the example declares in sampler_kw > "P". It lives on the
    # ESTIMATOR, so like N_theta it must NOT be splatted into the sampler.
    if feas_criterion is None:
        feas_criterion = kw.get('feas_criterion', 'VaR')
    # Same resolution order for the seed: an explicit argument wins, then
    # the example's own declaration. With neither, a fresh seed is DRAWN
    # rather than fixed, so repeated runs explore different trajectories —
    # and it is echoed in the header below, so any run stays reproducible
    # after the fact (pin it in sampler_kw['seed'] to repeat one).
    if run_seed is None:
        run_seed = kw.get('seed')
    _seed_drawn = run_seed is None
    if _seed_drawn:
        run_seed = int(np.random.SeedSequence().entropy % (2**31))
    safe_title   = (ex['title']
                    .replace(' ', '_')
                    .replace('·', '')
                    .replace('/', '-')
                    .strip('_')
                    + ('' if feas_criterion == 'P' else f"_{feas_criterion}"))

    # ── start capturing output ────────────────────────────────────
    tee = _TeeLogger()
    _sys.stdout = tee

    print(f"\n{'='*62}")
    print(f"  Example : {ex['title']}")
    print(f"  {ex['description']}")
    print(f"  Criterion : {feas_criterion}"
          + ("" if feas_criterion == 'P' else
             f"  (alpha = {1.0 - alpha_star:g}; feasible <= 0)"))
    print(f"  Seed      : {run_seed}"
          + ("  (drawn; pin it in sampler_kw['seed'] to repeat this run)"
             if _seed_drawn else ""))
    if viz_mode == 'step':
        print(f"  Step-by-step visualization : ON "
              f"(watch the 'MultiNest — step-by-step' window)")
    elif viz_mode == 'gif':
        print("  Run recording : ON — an animated GIF of the whole "
              "run will be saved")
    elif viz_mode == 'player':
        print("  Run recording : ON — the interactive frame player "
              "opens after the run")
    print(f"{'='*62}\n")

    model = ProcessModel(
        equation    = ex['equation'],
        uncertainty = ex['uncertainty'],
        constraints = ex['constraints'],
        name        = ex['title'],
    )
    estimator = model.make_estimator(
        uncertainty    = ex['uncertainty'],
        N_theta        = kw['N_theta'],
        feas_criterion = feas_criterion,
    )

    # ── landscape plot (always saved to disk) ───────────────────────
    viz = Visualizer(
        model          = model,
        alpha_star     = alpha_star,
        design_space   = ex['design_space'],
        n_grid         = 60,
        n_grid_theta   = 400,
        feas_criterion = feas_criterion,
    )
    viz.plot_landscape(save_path=f"landscape_{safe_title}.png")

    # ── run MultiNest ─────────────────────────────────────────────
    _t_start = _time.perf_counter()

    recorder = RunRecorder() if viz_mode in ('gif', 'player') else None

    # Forward EVERY sampler_kw key except ``N_theta`` (which belongs to
    # the estimator, built above — not to the sampler). Hand-picking the
    # keys here instead, as an earlier version did, silently DROPPED any
    # option an example declared but this function didn't know about
    # (min_pt, ef, multimodal, …): no error, just no effect.
    # Splatting keeps examples
    # and sampler in sync automatically — a bad key now raises a normal
    # TypeError from MultiNestSampler.__init__ instead of vanishing.
    # ``log_every``/``log_heartbeat`` are the exception: they belong to
    # run(), not to the constructor (how much ONE invocation prints is a
    # presentation choice, not sampler state). An example may still
    # declare them in its sampler_kw, so route just those two on to the
    # run() call rather than making examples reach a different API.
    _RUN_ONLY  = ('log_every', 'log_heartbeat')
    run_kw     = {k: kw[k] for k in _RUN_ONLY if k in kw}
    sampler_kw = {k: v for k, v in kw.items()
                  if k not in ('N_theta', 'feas_criterion', 'seed',
                               'snapshots', 'snapshot_kwargs')
                  + _RUN_ONLY}

    # ``snapshots`` is a FIGURE request, not sampler state, so it is
    # stripped like seed/N_theta. It rides along with any viz_mode --
    # SnapshotSaver keeps only the named sweeps, so it costs nothing even
    # on a plain run, and the two callbacks are fanned out below when a
    # gif/player recording is also running.
    saver = None
    if kw.get('snapshots'):
        saver = SnapshotSaver(kw['snapshots'],
                              path=f"snapshots/{safe_title}.png",
                              **(kw.get('snapshot_kwargs') or {}))
        print(f"  Snapshots : sweeps {saver.wanted} -> "
              f"snapshots/{safe_title}_sweep*.png")

    if saver is not None and recorder is not None:
        def _callback(fr, _r=recorder, _s=saver):
            _r(fr)
            _s(fr)
    else:
        _callback = saver or recorder

    result = MultiNestSampler(
        estimator    = estimator,
        design_space = ex['design_space'],
        **sampler_kw,
    ).run(frame_callback=_callback, seed=run_seed, **run_kw)

    if saver is not None:
        # viz was built before the run, so the criterion field is already
        # on hand -- the stills get the same background as the landscape
        # figure and the GIF, with nothing recomputed.
        saver.save(P_grid=viz.V_grid, D1=viz.D1, D2=viz.D2,
                   grid_vlim=viz.grid_vlim)

    # Sampling efficiency and wall time are already in the RESULT block the
    # sampler just printed; repeating them here only invited the reader to
    # check whether the two agreed.

    # Written now so the figure exists, but announced at the very end
    # alongside the console log -- the two "saved" lines belong together.
    _plot_path = f"multinest_{safe_title}.png"
    viz.plot_multinest_result(result, save_path=_plot_path, announce=False)
    print(result.reliability_table())
    if feas_criterion != "P":
        # The classical five-bin reliability-range table on the
        # feasibility probability P — available for free on a VaR/CVaR
        # run because merit_and_P records P from the same scenario sweep
        # that produced the risk measure (zero extra model runs). The
        # table above bins by the DRIVING criterion; this one is the
        # companion P view of the same sampled points.
        print(f"\n  Reliability-range table on P "
              f"(recorded alongside {feas_criterion}):")
        print(result.reliability_table_P())
    _mt = result.mode_table()
    if len(_mt) > 2:          # >1 real mode + Total: only worth printing then
        print()
        print(_mt.to_string(index=False))

    # ── algorithm movie: GIF export / interactive player ─────────
    if recorder is not None:
        print(f"  Recorded frames      : {len(recorder.frames)} "
              f"(events: init/candidate/accepted/refit/mode_split/"
              f"converged)")
        if viz_mode == 'gif':
            gkw = dict(fps=8.0, max_frames=300, include_candidates=False)
            gkw.update(gif_kwargs or {})
            save_run_gif(
                recorder.frames,
                path      = f"multinest_run_{safe_title}.gif",
                P_grid    = viz.V_grid, D1=viz.D1, D2=viz.D2,
                grid_vlim = viz.grid_vlim,
                title     = ex['title'],
                **gkw,
            )
        else:   # 'player'
            print("  Opening the interactive frame player "
                  "(Prev/Next/Auto/slider, arrow keys) …")

    # ── artefacts written, announced together ─────────────────────
    # The log line comes first because tee.save() is what closes the log;
    # anything printed after it is terminal-only by construction.
    tee.save(f"results_{safe_title}.txt")
    print(f"  Plot saved        : {_plot_path}")

    if viz_mode == 'player' and recorder is not None:
        try:
            RunPlayer(recorder.frames,
                      P_grid=viz.V_grid, D1=viz.D1, D2=viz.D2,
                      grid_vlim=viz.grid_vlim,
                      title=f"MultiNest — run player — {ex['title']}"
                      ).show()
        except Exception as exc:                     # headless fallback
            print(f"  Could not open the player window ({exc}).")
            print("  Falling back to a GIF instead …")
            save_run_gif(recorder.frames,
                         path=f"multinest_run_{safe_title}.gif",
                         P_grid=viz.V_grid, D1=viz.D1, D2=viz.D2,
                         grid_vlim=viz.grid_vlim,
                         title=ex['title'])

    if viz_mode == 'step':
        print("\n  Done. The step-by-step window stays open — close it "
              "manually or press Enter here to exit.")
        input()

    return result


def _run_all_examples() -> None:
    """
    Run ALL examples back-to-back without any user interaction
    (no step-by-step visualization — this is the old default
    ``main()`` behaviour, kept available for batch/regression runs).

    For each example the following files are saved in the working directory:
      landscape_<title>.png   — MC ground-truth P-field
      multinest_<title>.png   — MultiNest scatter vs. ground truth
      results_<title>.txt     — full console output incl. runtime
    """
    import time as _time

    t_total = _time.perf_counter()

    print("\n  MultiNest — Batch Run")
    print(f"  Running all {len(EXAMPLES)} examples sequentially ...\n")

    for i, ex in enumerate(EXAMPLES, 1):
        print(f"  [{i}/{len(EXAMPLES)}]  Starting: {ex['title']} ...")
        _run_example(ex, interactive=False)
        print(f"  [{i}/{len(EXAMPLES)}]  Done.\n")

    elapsed  = _time.perf_counter() - t_total
    h, rem   = divmod(int(elapsed), 3600)
    m, s     = divmod(rem, 60)
    print(f"  All {len(EXAMPLES)} examples finished.")
    print(f"  Total wall time : {h:02d}h {m:02d}m {s:02d}s")


LAST_RESULT = None  # set by main(): the most recent run, for post-run
                    # inspection (e.g. LAST_RESULT.decomp_traces).


def main() -> "SamplerResult":
    """
    Interactive entry point.

      1. Lets you pick ONE example from the numbered menu
         (``_select_example()``).
      2. Asks how to run it: plain batch run; the step-by-step
         debugger (pause on Enter after every algorithmic step); an
         animated GIF of the whole run (recorded through run()'s
         ``frame_callback`` hook and rendered by ``save_run_gif``,
         SECTION 12 — works on headless machines); or the interactive
         ``RunPlayer`` (replay the recorded run with Prev/Next/Auto
         buttons, a frame slider and arrow keys — the multi-mode
         counterpart of multinest_visualizer_v5.py's live window).
      3. Runs that one example.

    To run every example back-to-back non-interactively instead, call
    ``_run_all_examples()`` directly (e.g. from a separate script or
    ``python -c "import multinest_sampler as m; m._run_all_examples()"``).
    """
    ex = _select_example()

    # ── feasibility criterion ────────────────────────────────────
    # A per-RUN analysis choice, not a property of the model: the same
    # example is worth characterising under more than one criterion. So it
    # is asked here — UNLESS the example pins it in sampler_kw, in which
    # case that is the answer and there is nothing to ask.
    _kw = ex['sampler_kw']
    if 'feas_criterion' in _kw:
        feas_criterion = _kw['feas_criterion']
        print(f"\n  Feasibility criterion : {feas_criterion} "
              f"(from sampler_kw)")
    else:
        print("\n  Which feasibility criterion?")
        print("    [1] P      [2] VaR      [3] CVaR")
        _choices = {"": "VaR", "1": "P", "2": "VaR", "3": "CVaR"}
        raw = input("  Criterion [1-3, default VaR]: ").strip()
        feas_criterion = _choices.get(raw)
        while feas_criterion is None:
            raw = input("  ✗  Please enter 1, 2 or 3: ").strip()
            feas_criterion = _choices.get(raw)

    # ── random seed ──────────────────────────────────────────────
    # Enter draws a fresh one, so consecutive runs of the same example
    # explore different trajectories. ``_run_example`` echoes whichever
    # seed it used, so a run stays reproducible after the fact: read the
    # seed off the log and type it back (or pin sampler_kw['seed'], in
    # which case this is not asked for at all).
    if 'seed' in _kw:
        run_seed = _kw['seed']
        print(f"  Random seed           : {run_seed} (from sampler_kw)")
    else:
        raw = input("\n  Random seed [integer, Enter = draw a new "
                    "one]: ").strip()
        while True:
            if raw == "":
                run_seed = None          # _run_example draws and reports it
                break
            try:
                run_seed = int(raw)
                break
            except ValueError:
                raw = input("  ✗  Please enter an integer (or Enter to "
                            "draw a new one): ").strip()

    print("\n  How do you want to run it?")
    print("    [1] Plain run — just save the summary PNGs/log "
          "(default)")
    print("    [2] Step-by-step debugger — every ellipsoid fit, EM")
    print("        reassignment, candidate draw and eviction printed")
    print("        AND drawn live, pausing on Enter after each one")
    print("    [3] Animated GIF of the whole run — records the")
    print("        algorithm via the frame hook and saves")
    print("        multinest_run_<title>.gif (works headless)")
    print("    [4] Interactive frame player — records the run, then")
    print("        replay it with Prev/Next/Auto buttons + slider")
    print("        (needs a display; falls back to a GIF if not)")
    raw = input("  Visualization mode [1-4, default 1]: ").strip()
    viz_mode = {"": "plain", "1": "plain", "2": "step",
                "3": "gif",  "4": "player"}.get(raw)
    while viz_mode is None:
        raw = input("  ✗  Please enter 1, 2, 3 or 4: ").strip()
        viz_mode = {"": "plain", "1": "plain", "2": "step",
                    "3": "gif", "4": "player"}.get(raw)

    gif_kwargs = None
    if viz_mode == "gif":
        raw = input("  Also render every REJECTED candidate draw as "
                    "its own frame?\n"
                    "  (sub-iteration detail — much longer GIF) "
                    "[y/N]: ").strip().lower()
        include_candidates = raw in ("y", "yes")
        raw = input("  Playback speed in frames/second "
                    "[default 8]: ").strip()
        try:
            fps = float(raw) if raw else 8.0
        except ValueError:
            fps = 8.0
        gif_kwargs = dict(include_candidates=include_candidates, fps=fps)

    result = _run_example(ex, viz_mode=viz_mode, gif_kwargs=gif_kwargs,
                          feas_criterion=feas_criterion, run_seed=run_seed)
    # Exposed at module scope so that after ``main()`` returns you can
    # still inspect the run from an interactive session — in particular
    # ``LAST_RESULT.decomp_traces`` when ``sampler_kw`` set
    # ``trace_decomposition`` (each trace has .summary() / .plot()).
    global LAST_RESULT
    LAST_RESULT = result
    if result.decomp_traces:
        print(f"\n  {len(result.decomp_traces)} decomposition trace(s) "
              f"captured. In an interactive session: "
              f"for t in LAST_RESULT.decomp_traces: print(t.summary())  "
              f"(or t.plot('trace.png')). They are also printed above and "
              f"saved in the results_*.txt log.")

        for i, trace in enumerate(result.decomp_traces):
            print("\n" + trace.summary())

            if not trace.nodes:
                continue

            base_name = (
                f"trace_"
                f"sweep_{trace.sweep}_"
                f"mode_{trace.mode_label}_{i}"
            )

            # Normal decomposition trace plot
            trace_plot_path = f"{base_name}.png"
            trace.plot(save_path=trace_plot_path)
            print("Saved trace plot:", trace_plot_path)

            # Recursive decomposition animation
            gif_path = f"{base_name}_recursion.gif"
            trace.animate_recursion(gif_path)
            print("Saved recursion animation:", gif_path)

            # Aynı recursion adımlarının statik grid versiyonu
            recursion_grid_path = f"{base_name}_recursion_steps.png"
            trace.plot_recursion_steps(recursion_grid_path)
            print("Saved recursion steps:", recursion_grid_path)

            # Frame'lere ayrıca erişmek istersen
            frames = trace.recursion_frames()
            print("Number of animation frames:", len(frames))
    return result


if __name__ == "__main__":
    main()

