"""Coarse-grid pRF search, nonlinear refinement, and local parameter uncertainty.

The optimizer searches x0, y0, and sigma. Linear least squares projects amplitude and baseline
at every trial geometry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from ..arrays import FloatArray
from ..config import ConfigError, FitConfig
from ..geometry import Grid
from .model import GaussianReceptiveField, design_matrix, predict

N_PARAMETERS = 5
FITTED_PARAMETERS = ("x0", "y0", "sigma")
BOUND_TOLERANCE = 1e-3
MIN_ON_GRID_VOLUME = 0.99
RESOLUTION_PER_SIGMA = 6.1


@dataclass(frozen=True)
class UnitFit:
    """One fitted unit and its uncertainty.

    Convergence reports optimizer termination; acceptance applies the reporting criteria.
    """

    x0: float
    y0: float
    sigma: float
    beta: float
    baseline: float
    r2: float
    converged: bool
    n_fev: int
    se_x0: float = float("nan")
    se_y0: float = float("nan")
    se_sigma: float = float("nan")
    second_field_r2: float = float("nan")
    """Incremental R² from an additional candidate field.

    This model-mismatch diagnostic has no acceptance cutoff.
    """
    x0_at_bound: bool = False
    y0_at_bound: bool = False
    sigma_at_bound: bool = False
    dof: int = 0
    r2_threshold: float = float("inf")
    """Minimum R² for acceptance. An omitted threshold prevents acceptance."""

    @classmethod
    def failed(cls) -> UnitFit:
        """A unit that could not be fitted. Never reported as a pRF."""
        nan = float("nan")
        return cls(
            x0=nan,
            y0=nan,
            sigma=nan,
            beta=nan,
            baseline=nan,
            r2=nan,
            converged=False,
            n_fev=0,
        )

    @property
    def eccentricity(self) -> float:
        return float(np.hypot(self.x0, self.y0))

    @property
    def at_bound(self) -> bool:
        """Whether any fitted parameter is pinned against its search bound."""
        return self.x0_at_bound or self.y0_at_bound or self.sigma_at_bound

    @property
    def accepted(self) -> bool:
        """Converged at or above the R² threshold, with positive beta and unpinned sigma.

        Either sigma bound disqualifies a size estimate. Center bounds, undefined uncertainty,
        and the second-field diagnostic do not alter this policy. Negative beta is suppression.
        """
        return (
            self.converged
            and self.r2 >= self.r2_threshold
            and self.beta > 0.0
            and not self.sigma_at_bound
        )

    def confidence_interval(self, parameter: str, level: float = 0.95) -> tuple[float, float]:
        """Two-sided interval from local linear covariance and a Student-t critical value.

        Approximately Gaussian residuals and local linearity are assumed. Model error can
        invalidate coverage; unidentifiable parameters have undefined intervals.
        """
        if parameter not in FITTED_PARAMETERS:
            raise ValueError(f"parameter must be one of {FITTED_PARAMETERS}; got {parameter!r}")
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        estimate = getattr(self, parameter)
        error = getattr(self, f"se_{parameter}")
        if not np.isfinite(estimate) or not np.isfinite(error) or self.dof <= 0:
            return float("nan"), float("nan")
        critical = float(student_t.ppf(0.5 + level / 2.0, self.dof))
        return estimate - critical * error, estimate + critical * error


def _restore_amplitude(beta: float, response_scale: float, aperture_scale: float) -> float:
    """Convert amplitude units without overflowing an intermediate scale ratio."""
    response_mantissa, response_exponent = math.frexp(response_scale)
    aperture_mantissa, aperture_exponent = math.frexp(aperture_scale)
    try:
        return math.ldexp(
            beta * response_mantissa / aperture_mantissa,
            response_exponent - aperture_exponent,
        )
    except OverflowError:
        return math.copysign(float("inf"), beta)


def _solve_amplitude(
    prediction: FloatArray, response: FloatArray, *, aperture_scale: float = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Least-squares amplitude and baseline, with beta in the original aperture units."""
    design = np.column_stack([prediction, np.ones_like(prediction)])
    # Shift before scaling to retain small variations around a large baseline.
    response_anchor = float(response[0])
    with np.errstate(over="ignore"):
        shifted_response = response - response_anchor
    finite_shift = np.isfinite(shifted_response).all()
    response_peak = float(np.max(np.abs(response)))
    precision_limit = math.sqrt(np.finfo(np.float64).eps)
    if finite_shift and response_peak > 0.0:
        relative_variation = float(np.max(np.abs(shifted_response))) / response_peak
        if relative_variation >= precision_limit:
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
                    fitted = design @ coefficients
                if np.isfinite(coefficients).all() and np.isfinite(fitted).all():
                    if aperture_scale != 1.0:
                        coefficients[0] = _restore_amplitude(
                            float(coefficients[0]), 1.0, aperture_scale
                        )
                    return coefficients, fitted
            except np.linalg.LinAlgError:
                pass

    if not finite_shift:
        response_anchor = 0.0
        shifted_response = response
    response_scale = float(np.max(np.abs(shifted_response))) or 1.0
    coefficients, *_ = np.linalg.lstsq(design, shifted_response / response_scale, rcond=None)
    beta = _restore_amplitude(float(coefficients[0]), response_scale, aperture_scale)
    offsets = np.concatenate(([coefficients[1]], design @ coefficients))
    with np.errstate(over="ignore", invalid="ignore"):
        restored = response_anchor + response_scale * offsets
        if not np.isfinite(restored).all():
            output_scale = max(abs(response_anchor), response_scale)
            restored = (
                response_anchor / output_scale + (response_scale / output_scale) * offsets
            ) * output_scale
    return np.array([beta, restored[0]]), restored[1:]


def _r_squared(response: FloatArray, fitted: FloatArray) -> float:
    """R², or NaN for constant targets or non-finite inputs."""
    if not np.isfinite(response).all() or not np.isfinite(fitted).all():
        return float("nan")
    scale = float(np.max(np.abs(response)))
    if scale == 0.0:
        return float("nan")
    response = response / scale
    total = float(np.sum((response - response.mean()) ** 2))
    if total <= 0.0:
        return float("nan")
    # An excessively poor prediction has a negative infinite score.
    with np.errstate(over="ignore"):
        residual = float(np.sum((response - fitted / scale) ** 2))
    return 1.0 - residual / total


def _even_at_least(value: float) -> int:
    """Smallest even integer at or above ``value``."""
    return 2 * math.ceil(value / 2)


def _count_distinct_frames(apertures: FloatArray) -> int:
    """Count distinct aperture masks for residual degrees of freedom.

    Duplicate masks, including signed-zero copies, count once. Independent noise can add
    information, which this conservative count does not model.
    """
    flat = apertures.reshape(len(apertures), -1)
    return len({np.where(row == 0.0, 0.0, row).tobytes() for row in flat})


def _standard_errors(
    jacobian: FloatArray, cost: float, n_independent: int
) -> tuple[float, float, float]:
    """Linearised errors for a finite, full-rank three-parameter Jacobian.

    Singular directions return undefined errors; a pseudoinverse alone would give them zero
    variance. ``n_independent`` counts distinct masks; cost and Jacobian use all observations.
    Copying masks and responses scales cost and ``J^T J`` equally, preserving covariance.
    This does not apply to repeated masks with independent response noise.
    """
    if (
        jacobian.ndim != 2
        or jacobian.shape[1] != len(FITTED_PARAMETERS)
        or not np.isfinite(jacobian).all()
        or not np.isfinite(cost)
        or cost < 0.0
        or not isinstance(n_independent, int)
        or isinstance(n_independent, bool)
        or n_independent > len(jacobian)
    ):
        return (float("nan"),) * 3
    dof = n_independent - N_PARAMETERS
    if dof <= 0:
        return (float("nan"),) * 3
    try:
        if np.linalg.matrix_rank(jacobian) < len(FITTED_PARAMETERS):
            return (float("nan"),) * 3
        information = jacobian.T @ jacobian
        if not np.isfinite(information).all():
            return (float("nan"),) * 3
        singular_values = np.linalg.svd(information, compute_uv=False)
        if singular_values[-1] <= 1e-15 * singular_values[0]:
            return (float("nan"),) * 3
        residual_variance = 2.0 * cost / dof
        covariance = residual_variance * np.linalg.pinv(information)
    except np.linalg.LinAlgError:
        return (float("nan"),) * 3
    variances = np.diag(covariance)
    if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
        return (float("nan"),) * 3
    errors = np.sqrt(variances)
    return float(errors[0]), float(errors[1]), float(errors[2])


class PRFFitter:
    """Fit Gaussian pRFs using candidate predictions cached for an aperture snapshot."""

    def __init__(self, grid: Grid, apertures: FloatArray, config: FitConfig) -> None:
        if apertures.ndim != 3:
            raise ValueError("apertures must have shape (frames, height, width)")
        if apertures.shape[1:] != grid.shape:
            raise ValueError("aperture frames must match the grid")
        if len(apertures) <= N_PARAMETERS:
            raise ValueError(
                f"need more than {N_PARAMETERS} frames to fit a pRF; got {len(apertures)}"
            )
        if not np.isfinite(apertures).all():
            raise ValueError("apertures must be finite; found NaN or inf")
        self.grid = grid
        self._apertures = apertures.astype(np.float64, copy=True)
        self._apertures.setflags(write=False)
        self._aperture_scale = float(np.max(np.abs(self._apertures))) or 1.0
        self._fit_apertures = (
            self._apertures
            if self._aperture_scale == 1.0
            else self._apertures / self._aperture_scale
        )
        self._fit_apertures.setflags(write=False)
        self.config = config
        self._check_sigma_ceiling()
        self.n_independent_frames = _count_distinct_frames(self.apertures)
        self._candidates = tuple(self._build_candidates())
        self._predictions = design_matrix(self.candidates, grid, self._fit_apertures)

    def _check_sigma_ceiling(self) -> None:
        """Check sampled circular-field mass at the centered sigma ceiling.

        Off-center fields can still lose mass. This guard does not establish size bias.
        """
        _, sigma_high = self.config.sigma_bounds
        weights = GaussianReceptiveField(0.0, 0.0, sigma_high).weights(self.grid)
        volume = float(weights[self.grid.field_mask].sum())
        if volume < MIN_ON_GRID_VOLUME:
            raise ConfigError(
                f"upper sigma bound {sigma_high:g} px keeps only {volume:.3f} of its unit "
                f"volume inside a {self.grid.resolution} px field, under the "
                f"{MIN_ON_GRID_VOLUME} tolerance for a centered Gaussian. "
                "This exceeds the sampled-mass design limit. Lower the bound to about "
                f"{math.floor(self.grid.resolution / RESOLUTION_PER_SIGMA)} px, or raise the "
                f"grid resolution to about {_even_at_least(sigma_high * RESOLUTION_PER_SIGMA)} px."
            )

    @property
    def apertures(self) -> FloatArray:
        """Read-only stimulus snapshot used by cached and refined predictions."""
        return self._apertures.view()

    @property
    def candidates(self) -> tuple[GaussianReceptiveField, ...]:
        """Candidate geometries in the same order as their cached predictions."""
        return self._candidates

    @property
    def n_frames(self) -> int:
        return len(self.apertures)

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Square search box for centers, with the configured sigma range."""
        radius = self.grid.radius
        sigma_low, sigma_high = self.config.sigma_bounds
        return (-radius, -radius, sigma_low), (radius, radius, sigma_high)

    def _build_candidates(self) -> list[GaussianReceptiveField]:
        radius = self.grid.radius
        sigma_low, sigma_high = self.config.sigma_bounds
        n_sigma = max(3, self.config.grid_size // 2)

        centres = np.linspace(-radius, radius, self.config.grid_size)
        sigmas = np.geomspace(sigma_low, sigma_high, n_sigma)

        return [
            GaussianReceptiveField(float(x), float(y), float(sigma))
            for x in centres
            for y in centres
            if np.hypot(x, y) <= radius
            for sigma in sigmas
        ]

    def _bounds_hit(self, x0: float, y0: float, sigma: float) -> tuple[bool, bool, bool]:
        """Flag each fitted parameter within the tolerance of either search bound."""
        lower, upper = self.bounds
        pinned = [
            abs(value - low) <= BOUND_TOLERANCE or abs(value - high) <= BOUND_TOLERANCE
            for value, low, high in zip((x0, y0, sigma), lower, upper)
        ]
        return pinned[0], pinned[1], pinned[2]

    def _second_field_r2(self, response: FloatArray, fitted: FloatArray) -> float:
        """Largest incremental R2 from a cached candidate after orthogonalization.

        Project candidates against the fitted prediction and intercept; no second search runs.
        Constant targets and non-finite inputs return NaN.
        """
        if (
            not np.isfinite(response).all()
            or not np.isfinite(fitted).all()
            or np.all(response == response[0])
        ):
            return float("nan")
        scale = max(float(np.max(np.abs(response))), float(np.max(np.abs(fitted))))
        response = response / scale
        fitted = fitted / scale
        total = float(np.sum((response - response.mean()) ** 2))
        if total <= 0.0:
            return float("nan")
        residual = response - fitted
        design = np.column_stack([fitted, np.ones_like(fitted)])
        coefficients, *_ = np.linalg.lstsq(design, self._predictions, rcond=None)
        perpendicular = self._predictions - design @ coefficients
        norms = np.sum(perpendicular**2, axis=0)
        usable = norms > 1e-12
        if not usable.any():
            return 0.0
        gains = (residual @ perpendicular[:, usable]) ** 2 / (norms[usable] * total)
        return float(np.max(gains))

    def _coarse_search(self, response: FloatArray) -> GaussianReceptiveField:
        best_index, best_r2 = 0, -np.inf
        for index in range(self._predictions.shape[1]):
            _, fitted = _solve_amplitude(self._predictions[:, index], response)
            score = _r_squared(response, fitted)
            if score > best_r2:
                best_index, best_r2 = index, score
        return self.candidates[best_index]

    def fit_unit(self, response: FloatArray) -> UnitFit:
        response = np.asarray(response, dtype=np.float64)
        if response.shape != (self.n_frames,):
            raise ValueError("response must have one value per stimulus frame")
        if not np.isfinite(response).all() or np.all(response == response[0]):
            return UnitFit.failed()

        # Keep optimizer tolerances independent of response units and baseline.
        normalized_response: FloatArray = response / np.max(np.abs(response))
        normalized_response -= normalized_response.mean()
        normalized_response /= normalized_response.std()
        start = self._coarse_search(normalized_response)

        def residual(params: Sequence[float]) -> FloatArray:
            x0, y0, sigma = params
            field = GaussianReceptiveField(float(x0), float(y0), float(sigma))
            _, fitted = _solve_amplitude(
                predict(field.weights(self.grid), self._fit_apertures), normalized_response
            )
            return fitted - normalized_response

        try:
            solution = least_squares(
                residual,
                x0=[start.x0, start.y0, start.sigma],
                bounds=self.bounds,
                max_nfev=self.config.max_nfev,
            )
        except (ValueError, np.linalg.LinAlgError):
            return UnitFit.failed()

        x0, y0, sigma = (float(value) for value in solution.x)
        field = GaussianReceptiveField(x0, y0, sigma)
        coefficients, fitted = _solve_amplitude(
            predict(field.weights(self.grid), self._fit_apertures),
            response,
            aperture_scale=self._aperture_scale,
        )
        if not np.isfinite(coefficients).all() or not np.isfinite(fitted).all():
            return UnitFit.failed()
        score = _r_squared(response, fitted)
        # Residual and Jacobian scaling cancel in the geometry covariance.
        se_x0, se_y0, se_sigma = _standard_errors(
            solution.jac, float(solution.cost), self.n_independent_frames
        )
        x0_at_bound, y0_at_bound, sigma_at_bound = self._bounds_hit(x0, y0, sigma)

        return UnitFit(
            x0=x0,
            y0=y0,
            sigma=sigma,
            beta=float(coefficients[0]),
            baseline=float(coefficients[1]),
            r2=score,
            converged=bool(solution.success),
            n_fev=int(solution.nfev),
            se_x0=se_x0,
            se_y0=se_y0,
            se_sigma=se_sigma,
            second_field_r2=self._second_field_r2(response, fitted),
            x0_at_bound=x0_at_bound,
            y0_at_bound=y0_at_bound,
            sigma_at_bound=sigma_at_bound,
            dof=self.n_independent_frames - N_PARAMETERS,
            r2_threshold=self.config.r2_threshold,
        )

    def fit_all(self, activations: FloatArray) -> list[UnitFit]:
        """Fit every unit in a ``(frames, units)`` activation matrix."""
        activations = np.asarray(activations, dtype=np.float64)
        if activations.ndim != 2:
            raise ValueError("activations must have shape (frames, units)")
        if activations.shape[0] != self.n_frames:
            raise ValueError("activations must have one row per stimulus frame")
        return [self.fit_unit(response) for response in activations.T]
